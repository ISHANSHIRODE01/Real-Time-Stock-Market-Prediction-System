"""
LSTM training with leakage-free preprocessing and honest baselines.

Three defects in the previous version of this script are fixed here.

1. **Scaler leakage.** `StandardScaler().fit_transform(df[features])` was
   called on the entire series *before* any split, so the mean and variance of
   the test period leaked into training. The scaler is now fit on the training
   split only and applied to val/test.

2. **No split.** Every window was used for training and loss was reported on
   the training set, so the printed number said nothing about generalisation.
   There is now a chronological 70/15/15 train/val/test split — no shuffling
   across time, which would let the model interpolate between neighbouring
   days.

3. **Predicting the price level.** The old target was the scaled `Close`,
   which is near-perfectly autocorrelated: predicting "tomorrow == today"
   scores an excellent MSE and R^2 while being worthless. The target is now
   the **next-day return**, and a persistence baseline is always reported.

On what to expect
-----------------
Next-day equity returns are close to unpredictable from technical indicators
alone. A directional accuracy near 50% and an R^2 near or below zero is the
normal, correct result — not a bug. This script reports that honestly rather
than reframing it. A negative R^2 means the model is worse than predicting the
mean, which is genuinely common here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.lstm_model import StockLSTM  # noqa: E402

FEATURES = ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Volume"]


@dataclass
class TrainConfig:
    ticker: str = "AAPL"
    seq_len: int = 60
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    hidden_dim: int = 64
    patience: int = 8
    train_frac: float = 0.70
    val_frac: float = 0.15
    seed: int = 42


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_windows(feats: np.ndarray, target: np.ndarray, seq_len: int):
    """Build sliding windows. Window ending at t predicts target[t]."""
    X, y = [], []
    for i in range(seq_len, len(feats)):
        X.append(feats[i - seq_len : i])
        y.append(target[i])
    return np.asarray(X), np.asarray(y)


def chronological_split(n: int, train_frac: float, val_frac: float):
    tr = int(n * train_frac)
    va = int(n * (train_frac + val_frac))
    return slice(0, tr), slice(tr, va), slice(va, n)


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """Return regression and directional metrics on a loader."""
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).squeeze(-1).cpu().numpy().ravel())
            actuals.append(yb.cpu().numpy().ravel())

    p = np.concatenate(preds)
    a = np.concatenate(actuals)

    mse = float(np.mean((p - a) ** 2))
    mae = float(np.mean(np.abs(p - a)))
    var = float(np.var(a))
    r2 = float(1.0 - mse / var) if var > 0 else float("nan")
    # Directional accuracy: does the sign match? This is what a trader cares
    # about, and it is the metric a coin flip sets at ~50%.
    dir_acc = float(np.mean(np.sign(p) == np.sign(a)) * 100)
    return {"mse": mse, "mae": mae, "r2": r2, "directional_accuracy_pct": dir_acc}


def persistence_baseline(y: np.ndarray) -> dict[str, float]:
    """Predict "next return == previous return".

    Any model that cannot beat this has learned nothing useful.
    """
    pred = np.concatenate([[0.0], y[:-1]])
    mse = float(np.mean((pred - y) ** 2))
    var = float(np.var(y))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(pred - y))),
        "r2": float(1.0 - mse / var) if var > 0 else float("nan"),
        "directional_accuracy_pct": float(np.mean(np.sign(pred) == np.sign(y)) * 100),
    }


def zero_baseline(y: np.ndarray) -> dict[str, float]:
    """Predict zero return every day — the unconditional mean is ~0."""
    mse = float(np.mean(y**2))
    var = float(np.var(y))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(y))),
        "r2": float(1.0 - mse / var) if var > 0 else float("nan"),
        "directional_accuracy_pct": float("nan"),
    }


def train_model(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    path = Path(f"data/processed/{cfg.ticker}_cleaned.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            f"  python src/data_pipeline/fetch_stock_data.py\n"
            f"  python src/data_pipeline/preprocess_data.py"
        )

    df = pd.read_csv(path)
    df.columns = [str(c) for c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; found {list(df.columns)}")

    # Target: next-day simple return, not the price level.
    df["target_return"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df = df.dropna(subset=FEATURES + ["target_return"]).reset_index(drop=True)

    feats_raw = df[FEATURES].to_numpy(dtype=float)
    target = df["target_return"].to_numpy(dtype=float)

    n = len(df)
    tr_s, va_s, te_s = chronological_split(n, cfg.train_frac, cfg.val_frac)
    print(f"[*] {n} rows -> train {tr_s.stop - tr_s.start} | "
          f"val {va_s.stop - va_s.start} | test {te_s.stop - te_s.start} (chronological)")

    # Fit the scaler on training rows ONLY, then apply to all splits.
    scaler = StandardScaler().fit(feats_raw[tr_s])
    feats = scaler.transform(feats_raw)

    def build(sl: slice, seq_len: int):
        # Include seq_len rows of lead-in so the first window of val/test is
        # complete, without letting their targets into an earlier split.
        start = max(sl.start - seq_len, 0)
        Xw, yw = make_windows(feats[start : sl.stop], target[start : sl.stop], seq_len)
        drop = sl.start - start
        return Xw[drop:] if drop else Xw, yw[drop:] if drop else yw

    X_tr, y_tr = make_windows(feats[tr_s], target[tr_s], cfg.seq_len)
    X_va, y_va = build(va_s, cfg.seq_len)
    X_te, y_te = build(te_s, cfg.seq_len)

    if min(len(X_tr), len(X_va), len(X_te)) == 0:
        raise ValueError(f"a split produced no windows; seq_len={cfg.seq_len} too large for {n} rows")

    # shuffle=True is safe here: windows are already built, so no future row
    # can enter a past window. Ordering only affects gradient noise.
    tr_loader = DataLoader(WindowDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(WindowDataset(X_va, y_va), batch_size=cfg.batch_size)
    te_loader = DataLoader(WindowDataset(X_te, y_te), batch_size=cfg.batch_size)

    model = StockLSTM(input_dim=len(FEATURES))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()

        tr_loss = running / max(len(tr_loader), 1)
        val = evaluate(model, va_loader)
        history.append({"epoch": epoch, "train_mse": tr_loss, "val_mse": val["mse"]})
        print(f"    epoch {epoch:3d}/{cfg.epochs} | train {tr_loss:.3e} | "
              f"val {val['mse']:.3e} | val dir-acc {val['directional_accuracy_pct']:.1f}%")

        # Early stopping on validation loss keeps the test split untouched
        # during model selection.
        if val["mse"] < best_val - 1e-12:
            best_val, bad_epochs = val["mse"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"[*] early stop at epoch {epoch} (no val improvement in {cfg.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    results = {
        "config": asdict(cfg),
        "test": evaluate(model, te_loader),
        "val": evaluate(model, va_loader),
        "baselines": {
            "persistence": persistence_baseline(y_te),
            "predict_zero": zero_baseline(y_te),
        },
        "history": history,
        "n_test_windows": int(len(X_te)),
    }

    os.makedirs("models", exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(cfg), "features": FEATURES},
        f"models/lstm_{cfg.ticker}.pt",
    )
    import joblib

    joblib.dump(scaler, f"models/scaler_{cfg.ticker}.joblib")

    os.makedirs("reports", exist_ok=True)
    with open(f"reports/lstm_{cfg.ticker}_metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    t, b = results["test"], results["baselines"]
    print("\n" + "=" * 66)
    print(f"TEST RESULTS — {cfg.ticker} (next-day return)")
    print("=" * 66)
    print(f"{'model':<22}{'MSE':>14}{'R^2':>10}{'dir-acc':>12}")
    print("-" * 66)
    print(f"{'LSTM':<22}{t['mse']:>14.3e}{t['r2']:>10.3f}{t['directional_accuracy_pct']:>11.1f}%")
    print(f"{'persistence':<22}{b['persistence']['mse']:>14.3e}"
          f"{b['persistence']['r2']:>10.3f}{b['persistence']['directional_accuracy_pct']:>11.1f}%")
    print(f"{'predict zero':<22}{b['predict_zero']['mse']:>14.3e}{b['predict_zero']['r2']:>10.3f}{'n/a':>12}")
    print("-" * 66)
    beat_zero = t["mse"] < b["predict_zero"]["mse"]
    print(f"beats predict-zero: {'YES' if beat_zero else 'NO'}")
    if t["r2"] < 0:
        print("R^2 < 0: worse than predicting the mean. Expected for next-day\n"
              "returns from technical features alone — reported, not hidden.")
    print("=" * 66)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Train LSTM on next-day returns")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()
    train_model(TrainConfig(ticker=a.ticker, epochs=a.epochs, seq_len=a.seq_len,
                            batch_size=a.batch_size, lr=a.lr))


if __name__ == "__main__":
    main()
