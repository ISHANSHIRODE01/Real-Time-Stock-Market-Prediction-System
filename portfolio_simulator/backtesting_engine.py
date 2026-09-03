"""
Walk-forward backtesting engine.

Replaces an earlier stub whose `compute_roi` returned
`initial_capital * 1.082` — a hardcoded 8.2% that was independent of any
model, data, or trade.

What this engine does differently
---------------------------------
1. **Walk-forward, not single split.** The model is refit on an expanding
   window and only ever predicts the immediately following out-of-sample
   block. Nothing from the future reaches the fitting step.
2. **Costs are charged.** Commission and slippage are applied on every
   transition. Ignoring them is what turns most "profitable" retail
   backtests into losers.
3. **Execution lags the signal.** A signal computed from bar `t`'s close
   executes at bar `t+1`'s open. Trading on the close you used to decide is
   look-ahead bias.
4. **A baseline is always reported.** Strategy return is meaningless without
   buy-and-hold on the same window over the same period. A strategy that
   returns +20% while the asset returned +45% has destroyed value.

Metrics reported: total & annualised return, Sharpe, Sortino, max drawdown,
Calmar, hit rate, turnover, and the same for the baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    """Cost and sizing assumptions. Defaults model a retail equity account."""

    initial_capital: float = 100_000.0
    commission_bps: float = 5.0        # 0.05% per side
    slippage_bps: float = 5.0          # 0.05% adverse fill per side
    train_window: int = 500            # min bars before first prediction
    test_window: int = 20              # bars predicted per refit
    max_position: float = 1.0          # 1.0 = fully invested, no leverage
    allow_short: bool = False
    risk_free_rate: float = 0.04       # annual, for Sharpe

    @property
    def cost_per_side(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: int
    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["", "=" * 62, "WALK-FORWARD BACKTEST", "=" * 62]
        lines.append(f"{'metric':<26}{'strategy':>16}{'buy & hold':>18}")
        lines.append("-" * 62)
        for key, label, fmt in [
            ("total_return_pct", "total return", "{:+.2f}%"),
            ("annualised_return_pct", "annualised return", "{:+.2f}%"),
            ("volatility_pct", "volatility (ann.)", "{:.2f}%"),
            ("sharpe", "Sharpe", "{:.2f}"),
            ("sortino", "Sortino", "{:.2f}"),
            ("max_drawdown_pct", "max drawdown", "{:.2f}%"),
            ("calmar", "Calmar", "{:.2f}"),
        ]:
            s = self.metrics.get(key)
            b = self.baseline_metrics.get(key)
            s_str = fmt.format(s) if isinstance(s, (int, float)) else "n/a"
            b_str = fmt.format(b) if isinstance(b, (int, float)) else "n/a"
            lines.append(f"{label:<26}{s_str:>16}{b_str:>18}")
        lines.append("-" * 62)
        lines.append(f"{'hit rate':<26}{self.metrics.get('hit_rate_pct', float('nan')):>15.2f}%")
        lines.append(f"{'trades':<26}{self.trades:>16d}")
        lines.append(f"{'turnover (ann.)':<26}{self.metrics.get('annual_turnover', float('nan')):>16.2f}")
        lines.append("=" * 62)
        excess = self.metrics.get("total_return_pct", 0) - self.baseline_metrics.get("total_return_pct", 0)
        verdict = "OUTPERFORMED" if excess > 0 else "UNDERPERFORMED"
        lines.append(f"vs buy & hold: {excess:+.2f} pp  ->  {verdict}")
        lines.append("=" * 62)
        return "\n".join(lines)


# --- metrics --------------------------------------------------------------

def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    return float(((equity - running_max) / running_max).min())


def compute_metrics(returns: pd.Series, equity: pd.Series, rf: float = 0.04) -> dict[str, Any]:
    """Standard performance statistics from a daily return series."""
    r = returns.dropna()
    if len(r) < 2:
        return {"error": "insufficient data", "n_periods": len(r)}

    n = len(r)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = n / TRADING_DAYS
    ann_ret = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))

    daily_rf = rf / TRADING_DAYS
    excess = r - daily_rf
    sharpe = float(excess.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else float("nan")

    downside = r[r < 0]
    sortino = (
        float(excess.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else float("nan")
    )

    mdd = _max_drawdown(equity)
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 and not np.isnan(ann_ret) else float("nan")

    return {
        "total_return_pct": total * 100,
        "annualised_return_pct": ann_ret * 100,
        "volatility_pct": vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": mdd * 100,
        "calmar": calmar,
        "hit_rate_pct": float((r > 0).mean() * 100),
        "n_periods": n,
    }


# --- engine ---------------------------------------------------------------

class WalkForwardBacktester:
    """Expanding-window walk-forward backtester.

    `model_factory` must return an object exposing `fit(X, y)` and
    `predict(X)`. A fresh instance is built for every fold, so no state — and
    no fitted scaler — leaks across the boundary.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def _generate_folds(self, n: int) -> list[tuple[int, int, int]]:
        """Yield (train_start, train_end, test_end) index triples."""
        cfg = self.config
        folds: list[tuple[int, int, int]] = []
        train_end = cfg.train_window
        while train_end < n:
            test_end = min(train_end + cfg.test_window, n)
            if test_end - train_end < 1:
                break
            folds.append((0, train_end, test_end))
            train_end = test_end
        return folds

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        model_factory: Callable[[], Any],
        price_col: str = "Close",
        exec_col: str | None = "Open",
        target_horizon: int = 1,
    ) -> BacktestResult:
        """Execute the walk-forward simulation.

        The target is the forward return over `target_horizon` bars. The model
        predicts it; the sign of the prediction sets the position. Position for
        bar t+1 is decided from information available at bar t only.
        """
        cfg = self.config
        data = df.copy().reset_index(drop=True)

        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        if price_col not in data.columns:
            raise ValueError(f"missing price column: {price_col}")

        # Forward return is the prediction target.
        data["fwd_return"] = data[price_col].shift(-target_horizon) / data[price_col] - 1.0

        # Realised return actually earned per bar. If an execution column is
        # supplied we assume fills at the next bar's open.
        if exec_col and exec_col in data.columns:
            data["realised_return"] = data[exec_col].shift(-1) / data[exec_col] - 1.0
        else:
            data["realised_return"] = data[price_col].pct_change().shift(-1)

        n = len(data)
        folds = self._generate_folds(n)
        if not folds:
            raise ValueError(
                f"not enough data: {n} bars for train_window={cfg.train_window}. "
                "Reduce train_window or fetch a longer history."
            )
        logger.info("Running %d walk-forward folds over %d bars", len(folds), n)

        raw_positions = pd.Series(0.0, index=data.index)

        for train_start, train_end, test_end in folds:
            train = data.iloc[train_start:train_end].dropna(subset=list(feature_cols) + ["fwd_return"])
            if len(train) < 50:
                continue

            X_tr = train[list(feature_cols)].to_numpy(dtype=float)
            y_tr = train["fwd_return"].to_numpy(dtype=float)

            test = data.iloc[train_end:test_end]
            valid = test.dropna(subset=list(feature_cols))
            if valid.empty:
                continue

            model = model_factory()          # fresh per fold: no leakage
            model.fit(X_tr, y_tr)
            preds = np.asarray(model.predict(valid[list(feature_cols)].to_numpy(dtype=float))).ravel()

            for idx, pred in zip(valid.index, preds):
                if pred > 0:
                    raw_positions.loc[idx] = cfg.max_position
                elif cfg.allow_short:
                    raw_positions.loc[idx] = -cfg.max_position
                else:
                    raw_positions.loc[idx] = 0.0

        # Signal from bar t is acted on at t+1.
        positions = raw_positions.shift(1).fillna(0.0)

        realised = data["realised_return"].fillna(0.0)
        gross = positions * realised

        turn = positions.diff().abs().fillna(positions.abs())
        costs = turn * cfg.cost_per_side
        net = gross - costs

        first = folds[0][1]
        net = net.iloc[first:]
        positions_out = positions.iloc[first:]

        equity = cfg.initial_capital * (1.0 + net).cumprod()
        trades = int((positions_out.diff().abs() > 1e-9).sum())

        metrics = compute_metrics(net, equity, cfg.risk_free_rate)
        years = max(len(net) / TRADING_DAYS, 1e-9)
        metrics["annual_turnover"] = float(turn.iloc[first:].sum() / years)
        metrics["total_costs_pct"] = float(costs.iloc[first:].sum() * 100)
        metrics["final_equity"] = float(equity.iloc[-1])
        metrics["time_in_market_pct"] = float((positions_out.abs() > 0).mean() * 100)

        bh_ret = realised.iloc[first:]
        bh_equity = cfg.initial_capital * (1.0 + bh_ret).cumprod()
        baseline = compute_metrics(bh_ret, bh_equity, cfg.risk_free_rate)

        if "date" in data.columns:
            idx = pd.to_datetime(data["date"].iloc[first:], utc=True)
            equity.index = idx
            net.index = idx
            positions_out.index = idx

        return BacktestResult(
            equity_curve=equity,
            returns=net,
            positions=positions_out,
            trades=trades,
            metrics=metrics,
            baseline_metrics=baseline,
        )


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_pipeline.fetch_stock_data import fetch_data
    from src.data_pipeline.preprocess_data import calculate_technical_indicators

    raw = fetch_data("AAPL", period="5y")
    df = calculate_technical_indicators(raw)
    df.columns = [str(c) for c in df.columns]
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

    features = ["SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal", "Volume"]
    bt = WalkForwardBacktester(BacktestConfig(train_window=500, test_window=20))
    res = bt.run(
        df,
        features,
        lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    )
    print(res.summary())
