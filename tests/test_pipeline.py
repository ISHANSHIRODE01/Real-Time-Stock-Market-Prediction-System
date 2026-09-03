"""
Behavioural tests for the data pipeline, sentiment aggregation, and backtester.

These replace an earlier file whose tests only asserted that directories and
`requirements.txt` existed — which pass on any repo with the right folder names
and verify nothing about correctness.

Nothing here needs network access or model downloads: news retrieval and FinBERT
scoring are exercised through synthetic `Headline` / `ScoredHeadline` objects.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_simulator.backtesting_engine import (  # noqa: E402
    BacktestConfig,
    WalkForwardBacktester,
    _max_drawdown,
    compute_metrics,
)
from sentiment_analysis.news_sentiment import (  # noqa: E402
    Headline,
    ScoredHeadline,
    aggregate_sentiment,
    build_daily_sentiment_series,
)
from src.data_pipeline.preprocess_data import calculate_technical_indicators  # noqa: E402


# --- fixtures -------------------------------------------------------------

@pytest.fixture
def price_frame() -> pd.DataFrame:
    """400 bars of synthetic OHLCV with a mild upward drift."""
    rng = np.random.default_rng(7)
    n = 400
    returns = rng.normal(0.0004, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods=n, freq="B"),
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        }
    )


def _headline(hours_ago: float, label: str, conf: float) -> ScoredHeadline:
    sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}[label]
    return ScoredHeadline(
        title=f"{label} headline {hours_ago}h ago",
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        source="test",
        label=label,
        confidence=conf,
        signed_score=conf * sign,
    )


# --- technical indicators -------------------------------------------------

class TestTechnicalIndicators:
    def test_adds_expected_columns(self, price_frame):
        out = calculate_technical_indicators(price_frame)
        for col in ("SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal"):
            assert col in out.columns

    def test_rsi_stays_in_bounds(self, price_frame):
        rsi = calculate_technical_indicators(price_frame)["RSI"]
        assert rsi.between(0, 100).all(), "RSI must lie in [0, 100]"

    def test_sma20_matches_manual_rolling_mean(self, price_frame):
        out = calculate_technical_indicators(price_frame)
        expected = price_frame["Close"].rolling(20).mean().loc[out.index]
        assert np.allclose(out["SMA_20"].to_numpy(), expected.to_numpy(), rtol=1e-9)

    def test_macd_is_fast_minus_slow_ema(self, price_frame):
        out = calculate_technical_indicators(price_frame)
        c = price_frame["Close"]
        expected = (
            c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        ).loc[out.index]
        assert np.allclose(out["MACD"].to_numpy(), expected.to_numpy(), rtol=1e-9)

    def test_no_nans_survive(self, price_frame):
        out = calculate_technical_indicators(price_frame)
        assert not out[["SMA_20", "SMA_50", "RSI", "MACD"]].isna().any().any()

    def test_rsi_saturates_high_on_monotonic_rise(self):
        df = pd.DataFrame({"Close": np.arange(100, 200, dtype=float), "Volume": np.ones(100)})
        out = calculate_technical_indicators(df)
        # Every period is a gain, so average loss is ~0 and RSI pins near 100.
        assert out["RSI"].iloc[-1] > 99


# --- sentiment aggregation ------------------------------------------------

class TestSentimentAggregation:
    def test_no_headlines_returns_none_not_zero(self):
        """Critical: absence of news must not read as neutral news."""
        res = aggregate_sentiment([])
        assert res["score"] is None
        assert res["n_headlines"] == 0

    def test_all_positive_gives_positive_score(self):
        res = aggregate_sentiment([_headline(1, "positive", 0.9), _headline(2, "positive", 0.8)])
        assert res["score"] > 0.5

    def test_all_negative_gives_negative_score(self):
        res = aggregate_sentiment([_headline(1, "negative", 0.9), _headline(2, "negative", 0.8)])
        assert res["score"] < -0.5

    def test_score_bounded_in_unit_interval(self):
        heads = [_headline(h, "positive", 1.0) for h in range(1, 20)]
        assert -1.0 <= aggregate_sentiment(heads)["score"] <= 1.0

    def test_recent_news_outweighs_stale_news(self):
        """A fresh negative headline must dominate a week-old positive one."""
        res = aggregate_sentiment(
            [_headline(1, "negative", 0.95), _headline(168, "positive", 0.95)],
            half_life_hours=24.0,
        )
        assert res["score"] < 0

    def test_half_life_controls_decay(self):
        heads = [_headline(1, "negative", 0.9), _headline(72, "positive", 0.9)]
        fast = aggregate_sentiment(heads, half_life_hours=6.0)["score"]
        slow = aggregate_sentiment(heads, half_life_hours=500.0)["score"]
        # A long half-life nearly equal-weights them, pulling the score up toward 0.
        assert fast < slow

    def test_distribution_counts_labels(self):
        res = aggregate_sentiment(
            [
                _headline(1, "positive", 0.9),
                _headline(2, "negative", 0.8),
                _headline(3, "neutral", 0.7),
            ]
        )
        assert res["distribution"] == {"positive": 1, "negative": 1, "neutral": 1}

    def test_daily_series_groups_by_date(self):
        heads = [
            _headline(2, "positive", 0.9),
            _headline(3, "positive", 0.7),
            _headline(50, "negative", 0.8),
        ]
        series = build_daily_sentiment_series(heads)
        assert not series.empty
        assert {"sentiment", "n_headlines"} <= set(series.columns)
        assert series["n_headlines"].sum() == 3

    def test_daily_series_empty_input(self):
        assert build_daily_sentiment_series([]).empty

    def test_headline_roundtrip_preserves_timestamp(self):
        h = Headline("t", datetime(2025, 1, 2, 3, 4, tzinfo=timezone.utc), "src")
        assert h.to_dict()["published_at"].startswith("2025-01-02T03:04")


# --- metrics --------------------------------------------------------------

class TestMetrics:
    def test_max_drawdown_of_monotonic_rise_is_zero(self):
        assert _max_drawdown(pd.Series([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)

    def test_max_drawdown_known_value(self):
        # peak 200 -> trough 100 is exactly -50%
        assert _max_drawdown(pd.Series([100.0, 200.0, 100.0, 150.0])) == pytest.approx(-0.5)

    def test_zero_volatility_does_not_crash(self):
        m = compute_metrics(pd.Series([0.0] * 30), pd.Series(np.ones(30) * 100))
        assert np.isnan(m["sharpe"]) or m["sharpe"] == 0

    def test_insufficient_data_reports_error(self):
        assert "error" in compute_metrics(pd.Series([0.01]), pd.Series([100.0, 101.0]))

    def test_total_return_matches_equity_endpoints(self):
        eq = pd.Series([100.0, 110.0, 121.0])
        assert compute_metrics(eq.pct_change(), eq)["total_return_pct"] == pytest.approx(21.0, abs=1e-6)


# --- backtester -----------------------------------------------------------

class _AlwaysLong:
    """Predicts a constant positive return -> always long."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X))


class _AlwaysFlat:
    """Predicts a constant negative return -> always flat (long-only)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return -np.ones(len(X))


class TestBacktester:
    FEATURES = ["SMA_20", "SMA_50", "RSI", "MACD", "Volume"]

    def _prepared(self, price_frame):
        return calculate_technical_indicators(price_frame)

    def test_raises_when_history_too_short(self, price_frame):
        bt = WalkForwardBacktester(BacktestConfig(train_window=10_000))
        with pytest.raises(ValueError, match="not enough data"):
            bt.run(self._prepared(price_frame), self.FEATURES, _AlwaysLong)

    def test_raises_on_missing_feature_column(self, price_frame):
        bt = WalkForwardBacktester(BacktestConfig(train_window=200, test_window=20))
        with pytest.raises(ValueError, match="missing feature columns"):
            bt.run(self._prepared(price_frame), ["does_not_exist"], _AlwaysLong)

    def test_always_flat_strategy_never_trades(self, price_frame):
        bt = WalkForwardBacktester(
            BacktestConfig(train_window=200, test_window=20, allow_short=False)
        )
        res = bt.run(self._prepared(price_frame), self.FEATURES, _AlwaysFlat)
        assert res.trades == 0
        assert res.positions.abs().sum() == pytest.approx(0.0)

    def test_flat_strategy_preserves_capital_exactly(self, price_frame):
        """No position, no costs -> equity must not move."""
        cfg = BacktestConfig(train_window=200, test_window=20)
        res = WalkForwardBacktester(cfg).run(self._prepared(price_frame), self.FEATURES, _AlwaysFlat)
        assert res.equity_curve.iloc[-1] == pytest.approx(cfg.initial_capital, rel=1e-9)

    def test_costs_reduce_returns(self, price_frame):
        """Same signals, higher fees -> strictly worse outcome."""
        df = self._prepared(price_frame)
        cheap = WalkForwardBacktester(
            BacktestConfig(train_window=200, test_window=20, commission_bps=0, slippage_bps=0)
        ).run(df, self.FEATURES, _AlwaysLong)
        pricey = WalkForwardBacktester(
            BacktestConfig(train_window=200, test_window=20, commission_bps=50, slippage_bps=50)
        ).run(df, self.FEATURES, _AlwaysLong)
        assert pricey.metrics["total_return_pct"] < cheap.metrics["total_return_pct"]

    def test_baseline_is_always_reported(self, price_frame):
        res = WalkForwardBacktester(BacktestConfig(train_window=200, test_window=20)).run(
            self._prepared(price_frame), self.FEATURES, _AlwaysLong
        )
        assert "total_return_pct" in res.baseline_metrics
        assert res.baseline_metrics["n_periods"] == res.metrics["n_periods"]

    def test_always_long_approximates_buy_and_hold(self, price_frame):
        """With zero costs, always-long should track the benchmark closely."""
        res = WalkForwardBacktester(
            BacktestConfig(train_window=200, test_window=20, commission_bps=0, slippage_bps=0)
        ).run(self._prepared(price_frame), self.FEATURES, _AlwaysLong)
        assert res.metrics["total_return_pct"] == pytest.approx(
            res.baseline_metrics["total_return_pct"], rel=0.05
        )

    def test_folds_are_contiguous_and_expanding(self):
        bt = WalkForwardBacktester(BacktestConfig(train_window=100, test_window=25))
        folds = bt._generate_folds(200)
        assert folds[0] == (0, 100, 125)
        for prev, nxt in zip(folds, folds[1:]):
            assert nxt[0] == 0          # always trains from the start
            assert nxt[1] == prev[2]    # contiguous, no gaps or overlap

    def test_summary_renders(self, price_frame):
        res = WalkForwardBacktester(BacktestConfig(train_window=200, test_window=20)).run(
            self._prepared(price_frame), self.FEATURES, _AlwaysLong
        )
        text = res.summary()
        assert "buy & hold" in text
        assert "WALK-FORWARD BACKTEST" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
