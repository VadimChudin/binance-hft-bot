import numpy as np
import pandas as pd

from hftbot.research.features import build_features
from hftbot.research.labels import triple_barrier
from hftbot.research.trend import TrendConfig, backtest_trend


def _synth_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    price = 100 + np.cumsum(rng.normal(0, 0.1, n))
    high = price + np.abs(rng.normal(0, 0.05, n))
    low = price - np.abs(rng.normal(0, 0.05, n))
    vol = np.abs(rng.normal(100, 10, n))
    return pd.DataFrame(
        {
            "open": price, "high": high, "low": low, "close": price,
            "volume": vol, "close_time": 0, "quote_volume": vol * price,
            "count": rng.integers(10, 100, n),
            "taker_buy_volume": vol * rng.uniform(0.3, 0.7, n),
            "taker_buy_quote_volume": 0, "symbol": "X",
        },
        index=idx,
    )


def test_features_causal_and_finite():
    df = _synth_df()
    feat_df, feats = build_features(df)
    assert len(feats) > 20
    # Feature values must be defined (non-NaN) once warmed up.
    tail = feat_df[feats].iloc[200:]
    assert tail.notna().mean().mean() > 0.9


def test_triple_barrier_labels():
    idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
    # Strong up move: upper barrier should be hit first.
    close = pd.Series([100, 100, 100, 100, 105, 105, 105, 105, 105, 105], index=idx)
    df = pd.DataFrame({"high": close + 1, "low": close - 0.01, "close": close}, index=idx)
    lab = triple_barrier(df, barrier=0.02, horizon=5)
    assert lab["label"].iloc[0] == 1.0


def test_trend_backtest_profits_on_uptrend():
    n = 3000
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    price = 100 * (1.0002 ** np.arange(n))  # steady uptrend
    df = pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": 1.0},
        index=idx,
    )
    cfg = TrendConfig(ema_fast=24, ema_slow=168, vol_window=168, cost_per_turn=0.0)
    res = backtest_trend(df, cfg, "X")
    assert res.total_return > 0
    assert res.sharpe > 0
