"""Feature engineering for the ML strategy.

All features are strictly causal (computed from the current and past bars only)
so they can be reproduced live. Includes classic technical features plus
order-flow / microstructure features derived from taker-buy volume and trade
counts, and a few custom indicators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / sum(|moves|). Trend vs chop."""
    net = close.diff(window).abs()
    noise = close.diff().abs().rolling(window).sum()
    return net / noise.replace(0.0, np.nan)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (df_with_features, feature_column_names)."""
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    vol = out["volume"].replace(0.0, np.nan)

    feats: list[str] = []

    def add(name: str, series: pd.Series) -> None:
        out[name] = series
        feats.append(name)

    logret = np.log(close).diff()
    # Multi-horizon returns and their volatility.
    for h in (1, 3, 5, 15, 30, 60):
        add(f"ret_{h}", close.pct_change(h))
    for w in (5, 15, 30, 60):
        add(f"vol_{w}", logret.rolling(w).std())

    # Volatility regime: short vs long realized vol.
    add("vol_regime", logret.rolling(10).std() / logret.rolling(60).std())

    # EMA structure (momentum).
    for span in (9, 21, 50):
        ema = close.ewm(span=span, adjust=False).mean()
        add(f"ema_dist_{span}", close / ema - 1.0)
    add("ema_cross_9_21",
        close.ewm(span=9, adjust=False).mean() / close.ewm(span=21, adjust=False).mean() - 1.0)

    # RSI at two speeds.
    add("rsi_14", _rsi(close, 14) / 100.0)
    add("rsi_7", _rsi(close, 7) / 100.0)

    # MACD.
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    add("macd", macd / close)
    add("macd_hist", (macd - macd_sig) / close)

    # Bollinger position (z-score of price vs rolling mean).
    for w in (20, 60):
        mean = close.rolling(w).mean()
        std = close.rolling(w).std()
        add(f"bb_z_{w}", (close - mean) / std.replace(0.0, np.nan))

    # Bar-shape / range features.
    rng = (high - low) / close
    add("bar_range", rng)
    add("close_pos", (close - low) / (high - low).replace(0.0, np.nan))
    add("parkinson_vol", (np.log(high / low) ** 2).rolling(15).mean())

    # Breakout distance from rolling extremes.
    add("dist_high_60", close / high.rolling(60).max() - 1.0)
    add("dist_low_60", close / low.rolling(60).min() - 1.0)

    # Volume features.
    add("vol_z_30", (vol - vol.rolling(30).mean()) / vol.rolling(30).std())
    add("vol_change", vol.pct_change(5))

    # Order-flow / microstructure.
    taker_buy = out["taker_buy_volume"]
    of_delta = (2.0 * (taker_buy / vol) - 1.0)  # signed order-flow imbalance in [-1,1]
    add("of_imbalance", of_delta)
    add("of_imbalance_5", of_delta.rolling(5).mean())
    add("of_imbalance_15", of_delta.rolling(15).mean())
    add("of_persistence", of_delta.rolling(15).sum())
    count = out["count"].replace(0.0, np.nan)
    add("avg_trade_size", (vol / count) / (vol / count).rolling(60).mean())
    add("trade_count_z", (count - count.rolling(30).mean()) / count.rolling(30).std())

    # Custom: Kaufman efficiency ratio (trend quality).
    add("eff_ratio_30", _efficiency_ratio(close, 30))

    # Time-of-day seasonality.
    minutes = out.index.hour * 60 + out.index.minute
    add("tod_sin", pd.Series(np.sin(2 * np.pi * minutes / 1440.0), index=out.index))
    add("tod_cos", pd.Series(np.cos(2 * np.pi * minutes / 1440.0), index=out.index))

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, feats
