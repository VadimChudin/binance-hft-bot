"""Lightweight technical indicators (pure Python, no numpy dependency)."""

from __future__ import annotations

from collections.abc import Sequence


def ema(values: Sequence[float], period: int) -> float | None:
    """Exponential moving average of the last values. Returns None if too few."""
    if period <= 0 or len(values) < period:
        return None
    k = 2.0 / (period + 1.0)
    # Seed with SMA of the first `period` values.
    seed = sum(values[:period]) / period
    result = seed
    for v in values[period:]:
        result = v * k + result * (1.0 - k)
    return result


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
        period: int = 14) -> float | None:
    """Average true range over `period` candles."""
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, n):
        high, low, prev_close = highs[i], lows[i], closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def rsi(closes: Sequence[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(len(closes) - period, len(closes)):
        change = closes[i] - closes[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
