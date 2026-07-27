"""Mean-reversion strategy using RSI + Bollinger-band position.

Fades extremes: oversold (low RSI, price below lower band) -> long;
overbought (high RSI, price above upper band) -> short.
"""

from __future__ import annotations

from statistics import fmean, pstdev

from ..exchange.feed import SymbolBook
from ..indicators import rsi
from ..models import Signal
from .base import Strategy


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def evaluate(self, symbol: str, book: SymbolBook) -> Signal:
        period = int(self.params.get("period", 20))
        num_std = float(self.params.get("num_std", 2.0))
        rsi_period = int(self.params.get("rsi_period", 14))
        rsi_low = float(self.params.get("rsi_low", 30.0))
        rsi_high = float(self.params.get("rsi_high", 70.0))

        closes = book.closes()
        if len(closes) < max(period, rsi_period) + 1:
            return Signal(self.name, symbol, 0.0, "insufficient data")

        window = closes[-period:]
        mean = fmean(window)
        std = pstdev(window)
        if std == 0:
            return Signal(self.name, symbol, 0.0, "flat window")

        price = closes[-1]
        upper = mean + num_std * std
        lower = mean - num_std * std
        r = rsi(closes, rsi_period)
        if r is None:
            return Signal(self.name, symbol, 0.0, "no rsi")

        # z-score of price relative to the band; long when cheap, short when rich.
        z = (price - mean) / std
        if price <= lower and r <= rsi_low:
            score = min(1.0, abs(z) / num_std)
            return Signal(self.name, symbol, score, f"oversold rsi={r:.0f} z={z:.2f}")
        if price >= upper and r >= rsi_high:
            score = -min(1.0, abs(z) / num_std)
            return Signal(self.name, symbol, score, f"overbought rsi={r:.0f} z={z:.2f}")
        return Signal(self.name, symbol, 0.0, f"neutral rsi={r:.0f}")
