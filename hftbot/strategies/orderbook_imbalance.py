"""Order-book imbalance strategy.

Compares aggregated bid vs ask volume near the top of book. Heavy bid
pressure -> long bias; heavy ask pressure -> short bias.
"""

from __future__ import annotations

from ..exchange.feed import SymbolBook
from ..models import Signal
from .base import Strategy


class OrderBookImbalanceStrategy(Strategy):
    name = "orderbook_imbalance"

    def evaluate(self, symbol: str, book: SymbolBook) -> Signal:
        threshold = float(self.params.get("threshold", 0.65))
        depth_fraction = float(self.params.get("depth_fraction", 0.5))

        if not book.bids or not book.asks:
            return Signal(self.name, symbol, 0.0, "no book")

        levels = max(1, int(len(book.bids) * depth_fraction))
        bid_vol = sum(q for _, q in book.bids[:levels])
        ask_vol = sum(q for _, q in book.asks[:levels])
        total = bid_vol + ask_vol
        if total <= 0:
            return Signal(self.name, symbol, 0.0, "empty book")

        # imbalance in [0, 1]: fraction of volume on the bid side.
        imbalance = bid_vol / total
        if imbalance >= threshold:
            score = min(1.0, (imbalance - 0.5) / 0.5)
            return Signal(self.name, symbol, score, f"bid_pressure={imbalance:.2f}")
        if imbalance <= (1.0 - threshold):
            score = max(-1.0, (imbalance - 0.5) / 0.5)
            return Signal(self.name, symbol, score, f"ask_pressure={imbalance:.2f}")
        return Signal(self.name, symbol, 0.0, f"balanced={imbalance:.2f}")
