"""EMA-crossover momentum strategy.

Long when the fast EMA is meaningfully above the slow EMA, short when below.
The score scales with the relative EMA separation, capped at +/-1.
"""

from __future__ import annotations

from ..exchange.feed import SymbolBook
from ..indicators import ema
from ..models import Signal
from .base import Strategy


class MomentumStrategy(Strategy):
    name = "momentum"

    def evaluate(self, symbol: str, book: SymbolBook) -> Signal:
        fast_p = int(self.params.get("ema_fast", 9))
        slow_p = int(self.params.get("ema_slow", 21))
        min_sep = float(self.params.get("min_separation_pct", 0.0003))

        closes = book.closes()
        fast = ema(closes, fast_p)
        slow = ema(closes, slow_p)
        if fast is None or slow is None or slow == 0:
            return Signal(self.name, symbol, 0.0, "insufficient data")

        separation = (fast - slow) / slow
        if abs(separation) < min_sep:
            return Signal(self.name, symbol, 0.0, "flat EMAs")

        # Normalize: separation of ~0.3% saturates the score.
        score = max(-1.0, min(1.0, separation / 0.003))
        reason = f"ema_sep={separation:.4%}"
        return Signal(self.name, symbol, score, reason)
