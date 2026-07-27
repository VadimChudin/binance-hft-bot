"""Combine per-strategy signals into a single weighted decision."""

from __future__ import annotations

from .config import AggregatorConfig
from .models import Decision, Side, Signal


class SignalAggregator:
    def __init__(self, cfg: AggregatorConfig):
        self._cfg = cfg

    def aggregate(self, symbol: str, signals: list[Signal],
                  weights: dict[str, float]) -> Decision:
        total_weight = 0.0
        weighted = 0.0
        contributions: dict[str, float] = {}
        for sig in signals:
            w = weights.get(sig.strategy, 1.0)
            contributions[sig.strategy] = sig.score
            weighted += sig.score * w
            total_weight += w

        score = weighted / total_weight if total_weight else 0.0

        side = Side.FLAT
        if score >= self._cfg.entry_threshold:
            side = Side.LONG
        elif score <= -self._cfg.entry_threshold:
            side = Side.SHORT

        return Decision(symbol=symbol, side=side, score=score,
                        contributions=contributions)
