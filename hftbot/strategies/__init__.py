"""Strategy plugins. Each produces a Signal in [-1, 1] for a symbol."""

from __future__ import annotations

from ..config import Config
from .base import Strategy
from .momentum import MomentumStrategy
from .orderbook_imbalance import OrderBookImbalanceStrategy

_REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
    "orderbook_imbalance": OrderBookImbalanceStrategy,
}


def build_strategies(config: Config) -> list[Strategy]:
    strategies: list[Strategy] = []
    for name, cfg in config.strategies.items():
        if not cfg.enabled:
            continue
        cls = _REGISTRY.get(name)
        if cls is None:
            continue
        strategies.append(cls(weight=cfg.weight, params=cfg.params))
    return strategies


__all__ = [
    "MomentumStrategy",
    "OrderBookImbalanceStrategy",
    "Strategy",
    "build_strategies",
]
