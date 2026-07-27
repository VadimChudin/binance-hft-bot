"""Strategy plugins. Each produces a Signal in [-1, 1] for a symbol."""

from __future__ import annotations

from ..config import Config
from .base import Strategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .orderbook_imbalance import OrderBookImbalanceStrategy

_REGISTRY: dict[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
    "orderbook_imbalance": OrderBookImbalanceStrategy,
    "mean_reversion": MeanReversionStrategy,
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


def build_by_names(config: Config, names: list[str]) -> list[Strategy]:
    """Build the named strategies regardless of their ``enabled`` flag.

    Uses params/weight from config when present, else registry defaults.
    Useful for backtests where you want to force-run a specific strategy.
    """
    strategies: list[Strategy] = []
    for name in names:
        cls = _REGISTRY.get(name)
        if cls is None:
            continue
        cfg = config.strategies.get(name)
        if cfg is not None:
            strategies.append(cls(weight=cfg.weight, params=cfg.params))
        else:
            strategies.append(cls())
    return strategies


__all__ = [
    "MeanReversionStrategy",
    "MomentumStrategy",
    "OrderBookImbalanceStrategy",
    "Strategy",
    "build_by_names",
    "build_strategies",
]
