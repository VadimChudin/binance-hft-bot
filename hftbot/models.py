"""Shared data models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Signal:
    """A directional signal emitted by a single strategy.

    score is in [-1, 1]: positive = long conviction, negative = short.
    """

    strategy: str
    symbol: str
    score: float
    reason: str = ""

    @property
    def side(self) -> Side:
        if self.score > 0:
            return Side.LONG
        if self.score < 0:
            return Side.SHORT
        return Side.FLAT


@dataclass
class Decision:
    """Aggregated decision for a symbol."""

    symbol: str
    side: Side
    score: float
    contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class SymbolStats:
    """Liquidity/volatility snapshot produced by the scanner."""

    symbol: str
    quote_volume: float
    last_price: float
    spread_pct: float
    volatility_pct: float
    score: float = 0.0


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    quantity: float
    notional: float
    stop_loss: float
    take_profit: float
    opened_at: float = field(default_factory=time.time)
    strategy_tag: str = ""

    def unrealized_pnl(self, price: float) -> float:
        direction = 1.0 if self.side == Side.LONG else -1.0
        return (price - self.entry_price) * direction * self.quantity


@dataclass
class Trade:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    fees: float
    opened_at: float
    closed_at: float
    reason: str = ""

    @property
    def holding_sec(self) -> float:
        return self.closed_at - self.opened_at
