"""Executor interface shared by paper and live backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Position, Trade


class Executor(ABC):
    @abstractmethod
    async def open(self, position: Position) -> Position:
        """Open a position (simulated or real). Returns the filled position."""

    @abstractmethod
    async def close(self, position: Position, price: float, reason: str) -> Trade:
        """Close a position and return the resulting trade record."""

    @property
    @abstractmethod
    def equity(self) -> float:
        """Current account equity in quote asset."""

    async def prepare_symbol(self, symbol: str, leverage: int) -> None:
        """Optional per-symbol setup (e.g. set leverage). Default no-op."""
        return
