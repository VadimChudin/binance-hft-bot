"""Strategy base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..exchange.feed import SymbolBook
from ..models import Signal


class Strategy(ABC):
    name: str = "base"

    def __init__(self, weight: float = 1.0, params: dict[str, Any] | None = None):
        self.weight = weight
        self.params = params or {}

    @abstractmethod
    def evaluate(self, symbol: str, book: SymbolBook) -> Signal:
        """Return a Signal with score in [-1, 1] (0 = no opinion)."""
        raise NotImplementedError
