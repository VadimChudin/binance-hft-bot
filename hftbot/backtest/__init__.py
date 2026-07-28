"""Backtesting: historical data loading, simulation, and metrics."""

from __future__ import annotations

from .data import Candle, load_klines
from .engine import BacktestConfig, Backtester
from .metrics import BacktestResult, compute_metrics

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "Candle",
    "compute_metrics",
    "load_klines",
]
