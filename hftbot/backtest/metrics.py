"""Backtest performance metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..models import Trade


@dataclass
class BacktestResult:
    symbol: str
    start_equity: float
    end_equity: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    n_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    avg_holding_sec: float = 0.0
    total_fees: float = 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "symbol": self.symbol,
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "avg_pnl": round(self.avg_pnl, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "profit_factor": round(self.profit_factor, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe": round(self.sharpe, 3),
            "avg_holding_sec": round(self.avg_holding_sec, 1),
            "total_fees": round(self.total_fees, 2),
            "end_equity": round(self.end_equity, 2),
        }


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    # Per-trade Sharpe scaled by sqrt(n) as a rough annualization-agnostic figure.
    return (mean / std) * math.sqrt(len(returns))


def compute_metrics(
    symbol: str,
    start_equity: float,
    trades: list[Trade],
    equity_curve: list[float],
) -> BacktestResult:
    res = BacktestResult(
        symbol=symbol,
        start_equity=start_equity,
        end_equity=equity_curve[-1] if equity_curve else start_equity,
        trades=trades,
        equity_curve=equity_curve,
    )
    res.n_trades = len(trades)
    if not trades:
        return res

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    res.total_pnl = sum(t.pnl for t in trades)
    res.total_fees = sum(t.fees for t in trades)
    res.win_rate = len(wins) / len(trades)
    res.avg_pnl = res.total_pnl / len(trades)
    res.avg_win = (sum(wins) / len(wins)) if wins else 0.0
    res.avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    res.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else math.inf
    res.total_return_pct = (res.end_equity - start_equity) / start_equity
    res.max_drawdown_pct = _max_drawdown(equity_curve)
    res.sharpe = _sharpe([t.pnl / start_equity for t in trades])
    res.avg_holding_sec = sum(t.holding_sec for t in trades) / len(trades)
    return res
