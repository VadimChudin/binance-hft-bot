"""Trade/PnL logging and a periodic status summary."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from .logger import get_logger
from .models import Trade

log = get_logger(__name__)


class TradeLog:
    def __init__(self, log_dir: str = "logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(log_dir) / "trades.csv"
        self.trades: list[Trade] = []
        if not self._path.exists():
            with self._path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["closed_at", "symbol", "side", "entry", "exit",
                     "qty", "pnl", "fees", "holding_sec", "reason"]
                )

    def record(self, trade: Trade) -> None:
        self.trades.append(trade)
        with self._path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                f"{trade.closed_at:.0f}",
                trade.symbol,
                trade.side.value,
                f"{trade.entry_price:.6f}",
                f"{trade.exit_price:.6f}",
                f"{trade.quantity:.6f}",
                f"{trade.pnl:.4f}",
                f"{trade.fees:.4f}",
                f"{trade.holding_sec:.0f}",
                trade.reason,
            ])

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    def summary(self, equity: float, open_positions: int) -> str:
        return (
            f"trades={len(self.trades)} win_rate={self.win_rate:.0%} "
            f"realized_pnl={self.total_pnl:.2f} equity={equity:.2f} "
            f"open={open_positions}"
        )


class Heartbeat:
    def __init__(self, interval_sec: float = 30.0):
        self._interval = interval_sec
        self._last = 0.0

    def due(self) -> bool:
        now = time.time()
        if now - self._last >= self._interval:
            self._last = now
            return True
        return False
