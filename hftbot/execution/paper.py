"""Paper-trading executor: simulates fills against live prices with fees."""

from __future__ import annotations

import time

from ..logger import get_logger
from ..models import Position, Trade

log = get_logger(__name__)


class PaperExecutor:
    def __init__(self, starting_equity: float, taker_fee_pct: float = 0.0004):
        self._equity = starting_equity
        self._fee = taker_fee_pct

    @property
    def equity(self) -> float:
        return self._equity

    async def prepare_symbol(self, symbol: str, leverage: int) -> None:
        return None

    async def open(self, position: Position) -> Position:
        fee = position.notional * self._fee
        self._equity -= fee
        log.info(
            "[PAPER] OPEN %s %s qty=%.6f @ %.6f notional=%.2f fee=%.4f",
            position.side.value, position.symbol, position.quantity,
            position.entry_price, position.notional, fee,
        )
        return position

    async def close(self, position: Position, price: float, reason: str) -> Trade:
        gross = position.unrealized_pnl(price)
        exit_fee = abs(position.quantity * price) * self._fee
        entry_fee = position.notional * self._fee
        net = gross - exit_fee  # entry fee already deducted at open
        self._equity += net
        trade = Trade(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            pnl=net,
            fees=entry_fee + exit_fee,
            opened_at=position.opened_at,
            closed_at=time.time(),
            reason=reason,
        )
        log.info(
            "[PAPER] CLOSE %s %s @ %.6f pnl=%.4f (%s) equity=%.2f",
            position.side.value, position.symbol, price, net, reason, self._equity,
        )
        return trade
