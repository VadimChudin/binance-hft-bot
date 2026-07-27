"""Risk management: sizing, stop/target levels, and trade gating."""

from __future__ import annotations

import time

from .config import RiskConfig
from .logger import get_logger
from .models import Position, Side

log = get_logger(__name__)


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self._cfg = cfg
        self.realized_pnl_today: float = 0.0
        self._day = time.gmtime().tm_yday
        self._cooldown_until: dict[str, float] = {}

    def _roll_day(self) -> None:
        today = time.gmtime().tm_yday
        if today != self._day:
            self._day = today
            self.realized_pnl_today = 0.0
            log.info("new UTC day: realized PnL reset")

    @property
    def trading_halted(self) -> bool:
        self._roll_day()
        return self.realized_pnl_today <= -abs(self._cfg.max_daily_loss)

    def can_open(self, symbol: str, open_positions: int) -> tuple[bool, str]:
        if self.trading_halted:
            return False, "daily loss limit reached"
        if open_positions >= self._cfg.max_open_positions:
            return False, "max open positions"
        until = self._cooldown_until.get(symbol, 0.0)
        if time.time() < until:
            return False, "cooldown"
        return True, ""

    def build_position(
        self, symbol: str, side: Side, entry_price: float, equity: float,
        strategy_tag: str = "",
    ) -> Position:
        notional = equity * self._cfg.risk_per_trade_pct * self._cfg.leverage
        quantity = notional / entry_price if entry_price > 0 else 0.0
        if side == Side.LONG:
            stop = entry_price * (1.0 - self._cfg.stop_loss_pct)
            target = entry_price * (1.0 + self._cfg.take_profit_pct)
        else:
            stop = entry_price * (1.0 + self._cfg.stop_loss_pct)
            target = entry_price * (1.0 - self._cfg.take_profit_pct)
        return Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            notional=notional,
            stop_loss=stop,
            take_profit=target,
            strategy_tag=strategy_tag,
        )

    def should_close(self, pos: Position, price: float) -> str | None:
        """Return an exit reason if the position should close, else None."""
        if pos.side == Side.LONG:
            if price <= pos.stop_loss:
                return "stop_loss"
            if price >= pos.take_profit:
                return "take_profit"
        else:
            if price >= pos.stop_loss:
                return "stop_loss"
            if price <= pos.take_profit:
                return "take_profit"
        if time.time() - pos.opened_at >= self._cfg.max_holding_sec:
            return "max_holding"
        return None

    def register_close(self, symbol: str, pnl: float) -> None:
        self.realized_pnl_today += pnl
        self._cooldown_until[symbol] = time.time() + self._cfg.cooldown_sec
