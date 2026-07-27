"""Live executor: sends real market orders to Binance USD-M futures.

Positions are opened/closed with MARKET orders (taker). Quantities are
rounded to each symbol's LOT_SIZE step. Use with care and small size.
"""

from __future__ import annotations

import math
import time

from ..exchange.binance_client import BinanceFuturesClient
from ..logger import get_logger
from ..models import Position, Side, Trade

log = get_logger(__name__)


class LiveExecutor:
    def __init__(self, client: BinanceFuturesClient):
        self._client = client
        self._step: dict[str, float] = {}
        self._min_qty: dict[str, float] = {}
        self._equity_cache: float = 0.0
        self._prepared: set[str] = set()

    @property
    def equity(self) -> float:
        return self._equity_cache

    async def refresh_equity(self) -> float:
        acct = await self._client.account()
        self._equity_cache = float(acct.get("totalWalletBalance", 0.0))
        return self._equity_cache

    async def _load_filters(self) -> None:
        if self._step:
            return
        info = await self._client.exchange_info()
        for s in info.get("symbols", []):
            for f in s.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    self._step[s["symbol"]] = float(f["stepSize"])
                    self._min_qty[s["symbol"]] = float(f["minQty"])

    def _round_qty(self, symbol: str, qty: float) -> float:
        step = self._step.get(symbol)
        if not step:
            return qty
        rounded = math.floor(qty / step) * step
        # Avoid floating point dust.
        decimals = max(0, round(-math.log10(step))) if step < 1 else 0
        return round(rounded, decimals)

    async def prepare_symbol(self, symbol: str, leverage: int) -> None:
        await self._load_filters()
        if symbol in self._prepared:
            return
        try:
            await self._client.set_leverage(symbol, leverage)
        except Exception as exc:  # noqa: BLE001
            log.warning("set_leverage failed for %s: %s", symbol, exc)
        self._prepared.add(symbol)

    async def open(self, position: Position) -> Position:
        await self._load_filters()
        qty = self._round_qty(position.symbol, position.quantity)
        min_qty = self._min_qty.get(position.symbol, 0.0)
        if qty <= 0 or qty < min_qty:
            raise ValueError(
                f"quantity {qty} below min {min_qty} for {position.symbol}"
            )
        side = "BUY" if position.side == Side.LONG else "SELL"
        resp = await self._client.new_order(
            symbol=position.symbol, side=side, order_type="MARKET", quantity=qty
        )
        fill_price = float(resp.get("avgPrice") or position.entry_price)
        position.quantity = qty
        position.entry_price = fill_price or position.entry_price
        log.info("[LIVE] OPEN %s %s qty=%.6f @ %.6f",
                 position.side.value, position.symbol, qty, position.entry_price)
        return position

    async def close(self, position: Position, price: float, reason: str) -> Trade:
        side = "SELL" if position.side == Side.LONG else "BUY"
        resp = await self._client.new_order(
            symbol=position.symbol, side=side, order_type="MARKET",
            quantity=position.quantity, reduce_only=True,
        )
        fill_price = float(resp.get("avgPrice") or price)
        gross = position.unrealized_pnl(fill_price)
        log.info("[LIVE] CLOSE %s %s @ %.6f pnl~=%.4f (%s)",
                 position.side.value, position.symbol, fill_price, gross, reason)
        return Trade(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=fill_price,
            quantity=position.quantity,
            pnl=gross,
            fees=0.0,
            opened_at=position.opened_at,
            closed_at=time.time(),
            reason=reason,
        )
