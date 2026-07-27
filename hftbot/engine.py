"""Trading engine: orchestrates scanner, feed, strategies, risk and execution."""

from __future__ import annotations

import asyncio
import contextlib
import time

from .aggregator import SignalAggregator
from .config import Config
from .exchange.binance_client import BinanceFuturesClient
from .exchange.feed import MarketData
from .execution.base import Executor
from .execution.live import LiveExecutor
from .execution.paper import PaperExecutor
from .logger import get_logger
from .models import Decision, Position, Side, Signal
from .monitor import Heartbeat, TradeLog
from .risk import RiskManager
from .scanner import Scanner
from .strategies import build_strategies

log = get_logger(__name__)


class TradingEngine:
    def __init__(self, config: Config):
        self.config = config
        self.client = BinanceFuturesClient(config.credentials)
        self.scanner = Scanner(self.client, config.scanner, config.quote_asset)
        self.strategies = build_strategies(config)
        self.aggregator = SignalAggregator(config.aggregator)
        self.risk = RiskManager(config.risk)
        self.trade_log = TradeLog(config.logging.dir)
        self.heartbeat = Heartbeat(30.0)

        self.executor: Executor = self._build_executor()
        self.feed: MarketData | None = None
        self.symbols: list[str] = []
        self.positions: dict[str, Position] = {}
        self._weights = {s.name: s.weight for s in self.strategies}
        self._stop = asyncio.Event()
        self._last_scan = 0.0

    def _build_executor(self) -> Executor:
        if self.config.is_live:
            return LiveExecutor(self.client)
        return PaperExecutor(
            self.config.risk.paper_equity, self.config.risk.taker_fee_pct
        )

    async def _rescan(self) -> None:
        stats = await self.scanner.scan()
        new_symbols = [s.symbol for s in stats]
        if not new_symbols:
            log.warning("scanner returned no symbols; keeping current universe")
            return
        if set(new_symbols) == set(self.symbols) and self.feed is not None:
            self._last_scan = time.time()
            return

        # Do not drop symbols we currently hold a position in.
        held = [s for s in self.symbols if s in self.positions]
        merged = list(dict.fromkeys(new_symbols + held))

        if self.feed is not None:
            await self.feed.stop()
        self.symbols = merged
        self.feed = MarketData(
            self.client,
            self.symbols,
            kline_interval=self.config.feed.kline_interval,
            depth_levels=self.config.feed.depth_levels,
            testnet=self.config.credentials.testnet,
        )
        await self.feed.start()
        for symbol in self.symbols:
            with contextlib.suppress(Exception):
                await self.executor.prepare_symbol(symbol, self.config.risk.leverage)
        self._last_scan = time.time()

    def _decide(self, symbol: str) -> Decision:
        assert self.feed is not None
        book = self.feed.books[symbol]
        signals: list[Signal] = [s.evaluate(symbol, book) for s in self.strategies]
        return self.aggregator.aggregate(symbol, signals, self._weights)

    async def _manage_positions(self) -> None:
        assert self.feed is not None
        for symbol, pos in list(self.positions.items()):
            price = self.feed.books[symbol].mid_price
            if price <= 0:
                continue
            reason = self.risk.should_close(pos, price)
            if reason:
                trade = await self.executor.close(pos, price, reason)
                self.trade_log.record(trade)
                self.risk.register_close(symbol, trade.pnl)
                del self.positions[symbol]

    async def _open_positions(self) -> None:
        assert self.feed is not None
        for symbol in self.symbols:
            if symbol in self.positions:
                continue
            decision = self._decide(symbol)
            if decision.side == Side.FLAT:
                continue
            ok, why = self.risk.can_open(symbol, len(self.positions))
            if not ok:
                if why not in ("cooldown",):
                    log.debug("skip %s: %s", symbol, why)
                continue
            price = self.feed.books[symbol].mid_price
            if price <= 0:
                continue
            equity = self.executor.equity
            pos = self.risk.build_position(
                symbol, decision.side, price, equity,
                strategy_tag=f"score={decision.score:.2f}",
            )
            try:
                pos = await self.executor.open(pos)
            except Exception as exc:  # noqa: BLE001
                log.warning("open failed for %s: %s", symbol, exc)
                continue
            self.positions[symbol] = pos
            log.info("OPEN %s %s score=%.2f contrib=%s", decision.side.value,
                     symbol, decision.score, decision.contributions)

    async def run(self) -> None:
        if not self.strategies:
            raise RuntimeError("no strategies enabled")
        await self.client.connect()
        if self.config.is_live:
            if not self.config.credentials.has_keys:
                raise RuntimeError("live mode requires BINANCE_API_KEY/SECRET")
            await self.executor.refresh_equity()  # type: ignore[attr-defined]
            log.warning("LIVE MODE: trading real funds, equity=%.2f",
                        self.executor.equity)
        else:
            log.info("PAPER MODE: simulated equity=%.2f", self.executor.equity)

        await self._rescan()
        tick = self.config.engine.tick_sec
        try:
            while not self._stop.is_set():
                if time.time() - self._last_scan >= self.config.scanner.refresh_sec:
                    await self._rescan()
                if self.feed is not None:
                    await self._manage_positions()
                    await self._open_positions()
                if self.heartbeat.due():
                    log.info("status: %s", self.trade_log.summary(
                        self.executor.equity, len(self.positions)))
                await asyncio.sleep(tick)
        finally:
            await self.shutdown()

    async def stop(self) -> None:
        self._stop.set()

    async def shutdown(self) -> None:
        log.info("shutting down: closing %d open positions", len(self.positions))
        if self.feed is not None:
            for symbol, pos in list(self.positions.items()):
                price = self.feed.books[symbol].mid_price or pos.entry_price
                with contextlib.suppress(Exception):
                    trade = await self.executor.close(pos, price, "shutdown")
                    self.trade_log.record(trade)
            await self.feed.stop()
        await self.client.close()
        log.info("final: %s", self.trade_log.summary(
            self.executor.equity, len(self.positions)))
