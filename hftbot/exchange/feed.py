"""Real-time market data feed via Binance combined WebSocket streams.

Maintains, per symbol:
  * a rolling window of closed klines (bootstrapped from REST),
  * the latest partial order book (bids/asks),
  * the latest trade price.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections import deque
from dataclasses import dataclass, field

import websockets

from ..logger import get_logger
from .binance_client import BinanceFuturesClient

log = get_logger(__name__)

MAINNET_WS = "wss://fstream.binance.com/stream?streams="
TESTNET_WS = "wss://stream.binancefuture.com/stream?streams="


@dataclass
class Kline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SymbolBook:
    klines: deque[Kline] = field(default_factory=lambda: deque(maxlen=200))
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    last_price: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2.0
        return self.last_price

    def closes(self) -> list[float]:
        return [k.close for k in self.klines]

    def highs(self) -> list[float]:
        return [k.high for k in self.klines]

    def lows(self) -> list[float]:
        return [k.low for k in self.klines]


class MarketData:
    """Thread-free async market data manager for a fixed symbol universe."""

    def __init__(
        self,
        client: BinanceFuturesClient,
        symbols: list[str],
        kline_interval: str = "1m",
        depth_levels: int = 20,
        testnet: bool = False,
    ):
        self._client = client
        self.symbols = symbols
        self._interval = kline_interval
        self._depth_levels = depth_levels
        self._ws_base = TESTNET_WS if testnet else MAINNET_WS
        self.books: dict[str, SymbolBook] = {s: SymbolBook() for s in symbols}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def bootstrap(self) -> None:
        """Seed kline history via REST so indicators are usable immediately."""
        for symbol in self.symbols:
            try:
                raw = await self._client.klines(symbol, self._interval, limit=100)
            except Exception as exc:  # noqa: BLE001
                log.warning("bootstrap klines failed for %s: %s", symbol, exc)
                continue
            book = self.books[symbol]
            book.klines.clear()
            for k in raw:
                book.klines.append(
                    Kline(
                        open_time=int(k[0]),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                    )
                )
            if book.klines:
                book.last_price = book.klines[-1].close

    def _stream_names(self) -> list[str]:
        names: list[str] = []
        depth = min(self._depth_levels, 20)
        for symbol in self.symbols:
            low = symbol.lower()
            names.append(f"{low}@kline_{self._interval}")
            names.append(f"{low}@depth{depth}@100ms")
        return names

    async def start(self) -> None:
        await self.bootstrap()
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="market-data-ws")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        url = self._ws_base + "/".join(self._stream_names())
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=15, max_queue=1024) as ws:
                    log.info("WS connected (%d streams)", len(self.symbols) * 2)
                    backoff = 1.0
                    async for raw in ws:
                        if self._stop.is_set():
                            break
                        self._handle(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("WS error: %s (reconnect in %.0fs)", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle(self, msg: dict) -> None:
        stream = msg.get("stream", "")
        data = msg.get("data", {})
        if "@kline" in stream:
            self._handle_kline(data)
        elif "@depth" in stream:
            self._handle_depth(data)

    def _handle_kline(self, data: dict) -> None:
        k = data.get("k", {})
        symbol = data.get("s") or k.get("s")
        if not symbol or symbol not in self.books:
            return
        book = self.books[symbol]
        candle = Kline(
            open_time=int(k["t"]),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )
        book.last_price = candle.close
        if book.klines and book.klines[-1].open_time == candle.open_time:
            book.klines[-1] = candle
        else:
            book.klines.append(candle)

    def _handle_depth(self, data: dict) -> None:
        symbol = data.get("s")
        if not symbol or symbol not in self.books:
            return
        book = self.books[symbol]
        bids = data.get("b", [])
        asks = data.get("a", [])
        if bids:
            book.bids = [(float(p), float(q)) for p, q in bids]
        if asks:
            book.asks = [(float(p), float(q)) for p, q in asks]
