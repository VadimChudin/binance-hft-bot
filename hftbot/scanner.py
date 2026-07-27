"""Liquidity/volatility scanner.

Ranks the USD-M futures universe and returns the most tradable symbols:
strong 24h volume, tight spread, and enough short-term volatility to scalp.
"""

from __future__ import annotations

import asyncio

from .config import ScannerConfig
from .exchange.binance_client import BinanceFuturesClient
from .indicators import atr
from .logger import get_logger
from .models import SymbolStats

log = get_logger(__name__)


class Scanner:
    def __init__(
        self,
        client: BinanceFuturesClient,
        cfg: ScannerConfig,
        quote_asset: str = "USDT",
    ):
        self._client = client
        self._cfg = cfg
        self._quote = quote_asset
        self._perpetuals: set[str] | None = None

    async def _load_perpetuals(self) -> set[str]:
        info = await self._client.exchange_info()
        symbols: set[str] = set()
        for s in info.get("symbols", []):
            if s.get("quoteAsset") != self._quote:
                continue
            if s.get("status") != "TRADING":
                continue
            if self._cfg.perpetual_only and s.get("contractType") != "PERPETUAL":
                continue
            symbols.add(s["symbol"])
        return symbols

    async def scan(self) -> list[SymbolStats]:
        if self._perpetuals is None:
            self._perpetuals = await self._load_perpetuals()

        tickers, books = await asyncio.gather(
            self._client.ticker_24hr(), self._client.book_ticker()
        )
        book_map = {b["symbol"]: b for b in books}

        prelim: list[SymbolStats] = []
        for t in tickers:
            symbol = t["symbol"]
            if symbol not in self._perpetuals:
                continue
            quote_volume = float(t.get("quoteVolume", 0.0))
            if quote_volume < self._cfg.min_quote_volume:
                continue
            book = book_map.get(symbol)
            if not book:
                continue
            bid = float(book.get("bidPrice", 0.0))
            ask = float(book.get("askPrice", 0.0))
            if bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid if mid else 1.0
            if spread_pct > self._cfg.max_spread_pct:
                continue
            prelim.append(
                SymbolStats(
                    symbol=symbol,
                    quote_volume=quote_volume,
                    last_price=float(t.get("lastPrice", mid)),
                    spread_pct=spread_pct,
                    volatility_pct=0.0,
                )
            )

        # Fetch short-term volatility only for the most liquid candidates.
        prelim.sort(key=lambda s: s.quote_volume, reverse=True)
        candidates = prelim[: max(self._cfg.top_n * 3, self._cfg.top_n)]
        await self._enrich_volatility(candidates)

        qualified = [
            s for s in candidates if s.volatility_pct >= self._cfg.min_volatility_pct
        ]
        for s in qualified:
            s.score = self._composite_score(s)
        qualified.sort(key=lambda s: s.score, reverse=True)

        top = qualified[: self._cfg.top_n]
        log.info(
            "scanner: %d perpetuals, %d liquid, %d volatile -> top %d: %s",
            len(self._perpetuals),
            len(prelim),
            len(qualified),
            len(top),
            ", ".join(s.symbol for s in top),
        )
        return top

    async def _enrich_volatility(self, stats: list[SymbolStats]) -> None:
        async def one(s: SymbolStats) -> None:
            try:
                raw = await self._client.klines(s.symbol, "1m", limit=30)
            except Exception as exc:  # noqa: BLE001
                log.debug("volatility klines failed for %s: %s", s.symbol, exc)
                return
            highs = [float(k[2]) for k in raw]
            lows = [float(k[3]) for k in raw]
            closes = [float(k[4]) for k in raw]
            a = atr(highs, lows, closes, period=14)
            if a and s.last_price:
                s.volatility_pct = a / s.last_price

        await asyncio.gather(*(one(s) for s in stats))

    def _composite_score(self, s: SymbolStats) -> float:
        # Favor volatility (scalping edge) and volume, penalize spread.
        vol_ref = self._cfg.min_quote_volume or 1.0
        spread_ref = self._cfg.max_spread_pct or 1.0
        vol_term = min(s.volatility_pct / 0.01, 3.0)
        liq_term = min(s.quote_volume / vol_ref, 5.0) / 5.0
        spread_term = 1.0 - min(s.spread_pct / spread_ref, 1.0)
        return vol_term * 0.6 + liq_term * 0.3 + spread_term * 0.1
