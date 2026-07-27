"""Async Binance USD-M Futures REST client.

Public endpoints work without credentials; signed endpoints (orders, account)
require an API key/secret with futures trading enabled.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Self
from urllib.parse import urlencode

import aiohttp

from ..config import Credentials
from ..logger import get_logger

log = get_logger(__name__)

MAINNET = "https://fapi.binance.com"
TESTNET = "https://testnet.binancefuture.com"


class BinanceError(RuntimeError):
    def __init__(self, status: int, payload: Any):
        super().__init__(f"Binance API error {status}: {payload}")
        self.status = status
        self.payload = payload


class BinanceFuturesClient:
    """Thin async wrapper around the USD-M futures REST API."""

    def __init__(self, credentials: Credentials, recv_window: int = 5000):
        self._creds = credentials
        self._recv_window = recv_window
        self._base = TESTNET if credentials.testnet else MAINNET
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            headers = {}
            if self._creds.api_key:
                headers["X-MBX-APIKEY"] = self._creds.api_key
            self._session = aiohttp.ClientSession(headers=headers)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self._recv_window
        query = urlencode(params)
        signature = hmac.new(
            self._creds.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        if self._session is None:
            await self.connect()
        assert self._session is not None
        params = dict(params or {})
        if signed:
            if not self._creds.has_keys:
                raise BinanceError(401, "signed request requires API credentials")
            params = self._sign(params)
        url = f"{self._base}{path}"
        async with self._session.request(method, url, params=params) as resp:
            payload = await resp.json(content_type=None)
            if resp.status >= 400:
                raise BinanceError(resp.status, payload)
            return payload

    # ---- Public market data -------------------------------------------------

    async def exchange_info(self) -> dict[str, Any]:
        return await self._request("GET", "/fapi/v1/exchangeInfo")

    async def ticker_24hr(self) -> list[dict[str, Any]]:
        return await self._request("GET", "/fapi/v1/ticker/24hr")

    async def book_ticker(self) -> list[dict[str, Any]]:
        """Best bid/ask for all symbols."""
        return await self._request("GET", "/fapi/v1/ticker/bookTicker")

    async def klines(self, symbol: str, interval: str, limit: int = 100) -> list[list[Any]]:
        return await self._request(
            "GET",
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )

    async def depth(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return await self._request(
            "GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit}
        )

    # ---- Signed / trading ---------------------------------------------------

    async def account(self) -> dict[str, Any]:
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": leverage},
            signed=True,
        )

    async def new_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        reduce_only: bool = False,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if price is not None:
            params["price"] = price
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        return await self._request("POST", "/fapi/v1/order", params, signed=True)
