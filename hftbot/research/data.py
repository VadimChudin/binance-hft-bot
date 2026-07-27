"""Pandas-based historical loader with order-flow columns.

Reuses the same public daily kline archives as ``hftbot.backtest.data`` but
keeps the full column set (taker-buy volume, trade count, quote volume) which
carries microstructure / order-flow information useful as ML features.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..logger import get_logger

log = get_logger(__name__)

DAILY_URL = "https://data.binance.vision/data/futures/um/daily/klines"
MONTHLY_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
BASE_URL = DAILY_URL
DEFAULT_CACHE = Path("data/klines")

_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def _daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def _day_csv(symbol: str, interval: str, day: date, cache_dir: Path) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}-{interval}-{day.isoformat()}.csv"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    url = f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{day.isoformat()}.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            blob = resp.read()
    except Exception as exc:  # noqa: BLE001
        log.warning("no data for %s %s: %s", symbol, day, exc)
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8")
    cache_file.write_text(text, encoding="utf-8")
    return text


def _month_csv(symbol: str, interval: str, year: int, month: int,
               cache_dir: Path) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{year:04d}-{month:02d}"
    cache_file = cache_dir / f"{symbol}-{interval}-{tag}.csv"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    url = f"{MONTHLY_URL}/{symbol}/{interval}/{symbol}-{interval}-{tag}.zip"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            blob = resp.read()
    except Exception as exc:  # noqa: BLE001
        log.warning("no monthly data for %s %s: %s", symbol, tag, exc)
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8")
    cache_file.write_text(text, encoding="utf-8")
    return text


def _parse_csv(text: str) -> pd.DataFrame:
    first = text.lstrip()[:9].lower()
    header = 0 if first.startswith("open_time") else None
    return pd.read_csv(io.StringIO(text), header=header, names=_COLUMNS)


def _finalize(frames: list[pd.DataFrame], symbol: str, interval: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop(columns=["ignore"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time", "close"])
    df["open_time"] = df["open_time"].astype("int64")
    df = df.drop_duplicates(subset="open_time").sort_values("open_time")
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("dt")
    df["symbol"] = symbol
    log.info("loaded %d rows for %s %s", len(df), symbol, interval)
    return df


def load_ohlcv_df(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    cache_dir: Path | str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Return a time-indexed DataFrame with OHLCV + order-flow columns (daily archives)."""
    cache_dir = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    for day in _daterange(start, end):
        text = _day_csv(symbol, interval, day, cache_dir)
        if text:
            frames.append(_parse_csv(text))
    df = _finalize(frames, symbol, interval)
    return df[(df.index >= pd.Timestamp(start, tz="UTC"))] if not df.empty else df


def load_ohlcv_df_monthly(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    cache_dir: Path | str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Return a time-indexed DataFrame using monthly archives (for long spans)."""
    cache_dir = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        text = _month_csv(symbol, interval, y, m, cache_dir)
        if text:
            frames.append(_parse_csv(text))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return _finalize(frames, symbol, interval)
