"""Historical kline loader using Binance public data dumps.

Downloads daily USD-M futures kline archives from https://data.binance.vision
(no auth, no geo-restriction), caches the raw CSVs locally, and returns a list
of Candle objects. This is the data source used for backtesting.
"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from ..logger import get_logger

log = get_logger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
DEFAULT_CACHE = Path("data/klines")


@dataclass
class Candle:
    open_time: int  # ms
    open: float
    high: float
    low: float
    close: float
    volume: float


def _daterange(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def _download_day(symbol: str, interval: str, day: date,
                  cache_dir: Path) -> list[Candle]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol}-{interval}-{day.isoformat()}.csv"
    cache_file = cache_dir / fname
    text: str | None = None

    if cache_file.exists():
        text = cache_file.read_text(encoding="utf-8")
    else:
        url = f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{day.isoformat()}.zip"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                blob = resp.read()
        except Exception as exc:  # noqa: BLE001
            log.warning("no data for %s %s: %s", symbol, day, exc)
            return []
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = zf.namelist()[0]
            text = zf.read(name).decode("utf-8")
        cache_file.write_text(text, encoding="utf-8")

    candles: list[Candle] = []
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0] or row[0].lower().startswith("open_time"):
            continue
        try:
            candles.append(
                Candle(
                    open_time=int(float(row[0])),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        except (ValueError, IndexError):
            continue
    return candles


def load_klines(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    cache_dir: Path | str = DEFAULT_CACHE,
) -> list[Candle]:
    """Load and concatenate daily kline archives over [start, end] inclusive."""
    cache_dir = Path(cache_dir)
    all_candles: list[Candle] = []
    for day in _daterange(start, end):
        all_candles.extend(_download_day(symbol, interval, day, cache_dir))
    all_candles.sort(key=lambda c: c.open_time)
    # De-duplicate by open_time in case of overlapping archives.
    deduped: list[Candle] = []
    seen: set[int] = set()
    for c in all_candles:
        if c.open_time in seen:
            continue
        seen.add(c.open_time)
        deduped.append(c)
    log.info("loaded %d candles for %s %s (%s..%s)",
             len(deduped), symbol, interval, start, end)
    return deduped
