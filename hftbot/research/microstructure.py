"""L2 / order-flow microstructure features from free Binance data dumps.

Sources (https://data.binance.vision, futures/um/daily):
  * aggTrades  -> signed trade flow (is_buyer_maker gives the aggressor side):
                  order-flow imbalance, trade intensity, large-trade share.
  * bookDepth  -> order-book depth snapshots at +-1..5% from mid (~30s cadence):
                  depth imbalance at several levels, book slope.
  * metrics    -> open interest & long/short ratios (5-min cadence).

Everything is aggregated to 1-minute bars and cached per day so repeated runs
are fast.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..logger import get_logger

log = get_logger(__name__)

DAILY = "https://data.binance.vision/data/futures/um/daily"
MICRO_CACHE = Path("data/micro")

MICRO_FEATURES = [
    "ofi", "buy_ratio", "trade_count_z", "avg_trade_size_z", "big_trade_ratio",
    "depth_imb_1", "depth_imb_2", "depth_imb_5", "depth_imb_near", "book_slope",
    "oi_change", "taker_ls_ratio", "toptrader_ls_ratio",
]


def _download_csv(kind: str, symbol: str, day: date) -> str | None:
    url = f"{DAILY}/{kind}/{symbol}/{symbol}-{kind}-{day.isoformat()}.zip"
    try:
        with urllib.request.urlopen(url, timeout=90) as resp:
            blob = resp.read()
    except Exception as exc:  # noqa: BLE001
        log.warning("no %s for %s %s: %s", kind, symbol, day, exc)
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return zf.read(zf.namelist()[0]).decode("utf-8")


def _aggtrades_minute(symbol: str, day: date) -> pd.DataFrame:
    text = _download_csv("aggTrades", symbol, day)
    if not text:
        return pd.DataFrame()
    cols = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker"]
    header = 0 if text.lstrip()[:6].lower().startswith("agg_tr") else None
    df = pd.read_csv(io.StringIO(text), header=header, names=cols)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    # is_buyer_maker == True  => aggressor is a SELLER (market sell).
    is_maker = df["is_buyer_maker"].astype(str).str.lower().isin(["true", "1"])
    df["buy_qty"] = np.where(~is_maker, df["qty"], 0.0)
    df["sell_qty"] = np.where(is_maker, df["qty"], 0.0)
    df["minute"] = df["ts"].dt.floor("1min")

    g = df.groupby("minute")
    out = pd.DataFrame({
        "buy_vol": g["buy_qty"].sum(),
        "sell_vol": g["sell_qty"].sum(),
        "trade_count": g["qty"].count(),
        "max_trade": g["qty"].max(),
        "sum_qty": g["qty"].sum(),
    })
    tot = (out["buy_vol"] + out["sell_vol"]).replace(0.0, np.nan)
    out["ofi"] = (out["buy_vol"] - out["sell_vol"]) / tot
    out["buy_ratio"] = out["buy_vol"] / tot
    out["avg_trade_size"] = out["sum_qty"] / out["trade_count"].replace(0, np.nan)
    out["big_trade_ratio"] = out["max_trade"] / out["sum_qty"].replace(0.0, np.nan)
    return out


def _bookdepth_minute(symbol: str, day: date) -> pd.DataFrame:
    text = _download_csv("bookDepth", symbol, day)
    if not text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce")
    df["minute"] = df["timestamp"].dt.floor("1min")

    def imb(level: int, sub: pd.DataFrame) -> pd.Series:
        bid = sub[sub["percentage"] == -level].groupby("minute")["notional"].mean()
        ask = sub[sub["percentage"] == level].groupby("minute")["notional"].mean()
        j = pd.concat({"bid": bid, "ask": ask}, axis=1)
        tot = (j["bid"] + j["ask"]).replace(0.0, np.nan)
        return (j["bid"] - j["ask"]) / tot

    i1, i2, i5 = imb(1, df), imb(2, df), imb(5, df)
    # Book slope: how depth builds up 1%->5% on ask vs bid (negative => bid-heavy).
    ask_all = df[df["percentage"] > 0].groupby("minute")["notional"].sum()
    bid_all = df[df["percentage"] < 0].groupby("minute")["notional"].sum()
    tot_all = (ask_all + bid_all).replace(0.0, np.nan)
    out = pd.DataFrame({
        "depth_imb_1": i1,
        "depth_imb_2": i2,
        "depth_imb_5": i5,
        "depth_imb_near": (i1 + i2) / 2.0,
        "book_slope": (bid_all - ask_all) / tot_all,
    })
    return out


def _metrics_minute(symbol: str, day: date) -> pd.DataFrame:
    text = _download_csv("metrics", symbol, day)
    if not text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
    df = df.set_index("create_time")
    oi = pd.to_numeric(df["sum_open_interest"], errors="coerce")
    out = pd.DataFrame({
        "open_interest": oi,
        "taker_ls_ratio": pd.to_numeric(df["sum_taker_long_short_vol_ratio"], errors="coerce"),
        "toptrader_ls_ratio": pd.to_numeric(df["sum_toptrader_long_short_ratio"], errors="coerce"),
    })
    # 5-min cadence -> resample to 1-min and forward-fill.
    out = out.resample("1min").ffill()
    out["oi_change"] = out["open_interest"].pct_change(5)
    return out


def _build_day(symbol: str, day: date) -> pd.DataFrame:
    agg = _aggtrades_minute(symbol, day)
    book = _bookdepth_minute(symbol, day)
    met = _metrics_minute(symbol, day)
    if agg.empty and book.empty:
        return pd.DataFrame()
    df = agg.join(book, how="outer").join(met, how="outer")
    return df


def build_micro_features(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path | str = MICRO_CACHE,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (minute-indexed micro-feature DataFrame, feature names)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    day = start
    while day <= end:
        cache_file = cache_dir / f"{symbol}-{day.isoformat()}.csv"
        if cache_file.exists():
            d = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            d = _build_day(symbol, day)
            if not d.empty:
                d.to_csv(cache_file)
        if not d.empty:
            frames.append(d)
        day += timedelta(days=1)

    if not frames:
        return pd.DataFrame(), []

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    # Normalize a couple of raw counts to rolling z-scores.
    tc = df["trade_count"]
    df["trade_count_z"] = (tc - tc.rolling(120).mean()) / tc.rolling(120).std()
    ats = df["avg_trade_size"]
    df["avg_trade_size_z"] = (ats - ats.rolling(120).mean()) / ats.rolling(120).std()
    df = df.replace([np.inf, -np.inf], np.nan)
    feats = [f for f in MICRO_FEATURES if f in df.columns]
    log.info("micro features for %s: %d minutes, %d feats", symbol, len(df), len(feats))
    return df, feats
