"""Trend-following backtest CLI (higher timeframe, vol-targeted).

Example:
    python trend_backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT \
        --interval 1h --start 2021-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from hftbot.logger import get_logger, setup_logging
from hftbot.research.data import load_ohlcv_df_monthly
from hftbot.research.trend import TrendConfig, backtest_trend

log = get_logger("trend")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vol-targeted trend-following backtest")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    p.add_argument("--interval", default="1h")
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--ema-fast", type=int, default=24)
    p.add_argument("--ema-slow", type=int, default=168)
    p.add_argument("--target-vol", type=float, default=0.5)
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--cost", type=float, default=0.0005)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO", "logs")
    bars_per_year = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365,
                     "15m": 4 * 24 * 365}.get(args.interval, 24 * 365)
    cfg = TrendConfig(
        ema_fast=args.ema_fast, ema_slow=args.ema_slow,
        target_vol_annual=args.target_vol, max_leverage=args.max_leverage,
        cost_per_turn=args.cost, vol_window=args.ema_slow, bars_per_year=bars_per_year,
    )
    print(f"\ntrend cfg: ema={cfg.ema_fast}/{cfg.ema_slow} target_vol={cfg.target_vol_annual} "
          f"maxLev={cfg.max_leverage} cost={cfg.cost_per_turn} interval={args.interval}\n")

    port_returns: pd.DataFrame = pd.DataFrame()
    for symbol in args.symbols:
        df = load_ohlcv_df_monthly(symbol, args.interval, args.start, args.end)
        if df.empty or len(df) < cfg.ema_slow * 3:
            log.warning("not enough data for %s, skipping", symbol)
            continue
        res = backtest_trend(df, cfg, symbol)
        print(res.summary())
        port_returns[symbol] = res.net_returns

    if not port_returns.empty:
        port = port_returns.fillna(0.0).mean(axis=1)  # equal-weight portfolio
        equity = (1.0 + port).cumprod()
        years = len(port) / bars_per_year
        cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else 0.0
        sharpe = port.mean() / port.std() * np.sqrt(bars_per_year) if port.std() > 0 else 0.0
        peak = equity.cummax()
        max_dd = ((peak - equity) / peak).max()
        print("\n" + "=" * 60)
        print(f"PORTFOLIO (equal-weight)  CAGR={cagr:.1%} sharpe={sharpe:.2f} "
              f"maxDD={max_dd:.1%} totRet={equity.iloc[-1]-1:.1%}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
