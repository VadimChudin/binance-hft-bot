"""Backtest CLI.

Examples:
    python backtest.py --symbols BTCUSDT ETHUSDT --interval 1m \
        --start 2024-06-01 --end 2024-06-07
    python backtest.py --symbols SOLUSDT --strategies momentum mean_reversion \
        --start 2024-06-01 --end 2024-06-03
"""

from __future__ import annotations

import argparse
from datetime import date

from hftbot.backtest import Backtester, load_klines
from hftbot.backtest.engine import BacktestConfig
from hftbot.config import load_config
from hftbot.logger import get_logger, setup_logging
from hftbot.strategies import build_by_names, build_strategies

log = get_logger("backtest")


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest strategies on historical klines")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    p.add_argument("--interval", default="1m")
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, required=True)
    p.add_argument("--strategies", nargs="+",
                   help="subset of strategy names to run (default: all kline-based)")
    p.add_argument("--cache-dir", default="data/klines")
    return p.parse_args()


def _format_row(d: dict) -> str:
    return (
        f"{d['symbol']:<12} {d['n_trades']:>6} {d['win_rate']*100:>6.1f}% "
        f"{d['total_return_pct']*100:>8.2f}% {d['total_pnl']:>9.2f} "
        f"{d['profit_factor']:>7.2f} {d['max_drawdown_pct']*100:>7.2f}% "
        f"{d['sharpe']:>7.2f} {d['avg_holding_sec']:>7.0f}s {d['total_fees']:>8.2f}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.logging.level, config.logging.dir)

    if args.strategies:
        strategies = build_by_names(config, args.strategies)
    else:
        strategies = build_strategies(config)
    # Order-book strategies cannot be backtested (no historical depth).
    skipped = [s.name for s in strategies if s.name == "orderbook_imbalance"]
    strategies = [s for s in strategies if s.name != "orderbook_imbalance"]
    if skipped:
        log.warning("skipping order-book strategies in backtest: %s", skipped)
    if not strategies:
        raise SystemExit("no kline-based strategies to backtest")

    log.info("backtesting strategies: %s", [s.name for s in strategies])
    bt_cfg = BacktestConfig.from_risk(config.risk)
    bt = Backtester(strategies, bt_cfg, config.aggregator)

    header = (
        f"{'symbol':<12} {'trades':>6} {'win':>7} {'return':>9} {'pnl':>9} "
        f"{'PF':>7} {'maxDD':>8} {'sharpe':>7} {'avg_hold':>8} {'fees':>8}"
    )
    print("\n" + header)
    print("-" * len(header))

    agg_pnl = 0.0
    agg_trades = 0
    for symbol in args.symbols:
        candles = load_klines(symbol, args.interval, args.start, args.end,
                              cache_dir=args.cache_dir)
        if not candles:
            log.warning("no candles for %s, skipping", symbol)
            continue
        result = bt.run(symbol, candles)
        d = result.as_dict()
        print(_format_row(d))
        agg_pnl += result.total_pnl
        agg_trades += result.n_trades

    print("-" * len(header))
    print(f"TOTAL trades={agg_trades} pnl={agg_pnl:.2f} "
          f"(start_equity per symbol={bt_cfg.start_equity:.0f})\n")


if __name__ == "__main__":
    main()
