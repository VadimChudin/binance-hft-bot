"""ML research CLI: walk-forward train + out-of-sample simulation.

Example:
    python research.py --symbols BTCUSDT ETHUSDT SOLUSDT \
        --start 2024-01-01 --end 2024-05-31 \
        --barrier 0.005 --horizon 60 --thr-long 0.58 --thr-short 0.42
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from hftbot.logger import get_logger, setup_logging
from hftbot.research.data import load_ohlcv_df
from hftbot.research.train import WalkForwardConfig, run

log = get_logger("research")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML walk-forward research")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p.add_argument("--interval", default="1m")
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--barrier", type=float, default=0.005)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--train-bars", type=int, default=60_000)
    p.add_argument("--test-bars", type=int, default=20_000)
    p.add_argument("--cost", type=float, default=0.0005, help="per-side cost")
    p.add_argument("--thr-long", type=float, default=0.58)
    p.add_argument("--thr-short", type=float, default=0.42)
    p.add_argument("--out", default="research_out")
    p.add_argument("--save-importance", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO", "logs")
    cfg = WalkForwardConfig(
        barrier=args.barrier,
        horizon=args.horizon,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        cost_per_trade=args.cost,
        thr_long=args.thr_long,
        thr_short=args.thr_short,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nconfig: barrier={cfg.barrier} horizon={cfg.horizon} "
          f"cost/side={cfg.cost_per_trade} thr=({cfg.thr_short},{cfg.thr_long})\n")

    summary: dict[str, dict] = {}
    agg_returns: list[float] = []
    for symbol in args.symbols:
        df = load_ohlcv_df(symbol, args.interval, args.start, args.end)
        if df.empty or len(df) < cfg.train_bars + cfg.test_bars:
            log.warning("not enough data for %s (%d rows), skipping", symbol, len(df))
            continue
        result, _feat_df, _feats = run(df, cfg)
        print(f"{symbol:<10} {result.summary()}")
        summary[symbol] = {
            "n_trades": result.n_trades,
            "win_rate": round(result.win_rate, 4),
            "total_return": round(result.total_return, 4),
            "avg_return_bp": round(result.avg_return * 1e4, 2),
            "sharpe": round(result.sharpe, 3),
            "max_drawdown": round(result.max_drawdown, 4),
            "auc": round(result.auc, 4),
        }
        agg_returns.extend(result.returns)

    if agg_returns:
        import numpy as np
        arr = np.array(agg_returns)
        total = float(np.prod(1.0 + arr) - 1.0)
        sharpe = float(arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0.0
        print("\n" + "=" * 60)
        print(f"POOLED  trades={len(arr)} win={ (arr>0).mean():.1%} "
              f"total_ret={total:.2%} avg={arr.mean()*1e4:.1f}bp sharpe={sharpe:.2f}")
        summary["_pooled"] = {
            "n_trades": len(arr),
            "win_rate": round(float((arr > 0).mean()), 4),
            "total_return": round(total, 4),
            "sharpe": round(sharpe, 3),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {out_dir/'summary.json'}\n")


if __name__ == "__main__":
    main()
