"""ML research WITH L2 / order-flow microstructure features.

Combines candle-based features with order-book depth imbalance, signed
order-flow (aggTrades) and open-interest/long-short metrics, then runs the same
purged walk-forward + realistic simulation as research.py.

Example:
    python micro_research.py --symbols BTCUSDT ETHUSDT \
        --start 2024-05-01 --end 2024-06-30 \
        --barrier 0.0018 --horizon 15 --thr-long 0.6 --thr-short 0.4
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from hftbot.logger import get_logger, setup_logging
from hftbot.research.data import load_ohlcv_df
from hftbot.research.features import build_features
from hftbot.research.labels import triple_barrier
from hftbot.research.microstructure import build_micro_features
from hftbot.research.train import (
    WalkForwardConfig,
    simulate,
    walk_forward_predict,
)

log = get_logger("micro")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML research with L2 microstructure")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--barrier", type=float, default=0.0018)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--train-bars", type=int, default=40_000)
    p.add_argument("--test-bars", type=int, default=15_000)
    p.add_argument("--cost", type=float, default=0.0005)
    p.add_argument("--thr-long", type=float, default=0.60)
    p.add_argument("--thr-short", type=float, default=0.40)
    p.add_argument("--price-only", action="store_true",
                   help="ablation: skip microstructure features")
    p.add_argument("--out", default="research_out")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO", "logs")
    cfg = WalkForwardConfig(
        barrier=args.barrier, horizon=args.horizon,
        train_bars=args.train_bars, test_bars=args.test_bars,
        cost_per_trade=args.cost, thr_long=args.thr_long, thr_short=args.thr_short,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nconfig: barrier={cfg.barrier} horizon={cfg.horizon} "
          f"cost/side={cfg.cost_per_trade} thr=({cfg.thr_short},{cfg.thr_long}) "
          f"micro={'OFF' if args.price_only else 'ON'}\n")

    summary: dict[str, dict] = {}
    agg_returns: list[float] = []
    for symbol in args.symbols:
        df = load_ohlcv_df(symbol, "1m", args.start, args.end)
        if df.empty or len(df) < cfg.train_bars + cfg.test_bars:
            log.warning("not enough klines for %s (%d), skipping", symbol, len(df))
            continue
        feat_df, feats = build_features(df)

        if not args.price_only:
            micro_df, micro_feats = build_micro_features(symbol, args.start, args.end)
            if micro_feats:
                overlap = [c for c in micro_feats if c in feat_df.columns]
                if overlap:
                    micro_df = micro_df.rename(columns={c: f"mx_{c}" for c in overlap})
                    micro_feats = [f"mx_{c}" if c in overlap else c for c in micro_feats]
                feat_df = feat_df.join(micro_df[micro_feats])
                feats = feats + micro_feats

        lab = triple_barrier(feat_df, cfg.barrier, cfg.horizon)
        feat_df = feat_df.join(lab)
        proba = walk_forward_predict(feat_df, feats, cfg)
        result = simulate(feat_df, proba, cfg)
        print(f"{symbol:<10} {result.summary()}")
        summary[symbol] = {
            "n_trades": result.n_trades,
            "win_rate": round(result.win_rate, 4),
            "total_return": round(result.total_return, 4),
            "avg_return_bp": round(result.avg_return * 1e4, 2),
            "sharpe": round(result.sharpe, 3),
            "auc": round(result.auc, 4),
            "n_features": len(feats),
        }
        agg_returns.extend(result.returns)

    if agg_returns:
        arr = np.array(agg_returns)
        total = float(np.prod(1.0 + arr) - 1.0)
        sharpe = float(arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0.0
        print("\n" + "=" * 60)
        print(f"POOLED  trades={len(arr)} win={(arr>0).mean():.1%} "
              f"total_ret={total:.2%} avg={arr.mean()*1e4:.1f}bp sharpe={sharpe:.2f}")
        summary["_pooled"] = {"n_trades": len(arr),
                              "win_rate": round(float((arr > 0).mean()), 4),
                              "total_return": round(total, 4), "sharpe": round(sharpe, 3)}

    tag = "price_only" if args.price_only else "micro"
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved {out_dir}/summary_{tag}.json\n")


if __name__ == "__main__":
    main()
