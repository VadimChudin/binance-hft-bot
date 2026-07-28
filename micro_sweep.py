"""Sweep barrier/horizon x threshold x cost on cached micro features.

Trains the walk-forward model once per (barrier, horizon) and evaluates many
threshold/cost operating points from the same OOS probabilities.
"""

from __future__ import annotations

import argparse
from datetime import date

from hftbot.logger import setup_logging
from hftbot.research.data import load_ohlcv_df
from hftbot.research.features import build_features
from hftbot.research.labels import triple_barrier
from hftbot.research.microstructure import build_micro_features
from hftbot.research.train import WalkForwardConfig, simulate, walk_forward_predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--start", type=date.fromisoformat, default=date(2024, 5, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2024, 6, 30))
    ap.add_argument("--train-bars", type=int, default=40000)
    ap.add_argument("--test-bars", type=int, default=12000)
    args = ap.parse_args()
    setup_logging("WARNING", "logs")

    # Preload features per symbol.
    data = {}
    for sym in args.symbols:
        df = load_ohlcv_df(sym, "1m", args.start, args.end)
        feat_df, feats = build_features(df)
        micro_df, micro_feats = build_micro_features(sym, args.start, args.end)
        overlap = [c for c in micro_feats if c in feat_df.columns]
        if overlap:
            micro_df = micro_df.rename(columns={c: f"mx_{c}" for c in overlap})
            micro_feats = [f"mx_{c}" if c in overlap else c for c in micro_feats]
        feat_df = feat_df.join(micro_df[micro_feats])
        data[sym] = (feat_df, feats + micro_feats)

    grids = [(0.0010, 6), (0.0012, 8), (0.0015, 10), (0.002, 15)]
    thr_pairs = [(0.70, 0.30), (0.75, 0.25), (0.80, 0.20), (0.85, 0.15)]
    costs = [0.0004, 0.0002, 0.0001]

    print(f"\n{'bar':>6} {'hor':>4} {'cost':>7} {'thr':>5} "
          f"{'trades':>7} {'win':>6} {'totRet':>9} {'avg_bp':>7} {'sharpe':>7} {'auc':>6}")
    print("-" * 80)
    for barrier, horizon in grids:
        cfg = WalkForwardConfig(barrier=barrier, horizon=horizon,
                                train_bars=args.train_bars, test_bars=args.test_bars)
        # Train once per (barrier,horizon), pooled over symbols.
        per_sym = []
        for sym, (feat_df, feats) in data.items():
            lab = triple_barrier(feat_df, barrier, horizon)
            fdf = feat_df.join(lab)
            proba = walk_forward_predict(fdf, feats, cfg)
            per_sym.append((fdf, proba))
        for cost in costs:
            for tl, ts in thr_pairs:
                cfg.cost_per_trade = cost
                cfg.thr_long, cfg.thr_short = tl, ts
                rets = []
                aucs = []
                for fdf, proba in per_sym:
                    r = simulate(fdf, proba, cfg)
                    rets.extend(r.returns)
                    if r.auc == r.auc:  # not NaN
                        aucs.append(r.auc)
                if not rets:
                    continue
                import numpy as np
                arr = np.array(rets)
                tot = float(np.prod(1 + arr) - 1)
                shp = float(arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0.0
                auc = sum(aucs) / len(aucs) if aucs else float("nan")
                print(f"{barrier:>6.3f} {horizon:>4} {cost:>7.4f} {tl:>5.2f} "
                      f"{len(arr):>7} {(arr>0).mean():>6.1%} {tot:>9.1%} "
                      f"{arr.mean()*1e4:>7.1f} {shp:>7.2f} {auc:>6.3f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
