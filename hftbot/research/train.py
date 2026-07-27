"""Walk-forward training and realistic out-of-sample simulation.

Pipeline:
  1. Build features + triple-barrier labels.
  2. Walk-forward: repeatedly train a LightGBM classifier on a trailing window
     and predict the next (unseen) block, purging the horizon-length gap between
     train and test to avoid label leakage.
  3. Simulate trading on the OOS probabilities with round-trip costs, taking
     non-overlapping positions and exiting at the realized triple-barrier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from ..logger import get_logger
from .features import build_features
from .labels import triple_barrier

log = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    barrier: float = 0.005
    horizon: int = 60
    train_bars: int = 60_000
    test_bars: int = 20_000
    cost_per_trade: float = 0.0005  # per side (fee + slippage); round trip = 2x
    thr_long: float = 0.58
    thr_short: float = 0.42
    min_train_pos: int = 500
    model_params: dict = field(default_factory=lambda: {
        "max_iter": 500,
        "learning_rate": 0.03,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 200,
        "l2_regularization": 5.0,
        "max_bins": 255,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 30,
        "random_state": 42,
    })


@dataclass
class SimResult:
    n_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    avg_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    auc: float = float("nan")
    long_trades: int = 0
    short_trades: int = 0
    equity_curve: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"trades={self.n_trades} (L={self.long_trades}/S={self.short_trades}) "
            f"win={self.win_rate:.1%} total_ret={self.total_return:.2%} "
            f"avg={self.avg_return*1e4:.1f}bp sharpe={self.sharpe:.2f} "
            f"maxDD={self.max_drawdown:.2%} auc={self.auc:.3f}"
        )


def prepare(df: pd.DataFrame, cfg: WalkForwardConfig):
    feat_df, feats = build_features(df)
    lab = triple_barrier(feat_df, cfg.barrier, cfg.horizon)
    feat_df = feat_df.join(lab)
    return feat_df, feats


def walk_forward_predict(
    feat_df: pd.DataFrame, feats: list[str], cfg: WalkForwardConfig
) -> pd.Series:
    n = len(feat_df)
    proba = pd.Series(np.nan, index=feat_df.index)
    X_all = feat_df[feats]
    y_all = feat_df["label"]

    start = 0
    while start + cfg.train_bars + 1 < n:
        tr_lo = start
        tr_hi = start + cfg.train_bars  # exclusive
        # Purge the horizon gap so training labels don't peek into the test block.
        purge_hi = max(tr_lo, tr_hi - cfg.horizon)
        te_lo = tr_hi
        te_hi = min(te_lo + cfg.test_bars, n)

        X_tr = X_all.iloc[tr_lo:purge_hi]
        y_tr = y_all.iloc[tr_lo:purge_hi]
        mask = y_tr.notna()  # HistGB handles NaN features natively
        X_tr, y_tr = X_tr[mask], y_tr[mask]

        X_te = X_all.iloc[te_lo:te_hi]

        if y_tr.sum() >= cfg.min_train_pos and (len(y_tr) - y_tr.sum()) >= cfg.min_train_pos:
            model = HistGradientBoostingClassifier(**cfg.model_params)
            model.fit(X_tr, y_tr.astype(int))
            pred = model.predict_proba(X_te)[:, 1]
            proba.loc[X_te.index] = pred
            log.info("fold train[%d:%d] test[%d:%d] pos=%d/%d",
                     tr_lo, purge_hi, te_lo, te_hi, int(y_tr.sum()), len(y_tr))
        start += cfg.test_bars

    return proba


def simulate(feat_df: pd.DataFrame, proba: pd.Series, cfg: WalkForwardConfig) -> SimResult:
    close = feat_df["close"].to_numpy()
    exit_idx = feat_df["exit_idx"].to_numpy()
    exit_price = feat_df["exit_price"].to_numpy()
    p = proba.to_numpy()
    n = len(feat_df)
    rt_cost = 2.0 * cfg.cost_per_trade

    returns: list[float] = []
    longs = shorts = 0
    i = 0
    while i < n:
        pi = p[i]
        if np.isnan(pi) or exit_idx[i] < 0:
            i += 1
            continue
        if pi >= cfg.thr_long:
            entry = close[i]
            gross = exit_price[i] / entry - 1.0
            returns.append(gross - rt_cost)
            longs += 1
            i = int(exit_idx[i]) + 1
        elif pi <= cfg.thr_short:
            entry = close[i]
            gross = entry / exit_price[i] - 1.0
            returns.append(gross - rt_cost)
            shorts += 1
            i = int(exit_idx[i]) + 1
        else:
            i += 1

    res = SimResult(returns=returns, long_trades=longs, short_trades=shorts)
    res.n_trades = len(returns)

    # Classification AUC over labeled OOS rows.
    lab = feat_df["label"]
    both = proba.notna() & lab.notna()
    if both.sum() > 100 and lab[both].nunique() == 2:
        res.auc = roc_auc_score(lab[both].astype(int), proba[both])

    if not returns:
        return res
    arr = np.array(returns)
    res.win_rate = float((arr > 0).mean())
    res.avg_return = float(arr.mean())
    res.total_return = float(np.prod(1.0 + arr) - 1.0)
    if arr.std() > 0:
        res.sharpe = float(arr.mean() / arr.std() * np.sqrt(len(arr)))
    equity = np.cumprod(1.0 + arr)
    res.equity_curve = equity.tolist()
    peak = np.maximum.accumulate(equity)
    res.max_drawdown = float(((peak - equity) / peak).max())
    return res


def run(df: pd.DataFrame, cfg: WalkForwardConfig) -> tuple[SimResult, pd.DataFrame, list[str]]:
    feat_df, feats = prepare(df, cfg)
    proba = walk_forward_predict(feat_df, feats, cfg)
    result = simulate(feat_df, proba, cfg)
    feat_df["proba"] = proba
    return result, feat_df, feats
