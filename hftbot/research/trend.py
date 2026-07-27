"""Volatility-targeted trend-following backtest (CTA-style).

Unlike minute-scale directional prediction (which shows ~0.5 AUC and loses to
costs), cross-sectional / time-series trend following on higher timeframes has a
long, well-documented positive expectancy — it rides big crypto trends and goes
short in bear markets. This module tests that honestly with turnover costs.

Signal (few, standard params to avoid overfitting):
  trend = sign(EMA_fast - EMA_slow), gated by price vs EMA_slow.
Position is scaled to a target volatility (vol targeting), capped at max_leverage.
Returns are computed on next-bar close-to-close, charging cost on turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendConfig:
    ema_fast: int = 24
    ema_slow: int = 168  # ~1 week on 1h bars
    vol_window: int = 168
    target_vol_annual: float = 0.5  # 50% annualized target
    max_leverage: float = 3.0
    cost_per_turn: float = 0.0005  # per unit position change (fee+slippage)
    bars_per_year: float = 24 * 365


@dataclass
class TrendResult:
    symbol: str
    n_bars: int
    cagr: float
    sharpe: float
    max_drawdown: float
    total_return: float
    turnover: float
    net_returns: pd.Series

    def summary(self) -> str:
        return (
            f"{self.symbol:<10} bars={self.n_bars} CAGR={self.cagr:.1%} "
            f"sharpe={self.sharpe:.2f} maxDD={self.max_drawdown:.1%} "
            f"totRet={self.total_return:.1%} turnover={self.turnover:.0f}"
        )


def backtest_trend(df: pd.DataFrame, cfg: TrendConfig, symbol: str = "") -> TrendResult:
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    ema_f = close.ewm(span=cfg.ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=cfg.ema_slow, adjust=False).mean()
    raw = np.sign(ema_f - ema_s)
    # Trend gate: only long above slow EMA, only short below it.
    raw = raw.where(((raw > 0) & (close > ema_s)) | ((raw < 0) & (close < ema_s)), 0.0)

    realized_vol = ret.rolling(cfg.vol_window).std() * np.sqrt(cfg.bars_per_year)
    target_bar_scale = cfg.target_vol_annual / realized_vol.replace(0.0, np.nan)
    scale = target_bar_scale.clip(upper=cfg.max_leverage).fillna(0.0)

    position = (raw * scale).clip(-cfg.max_leverage, cfg.max_leverage)
    position = position.shift(1).fillna(0.0)  # trade on next bar (no look-ahead)

    turnover = position.diff().abs().fillna(position.abs())
    net = position * ret - cfg.cost_per_turn * turnover

    equity = (1.0 + net).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(df) / cfg.bars_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else -1.0
    sharpe = float(net.mean() / net.std() * np.sqrt(cfg.bars_per_year)) if net.std() > 0 else 0.0
    peak = equity.cummax()
    max_dd = float(((peak - equity) / peak).max())

    return TrendResult(
        symbol=symbol, n_bars=len(df), cagr=cagr, sharpe=sharpe,
        max_drawdown=max_dd, total_return=total_return,
        turnover=float(turnover.sum()), net_returns=net,
    )
