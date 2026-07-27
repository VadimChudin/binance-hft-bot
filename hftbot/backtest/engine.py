"""Event-driven backtester.

Replays historical candles bar-by-bar, reusing the *same* strategies,
aggregator and risk parameters as the live bot. Entries fill at the close of
the signal bar; exits (stop-loss / take-profit) are checked against each
subsequent bar's high/low, and a max-holding timeout closes at the bar close.

Order-book strategies emit no signal here (no historical depth), so backtests
exercise the kline-based strategies (momentum, mean_reversion).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..aggregator import SignalAggregator
from ..config import AggregatorConfig, RiskConfig
from ..exchange.feed import Kline, SymbolBook
from ..logger import get_logger
from ..models import Position, Side, Trade
from ..strategies.base import Strategy
from .data import Candle
from .metrics import BacktestResult, compute_metrics

log = get_logger(__name__)


@dataclass
class BacktestConfig:
    start_equity: float = 1000.0
    leverage: int = 5
    risk_per_trade_pct: float = 0.02
    stop_loss_pct: float = 0.004
    take_profit_pct: float = 0.006
    max_holding_sec: int = 300
    cooldown_sec: int = 60
    taker_fee_pct: float = 0.0004
    warmup: int = 50
    window: int = 200

    @classmethod
    def from_risk(cls, risk: RiskConfig, **overrides) -> BacktestConfig:
        cfg = cls(
            start_equity=risk.paper_equity,
            leverage=risk.leverage,
            risk_per_trade_pct=risk.risk_per_trade_pct,
            stop_loss_pct=risk.stop_loss_pct,
            take_profit_pct=risk.take_profit_pct,
            max_holding_sec=risk.max_holding_sec,
            cooldown_sec=risk.cooldown_sec,
            taker_fee_pct=risk.taker_fee_pct,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


class Backtester:
    def __init__(
        self,
        strategies: list[Strategy],
        cfg: BacktestConfig,
        aggregator_cfg: AggregatorConfig,
    ):
        self.strategies = strategies
        self.cfg = cfg
        self.aggregator = SignalAggregator(aggregator_cfg)
        self._weights = {s.name: s.weight for s in strategies}

    def run(self, symbol: str, candles: list[Candle]) -> BacktestResult:
        equity = self.cfg.start_equity
        book = SymbolBook()
        book.klines = deque(maxlen=self.cfg.window)

        position: Position | None = None
        opened_at_sec = 0.0
        trades: list[Trade] = []
        equity_curve: list[float] = []
        cooldown_until = 0.0

        for i, c in enumerate(candles):
            book.klines.append(
                Kline(c.open_time, c.open, c.high, c.low, c.close, c.volume)
            )
            book.last_price = c.close
            now_sec = c.open_time / 1000.0

            # 1) Manage an open position against THIS bar.
            if position is not None:
                exit_price, reason = self._check_exit(position, c, opened_at_sec, now_sec)
                if reason:
                    trade, equity = self._close(position, exit_price, reason,
                                                opened_at_sec, now_sec, equity)
                    trades.append(trade)
                    position = None
                    cooldown_until = now_sec + self.cfg.cooldown_sec

            # 2) Consider a new entry (only when flat, warmed up, off cooldown).
            if position is None and i >= self.cfg.warmup and now_sec >= cooldown_until:
                signals = [s.evaluate(symbol, book) for s in self.strategies]
                decision = self.aggregator.aggregate(symbol, signals, self._weights)
                if decision.side != Side.FLAT:
                    position, opened_at_sec = self._open(
                        symbol, decision.side, c.close, equity, now_sec
                    )

            # 3) Mark-to-market equity for the curve.
            mtm = equity
            if position is not None:
                mtm += position.unrealized_pnl(c.close)
            equity_curve.append(mtm)

        # Close any dangling position at the last close.
        if position is not None and candles:
            last = candles[-1]
            trade, equity = self._close(
                position, last.close, "end_of_data",
                opened_at_sec, last.open_time / 1000.0, equity,
            )
            trades.append(trade)
            equity_curve.append(equity)

        return compute_metrics(symbol, self.cfg.start_equity, trades, equity_curve)

    def _open(self, symbol: str, side: Side, price: float, equity: float,
              now_sec: float) -> tuple[Position, float]:
        notional = equity * self.cfg.risk_per_trade_pct * self.cfg.leverage
        qty = notional / price if price > 0 else 0.0
        if side == Side.LONG:
            stop = price * (1.0 - self.cfg.stop_loss_pct)
            target = price * (1.0 + self.cfg.take_profit_pct)
        else:
            stop = price * (1.0 + self.cfg.stop_loss_pct)
            target = price * (1.0 - self.cfg.take_profit_pct)
        pos = Position(
            symbol=symbol, side=side, entry_price=price, quantity=qty,
            notional=notional, stop_loss=stop, take_profit=target,
            opened_at=now_sec,
        )
        return pos, now_sec

    def _check_exit(self, pos: Position, c: Candle, opened_at_sec: float,
                    now_sec: float) -> tuple[float, str | None]:
        # Conservative: if both stop and target are touched in the same bar,
        # assume the stop is hit first.
        if pos.side == Side.LONG:
            if c.low <= pos.stop_loss:
                return pos.stop_loss, "stop_loss"
            if c.high >= pos.take_profit:
                return pos.take_profit, "take_profit"
        else:
            if c.high >= pos.stop_loss:
                return pos.stop_loss, "stop_loss"
            if c.low <= pos.take_profit:
                return pos.take_profit, "take_profit"
        if now_sec - opened_at_sec >= self.cfg.max_holding_sec:
            return c.close, "max_holding"
        return c.close, None

    def _close(self, pos: Position, price: float, reason: str,
               opened_at_sec: float, now_sec: float,
               equity: float) -> tuple[Trade, float]:
        gross = pos.unrealized_pnl(price)
        entry_fee = pos.notional * self.cfg.taker_fee_pct
        exit_fee = abs(pos.quantity * price) * self.cfg.taker_fee_pct
        net = gross - entry_fee - exit_fee
        equity += net
        trade = Trade(
            symbol=pos.symbol, side=pos.side, entry_price=pos.entry_price,
            exit_price=price, quantity=pos.quantity, pnl=net,
            fees=entry_fee + exit_fee, opened_at=opened_at_sec,
            closed_at=now_sec, reason=reason,
        )
        return trade, equity
