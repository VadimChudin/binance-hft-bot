from hftbot.backtest.data import Candle
from hftbot.backtest.engine import BacktestConfig, Backtester
from hftbot.backtest.metrics import compute_metrics
from hftbot.config import AggregatorConfig
from hftbot.exchange.feed import SymbolBook
from hftbot.models import Side, Signal, Trade
from hftbot.strategies.base import Strategy
from hftbot.strategies.mean_reversion import MeanReversionStrategy


class AlwaysLong(Strategy):
    name = "always_long"

    def evaluate(self, symbol: str, book: SymbolBook) -> Signal:
        return Signal(self.name, symbol, 1.0, "test")


def _candle(t, o, h, low, c):
    return Candle(open_time=t * 60_000, open=o, high=h, low=low, close=c, volume=1.0)


def test_backtester_take_profit():
    # Flat warmup then a strong up move to trigger the +0.6% take-profit.
    candles = [_candle(i, 100, 100, 100, 100) for i in range(60)]
    candles.append(_candle(60, 100, 100.1, 99.9, 100.0))  # entry bar
    candles.append(_candle(61, 100, 101.0, 100.0, 100.8))  # hits TP (100.6)
    cfg = BacktestConfig(warmup=50, take_profit_pct=0.006, stop_loss_pct=0.004,
                         taker_fee_pct=0.0, cooldown_sec=0, max_holding_sec=1_000_000)
    bt = Backtester([AlwaysLong()], cfg, AggregatorConfig(entry_threshold=0.5))
    res = bt.run("TESTUSDT", candles)
    assert res.n_trades >= 1
    assert res.trades[0].side == Side.LONG
    assert res.trades[0].reason == "take_profit"
    assert res.total_pnl > 0


def test_metrics_basic():
    trades = [
        Trade("X", Side.LONG, 100, 101, 1, 1.0, 0.0, 0, 60),
        Trade("X", Side.LONG, 100, 99, 1, -1.0, 0.0, 60, 120),
        Trade("X", Side.LONG, 100, 102, 1, 2.0, 0.0, 120, 180),
    ]
    res = compute_metrics("X", 1000.0, trades, [1000, 1001, 1000, 1002])
    assert res.n_trades == 3
    assert abs(res.win_rate - 2 / 3) < 1e-9
    assert res.total_pnl == 2.0
    assert res.profit_factor == 3.0


def test_mean_reversion_oversold_long():
    strat = MeanReversionStrategy(params={"period": 20, "rsi_period": 14,
                                          "rsi_low": 35, "num_std": 2.0})
    book = SymbolBook()
    closes = [100.0] * 20 + [99, 98, 97, 96, 94, 92, 90, 88, 86, 84]
    for i, c in enumerate(closes):
        from hftbot.exchange.feed import Kline
        book.klines.append(Kline(i, c, c, c, c, 1.0))
    sig = strat.evaluate("X", book)
    assert sig.side == Side.LONG
