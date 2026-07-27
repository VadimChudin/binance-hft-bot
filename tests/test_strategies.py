from hftbot.exchange.feed import Kline, SymbolBook
from hftbot.models import Side
from hftbot.strategies.momentum import MomentumStrategy
from hftbot.strategies.orderbook_imbalance import OrderBookImbalanceStrategy


def _book_with_closes(closes):
    book = SymbolBook()
    for i, c in enumerate(closes):
        book.klines.append(Kline(i, c, c, c, c, 1.0))
    book.last_price = closes[-1]
    return book


def test_momentum_long_on_uptrend():
    strat = MomentumStrategy(params={"ema_fast": 5, "ema_slow": 15})
    book = _book_with_closes([100 + i for i in range(40)])
    sig = strat.evaluate("BTCUSDT", book)
    assert sig.side == Side.LONG
    assert sig.score > 0


def test_momentum_short_on_downtrend():
    strat = MomentumStrategy(params={"ema_fast": 5, "ema_slow": 15})
    book = _book_with_closes([200 - i for i in range(40)])
    sig = strat.evaluate("BTCUSDT", book)
    assert sig.side == Side.SHORT
    assert sig.score < 0


def test_orderbook_imbalance_bid_pressure():
    strat = OrderBookImbalanceStrategy(params={"threshold": 0.6})
    book = SymbolBook()
    book.bids = [(100.0, 50.0), (99.9, 40.0)]
    book.asks = [(100.1, 5.0), (100.2, 3.0)]
    sig = strat.evaluate("BTCUSDT", book)
    assert sig.side == Side.LONG


def test_orderbook_imbalance_balanced():
    strat = OrderBookImbalanceStrategy(params={"threshold": 0.65})
    book = SymbolBook()
    book.bids = [(100.0, 10.0)]
    book.asks = [(100.1, 10.0)]
    sig = strat.evaluate("BTCUSDT", book)
    assert sig.side == Side.FLAT
