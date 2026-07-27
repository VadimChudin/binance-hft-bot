import asyncio

from hftbot.execution.paper import PaperExecutor
from hftbot.models import Position, Side


def test_paper_profitable_long():
    ex = PaperExecutor(starting_equity=1000.0, taker_fee_pct=0.0)
    pos = Position(
        symbol="BTCUSDT", side=Side.LONG, entry_price=100.0, quantity=1.0,
        notional=100.0, stop_loss=99.0, take_profit=102.0,
    )
    asyncio.run(ex.open(pos))
    trade = asyncio.run(ex.close(pos, 102.0, "take_profit"))
    assert trade.pnl == 2.0
    assert ex.equity == 1002.0


def test_paper_fee_reduces_pnl():
    ex = PaperExecutor(starting_equity=1000.0, taker_fee_pct=0.001)
    pos = Position(
        symbol="BTCUSDT", side=Side.SHORT, entry_price=100.0, quantity=1.0,
        notional=100.0, stop_loss=101.0, take_profit=98.0,
    )
    asyncio.run(ex.open(pos))
    trade = asyncio.run(ex.close(pos, 98.0, "take_profit"))
    # gross = 2.0, minus fees -> less than 2.0
    assert trade.pnl < 2.0
    assert trade.fees > 0
