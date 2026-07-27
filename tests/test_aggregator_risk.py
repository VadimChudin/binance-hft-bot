
from hftbot.aggregator import SignalAggregator
from hftbot.config import AggregatorConfig, RiskConfig
from hftbot.models import Side, Signal
from hftbot.risk import RiskManager


def test_aggregator_long():
    agg = SignalAggregator(AggregatorConfig(entry_threshold=0.5))
    signals = [
        Signal("momentum", "BTCUSDT", 0.8),
        Signal("orderbook_imbalance", "BTCUSDT", 0.6),
    ]
    dec = agg.aggregate("BTCUSDT", signals, {"momentum": 1.0, "orderbook_imbalance": 1.0})
    assert dec.side == Side.LONG
    assert dec.score > 0.5


def test_aggregator_flat_when_conflicting():
    agg = SignalAggregator(AggregatorConfig(entry_threshold=0.5))
    signals = [Signal("a", "X", 0.8), Signal("b", "X", -0.8)]
    dec = agg.aggregate("X", signals, {"a": 1.0, "b": 1.0})
    assert dec.side == Side.FLAT


def test_risk_position_levels_long():
    rm = RiskManager(RiskConfig(stop_loss_pct=0.01, take_profit_pct=0.02, leverage=5,
                                risk_per_trade_pct=0.02))
    pos = rm.build_position("BTCUSDT", Side.LONG, 100.0, 1000.0)
    assert pos.stop_loss < 100.0 < pos.take_profit
    assert pos.quantity > 0


def test_risk_daily_loss_halt():
    rm = RiskManager(RiskConfig(max_daily_loss=50.0))
    assert not rm.trading_halted
    rm.register_close("BTCUSDT", -60.0)
    assert rm.trading_halted
    ok, why = rm.can_open("BTCUSDT", 0)
    assert not ok and "loss" in why


def test_risk_max_positions_and_cooldown():
    rm = RiskManager(RiskConfig(max_open_positions=2, cooldown_sec=1000))
    ok, _ = rm.can_open("BTCUSDT", 2)
    assert not ok
    rm.register_close("ETHUSDT", 1.0)
    ok, why = rm.can_open("ETHUSDT", 0)
    assert not ok and why == "cooldown"


def test_risk_should_close_take_profit():
    rm = RiskManager(RiskConfig(take_profit_pct=0.02, stop_loss_pct=0.01,
                                max_holding_sec=9999))
    pos = rm.build_position("BTCUSDT", Side.LONG, 100.0, 1000.0)
    assert rm.should_close(pos, 103.0) == "take_profit"
    assert rm.should_close(pos, 98.0) == "stop_loss"
    assert rm.should_close(pos, 100.5) is None
