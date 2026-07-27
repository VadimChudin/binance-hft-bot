from hftbot.indicators import atr, ema, rsi


def test_ema_insufficient():
    assert ema([1, 2], 5) is None


def test_ema_trends_up():
    rising = list(range(1, 30))
    value = ema(rising, 9)
    assert value is not None
    assert value > 20  # tracks recent (higher) values


def test_rsi_all_gains():
    closes = [float(i) for i in range(1, 20)]
    assert rsi(closes, 14) == 100.0


def test_atr_positive():
    highs = [10 + i * 0.5 for i in range(20)]
    lows = [9 + i * 0.5 for i in range(20)]
    closes = [9.5 + i * 0.5 for i in range(20)]
    a = atr(highs, lows, closes, 14)
    assert a is not None and a > 0
