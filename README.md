# binance-hft-bot

Roboticized scalping bot for **Binance USD-M Futures**. It scans the whole
futures universe, picks the most liquid & volatile symbols, streams live market
data, runs a pool of analyzer strategies, aggregates their signals, applies risk
management, and executes fast (few-minute) trades.

Starts in **paper-trading** mode (simulated fills on live market data). Switch to
**live** with a single config flag once you're comfortable.

> ⚠️ Trading futures with leverage is risky. Start on paper / testnet and with a
> tiny deposit. This software comes with no warranty.

## Architecture

```
                +------------------+
                |     Scanner      |  rank universe by volume/spread/volatility
                +--------+---------+
                         | top-N symbols
                +--------v---------+
                |   Market Data    |  WebSocket klines + order book (+ REST seed)
                +--------+---------+
                         |
        +----------------+----------------+
        |                                 |
+-------v--------+              +---------v--------+
|  Momentum EMA  |              | OrderBook Imbal. |   ... more strategies
+-------+--------+              +---------+--------+
        |            signals              |
        +----------------+----------------+
                         |
                +--------v---------+
                |   Aggregator     |  weighted score -> LONG / SHORT / FLAT
                +--------+---------+
                         |
                +--------v---------+
                |  Risk Manager    |  sizing, SL/TP, limits, cooldown, daily loss
                +--------+---------+
                         |
                +--------v---------+
                |    Executor      |  Paper (simulate) | Live (real orders)
                +------------------+
```

### Modules
- `hftbot/scanner.py` — liquidity/volatility ranking of all USD-M perpetuals.
- `hftbot/exchange/binance_client.py` — async REST client (public + signed).
- `hftbot/exchange/feed.py` — combined WebSocket feed (klines + partial depth).
- `hftbot/strategies/` — pluggable analyzer bots (momentum, order-book imbalance, mean-reversion).
- `hftbot/backtest/` — historical data loader, bar-by-bar simulator, metrics.
- `hftbot/aggregator.py` — weighted combination of strategy signals.
- `hftbot/risk.py` — position sizing, stops/targets, exposure & loss limits.
- `hftbot/execution/` — `paper.py` (simulation) and `live.py` (real orders).
- `hftbot/engine.py` — orchestration loop.

## Setup

```bash
# requires Python >= 3.11
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill keys only when going live
```

## Run (paper)

```bash
python run.py            # mode: paper (from config.yaml)
```

Paper mode needs **no API keys** — it uses public market data and simulates fills
(including taker fees) against the live mid price. Trades are logged to
`logs/trades.csv` and a status line prints every 30s.

## Go live (later)

1. Create Binance API keys with **Futures trading enabled** and **withdrawals
   disabled**. Put them in `.env`.
2. Set `mode: live` in `config.yaml` (or `python run.py --mode live`).
3. Start with tiny `risk.risk_per_trade_pct` and low `risk.leverage`.

Optionally set `BINANCE_TESTNET=true` to use the futures testnet first.

## Configuration

All tunables live in `config.yaml`: scanner thresholds, strategy params/weights,
aggregator entry threshold, and full risk settings (leverage, SL/TP, max holding
time, max open positions, daily loss limit, cooldown).

## Backtesting

Before risking money, validate strategies on historical data. The backtester
downloads public USD-M futures klines from `data.binance.vision` (no auth, no
geo-restriction), caches them under `data/klines/`, and replays them bar-by-bar
through the **same** strategies, aggregator and risk parameters as the live bot.

```bash
python backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --interval 1m \
    --start 2024-06-01 --end 2024-06-07

# pick specific strategies (order-book imbalance can't be backtested — no
# historical depth data — so it's skipped automatically):
python backtest.py --symbols SOLUSDT --strategies momentum mean_reversion \
    --start 2024-06-01 --end 2024-06-03
```

Reported metrics per symbol: trades, win rate, return %, PnL, profit factor,
max drawdown, Sharpe, average holding time, fees.

**Reality check:** entries fill at the signal bar's close; stop-loss/take-profit
are checked against each later bar's high/low (stop assumed first if both are
touched); taker fees are charged both sides. With the *default* parameters the
simple momentum/mean-reversion bots do **not** print money on recent data —
fees and noise dominate. Treat the framework as a foundation to research and
tune real edges, not a turnkey profit machine. Recommended path: backtest →
tune → paper on live market → tiny live size.

## Tests

```bash
pip install pytest
pytest
```

## Roadmap
- More analyzer bots (volume breakout, funding-rate, VWAP reversion).
- Parameter optimization / walk-forward over the backtester.
- Redis-backed data bus + multi-process scaling.
- FastAPI dashboard with live PnL and an emergency stop.
