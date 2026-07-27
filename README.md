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
- `hftbot/strategies/` — pluggable analyzer bots (momentum, order-book imbalance).
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

## Tests

```bash
pip install pytest
pytest
```

## Roadmap
- More analyzer bots (mean-reversion RSI/Bollinger, volume breakout, funding-rate).
- Redis-backed data bus + multi-process scaling.
- FastAPI dashboard with live PnL and an emergency stop.
- Backtesting harness over historical klines.
