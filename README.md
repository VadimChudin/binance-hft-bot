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

## Quant research: does any edge survive costs?

The `hftbot/research/` package + `research.py` / `trend_backtest.py` are a
proper quant research harness (pandas + scikit-learn). Install extras:

```bash
pip install -r requirements-research.txt
```

### 1) ML directional model at 1-minute scale — NO edge

`research.py` builds ~35 causal features (multi-horizon returns, EMA/RSI/MACD,
Bollinger, realized vol, **order-flow imbalance** from taker-buy volume, trade
counts, Kaufman efficiency ratio, time-of-day), labels bars with a
**triple-barrier** (up/down/timeout), and trains a gradient-boosted classifier
in a **purged walk-forward** (train on the past, predict the unseen future).

```bash
python research.py --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT \
    --start 2024-01-01 --end 2024-06-30 --barrier 0.008 --horizon 120
```

Result on 4 symbols / 6 months: **out-of-sample AUC ≈ 0.50** (0.498–0.515).
Translation: minute-scale direction is ~unpredictable from candle+order-flow
data, and after round-trip costs the strategy loses. This matches reality —
real HFT edge needs L2 order-book depth and latency we cannot get from
historical klines. **Conclusion: don't scalp minutes with this data.**

### 2) Vol-targeted trend-following on 1h — robust positive edge

Where edge historically *does* survive costs is time-series trend following.
`trend_backtest.py` sizes a `sign(EMA_fast − EMA_slow)` signal (trend-gated) to a
target volatility, charges turnover costs, and reports risk-adjusted returns.

```bash
python trend_backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT \
    --interval 1h --start 2021-01-01 --end 2024-12-31
```

Equal-weight portfolio, 2021–2024 (net of 5bp/turn costs), robust across EMA
params (Sharpe 0.8–1.0):

| period | CAGR | Sharpe | max DD |
|---|---|---|---|
| 2021–2024 (full) | ~28–37% | 0.82–0.99 | ~32–40% |
| 2022 bear only | ~39% | ~0.99 | ~20% |
| 2023–2024 | ~25% | ~0.76 | ~32% |

Positive in **both** the bull and the 2022 bear (it shorts) — a genuine,
regime-independent edge.

**Honest caveats:** Sharpe ~0.9 with 30%+ drawdowns is good, not a money
printer; this holds positions for **hours–days** (not minutes), so it is *not*
the original scalping idea; funding fees and slippage beyond 5bp are not yet
modeled; and past performance never guarantees future results. Use small size
and paper-trade first.

### 3) L2 order-flow microstructure — a real short-horizon edge

Binance publishes **free** order-book / order-flow history at
`data.binance.vision` (futures `bookDepth`, `aggTrades`, `metrics`).
`hftbot/research/microstructure.py` turns them into 1-minute features:
order-flow imbalance (signed via `aggTrades`), depth imbalance at ±1/2/5%
(`bookDepth`), book slope, trade intensity, open-interest change, long/short
ratios. `micro_research.py` adds them to the candle features; `micro_sweep.py`
sweeps barrier/horizon × threshold × cost.

**Ablation (BTCUSDT, purged walk-forward, May–Jun 2024):**

| features | OOS AUC |
|---|---|
| candles only | **0.53** |
| candles + L2 order-flow | **0.64** |

Pooled BTC+ETH at a ~6-minute horizon reaches **AUC ≈ 0.70**. The edge is
strongest at *very short* horizons (minutes) and decays as the horizon grows —
i.e. order flow really does predict the next few minutes. This is a large,
genuine improvement over candles alone.

**Is it profitable? Only with maker-grade fees — honest breakdown:**
break-even is ≈ **0.023% per side**. So:

| execution | fee/side | result |
|---|---|---|
| retail taker | ~0.05% | **loses** (edge too thin for the fee) |
| maker / VIP+BNB | ~0.018% | marginally **positive** (~1–3 bp/trade) |

Caveat: the eye-popping "+600% / Sharpe 28" cells `micro_sweep.py` can print at
near-zero fees are **optimistic artifacts** — they compound thousands of tiny
trades and assume idealized fills at the exact barrier price. The trustworthy
number is **~1–3 basis points of *gross* edge per trade**. To capture it you must
(a) trade as a **maker** (limit orders, ~0.018% fee) and (b) actually get filled
without slippage — neither is guaranteed. So: the signal is real and deployable
(the live bot already streams the depth data needed to compute it), but turning
it into stable net profit hinges on execution quality, not on the model.

```bash
pip install -r requirements-research.txt
python micro_research.py --symbols BTCUSDT --start 2024-05-01 --end 2024-06-30 \
    --barrier 0.002 --horizon 15               # add --price-only for the ablation
python micro_sweep.py --symbols BTCUSDT ETHUSDT # barrier/horizon x threshold x cost
```

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
