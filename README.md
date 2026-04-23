# Ez-Money-Glitch

Collaborative trading research project — Martin & William.

Distilled from production-grade research infrastructure running 24/7. Everything runs locally on your machine, no cloud needed, no API keys required for the core features.

---

## What This Is

We have a problem: *what's the optimal combination of trading strategies and their weights?*

The solution: treat each strategy like an ETF, then run portfolio optimization on them. The **auto-researcher** does this continuously — it proposes parameter combinations, walk-forward backtests them (no look-ahead bias), and promotes whatever beats the current champion. Leave it running overnight and come back to results.

The **neural network** (SwingTradeNet v6) adds a learned signal on top of the classic strategies. It's a 4-branch CNN + Transformer architecture trained on OHLCV + technical indicators. Currently near-random (~39% val accuracy vs 33% baseline), which means there's a lot of room to improve — that's the interesting research problem.

---

## Setup

**Requirements:** Python 3.10+, pip. GPU optional (speeds up NN training significantly).

```bash
git clone https://github.com/rookymartin/Ez-money-glitch.git
cd Ez-money-glitch
pip install -r requirements.txt
```

That's it. Data downloads automatically from Yahoo Finance and caches locally in `data/cache/` so you don't re-download on every run.

---

## How to Run Everything

### 1. See signals for a ticker

```bash
python run.py signals AAPL
python run.py signals AAPL NVDA TSLA EQNR.OL
python run.py signals --mode swing     # scans all ~42 swing candidates
python run.py signals AAPL --no-nn     # TA only, no neural network needed
```

Output: RSI, MACD, Bollinger, Volume Breakout, NN signal (if model trained), composite BUY/NEUTRAL/SELL.

---

### 2. Backtest all strategies

```bash
python run.py backtest
```

Runs 13 strategies (RSI, MACD, Bollinger, EMA cross, Volume Breakout, Momentum, Kalman Trend, 52-week Momentum, Gap Fade, RSI Divergence, and more) against 20 swing-candidate tickers over 13 years.

Walk-forward: the strategy never sees future data. Each bar uses only past information.

Results saved to `results/backtest_results.json` and visible in the dashboard under **Strategies**.

```bash
# Faster: fewer tickers
python run.py backtest --tickers AAPL MSFT NVDA TSLA GOOGL

# Specific timeframe
python run.py backtest --years 5
```

Takes ~5–15 minutes depending on your machine.

---

### 3. Run the auto-researcher

```bash
python run.py research
```

Continuously proposes strategy + parameter combinations, backtests them, and keeps track of the champion. Runs until you press Ctrl+C.

The loop:
1. **Explore** — pick a random strategy from the library, sample random parameters
2. **Test** — walk-forward backtest on all tickers
3. **Score** — composite: 60% Sharpe + 40% CAGR-toward-21.8% target
4. **Promote** — if score beats current champion, it becomes the new champion
5. **Exploit** — run 5 parameter perturbations around the new champion before next random sample
6. **Repeat**

Status written to `results/researcher_status.json` every iteration — the dashboard reads this live.

```bash
python run.py research --hours 8          # run for 8 hours then stop
python run.py research --dry-run          # one iteration only, for testing
```

Good to leave running overnight. Typically does 20–50 experiments/hour depending on the strategy.

---

### 4. Train the neural network

```bash
python run.py train
```

Trains SwingTradeNet v6 on the swing candidates (42 tickers, 13 years of data). Uses GPU automatically if available (CUDA). Falls back to CPU — works fine, just slower.

Architecture:
- **4-branch Multi-Scale CNN** (kernels 3, 5, 7, 11) → captures patterns at different time scales
- **Transformer encoder** (4 layers, 6 heads, Pre-LayerNorm)
- **DualPooling** (attention + global average → richer context)
- **MetaBranch** for macro features (VIX, rates, etc.)
- **3-class output**: BUY / NEUTRAL / SELL

Labels: forward 5-day return > 2% = BUY, < -2% = SELL, otherwise NEUTRAL.

```bash
python run.py train --epochs 80           # default
python run.py train --sp500               # train on full S&P 500 (~500 tickers, hours)
python run.py train --years 5             # less data, faster
python run.py train --weight-decay 5e-3   # more regularization if overfitting
```

Training log written to `results/training_log.json` — visible live in the dashboard under **Training**.
Model saved to `models/swing_model.pt` (gitignored — train locally, don't commit).

Current state from research: ~39% val accuracy. Random baseline is 33%. Overfitting is the main challenge (train 47% vs val 39%). Things to try: higher dropout, more weight decay, more data, longer sequences.

---

### 5. Open the dashboard

```bash
python run.py dashboard
```

Opens at **http://localhost:8080/index.html**

Four pages:
- **Hub** — status overview (experiments, champion, NN accuracy, best strategy)
- **Training** — live epoch curves, val accuracy, train/val loss (auto-refreshes every 10s)
- **Strategies** — equity curve comparison + Sharpe/CAGR rankings table
- **Researcher** — champion card, top 10 strategy rankings, experiments/hr rate

The dashboard reads from `results/` JSON files. It auto-refreshes — run backtest/researcher/train in one terminal, dashboard in another.

---

## Project Structure

```
Ez-money-glitch/
├── run.py                      ← CLI entrypoint (start here)
├── requirements.txt
│
├── nn/                         ← Neural network
│   ├── model.py                    SwingTradeNet v6 architecture
│   ├── train.py                    Training pipeline (writes results/training_log.json)
│   ├── signals.py                  Inference: load model, generate signals
│   └── indicators.py               55 technical indicators used as NN features
│
├── strategies/                 ← Strategy library
│   ├── classic.py                  RSI, MACD, Bollinger, EMA cross, Volume Breakout, Momentum
│   ├── advanced.py                 BreadthMomentum, Kalman, Hurst Adaptive, 52wk, Gap Fade, RSI Divergence
│   └── backtest.py                 Walk-forward backtester (writes results/backtest_results.json)
│
├── data/                       ← Data layer
│   ├── fetch.py                    yfinance downloader + parquet cache (data/cache/)
│   └── universe.py                 S&P 500, NASDAQ 100, Oslo Børs top 60 ticker lists
│
├── researcher/                 ← Auto-researcher
│   └── auto_researcher.py          Continuous discovery loop (writes results/researcher_status.json)
│
├── dashboard/                  ← Static HTML dashboard
│   ├── index.html                  Hub
│   ├── training.html               NN training monitor
│   ├── strategies.html             Strategy hub + equity curves
│   ├── researcher.html             Auto-researcher live feed
│   └── serve.py                    Simple HTTP server (localhost:8080)
│
├── results/                    ← JSON output (gitignored — generate locally)
│   ├── backtest_results.json
│   ├── training_log.json
│   ├── auto_research_log.json
│   ├── auto_research_best.json
│   └── researcher_status.json
│
└── models/                     ← NN checkpoints (gitignored — train locally)
    ├── swing_model.pt
    └── scaler.pkl
```

`results/` and `models/` are gitignored — each person generates these locally by running the code. Don't commit them.

---

## Strategies in the Library

### Classic (strategies/classic.py)

| Strategy | How it works | Best params found |
|---|---|---|
| RSI_MR | RSI mean-reversion, buy oversold zone + MA filter | period=14, oversold=35, overbought=65 |
| RSIAlpha | RSI momentum (not reversal), zone 50-70 + MA200 | period=10, zone=50-70, ma=200 |
| MACD_Cross | Histogram crossover, ATR-normalized | fast=12, slow=26, signal=9 |
| Bollinger_MR | %B mean-reversion | period=20, std=2.0 |
| EMA_Cross | Fast/slow EMA crossover (trend following) | 20/50 |
| VolBreakout | High volume + price move = strong entry | threshold=1.5x avg vol |
| Momentum | N-day price momentum, vol-normalized | period=21 |

### Advanced (strategies/advanced.py)

| Strategy | How it works | Notes |
|---|---|---|
| BreadthMomentum | Regime filter (≥65% tickers trending up) + RSI entry | Champion: Sharpe 1.02, CAGR 16% |
| Kalman_Trend | Kalman filter smoothed price → trend signal | Less whipsaw than MA in choppy markets |
| Hurst_Adaptive | Hurst exponent detects trending vs mean-reverting regime | Switches strategy based on market character |
| Momentum_52wk | Proximity to 52-week high × MA200 filter | Faber/GEM style momentum |
| Gap_Fade | Fade large opening gaps (mean reversion) | Works on gap > 2% |
| RSI_Divergence | Price makes new high but RSI doesn't confirm → fade | Bearish/bullish divergence |
| Overnight_Effect | Systematic overnight premium capture | Stocks systematically gap up at open |

---

## Key Results (13yr backtest, S&P 500, walk-forward)

Benchmarks: SPY ~11.8% CAGR | 2x Leveraged Index ~21.8% CAGR

| Strategy | Sharpe | CAGR | Win Rate | Notes |
|---|---|---|---|---|
| **BreadthMomentum** | **1.02** | **16.0%** | — | Champion — regime filter is the key |
| NN × VolBreakout | 0.49 | 10.5% | 57% | NN as filter for vol breakout entries |
| VolBreakout(14,1.5) | 0.39 | 9.9% | 55% | Best pure TA strategy |
| NN Confident (>10%) | 0.37 | 6.8% | 65% | High precision, few trades |
| NN Pure | 0.25 | 6.8% | 57% | Model near-random, needs improvement |
| SPY Buy & Hold | ~0.55 | 11.8% | — | Benchmark |

**Key insight:** The BreadthMomentum regime filter is the single biggest alpha source. When ≥65% of tickers have positive 5-day returns, full allocation. When 40–65%, half allocation. Below 40%, cash. Combined with RSI momentum entry (period=10, zone 50-70, MA200 filter).

---

## How to Contribute

1. Add a new strategy in `strategies/advanced.py` — follow the same signature: `def my_signal(df: pd.DataFrame, **params) -> pd.Series`
2. Register it in `researcher/auto_researcher.py` under `STRATEGY_SPACE`
3. Run `python run.py research` and see if the researcher promotes it

The researcher will automatically test your strategy against all others and report if it's better than the current champion.

---

## Data Sources

All data via **yfinance** (free, no API key). Cached locally as parquet files in `data/cache/`.

- **Price data**: S&P 500, NASDAQ 100, Oslo Børs top 60 — 13 years of OHLCV
- **Universe**: Wikipedia-scraped (dynamic, refreshed every 7 days)
- First run downloads data; subsequent runs serve from cache and only fetch new bars

Cache location: `data/cache/ohlcv/{TICKER}.parquet`

---

## FAQ

**Q: Do I need a GPU?**
A: No. CPU works for everything. GPU (CUDA) speeds up NN training ~10x.

**Q: The NN accuracy is only 39%, is that good?**
A: Random baseline is 33% (3 classes). So 39% is better than random but not by much. The model is overfitting (train ~47% vs val ~39%). This is the main research challenge. Things to try: higher dropout (0.4→0.5), more weight decay, longer training sequences, more data.

**Q: How does the auto-researcher avoid overfitting?**
A: Walk-forward backtesting — the strategy is tested only on data it has never seen. The universe is split: test always uses out-of-sample future bars. No parameter fitting on the test set.

**Q: Why is BreadthMomentum so much better than everything else?**
A: Regime filtering. It avoids being in the market during bear phases entirely. Most strategies fail because they trade in both bull and bear markets. Sitting in cash when breadth is low is a huge edge.

**Q: Can we combine with William's XGBoost ensemble?**
A: Yes — the best approach is to treat William's model predictions as an additional signal and feed them into the auto-researcher as a strategy to be weighted alongside the others.
