# Ez-Money-Glitch 💹

Collaborative trading research project between Martin and William.

Distilled from production-grade research infrastructure:
- **SwingTradeNet v6** — 4-branch CNN + Transformer neural network for stock swing trading
- **Auto-Researcher** — continuous 24/7 strategy discovery loop (no ML API needed, pure walk-forward backtest)
- **Strategy Library** — RSI, MACD, Bollinger, Volume Breakout + innovative: Breadth Momentum, Kalman Trend, Hurst Adaptive
- **Dashboard** — live training monitor, strategy hub, researcher feed (localhost, no cloud)

## Quick Start

```bash
pip install -r requirements.txt

# Run signals for a ticker
python run.py signals AAPL

# Backtest all strategies
python run.py backtest

# Start the dashboard
python run.py dashboard
# → http://localhost:8080/index.html

# Train the neural network
python run.py train

# Run auto-researcher (continuous loop)
python run.py research
```

## Architecture

```
nn/             Neural network (SwingTradeNet v6)
strategies/     Strategy library + walk-forward backtester
data/           yfinance data layer + parquet cache
researcher/     Auto-researcher: discovers strategies autonomously
dashboard/      Static HTML dashboard (Chart.js, dark theme)
results/        JSON output (gitignored — generate locally)
models/         NN checkpoints (gitignored — train locally)
```

## Results So Far (OpenClaw research, 12yr backtest, S&P500)

| Strategy | Sharpe | CAGR | Win Rate |
|---|---|---|---|
| BreadthMomentum | **1.02** | 16.0% | — |
| NN × VolBreakout | 0.49 | 10.5% | 57% |
| VolBreakout | 0.39 | 9.9% | 55% |
| NN Confident | 0.37 | 6.8% | 65% |
| Benchmark (SPY) | — | 11.8% | — |

Best combo: regime filter (BreadthThrust ≥ 65%) + RSI momentum entry.

## Key Insight

> "The problem we've found is figuring out the optimal combination of strategies and their weights."
>
> Solution: treat each strategy as an ETF → portfolio optimization on them.
> Auto-researcher does this continuously: proposes combinations, walk-forward tests, promotes winners.
