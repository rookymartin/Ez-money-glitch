"""
backtest.py — Walk-forward backtester for trading strategies.

Simulates trading based on strategy signals with walk-forward validation
(train on past data, test on future data — no look-ahead bias).

Usage:
    from strategies.backtest import run_backtest, run_all_strategies
    results = run_all_strategies(tickers=["AAPL","MSFT","NVDA"])
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def _run_strategy_single(df: pd.DataFrame, signal_fn: Callable,
                          hold_days: int = 10, max_positions: int = 1,
                          signal_threshold: float = 0.1,
                          stop_loss_pct: float = 0.0,
                          **fn_kwargs) -> Dict:
    """
    Walk-forward backtest for a single ticker.

    Args:
        df: OHLCV DataFrame
        signal_fn: function(df, **kwargs) → pd.Series of signals
        hold_days: number of bars to hold each position
        signal_threshold: minimum absolute signal to enter
        stop_loss_pct: optional stop loss (0 = disabled)

    Returns:
        dict with equity curve, trades, and metrics
    """
    if len(df) < 100:
        return {"trades": [], "equity": [], "metrics": {}}

    signals = signal_fn(df, **fn_kwargs)
    signals = signals.fillna(0)

    close = df["Close"]
    dates = df.index

    capital = 100_000.0
    equity = [capital]
    equity_dates = [dates[0]]
    trades = []

    in_position = False
    entry_price = 0.0
    entry_date = None
    bars_held = 0
    position_size = capital  # always 100% in
    direction = 1

    for i in range(1, len(close)):
        cur_signal = signals.iloc[i]
        cur_price  = close.iloc[i]
        cur_date   = dates[i]

        if in_position:
            bars_held += 1
            cur_val = capital + (cur_price / entry_price - 1) * direction * position_size

            hit_stop = False
            if stop_loss_pct > 0:
                loss_pct = (cur_price / entry_price - 1) * direction
                if loss_pct < -stop_loss_pct:
                    hit_stop = True

            if bars_held >= hold_days or hit_stop:
                ret = (cur_price / entry_price - 1) * direction
                pnl = ret * position_size
                capital += pnl
                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date":  str(cur_date.date()),
                    "direction":  "long" if direction == 1 else "short",
                    "entry":      round(entry_price, 2),
                    "exit":       round(cur_price, 2),
                    "return_pct": round(ret * 100, 3),
                    "pnl":        round(pnl, 2),
                    "stop_hit":   hit_stop,
                })
                in_position = False
                position_size = capital

        else:
            if abs(cur_signal) >= signal_threshold:
                in_position = True
                entry_price = cur_price
                entry_date  = cur_date
                bars_held   = 0
                direction   = 1 if cur_signal > 0 else -1
                position_size = capital

        equity.append(capital if not in_position else
                      capital + (cur_price / entry_price - 1) * direction * position_size)
        equity_dates.append(cur_date)

    eq_series = pd.Series(equity, index=equity_dates)
    metrics = _compute_metrics(eq_series, trades)

    return {
        "trades": trades,
        "equity": [(str(d.date()), round(v, 2)) for d, v in zip(equity_dates, equity)],
        "metrics": metrics,
    }


def _compute_metrics(equity: pd.Series, trades: list) -> dict:
    """Compute Sharpe, CAGR, max drawdown, win rate."""
    if len(equity) < 2:
        return {}

    daily_ret = equity.pct_change().dropna()
    years = len(daily_ret) / 252

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(years, 0.1)) - 1
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    win_rate = 0.0
    if trades:
        winning = sum(1 for t in trades if t["return_pct"] > 0)
        win_rate = winning / len(trades)

    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    return {
        "sharpe":       round(float(sharpe), 3),
        "cagr_pct":     round(float(cagr) * 100, 2),
        "max_dd_pct":   round(float(max_dd) * 100, 2),
        "win_rate":     round(float(win_rate), 3),
        "total_ret_pct": round(float(total_return) * 100, 2),
        "n_trades":     len(trades),
        "years":        round(years, 1),
    }


# ---------------------------------------------------------------------------
# Multi-ticker portfolio backtest
# ---------------------------------------------------------------------------

def run_backtest(tickers: List[str], strategy_name: str,
                 signal_fn: Callable, hold_days: int = 10,
                 signal_threshold: float = 0.1, years: int = 13,
                 stop_loss_pct: float = 0.0,
                 verbose: bool = True, **fn_kwargs) -> dict:
    """
    Run a strategy against multiple tickers and combine equity curves.

    Returns:
        {
          "strategy": name,
          "tickers": tickers,
          "portfolio_equity": [...],
          "metrics": {...},
          "ticker_results": {ticker: result_dict}
        }
    """
    from data.fetch import get_prices

    ticker_results = {}
    all_equities = []

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i}/{len(tickers)}] {ticker}…", end=" ", flush=True)
        df = get_prices(ticker, years=years)
        if df is None or len(df) < 100:
            if verbose:
                print("skip (no data)")
            continue
        result = _run_strategy_single(df, signal_fn, hold_days=hold_days,
                                       signal_threshold=signal_threshold,
                                       stop_loss_pct=stop_loss_pct,
                                       **fn_kwargs)
        ticker_results[ticker] = result
        if result["equity"]:
            eq_df = pd.Series(
                [v for _, v in result["equity"]],
                index=pd.to_datetime([d for d, _ in result["equity"]])
            )
            all_equities.append(eq_df / eq_df.iloc[0])
        if verbose:
            m = result["metrics"]
            print(f"Sharpe={m.get('sharpe','?')}, CAGR={m.get('cagr_pct','?')}%")

    if not all_equities:
        return {"strategy": strategy_name, "metrics": {}, "ticker_results": {}}

    combined = pd.concat(all_equities, axis=1).mean(axis=1).ffill().bfill()
    combined = combined * 100_000
    all_trades = []
    for tr in ticker_results.values():
        all_trades.extend(tr.get("trades", []))

    port_metrics = _compute_metrics(combined, all_trades)

    return {
        "strategy":        strategy_name,
        "tickers":         tickers,
        "portfolio_equity": [(str(d.date()), round(v, 2))
                              for d, v in combined.items()],
        "metrics":         port_metrics,
        "ticker_results":  ticker_results,
        "n_tickers":       len(ticker_results),
    }


# ---------------------------------------------------------------------------
# Run all standard strategies
# ---------------------------------------------------------------------------

def run_all_strategies(tickers: Optional[List[str]] = None,
                       years: int = 13,
                       output_path: Optional[Path] = None,
                       verbose: bool = True) -> List[dict]:
    """
    Run all classic + advanced strategies and write results JSON.
    Returns list of result dicts sorted by Sharpe ratio.
    """
    from strategies.classic import (rsi_signal, macd_signal, bollinger_signal,
                                     ema_cross_signal, volume_breakout_signal,
                                     momentum_signal, composite_ta_signal)
    from strategies.advanced import (kalman_trend_signal, momentum_52wk_signal,
                                      gap_fade_signal, rsi_divergence_signal,
                                      overnight_effect_signal)
    from data.universe import get_swing_candidates
    import yfinance as yf

    if tickers is None:
        tickers = get_swing_candidates()[:20]

    strategies = [
        # --- Classic strategies ---
        ("RSI_MR",          rsi_signal,            {"period": 14, "oversold": 35, "overbought": 65}, 10),
        ("RSIAlpha",         rsi_signal,            {"period": 10, "oversold": 50, "overbought": 70, "ma_period": 200}, 21),
        ("MACD_Cross",       macd_signal,           {}, 10),
        ("Bollinger_MR",     bollinger_signal,      {"period": 20, "std_mult": 2.0}, 5),
        ("EMA_Cross_20_50",  ema_cross_signal,      {"fast": 20, "slow": 50}, 10),
        ("VolBreakout",      volume_breakout_signal, {"period": 20, "threshold": 1.5}, 10),
        ("Momentum_21",      momentum_signal,        {"period": 21}, 21),
        ("Composite_TA",     composite_ta_signal,    {}, 10),
        # --- Advanced strategies ---
        ("Kalman_Trend",     kalman_trend_signal,   {}, 21),
        ("Momentum_52wk",    momentum_52wk_signal,  {}, 30),
        ("Gap_Fade",         gap_fade_signal,        {"min_gap_pct": 0.02}, 3),
        ("RSI_Divergence",   rsi_divergence_signal, {}, 10),
        ("Overnight_Effect", overnight_effect_signal, {}, 5),
    ]

    results = []

    print(f"\nRunning {len(strategies)} strategies on {len(tickers)} tickers…\n")

    for strat_name, fn, kwargs, hold in strategies:
        print("-" * 50)
        print(f"Strategy: {strat_name}")
        result = run_backtest(tickers, strat_name, fn,
                               hold_days=hold, years=years,
                               verbose=verbose, **kwargs)
        results.append(result)
        m = result["metrics"]
        print(f"  -> Sharpe={m.get('sharpe','?')} | CAGR={m.get('cagr_pct','?')}% | "
              f"Win={m.get('win_rate','?'):.0%} | Trades={m.get('n_trades','?')}")

    results.sort(key=lambda x: x["metrics"].get("sharpe", -99), reverse=True)

    # Add benchmark (buy-and-hold SPY)
    try:
        spy_df = yf.download("SPY", period=f"{years}y", auto_adjust=True, progress=False)
        if isinstance(spy_df.columns[0], tuple):
            spy_df.columns = [c[0] for c in spy_df.columns]
        if not spy_df.empty:
            spy_ret = spy_df["Close"].pct_change().dropna()
            spy_years = len(spy_ret) / 252
            spy_cagr  = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[0]) ** (1/spy_years) - 1
            spy_sharpe = spy_ret.mean() / spy_ret.std() * np.sqrt(252)
            benchmark = {
                "strategy": "SPY_BuyHold",
                "metrics": {
                    "sharpe": round(float(spy_sharpe), 3),
                    "cagr_pct": round(float(spy_cagr) * 100, 2),
                    "n_trades": 1,
                }
            }
            results.insert(0, benchmark)
    except Exception:
        pass

    output = output_path or RESULTS_DIR / "backtest_results.json"
    with open(output, "w") as f:
        compact = []
        for r in results:
            compact.append({
                "strategy": r["strategy"],
                "metrics": r["metrics"],
                "n_tickers": r.get("n_tickers", 0),
                "portfolio_equity": r.get("portfolio_equity", [])[:500],
            })
        json.dump({
            "ts": datetime.utcnow().isoformat(),
            "tickers": tickers,
            "results": compact,
        }, f, indent=2)
    print(f"\nResults saved to: {output}")

    print(f"\n{'='*60}")
    print(f"  STRATEGY RANKINGS (by Sharpe)")
    print(f"{'='*60}")
    for r in results:
        m = r["metrics"]
        print(f"  {r['strategy']:25s} Sharpe={m.get('sharpe','?'):5}  "
              f"CAGR={m.get('cagr_pct','?'):6}%")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--years", type=int, default=13)
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 10 tickers only")
    args = parser.parse_args()

    from data.universe import get_swing_candidates
    tickers = args.tickers
    if tickers is None:
        tickers = get_swing_candidates()[:10 if args.quick else 30]

    run_all_strategies(tickers=tickers, years=args.years)
