"""
auto_researcher.py — Continuous strategy discovery loop.

Runs autonomously, proposing strategy parameter combinations, testing them
via walk-forward backtest, and keeping the champions.

Inspired by Karpathy's auto-research concept:
  "Propose → Test → Keep if better → Exploit winner → Repeat"

Scoring: 60% Sharpe + 40% CAGR-toward-2x-index (21.8% target)
Champion: whoever has the best composite score after walk-forward validation.

No Telegram. Writes status to results/researcher_status.json for dashboard.

Usage:
    python -m researcher.auto_researcher              # run forever
    python -m researcher.auto_researcher --hours 8   # run for 8 hours
    python -m researcher.auto_researcher --dry-run   # test one loop iteration
"""

import json
import math
import random
import signal
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = _PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH    = RESULTS_DIR / "auto_research_log.json"
BEST_PATH   = RESULTS_DIR / "auto_research_best.json"
STATUS_PATH = RESULTS_DIR / "researcher_status.json"

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

INDEX_1X_CAGR = 11.8    # SPY long-run CAGR %
INDEX_2X_CAGR = 21.8    # 2x leveraged index CAGR % (the target bar)
SHARPE_WEIGHT = 0.60
CAGR_WEIGHT   = 0.40

EXPLOIT_ROUNDS = 5       # how many param perturbations per champion before next random sample
MIN_TRADES     = 20      # minimum trades for a result to be valid


def composite_score(sharpe: float, cagr_pct: float) -> float:
    """60% Sharpe + 40% CAGR-toward-2x-index composite score."""
    sharpe_norm = (sharpe + 0.5) / 2.0
    cagr_norm   = max(0, min(1, cagr_pct / INDEX_2X_CAGR))
    return SHARPE_WEIGHT * sharpe_norm + CAGR_WEIGHT * cagr_norm


# ---------------------------------------------------------------------------
# Strategy parameter space
# ---------------------------------------------------------------------------

STRATEGY_SPACE = [
    # (name, signal_function_name, param_ranges)
    ("RSI_MR",         "rsi_signal",         {"period": (7, 21), "oversold": (25, 45), "overbought": (55, 75), "ma_period": [50, 100, 200]}),
    ("RSIAlpha",       "rsi_signal",         {"period": (5, 20), "oversold": (40, 60), "overbought": (60, 80), "ma_period": [100, 200]}),
    ("MACD_Cross",     "macd_signal",        {"fast": (8, 16), "slow": (20, 32), "signal_period": (6, 12)}),
    ("Bollinger_MR",   "bollinger_signal",   {"period": (10, 30), "std_mult": [1.5, 2.0, 2.5]}),
    ("EMA_Cross",      "ema_cross_signal",   {"fast": (10, 30), "slow": (40, 100)}),
    ("VolBreakout",    "volume_breakout_signal", {"period": (10, 30), "threshold": [1.2, 1.5, 2.0, 2.5]}),
    ("Momentum",       "momentum_signal",    {"period": (10, 42)}),
    ("Kalman_Trend",   "kalman_trend_signal", {"process_noise": [1e-4, 1e-3, 1e-2], "obs_noise": [0.05, 0.1, 0.2]}),
    ("Momentum_52wk",  "momentum_52wk_signal", {"lookback": [126, 252], "ma_period": [100, 200]}),
    ("Gap_Fade",       "gap_fade_signal",    {"min_gap_pct": [0.01, 0.02, 0.03], "hold_period": (2, 5)}),
    ("RSI_Divergence", "rsi_divergence_signal", {"rsi_period": (10, 20), "window": (10, 30)}),
    ("Overnight",      "overnight_effect_signal", {}),
]

PORTFOLIO_PARAM_GRID = {
    "hold_days":        [5, 10, 21, 30, 30],  # 30 weighted double
    "signal_threshold": [0.05, 0.1, 0.15, 0.2],
    "stop_loss_pct":    [0.0, 0.0, 0.05, 0.10],  # 0 weighted double
}


def _sample_params(param_ranges: dict) -> dict:
    """Sample a parameter combination from the space."""
    params = {}
    for key, val in param_ranges.items():
        if isinstance(val, list):
            params[key] = random.choice(val)
        elif isinstance(val, tuple) and len(val) == 2:
            lo, hi = val
            if isinstance(lo, float) or isinstance(hi, float):
                params[key] = round(random.uniform(lo, hi), 4)
            else:
                params[key] = random.randint(lo, hi)
    return params


def _perturb_params(params: dict, param_ranges: dict, step_frac: float = 0.3) -> dict:
    """Perturb champion params slightly for exploit rounds."""
    new_params = deepcopy(params)
    for key, val in param_ranges.items():
        if key not in new_params:
            continue
        if isinstance(val, list):
            if random.random() < 0.3:
                new_params[key] = random.choice(val)
        elif isinstance(val, tuple):
            lo, hi = val
            cur = new_params[key]
            span = hi - lo
            delta = span * step_frac * (random.random() * 2 - 1)
            if isinstance(lo, int):
                new_params[key] = int(max(lo, min(hi, round(cur + delta))))
            else:
                new_params[key] = round(max(lo, min(hi, cur + delta)), 4)
    return new_params


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _get_signal_fn(fn_name: str):
    """Import and return a strategy signal function by name."""
    from strategies.classic import (rsi_signal, macd_signal, bollinger_signal,
                                     ema_cross_signal, volume_breakout_signal,
                                     momentum_signal, composite_ta_signal)
    from strategies.advanced import (kalman_trend_signal, momentum_52wk_signal,
                                      gap_fade_signal, rsi_divergence_signal,
                                      overnight_effect_signal)
    fns = {
        "rsi_signal": rsi_signal,
        "macd_signal": macd_signal,
        "bollinger_signal": bollinger_signal,
        "ema_cross_signal": ema_cross_signal,
        "volume_breakout_signal": volume_breakout_signal,
        "momentum_signal": momentum_signal,
        "composite_ta_signal": composite_ta_signal,
        "kalman_trend_signal": kalman_trend_signal,
        "momentum_52wk_signal": momentum_52wk_signal,
        "gap_fade_signal": gap_fade_signal,
        "rsi_divergence_signal": rsi_divergence_signal,
        "overnight_effect_signal": overnight_effect_signal,
    }
    return fns.get(fn_name)


def run_experiment(strategy_name: str, fn_name: str, sig_params: dict,
                   port_params: dict, tickers: List[str],
                   years: int = 13) -> Optional[dict]:
    """
    Run one experiment: backtest strategy with given params on all tickers.
    Returns result dict or None on failure.
    """
    from strategies.backtest import run_backtest
    fn = _get_signal_fn(fn_name)
    if fn is None:
        return None

    try:
        result = run_backtest(
            tickers, strategy_name, fn,
            hold_days=port_params.get("hold_days", 10),
            signal_threshold=port_params.get("signal_threshold", 0.1),
            stop_loss_pct=port_params.get("stop_loss_pct", 0.0),
            years=years, verbose=False,
            **sig_params
        )
        m = result.get("metrics", {})
        if not m or m.get("n_trades", 0) < MIN_TRADES:
            return None
        return result
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_log() -> List[dict]:
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text())
        except Exception:
            pass
    return []


def _save_log(log: List[dict]) -> None:
    LOG_PATH.write_text(json.dumps(log[-2000:], indent=2))


def _load_best() -> Optional[dict]:
    if BEST_PATH.exists():
        try:
            return json.loads(BEST_PATH.read_text())
        except Exception:
            pass
    return None


def _save_best(best: dict) -> None:
    BEST_PATH.write_text(json.dumps(best, indent=2))


def _write_status(status: dict) -> None:
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(STATUS_PATH)


def _build_rankings(log: List[dict]) -> List[dict]:
    """Build top strategy rankings from experiment log."""
    by_name: Dict[str, list] = {}
    for exp in log:
        name = exp.get("strategy", "")
        if name not in by_name:
            by_name[name] = []
        by_name[name].append(exp)

    rankings = []
    for name, exps in by_name.items():
        best_exp = max(exps, key=lambda x: x.get("score", 0))
        rankings.append({
            "strategy": name,
            "best_sharpe": best_exp.get("sharpe", 0),
            "best_cagr":   best_exp.get("cagr_pct", 0),
            "best_score":  best_exp.get("score", 0),
            "n_experiments": len(exps),
            "sig_params":  best_exp.get("sig_params", {}),
            "port_params": best_exp.get("port_params", {}),
        })

    rankings.sort(key=lambda x: x["best_score"], reverse=True)
    return rankings


# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------

_RUNNING = True


def _signal_handler(sig, frame):
    global _RUNNING
    print("\n[researcher] Stopping gracefully…")
    _RUNNING = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_researcher(tickers: Optional[List[str]] = None,
                   years: int = 13,
                   max_hours: Optional[float] = None,
                   dry_run: bool = False) -> None:
    """
    Main research loop. Runs until stopped or max_hours reached.

    Args:
        tickers: list of tickers to test on (default: swing candidates)
        years: data history to use
        max_hours: stop after this many hours (None = run forever)
        dry_run: run one iteration and exit
    """
    global _RUNNING

    from data.universe import get_swing_candidates
    if tickers is None:
        tickers = get_swing_candidates()

    print(f"[researcher] Starting with {len(tickers)} tickers, {years}yr history")
    print(f"[researcher] Champion threshold: Sharpe>0.3, CAGR>{INDEX_1X_CAGR}%")
    print(f"[researcher] Results: {RESULTS_DIR}")
    if max_hours:
        print(f"[researcher] Will run for {max_hours}h")

    log = _load_log()
    best = _load_best()
    start_time = time.time()
    experiment_count = len(log)
    exploit_queue: List[Tuple[dict, dict, dict]] = []
    n_per_hour_window = []

    while _RUNNING:
        if max_hours and (time.time() - start_time) / 3600 >= max_hours:
            print(f"[researcher] Max hours reached ({max_hours}h). Stopping.")
            break

        t0 = time.time()
        n_per_hour_window.append(t0)
        n_per_hour_window = [t for t in n_per_hour_window if t > t0 - 3600]

        # Choose next experiment
        if exploit_queue:
            strat_name, fn_name, sig_params, port_params, param_ranges = exploit_queue.pop(0)
            sig_params  = _perturb_params(sig_params, param_ranges)
            port_params = {k: random.choice(v) for k, v in PORTFOLIO_PARAM_GRID.items()}
            mode = "exploit"
        else:
            idx = random.randint(0, len(STRATEGY_SPACE) - 1)
            strat_name, fn_name, param_ranges = STRATEGY_SPACE[idx]
            sig_params  = _sample_params(param_ranges)
            port_params = {k: random.choice(v) for k, v in PORTFOLIO_PARAM_GRID.items()}
            mode = "explore"

        experiment_count += 1
        print(f"[{experiment_count}] {mode}: {strat_name} hold={port_params['hold_days']}d "
              f"params={sig_params}", flush=True)

        result = run_experiment(strat_name, fn_name, sig_params, port_params,
                                 tickers, years=years)

        if result is None:
            print(f"  → failed/no data")
        else:
            m = result["metrics"]
            sharpe   = m.get("sharpe", 0)
            cagr_pct = m.get("cagr_pct", 0)
            score    = composite_score(sharpe, cagr_pct)

            print(f"  → Sharpe={sharpe:.3f} CAGR={cagr_pct:.1f}% Score={score:.3f} "
                  f"Trades={m.get('n_trades',0)}")

            exp_record = {
                "exp_id":    experiment_count,
                "ts":        datetime.utcnow().isoformat(),
                "strategy":  strat_name,
                "fn_name":   fn_name,
                "sig_params": sig_params,
                "port_params": port_params,
                "sharpe":    sharpe,
                "cagr_pct":  cagr_pct,
                "score":     round(score, 4),
                "n_trades":  m.get("n_trades", 0),
                "win_rate":  m.get("win_rate", 0),
                "max_dd_pct": m.get("max_dd_pct", 0),
                "mode":      mode,
            }
            log.append(exp_record)

            if best is None or score > best.get("score", 0):
                best = {**exp_record, "found_at": experiment_count}
                _save_best(best)
                print(f"  ★ NEW CHAMPION: {strat_name} Score={score:.3f}")

                # Queue exploit rounds on new champion
                if len(param_ranges) > 0:
                    for _ in range(EXPLOIT_ROUNDS):
                        exploit_queue.append((strat_name, fn_name,
                                               deepcopy(sig_params),
                                               deepcopy(port_params),
                                               param_ranges))

        _save_log(log)

        rankings = _build_rankings(log)
        status = {
            "running":         True,
            "experiments":     experiment_count,
            "per_hour":        len(n_per_hour_window),
            "ts":              datetime.utcnow().isoformat(),
            "uptime_h":        round((time.time() - start_time) / 3600, 2),
            "champion":        best,
            "top_strategies":  rankings[:10],
            "exploit_queue":   len(exploit_queue),
        }
        _write_status(status)

        if dry_run:
            print("[researcher] Dry-run complete.")
            break

    status = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}
    status["running"] = False
    _write_status(status)
    print(f"[researcher] Done. {experiment_count} experiments run.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--years", type=int, default=13)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_researcher(tickers=args.tickers, years=args.years,
                   max_hours=args.hours, dry_run=args.dry_run)
