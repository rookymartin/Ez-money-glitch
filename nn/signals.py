"""
signals.py — Generate trading signals for one or more tickers.

Combines:
  1. Neural network predictions (SwingTradeNet)
  2. Classic TA signals (RSI, MACD, Bollinger, Volume Breakout)

Usage:
    python -m nn.signals AAPL
    python -m nn.signals AAPL NVDA TSLA --no-nn
    python -m nn.signals --mode swing        # scan all swing candidates
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from nn.indicators import build_features, rsi, macd, bollinger_bands, volume_ratio, atr
from data.fetch import get_prices
from data.universe import get_swing_candidates

MODELS_DIR = _PROJECT_ROOT / "models"
MODEL_PATH  = MODELS_DIR / "swing_model.pt"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

SEQ_LEN = 30


def _load_model(device):
    """Load trained SwingTradeNet. Returns (model, scaler) or (None, None) if not found."""
    try:
        import torch
        import pickle
        from nn.model import SwingTradeNet, N_META

        if not MODEL_PATH.exists():
            return None, None

        model = SwingTradeNet(n_features=55, n_meta=N_META, seq_len=SEQ_LEN)
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        scaler = None
        if SCALER_PATH.exists():
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)

        return model, scaler
    except Exception as e:
        print(f"  [nn] could not load model: {e}")
        return None, None


def _nn_signal(ticker: str, df: pd.DataFrame, model, scaler, device) -> Optional[dict]:
    """Run NN inference on the last SEQ_LEN bars. Returns dict or None."""
    try:
        import torch
        from nn.model import N_META

        feat_df = build_features(df)
        feat_cols = [c for c in feat_df.columns if c not in ("Open","High","Low","Close","Volume")]
        features = feat_df[feat_cols].values.astype(np.float32)

        if len(features) < SEQ_LEN:
            return None

        seq = features[-SEQ_LEN:]
        if np.isnan(seq).any():
            seq = np.nan_to_num(seq, nan=0.0)

        if scaler is not None:
            n_feat = seq.shape[1]
            seq_flat = seq.reshape(-1, n_feat)
            seq_flat = scaler.transform(seq_flat)
            seq = seq_flat.reshape(SEQ_LEN, n_feat)

        x = torch.from_numpy(seq).unsqueeze(0).to(device)
        meta = torch.zeros(1, N_META).to(device)

        with torch.no_grad():
            preds = model.predict(x, meta)
        return preds[0]
    except Exception as e:
        return None


def _ta_signals(df: pd.DataFrame) -> dict:
    """Compute classic TA signals on recent data."""
    rsi_14 = rsi(df, 14).iloc[-1]
    macd_df = macd(df)
    macd_val  = macd_df["macd"].iloc[-1]
    macd_hist = macd_df["histogram"].iloc[-1]
    macd_prev = macd_df["histogram"].iloc[-2]
    bb = bollinger_bands(df)
    bb_pct_b = bb["bb_pct_b"].iloc[-1]
    vol_r = volume_ratio(df, 20).iloc[-1]
    atr_val = atr(df, 14).iloc[-1]
    close = df["Close"].iloc[-1]

    ret_5d = (df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100 if len(df) > 6 else 0

    rsi_signal = (
        "BUY" if rsi_14 < 30 else
        "SELL" if rsi_14 > 70 else
        "OVERSOLD_WATCH" if rsi_14 < 45 else
        "NEUTRAL"
    )
    macd_signal = (
        "BUY" if (macd_hist > 0 and macd_prev <= 0) else
        "SELL" if (macd_hist < 0 and macd_prev >= 0) else
        "BULLISH" if macd_hist > 0 else
        "BEARISH"
    )
    bb_signal = (
        "BUY" if bb_pct_b < 0.1 else
        "SELL" if bb_pct_b > 0.9 else
        "NEUTRAL"
    )
    vol_signal = "HIGH_VOLUME" if vol_r > 1.5 else "LOW_VOLUME" if vol_r < 0.7 else "NORMAL"

    ta_votes = sum([
        1 if rsi_signal in ("BUY","OVERSOLD_WATCH") else -1 if rsi_signal == "SELL" else 0,
        1 if macd_signal in ("BUY","BULLISH") else -1,
        1 if bb_signal == "BUY" else -1 if bb_signal == "SELL" else 0,
    ])
    ta_composite = "BUY" if ta_votes >= 2 else "SELL" if ta_votes <= -2 else "NEUTRAL"

    return {
        "rsi_14": round(float(rsi_14), 1),
        "rsi_signal": rsi_signal,
        "macd_signal": macd_signal,
        "macd_hist": round(float(macd_hist), 4),
        "bb_pct_b": round(float(bb_pct_b), 2),
        "bb_signal": bb_signal,
        "vol_ratio": round(float(vol_r), 2),
        "vol_signal": vol_signal,
        "atr_pct": round(float(atr_val / close * 100), 2),
        "ret_5d_pct": round(float(ret_5d), 2),
        "ta_composite": ta_composite,
        "ta_votes": ta_votes,
    }


def scan_ticker(ticker: str, model=None, scaler=None, device=None,
                years: int = 3) -> Optional[dict]:
    """Generate full signal report for one ticker. Returns dict or None on failure."""
    df = get_prices(ticker, years=years)
    if df is None or len(df) < SEQ_LEN + 20:
        return None

    ta = _ta_signals(df)
    nn_pred = None

    if model is not None:
        nn_pred = _nn_signal(ticker, df, model, scaler, device)

    close  = float(df["Close"].iloc[-1])
    date   = df.index[-1].strftime("%Y-%m-%d")

    composite = ta["ta_composite"]
    if nn_pred is not None:
        nn_vote = 1 if nn_pred["signal"] == "BUY" else -1 if nn_pred["signal"] == "SELL" else 0
        combined_votes = ta["ta_votes"] + nn_vote * 1.5
        composite = "BUY" if combined_votes >= 2 else "SELL" if combined_votes <= -2 else "NEUTRAL"

    result = {
        "ticker": ticker,
        "date": date,
        "close": round(close, 2),
        "composite": composite,
        "ta": ta,
    }
    if nn_pred is not None:
        result["nn"] = {
            "signal": nn_pred["signal"],
            "confidence": round(nn_pred["confidence"], 3),
            "prob_buy": round(nn_pred["prob_buy"], 3),
            "prob_sell": round(nn_pred["prob_sell"], 3),
            "regime": nn_pred.get("regime", "UNKNOWN"),
        }
    return result


def scan_universe(tickers: list, use_nn: bool = True) -> list:
    """Scan multiple tickers and return list of signal dicts."""
    device = None
    model = None
    scaler = None

    if use_nn:
        try:
            import torch
            from nn.model import get_device
            device = get_device()
            model, scaler = _load_model(device)
            if model is None:
                print("  [nn] No trained model found — running TA only. Train first: python run.py train")
        except ImportError:
            print("  [nn] torch not available — running TA only")

    results = []
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}…", end=" ", flush=True)
        sig = scan_ticker(ticker, model, scaler, device)
        if sig:
            results.append(sig)
            print(f"{sig['composite']} (RSI={sig['ta']['rsi_14']})")
        else:
            print("skip")

    results.sort(key=lambda x: (
        0 if x["composite"] == "BUY" else 1 if x["composite"] == "NEUTRAL" else 2,
        -x["ta"]["vol_ratio"]
    ))
    return results


def print_report(results: list):
    """Pretty-print signal results to terminal."""
    print(f"\n{'='*70}")
    print(f"  SIGNAL REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    for r in results:
        nn_str = ""
        if "nn" in r:
            nn_str = f"  NN={r['nn']['signal']}({r['nn']['confidence']:.0%})"
        ta = r["ta"]
        print(f"  {r['ticker']:12s} {r['composite']:8s}  "
              f"RSI={ta['rsi_14']:5.1f} MACD={ta['macd_signal']:8s} "
              f"Vol={ta['vol_ratio']:.1f}x  5d={ta['ret_5d_pct']:+.1f}%{nn_str}")
    print(f"{'='*70}")
    buys  = [r["ticker"] for r in results if r["composite"] == "BUY"]
    sells = [r["ticker"] for r in results if r["composite"] == "SELL"]
    if buys:
        print(f"\n  BUY signals:  {', '.join(buys)}")
    if sells:
        print(f"  SELL signals: {', '.join(sells)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*", default=None)
    parser.add_argument("--mode", choices=["swing","sp500"], default="swing")
    parser.add_argument("--no-nn", action="store_true")
    parser.add_argument("--output", default=None, help="Write results to JSON file")
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    elif args.mode == "sp500":
        from data.universe import get_sp500
        tickers = get_sp500()[:50]
    else:
        tickers = get_swing_candidates()

    results = scan_universe(tickers, use_nn=not args.no_nn)
    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
