"""
train.py — Train SwingTradeNet on historical OHLCV data.

Usage:
    python -m nn.train                      # train on 40 swing candidates
    python -m nn.train --years 13           # use 13 years of data
    python -m nn.train --tickers NVDA AMD   # specific tickers
    python -m nn.train --epochs 60          # custom epoch count
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from nn.model import SwingTradeNet, get_device, EarlyStopping, RegimeLoss, count_parameters, N_META
from nn.indicators import build_features
from data.fetch import load_tickers_batch
from data.universe import get_swing_candidates, get_sp500

MODELS_DIR  = _PROJECT_ROOT / "models"
RESULTS_DIR = _PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "swing_model.pt"
LOG_PATH   = RESULTS_DIR / "training_log.json"

SEQ_LEN      = 30
FORWARD_DAYS = 5
BUY_THRESH   = 0.02
SELL_THRESH  = -0.02


def _label(df: pd.DataFrame) -> np.ndarray:
    """Label each bar: 2=BUY, 0=SELL, 1=NEUTRAL based on 5-day forward return."""
    fwd = df["Close"].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)
    labels = np.where(fwd > BUY_THRESH, 2,
             np.where(fwd < SELL_THRESH, 0, 1)).astype(np.int64)
    return labels


def _regime_label(df: pd.DataFrame) -> np.ndarray:
    """Auxiliary task: BULL/SIDEWAYS/BEAR based on position relative to 200-day SMA."""
    sma200 = df["Close"].rolling(200, min_periods=50).mean()
    ret_20 = df["Close"].pct_change(20)
    regime = np.where((df["Close"] > sma200) & (ret_20 > 0.03), 2,
              np.where((df["Close"] < sma200) & (ret_20 < -0.03), 0, 1)).astype(np.int64)
    return regime


def _build_dataset(all_data: dict, verbose: bool = True):
    """Build (X, meta, y_signal, y_regime) arrays from ticker data dict."""
    import torch
    from sklearn.preprocessing import RobustScaler

    X_list, meta_list, sig_list, reg_list = [], [], [], []

    for ticker, df in all_data.items():
        if len(df) < SEQ_LEN + FORWARD_DAYS + 10:
            continue
        try:
            feat_df = build_features(df)
            feat_cols = [c for c in feat_df.columns
                         if c not in ("Open","High","Low","Close","Volume")]
            features = feat_df[feat_cols].values.astype(np.float32)
            signals  = _label(df)
            regimes  = _regime_label(df)

            n = len(features)
            for i in range(SEQ_LEN, n - FORWARD_DAYS):
                seq = features[i - SEQ_LEN:i]
                if np.isnan(seq).any():
                    continue
                sig = signals[i]
                reg = regimes[i]
                meta = np.zeros(N_META, dtype=np.float32)
                X_list.append(seq)
                meta_list.append(meta)
                sig_list.append(sig)
                reg_list.append(reg)
        except Exception as e:
            if verbose:
                print(f"  [train] skip {ticker}: {e}")

    if not X_list:
        raise ValueError("No training samples built — check data")

    X      = np.array(X_list, dtype=np.float32)
    meta   = np.array(meta_list, dtype=np.float32)
    y_sig  = np.array(sig_list, dtype=np.int64)
    y_reg  = np.array(reg_list, dtype=np.int64)

    print(f"  Dataset: {len(X):,} samples, {X.shape[2]} features, seq={SEQ_LEN}")
    class_counts = np.bincount(y_sig, minlength=3)
    print(f"  Class dist: SELL={class_counts[0]} NEUTRAL={class_counts[1]} BUY={class_counts[2]}")

    scaler = RobustScaler()
    n, seq, feat = X.shape
    X_flat = X.reshape(n * seq, feat)
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(n, seq, feat)

    return X, meta, y_sig, y_reg, scaler


def train(tickers: list, years: int = 13, epochs: int = 80,
          lr: float = 1e-3, weight_decay: float = 3e-3, batch_size: int = 128):
    import torch
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    import pickle

    device = get_device()
    print(f"\nDownloading data for {len(tickers)} tickers ({years}yr)…")
    all_data = load_tickers_batch(tickers, years=years)

    print("Building dataset…")
    X, meta, y_sig, y_reg, scaler = _build_dataset(all_data)

    split = int(len(X) * 0.85)
    X_tr, X_val   = X[:split], X[split:]
    m_tr, m_val   = meta[:split], meta[split:]
    s_tr, s_val   = y_sig[:split], y_sig[split:]
    r_tr, r_val   = y_reg[:split], y_reg[split:]

    tr_set = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(m_tr),
                           torch.from_numpy(s_tr), torch.from_numpy(r_tr))
    val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(m_val),
                            torch.from_numpy(s_val), torch.from_numpy(r_val))

    counts = np.bincount(s_tr, minlength=3)
    weights = 1.0 / (counts + 1)
    sample_weights = weights[s_tr]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    tr_loader  = DataLoader(tr_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    n_feat = X.shape[2]
    model = SwingTradeNet(n_features=n_feat, n_meta=N_META, seq_len=SEQ_LEN,
                          dropout=0.4).to(device)
    print(f"Model: {count_parameters(model):,} parameters, features={n_feat}")

    criterion = RegimeLoss(regime_weight=0.3, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stop = EarlyStopping(patience=12)

    best_val_acc = 0.0
    log = []

    print(f"\nTraining for up to {epochs} epochs…")
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for X_b, m_b, s_b, r_b in tr_loader:
            X_b, m_b, s_b, r_b = X_b.to(device), m_b.to(device), s_b.to(device), r_b.to(device)
            optimizer.zero_grad()
            sig_logits, reg_logits = model(X_b, m_b)
            loss, _, _ = criterion(sig_logits, reg_logits, s_b, r_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_b, m_b, s_b, r_b in val_loader:
                X_b, m_b, s_b, r_b = X_b.to(device), m_b.to(device), s_b.to(device), r_b.to(device)
                sig_logits, reg_logits = model(X_b, m_b)
                loss, _, _ = criterion(sig_logits, reg_logits, s_b, r_b)
                val_loss += loss.item()
                preds = sig_logits.argmax(dim=1)
                correct += (preds == s_b).sum().item()
                total += len(s_b)

        tr_l  = tr_loss / len(tr_loader)
        val_l = val_loss / len(val_loader)
        val_acc = correct / total * 100

        print(f"  Epoch {epoch:3d}/{epochs} | train={tr_l:.4f} val={val_l:.4f} acc={val_acc:.1f}%")

        entry = {"epoch": epoch, "train_loss": round(tr_l, 4),
                 "val_loss": round(val_l, 4), "val_acc": round(val_acc, 2),
                 "ts": datetime.utcnow().isoformat()}
        log.append(entry)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            with open(MODELS_DIR / "scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

        status = {
            "epoch": epoch, "epochs_total": epochs,
            "val_acc": round(val_acc, 2), "best_val_acc": round(best_val_acc, 2),
            "train_loss": round(tr_l, 4), "val_loss": round(val_l, 4),
            "status": "training", "ts": datetime.utcnow().isoformat(),
            "n_features": n_feat, "n_samples": len(X),
        }
        with open(LOG_PATH, "w") as f:
            json.dump({"log": log, "status": status}, f, indent=2)

        if early_stop.step(val_l):
            print(f"  Early stopping at epoch {epoch}")
            break

    status["status"] = "done"
    with open(LOG_PATH, "w") as f:
        json.dump({"log": log, "status": status}, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Log saved to:   {LOG_PATH}")
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--years", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sp500", action="store_true", help="Train on full S&P 500 (~500 tickers)")
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    elif args.sp500:
        tickers = get_sp500()
        print(f"Training on full S&P 500: {len(tickers)} tickers")
    else:
        tickers = get_swing_candidates()
        print(f"Training on swing candidates: {len(tickers)} tickers")

    train(tickers, years=args.years, epochs=args.epochs,
          lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size)
