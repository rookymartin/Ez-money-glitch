"""
classic.py — Classic technical analysis strategy signal generators.

Each function takes a price DataFrame (OHLCV with DatetimeIndex) and returns
a pd.Series of float signals aligned to the same index:
  > 0 = bullish / buy pressure
  < 0 = bearish / sell pressure
  = 0 = neutral

Usage:
    from strategies.classic import rsi_signal, macd_signal, bollinger_signal
    signals = rsi_signal(df)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _rsi_raw(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# RSI Strategy
# Signal: +1 in oversold zone (< oversold), -1 in overbought (> overbought)
# Best params found: period=10, oversold=40, overbought=65 (RSIAlpha champion)
# ---------------------------------------------------------------------------

def rsi_signal(df: pd.DataFrame, period: int = 14,
               oversold: float = 30, overbought: float = 70,
               ma_period: int = 200) -> pd.Series:
    """
    RSI mean-reversion signal.
    Positive when RSI is in the oversold zone AND price is above MA filter.
    Negative when RSI is in the overbought zone.
    """
    rsi_val = _rsi_raw(df["Close"], period)
    ma = _sma(df["Close"], ma_period)
    above_ma = (df["Close"] > ma).astype(float)

    signal = pd.Series(0.0, index=df.index)
    signal = signal.where(rsi_val >= oversold, (oversold - rsi_val) / oversold * above_ma)
    signal = signal.where(rsi_val <= overbought, -(rsi_val - overbought) / (100 - overbought))
    return signal.clip(-1, 1)


# ---------------------------------------------------------------------------
# MACD Crossover Strategy
# Signal: positive when MACD histogram is positive and rising
# ---------------------------------------------------------------------------

def macd_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                signal_period: int = 9) -> pd.Series:
    """
    MACD histogram crossover signal.
    Normalized by ATR to be scale-invariant across tickers.
    """
    ema_fast = _ema(df["Close"], fast)
    ema_slow = _ema(df["Close"], slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    atr_val = _atr(df).replace(0, np.nan)
    return (histogram / atr_val).clip(-2, 2) / 2


# ---------------------------------------------------------------------------
# Bollinger Band Mean Reversion
# Signal: positive when price near lower band, negative near upper
# ---------------------------------------------------------------------------

def bollinger_signal(df: pd.DataFrame, period: int = 20,
                     std_mult: float = 2.0) -> pd.Series:
    """
    Bollinger Band %B signal. 0=lower band (buy), 1=upper band (sell).
    Returns -(pct_b - 0.5) * 2, so: +1 at lower band, -1 at upper band.
    """
    mid = _sma(df["Close"], period)
    std = df["Close"].rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    pct_b = (df["Close"] - lower) / (upper - lower).replace(0, np.nan)
    return -(pct_b.fillna(0.5) - 0.5) * 2


# ---------------------------------------------------------------------------
# EMA Crossover (trend following)
# ---------------------------------------------------------------------------

def ema_cross_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    EMA fast/slow crossover. +1 when fast > slow (uptrend), -1 otherwise.
    Normalized by the magnitude of the crossover relative to price.
    """
    fast_ema = _ema(df["Close"], fast)
    slow_ema = _ema(df["Close"], slow)
    cross = (fast_ema - slow_ema) / df["Close"].replace(0, np.nan)
    return cross.clip(-0.1, 0.1) * 10


# ---------------------------------------------------------------------------
# Donchian Channel Breakout (trend + momentum)
# ---------------------------------------------------------------------------

def donchian_signal(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Donchian channel: +1 when price breaks above N-day high, -1 below N-day low.
    """
    high_n = df["High"].rolling(period).max().shift(1)
    low_n  = df["Low"].rolling(period).min().shift(1)
    mid_n  = (high_n + low_n) / 2

    signal = (df["Close"] - mid_n) / (high_n - low_n).replace(0, np.nan)
    return signal.clip(-1, 1).fillna(0)


# ---------------------------------------------------------------------------
# Volume Breakout (the OpenClaw champion standalone)
# Best combo: threshold=1.5, period=20 → Sharpe 0.39, 9.9% CAGR
# ---------------------------------------------------------------------------

def volume_breakout_signal(df: pd.DataFrame, period: int = 20,
                           threshold: float = 1.5) -> pd.Series:
    """
    Volume breakout signal: high volume + upward price move = strong BUY.
    Returns positive when volume spike accompanies positive price action.
    """
    avg_vol = df["Volume"].rolling(period).mean()
    vol_ratio = df["Volume"] / avg_vol.replace(0, np.nan)
    ret_1d = df["Close"].pct_change()

    vol_spike = (vol_ratio >= threshold).astype(float)
    signal = ret_1d * vol_spike
    return signal.clip(-0.1, 0.1) * 10


# ---------------------------------------------------------------------------
# Momentum (price rate of change)
# ---------------------------------------------------------------------------

def momentum_signal(df: pd.DataFrame, period: int = 21) -> pd.Series:
    """
    Simple N-day price momentum. Normalized by volatility.
    """
    ret = df["Close"].pct_change(period)
    vol = df["Close"].pct_change().rolling(period).std().replace(0, np.nan)
    return (ret / vol).clip(-3, 3) / 3


# ---------------------------------------------------------------------------
# Composite TA signal (weighted average of all classic strategies)
# ---------------------------------------------------------------------------

def composite_ta_signal(df: pd.DataFrame) -> pd.Series:
    """
    Weighted composite of all classic strategies.
    Weights tuned from OpenClaw research results.
    """
    signals = {
        "rsi":     (rsi_signal(df), 0.25),
        "macd":    (macd_signal(df), 0.20),
        "bb":      (bollinger_signal(df), 0.15),
        "ema":     (ema_cross_signal(df), 0.20),
        "vol_bo":  (volume_breakout_signal(df), 0.20),
    }
    composite = pd.Series(0.0, index=df.index)
    for name, (sig, weight) in signals.items():
        composite += sig.fillna(0) * weight
    return composite.clip(-1, 1)


if __name__ == "__main__":
    import yfinance as yf
    df = yf.download("AAPL", period="2y", auto_adjust=True, progress=False)
    df.columns = [c[0] for c in df.columns] if isinstance(df.columns[0], tuple) else df.columns
    print("RSI last:", rsi_signal(df).iloc[-1])
    print("MACD last:", macd_signal(df).iloc[-1])
    print("Composite:", composite_ta_signal(df).iloc[-1])
