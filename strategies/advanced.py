"""
advanced.py — Innovative strategy signal generators.

These are the strategies discovered through OpenClaw's auto-researcher
and academic research. Each returns a pd.Series of float signals.

Key findings:
  - BreadthMomentum (Sharpe 1.02) is the clear champion
  - Regime filtering is crucial: only trade when market breadth ≥ 65%
  - Kalman + Hurst adaptive help in choppy markets
  - RSI divergence and 52wk momentum are powerful standalone signals
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List


# ---------------------------------------------------------------------------
# Breadth Momentum (the champion — Sharpe 1.02 over 13yr backtest)
# Regime filter: only trade when ≥65% of tickers are trending up
# ---------------------------------------------------------------------------

def breadth_regime_signal(universe_prices: Dict[str, pd.DataFrame],
                           lookback: int = 5,
                           threshold: float = 0.65) -> pd.Series:
    """
    Market breadth regime signal.
    Returns fraction of tickers with positive N-day return.
    Signal > threshold = BULL (full capital), < threshold*0.6 = BEAR (cash).

    This is the key regime filter from the OpenClaw champion strategy:
    BreadthMomentum(lb=5, th=0.65) → Sharpe 1.02, CAGR 16.0%

    Args:
        universe_prices: dict of {ticker: OHLCV DataFrame}
        lookback: days for return calculation (5 = weekly)
        threshold: breadth fraction for BULL regime

    Returns:
        pd.Series indexed by date, values in [0, 1] where 1=full bull
    """
    all_returns = {}
    for ticker, df in universe_prices.items():
        if len(df) > lookback + 5:
            all_returns[ticker] = df["Close"].pct_change(lookback)

    if not all_returns:
        return pd.Series()

    breadth_df = pd.DataFrame(all_returns)
    positive_frac = (breadth_df > 0).mean(axis=1)

    regime = pd.Series(0.0, index=positive_frac.index)
    regime = regime.where(positive_frac < threshold * 0.6, positive_frac / threshold)
    regime = regime.clip(0, 1)
    return regime


def breadth_momentum_signal(df: pd.DataFrame,
                             breadth: Optional[pd.Series] = None,
                             rsi_period: int = 10,
                             rsi_zone_low: float = 50,
                             rsi_zone_high: float = 70,
                             ma_period: int = 200) -> pd.Series:
    """
    RSIAlpha signal gated by breadth regime.
    Best config: rsi_period=10, ma=200, zone=50-70

    For individual ticker screening when breadth series is not available,
    this falls back to pure RSI momentum with MA filter.
    """
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))

    ma = df["Close"].rolling(ma_period, min_periods=50).mean()
    above_ma = (df["Close"] > ma).astype(float)
    in_zone = ((rsi_val >= rsi_zone_low) & (rsi_val <= rsi_zone_high)).astype(float)

    signal = above_ma * in_zone * (rsi_val - rsi_zone_low) / (rsi_zone_high - rsi_zone_low)

    if breadth is not None:
        breadth_aligned = breadth.reindex(df.index, method="ffill").fillna(0.5)
        signal = signal * breadth_aligned

    return signal.clip(0, 1)


# ---------------------------------------------------------------------------
# Kalman Filter Trend Following
# Smooth price series and follow trend with less whipsaw
# ---------------------------------------------------------------------------

def kalman_trend_signal(df: pd.DataFrame,
                         process_noise: float = 1e-3,
                         obs_noise: float = 1e-1) -> pd.Series:
    """
    Kalman filter smoothed trend signal.
    State = smoothed price; signal = normalized deviation from smoothed trend.
    Less whipsawing than simple MA crossovers in choppy regimes.
    """
    closes = df["Close"].values.astype(float)
    n = len(closes)
    state = closes[0]
    covar = 1.0
    smoothed = np.zeros(n)

    for i in range(n):
        # Predict
        state_pred = state
        covar_pred = covar + process_noise
        # Update
        gain = covar_pred / (covar_pred + obs_noise)
        state = state_pred + gain * (closes[i] - state_pred)
        covar = (1 - gain) * covar_pred
        smoothed[i] = state

    smoothed_s = pd.Series(smoothed, index=df.index)
    trend = (df["Close"] - smoothed_s) / smoothed_s.replace(0, np.nan)

    smoothed_ret = smoothed_s.pct_change(5)
    smoothed_vol = smoothed_s.pct_change().rolling(20).std().replace(0, np.nan)
    momentum = (smoothed_ret / smoothed_vol).clip(-3, 3) / 3

    return momentum.fillna(0)


# ---------------------------------------------------------------------------
# Hurst Exponent Adaptive Strategy
# Switches between trend-following and mean-reversion based on Hurst
# H > 0.55 = trending → follow trend
# H < 0.45 = mean-reverting → fade moves
# ---------------------------------------------------------------------------

def hurst_adaptive_signal(df: pd.DataFrame,
                           hurst_window: int = 100,
                           lag_range: int = 20) -> pd.Series:
    """
    Hurst exponent regime detection.
    H > 0.55: trending → momentum signal
    H < 0.45: mean-reverting → contrarian signal
    0.45-0.55: mixed → zero signal
    """
    closes = df["Close"].values.astype(float)
    n = len(closes)
    signal = np.zeros(n)

    for i in range(hurst_window, n):
        window = closes[i - hurst_window:i]
        lags = range(2, min(lag_range, hurst_window // 4))
        rs_vals = []
        for lag in lags:
            subseries = [window[j:j+lag] for j in range(0, len(window)-lag, lag)]
            rs_sub = []
            for sub in subseries:
                if len(sub) < 2:
                    continue
                mean = sub.mean()
                dev = np.cumsum(sub - mean)
                r = dev.max() - dev.min()
                s = sub.std()
                if s > 0:
                    rs_sub.append(r / s)
            if rs_sub:
                rs_vals.append((lag, np.mean(rs_sub)))

        if len(rs_vals) < 3:
            continue
        lags_arr = np.log([x[0] for x in rs_vals])
        rs_arr   = np.log([x[1] for x in rs_vals])
        if lags_arr.std() < 1e-10:
            continue
        hurst = np.polyfit(lags_arr, rs_arr, 1)[0]

        ret = (closes[i] - closes[i-5]) / closes[i-5]
        if hurst > 0.55:
            signal[i] = np.sign(ret) * min(abs(ret * 10), 1.0)
        elif hurst < 0.45:
            signal[i] = -np.sign(ret) * min(abs(ret * 10), 1.0)

    return pd.Series(signal, index=df.index)


# ---------------------------------------------------------------------------
# 52-Week High Momentum (Faber/GEM style)
# Proximity to 52-week high is a strong momentum signal
# ---------------------------------------------------------------------------

def momentum_52wk_signal(df: pd.DataFrame,
                          lookback: int = 252,
                          ma_period: int = 200) -> pd.Series:
    """
    52-week high proximity momentum signal.
    Close / 52-week-high near 1.0 = strong momentum.
    Combined with 200-day MA filter.
    """
    high_52w = df["High"].rolling(lookback, min_periods=100).max()
    proximity = (df["Close"] / high_52w.replace(0, np.nan)).clip(0.5, 1.0)
    proximity_signal = (proximity - 0.5) * 2

    ma = df["Close"].rolling(ma_period, min_periods=50).mean()
    above_ma = (df["Close"] > ma).astype(float)

    return (proximity_signal * above_ma).clip(0, 1)


# ---------------------------------------------------------------------------
# Gap Fade (mean reversion after large opening gaps)
# ---------------------------------------------------------------------------

def gap_fade_signal(df: pd.DataFrame,
                    min_gap_pct: float = 0.02,
                    hold_period: int = 3) -> pd.Series:
    """
    Fade large opening gaps. Large up-gaps are often faded intraday/next-day.
    Signal: negative (sell) after large gap up, positive after large gap down.
    """
    gap = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    large_gap_up   = (gap > min_gap_pct).astype(float)
    large_gap_down = (gap < -min_gap_pct).astype(float)

    # Signal fades over hold_period days
    fade_up   = large_gap_up.rolling(hold_period).max()
    fade_down = large_gap_down.rolling(hold_period).max()

    signal = fade_down - fade_up
    return signal.fillna(0) * (gap.abs() / min_gap_pct).clip(1, 3)


# ---------------------------------------------------------------------------
# RSI Divergence (bearish: price high but RSI doesn't confirm)
# ---------------------------------------------------------------------------

def rsi_divergence_signal(df: pd.DataFrame,
                           rsi_period: int = 14,
                           window: int = 20) -> pd.Series:
    """
    RSI divergence detector.
    Bearish divergence: price makes new high but RSI doesn't → sells.
    Bullish divergence: price makes new low but RSI doesn't → buys.
    """
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))

    price_high = df["Close"].rolling(window).max()
    rsi_high   = rsi_val.rolling(window).max()
    price_low  = df["Close"].rolling(window).min()
    rsi_low    = rsi_val.rolling(window).min()

    at_price_high = (df["Close"] >= price_high * 0.99).astype(float)
    rsi_confirms_high = (rsi_val >= rsi_high * 0.97).astype(float)
    bearish_div = at_price_high * (1 - rsi_confirms_high)

    at_price_low = (df["Close"] <= price_low * 1.01).astype(float)
    rsi_confirms_low = (rsi_val <= rsi_low * 1.03).astype(float)
    bullish_div = at_price_low * (1 - rsi_confirms_low)

    signal = bullish_div - bearish_div
    return signal.clip(-1, 1)


# ---------------------------------------------------------------------------
# Breadth Thrust (rapid expansion of advancing tickers)
# Zweig Breadth Thrust indicator variant
# ---------------------------------------------------------------------------

def breadth_thrust_signal(universe_prices: Dict[str, pd.DataFrame],
                           lookback: int = 10,
                           thrust_threshold: float = 0.615) -> pd.Series:
    """
    Breadth thrust: triggers when advancing tickers surge from <40% to >61.5%
    within 10 trading days (Zweig Breadth Thrust variant).
    Returns 1.0 when thrust fires (strong bull signal), 0 otherwise.
    """
    all_advances = {}
    for ticker, df in universe_prices.items():
        if len(df) > 5:
            all_advances[ticker] = (df["Close"].pct_change() > 0).astype(float)

    if not all_advances:
        return pd.Series()

    adv_df = pd.DataFrame(all_advances)
    adv_frac = adv_df.mean(axis=1)

    low_10d  = adv_frac.rolling(lookback).min()
    high_10d = adv_frac.rolling(lookback).max()

    thrust = ((low_10d < 0.40) & (high_10d > thrust_threshold)).astype(float)
    decay = thrust.rolling(20).max().shift(1)
    return decay.fillna(0)


# ---------------------------------------------------------------------------
# Overnight Effect (systematic overnight premium capture)
# ---------------------------------------------------------------------------

def overnight_effect_signal(df: pd.DataFrame,
                             min_overnight_return: float = 0.001) -> pd.Series:
    """
    Overnight premium: stocks systematically gap up on open.
    Signal based on consistent overnight return pattern over rolling window.
    """
    overnight_ret = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    avg_overnight = overnight_ret.rolling(21).mean()
    signal = (avg_overnight / overnight_ret.rolling(21).std().replace(0, np.nan)).clip(-3, 3) / 3
    return signal.fillna(0)


if __name__ == "__main__":
    import yfinance as yf
    df = yf.download("NVDA", period="5y", auto_adjust=True, progress=False)
    if isinstance(df.columns[0], tuple):
        df.columns = [c[0] for c in df.columns]
    print("Kalman trend (last 5):", kalman_trend_signal(df).iloc[-5:].values)
    print("52wk momentum (last 5):", momentum_52wk_signal(df).iloc[-5:].values)
    print("RSI divergence (last 5):", rsi_divergence_signal(df).iloc[-5:].values)
