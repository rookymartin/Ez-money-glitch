"""
indicators.py — Technical Indicators
VWAP, SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Volume Ratio

All functions take a pandas DataFrame with columns: Open, High, Low, Close, Volume
Returns are added as new columns (in-place or new df).
"""

import numpy as np
import pandas as pd


# ── Moving Averages ──────────────────────────────────────────────────────────

def sma(df: pd.DataFrame, period: int, col: str = "Close") -> pd.Series:
    """Simple Moving Average."""
    return df[col].rolling(window=period).mean()


def ema(df: pd.DataFrame, period: int, col: str = "Close") -> pd.Series:
    """Exponential Moving Average."""
    return df[col].ewm(span=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP).
    Resets daily. Uses typical price = (H + L + C) / 3.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    # Group by date for daily reset
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date
    else:
        dates = pd.to_datetime(df.index).date

    cum_vol_price = (typical * df["Volume"]).groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(pd.Series(dates, index=df.index)).cumsum()
    return cum_vol_price / cum_vol


def mwa(df: pd.DataFrame, period: int = 20, col: str = "Close") -> pd.Series:
    """
    Modified Weighted Average — weights recent prices more heavily
    (triangular weighting: most recent bar gets highest weight).
    """
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    return df[col].rolling(window=period).apply(
        lambda x: np.dot(x, weights), raw=True
    )


# ── RSI ───────────────────────────────────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = 14, col: str = "Close") -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── MACD ─────────────────────────────────────────────────────────────────────

def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
         signal: int = 9, col: str = "Close") -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).
    Returns DataFrame with columns: macd, signal, histogram.
    """
    ema_fast = ema(df, fast, col)
    ema_slow = ema(df, slow, col)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram},
                        index=df.index)


# ── Bollinger Bands ──────────────────────────────────────────────────────────

def bollinger_bands(df: pd.DataFrame, period: int = 20,
                    std_mult: float = 2.0, col: str = "Close") -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame with: upper, middle, lower, bandwidth, %B.
    """
    middle = sma(df, period, col)
    std = df[col].rolling(window=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle
    pct_b = (df[col] - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper": upper, "bb_middle": middle, "bb_lower": lower,
        "bb_bandwidth": bandwidth, "bb_pct_b": pct_b
    }, index=df.index)


# ── ATR ───────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures volatility, used for stop sizing."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── Volume ────────────────────────────────────────────────────────────────────

def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume / N-day average volume. >1.5 = above-average volume."""
    avg_vol = df["Volume"].rolling(window=period).mean()
    return df["Volume"] / avg_vol.replace(0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — cumulative volume flow."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (df["Volume"] * direction).cumsum()


# ── NEW: Additional high-signal indicators ────────────────────────────────────

def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).
    %K = (Close - LowestLow) / (HighestHigh - LowestLow) * 100
    %D = 3-period SMA of %K
    Overbought >80, oversold <20. Crossovers signal entries/exits.
    """
    lowest_low   = df["Low"].rolling(window=k_period).min()
    highest_high = df["High"].rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = ((df["Close"] - lowest_low) / denom * 100).fillna(50)
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({"stoch_k": k / 100.0, "stoch_d": d / 100.0}, index=df.index)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R — oscillator similar to Stochastic but inverted.
    Range: -100 (oversold) to 0 (overbought). Normalised to [0,1] here.
    Effective at spotting extreme reversal points.
    """
    highest_high = df["High"].rolling(window=period).max()
    lowest_low   = df["Low"].rolling(window=period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    wr = ((highest_high - df["Close"]) / denom * -100).fillna(-50)
    return (wr + 100) / 100.0   # normalise to [0,1]


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index — deviation of price from its statistical mean.
    High (+200) = overbought momentum. Low (-200) = oversold.
    Clipped and scaled here to [-1, 1] for the NN.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp   = typical.rolling(window=period).mean()
    mean_dev = typical.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    raw_cci = (typical - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
    return raw_cci.fillna(0).clip(-3, 3) / 3.0   # scale to [-1,1]


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index — TREND STRENGTH (not direction).
    ADX > 25 = trending market; < 20 = ranging/choppy.
    Crucial for avoiding whipsaws in sideways markets.
    Normalised to [0,1] here (ADX rarely exceeds 60).
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    high_low   = high - low
    high_close = (high - close.shift(1)).abs()
    low_close  = (low  - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_s    = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di  = pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan) * 100
    minus_di = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_s.replace(0, np.nan) * 100
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100).fillna(0)
    adx_val  = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.clip(0, 60) / 60.0   # scale [0,1]


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index — volume-weighted RSI.
    Combines price and volume to measure buying/selling pressure.
    MFI > 80 = overbought (distribution); < 20 = oversold (accumulation).
    Normalised to [0,1].
    """
    typical   = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_mf    = typical * df["Volume"]
    pos_mf    = raw_mf.where(typical > typical.shift(1), 0.0)
    neg_mf    = raw_mf.where(typical < typical.shift(1), 0.0)
    pos_sum   = pos_mf.rolling(period).sum()
    neg_sum   = neg_mf.rolling(period).sum()
    mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return (100 - 100 / (1 + mfi_ratio)).fillna(50) / 100.0


def roc(df: pd.DataFrame, period: int, col: str = "Close") -> pd.Series:
    """Rate of Change over N periods — pure momentum signal."""
    return df[col].pct_change(period).clip(-0.5, 0.5)


def close_position(df: pd.DataFrame) -> pd.Series:
    """
    Close position within the day's range: (C - L) / (H - L).
    1.0 = closed at day high (bullish), 0.0 = closed at day low (bearish).
    Useful for detecting intraday buying/selling pressure.
    """
    denom = (df["High"] - df["Low"]).replace(0, np.nan)
    return ((df["Close"] - df["Low"]) / denom).fillna(0.5).clip(0, 1)


def gap_pct(df: pd.DataFrame) -> pd.Series:
    """
    Overnight gap: (Open - prior Close) / prior Close.
    Large gaps signal institutional activity, earnings surprises.
    """
    return ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)).clip(-0.2, 0.2).fillna(0)


def streak(df: pd.DataFrame) -> pd.Series:
    """
    Consecutive up/down days, normalised by /10.
    +0.3 = 3 consecutive up days, -0.4 = 4 consecutive down days.
    Captures momentum persistence and mean-reversion setups.
    Vectorized: no Python loop.
    """
    daily_ret = df["Close"].diff()
    direction = np.sign(daily_ret).fillna(0)
    # New group whenever direction changes
    change = (direction != direction.shift(1)).fillna(True)
    group_id = change.cumsum()
    # Within-group running count × direction = signed streak length
    streak_vals = (direction.groupby(group_id).cumcount() + 1) * direction
    # Flat bars (direction==0) get streak 0
    streak_vals = streak_vals.where(direction != 0, 0.0)
    return (streak_vals / 10.0).clip(-1, 1)


def vol_5d(df: pd.DataFrame) -> pd.Series:
    """5-day realised volatility (annualised). Captures short-term risk spikes."""
    return (df["Close"].pct_change().rolling(5).std() * np.sqrt(252)).fillna(0).clip(0, 2)


# ── Academic / factor-model features ─────────────────────────────────────────

def momentum_12_1(df: pd.DataFrame) -> pd.Series:
    """
    Jegadeesh-Titman 12-1 month momentum factor.
    12-month return MINUS last month's return (skips recent reversal noise).
    Strongest and most replicated equity factor in the literature.
    Returns clipped to [-1, 1].
    """
    ret_12m = df["Close"].pct_change(252)
    ret_1m  = df["Close"].pct_change(21)
    return (ret_12m - ret_1m).clip(-1, 1)


def z_score(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Mean-reversion Z-score: (Close - SMA_N) / rolling_std_N.
    Basis of Mean Reversion Z strategy (75% consistency, Sharpe 4.88 in backtests).
    Clipped to [-3, 3] then normalised to [-1, 1].
    """
    sma_n = df["Close"].rolling(period).mean()
    std_n = df["Close"].rolling(period).std().replace(0, np.nan)
    return ((df["Close"] - sma_n) / std_n).clip(-3, 3).fillna(0) / 3.0


def sharpe_21d(df: pd.DataFrame) -> pd.Series:
    """
    21-day annualised Sharpe ratio of daily returns.
    Quality-of-momentum signal: high Sharpe = smooth uptrend (not just a spike).
    Clipped to [-3, 3] and normalised to [-1, 1].
    """
    ret = df["Close"].pct_change()
    mean_r = ret.rolling(21).mean()
    std_r  = ret.rolling(21).std().replace(0, np.nan)
    sharpe = (mean_r / std_r * np.sqrt(252)).clip(-3, 3)
    return sharpe.fillna(0) / 3.0


def vol_60d(df: pd.DataFrame) -> pd.Series:
    """
    60-day realised annualised volatility.
    Frazzini-Pedersen low-volatility factor: low-vol stocks outperform on
    a risk-adjusted basis. Used as the input feature; the model learns the sign.
    Clipped at 2.0 (200% annualised vol = effectively 0 after clip).
    """
    return (df["Close"].pct_change().rolling(60).std() * np.sqrt(252)).fillna(0).clip(0, 2)


# ── Composite feature builder ─────────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   ff5_daily: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build all features for the neural network.
    Input: OHLCV DataFrame with DatetimeIndex.
    ff5_daily: optional DataFrame with FF5 factor columns aligned to dates
               (columns: ff5_mktrf, ff5_smb, ff5_hml, ff5_rmw, ff5_cma).
               When provided, 5 extra features are appended (n_features 50→55).
               When None, 5 zero columns are appended to keep shape consistent.
    Returns: DataFrame with 55 features, NaN rows dropped.
    """
    out = df.copy()

    # Price-based features (normalised as % deviations — important for NN)
    out["sma_20"]  = sma(df, 20)
    out["sma_50"]  = sma(df, 50)
    out["sma_200"] = sma(df, 200)
    out["ema_12"]  = ema(df, 12)
    out["ema_26"]  = ema(df, 26)
    out["mwa_20"]  = mwa(df, 20)

    # 20-day Volume-Weighted Moving Average deviation
    # (daily VWAP is meaningless — same bar H+L+C/3 vs close; VWMA gives real signal)
    _typical = (df["High"] + df["Low"] + df["Close"]) / 3
    _vwma_20 = ((_typical * df["Volume"]).rolling(20).sum() /
                df["Volume"].rolling(20).sum().replace(0, np.nan))
    out["vwap_dev"] = ((df["Close"] - _vwma_20) / _vwma_20.replace(0, np.nan)
                       ).clip(-0.3, 0.3).fillna(0)

    # Moving average deviations (normalised)
    out["dev_sma20"]  = (df["Close"] - out["sma_20"])  / out["sma_20"]
    out["dev_sma50"]  = (df["Close"] - out["sma_50"])  / out["sma_50"]
    out["dev_sma200"] = (df["Close"] - out["sma_200"]) / out["sma_200"]

    # MA crossover signals
    out["ma_cross_20_50"]  = (out["sma_20"] / out["sma_50"] - 1)
    out["ma_cross_50_200"] = (out["sma_50"] / out["sma_200"] - 1)

    # Volatility — computed before MACD so we can normalise MACD by ATR
    out["atr_14"]    = atr(df, 14)
    out["atr_pct"]   = out["atr_14"] / df["Close"]  # normalised ATR

    # Momentum indicators
    out["rsi_14"]    = rsi(df, 14)
    _atr_safe        = out["atr_14"].replace(0, np.nan).ffill().fillna(1.0)
    macd_df          = macd(df)
    # Normalise MACD by ATR: makes values scale-invariant across tickers
    out["macd"]      = (macd_df["macd"]      / _atr_safe).clip(-5, 5).fillna(0)
    out["macd_sig"]  = (macd_df["signal"]    / _atr_safe).clip(-5, 5).fillna(0)
    out["macd_hist"] = (macd_df["histogram"] / _atr_safe).clip(-5, 5).fillna(0)

    # Bollinger Bands
    bb = bollinger_bands(df)
    out["bb_pct_b"]     = bb["bb_pct_b"]      # 0=at lower, 1=at upper band
    out["bb_bandwidth"] = bb["bb_bandwidth"]   # wider = more volatile

    # Volume
    out["vol_ratio"] = volume_ratio(df, 20)
    out["obv_norm"]  = obv(df)
    out["obv_norm"]  = (out["obv_norm"] - out["obv_norm"].rolling(20).mean()) / \
                        out["obv_norm"].rolling(20).std().replace(0, 1)

    # Price momentum (returns over different windows)
    out["ret_1d"]   = df["Close"].pct_change(1)
    out["ret_5d"]   = df["Close"].pct_change(5)
    out["ret_20d"]  = df["Close"].pct_change(20)

    # Candlestick body size
    out["body_size"]  = (df["Close"] - df["Open"]).abs() / df["Open"]
    out["upper_wick"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Open"]
    out["lower_wick"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Open"]

    # ── NEW: High-signal additional indicators ────────────────────────────────
    stoch_df        = stochastic(df, k_period=14, d_period=3)
    out["stoch_k"]  = stoch_df["stoch_k"]   # [0,1]: <0.2 oversold, >0.8 overbought
    # stoch_d and williams_r dropped — highly correlated with stoch_k / rsi_14
    # cci_20 dropped — correlated with rsi/stoch; mfi_14 kept (adds volume dimension)

    out["adx_14"]   = adx(df, 14)           # [0,1]: >0.42 = trending (ADX>25), crucial filter

    out["mfi_14"]   = mfi(df, 14)           # [0,1]: volume-weighted RSI — smart money proxy

    out["roc_3"]    = roc(df, 3)            # 3D rate of change — very short-term momentum
    out["roc_10"]   = roc(df, 10)           # 10D rate of change — swing trade momentum

    out["close_pos"] = close_position(df)   # [0,1]: close position in day range

    out["gap_pct"]  = gap_pct(df)           # overnight gap — institutional activity signal

    out["streak"]   = streak(df)            # consecutive up/down days [-1,1]

    out["vol_5d"]   = vol_5d(df)            # 5-day realized vol — short-term risk spike

    # ── Academic / factor-model features ─────────────────────────────────────
    out["mom_12_1"]   = momentum_12_1(df)   # Jegadeesh-Titman 12-1 momentum [-1,1]
    out["z_score_20"] = z_score(df, 20)     # 20D mean-reversion Z [-1,1]
    out["z_score_50"] = z_score(df, 50)     # 50D mean-reversion Z [-1,1]
    out["sharpe_21d"] = sharpe_21d(df)      # 21D Sharpe ratio (quality of momentum) [-1,1]
    out["vol_60d"]    = vol_60d(df)         # 60D realized vol (Frazzini-Pedersen factor) [0,2]

    # ── NEW: 15 additional high-signal features ────────────────────────────────
    close = df["Close"].values.astype(float)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    n = len(close)

    # 1. Williams %R (mean-reversion, different view from RSI)
    wr_period = 14
    wr = np.full(n, np.nan)
    for i in range(wr_period - 1, n):
        hh = high[i - wr_period + 1:i + 1].max()
        ll = low[i - wr_period + 1:i + 1].min()
        wr[i] = (hh - close[i]) / (hh - ll + 1e-10)
    out["williams_r"] = pd.Series(wr, index=df.index).clip(0, 1)

    # 2. CCI (Commodity Channel Index — trend breakout detection)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(20).mean()
    mad_tp = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out["cci_20"] = ((tp - sma_tp) / (0.015 * mad_tp + 1e-10)).clip(-3, 3) / 3

    # 3. Chaikin Money Flow (accumulation/distribution pressure)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-10)
    mfv = mfm * df["Volume"]
    out["cmf_20"] = (mfv.rolling(20).sum() / (df["Volume"].rolling(20).sum() + 1e-10)).clip(-1, 1)

    # 4. Keltner Channel position (like Bollinger but ATR-based)
    ema_20 = pd.Series(close, index=df.index).ewm(span=20).mean()
    atr_vals = out["atr_pct"].values * close
    atr_s = pd.Series(atr_vals, index=df.index)
    kc_upper = ema_20 + 2 * atr_s
    kc_lower = ema_20 - 2 * atr_s
    kc_range = kc_upper - kc_lower + 1e-10
    out["keltner_pos"] = ((df["Close"].values - kc_lower.values) / kc_range.values).clip(0, 1)

    # 5. Volume-weighted momentum (institutional signal)
    vol_ma = pd.Series(volume).rolling(20).mean()
    vol_weight = (pd.Series(volume) / (vol_ma + 1e-10)).clip(0, 5)
    ret_1d_raw = pd.Series(close).pct_change()
    out["vol_weighted_mom"] = (ret_1d_raw * vol_weight).rolling(10).mean().clip(-0.1, 0.1) * 10

    # 6. Price acceleration (second derivative of price)
    ret_5d_raw = pd.Series(close).pct_change(5)
    ret_5d_prev = ret_5d_raw.shift(5)
    out["price_accel"] = (ret_5d_raw - ret_5d_prev).clip(-0.1, 0.1) * 5

    # 7. Relative volume spike (institutional entry detection)
    vol_20 = pd.Series(volume).rolling(20).mean()
    vol_50 = pd.Series(volume).rolling(50).mean()
    out["vol_spike"] = ((pd.Series(volume) / (vol_20 + 1e-10)) - 1).clip(-1, 3) / 3

    # 8. Intraday range vs historical range (volatility expansion/contraction)
    day_range = (df["High"] - df["Low"]) / (df["Close"] + 1e-10)
    avg_range = day_range.rolling(20).mean()
    out["range_ratio"] = (day_range / (avg_range + 1e-10)).clip(0, 3) / 3

    # 9. Distance from 52-week high (trend position)
    high_252 = pd.Series(high, index=df.index).rolling(252, min_periods=20).max()
    out["dist_52w_high"] = ((df["Close"] - high_252) / (high_252 + 1e-10)).clip(-1, 0)

    # 10. Distance from 52-week low
    low_252 = pd.Series(low, index=df.index).rolling(252, min_periods=20).min()
    out["dist_52w_low"] = ((df["Close"] - low_252) / (low_252 + 1e-10)).clip(0, 3) / 3

    # 11. Price efficiency ratio (Kaufman) — trending vs choppy
    direction = abs(pd.Series(close).diff(10))
    volatility = pd.Series(close).diff().abs().rolling(10).sum()
    out["efficiency_ratio"] = (direction / (volatility + 1e-10)).clip(0, 1)

    # 12. Volume trend (is volume increasing or decreasing with price?)
    vol_trend = pd.Series(volume).rolling(10).mean() / (pd.Series(volume).rolling(50).mean() + 1e-10)
    out["vol_trend"] = (vol_trend - 1).clip(-1, 1)

    # 13. High-low range percentile (where is today's range vs history)
    range_pct = day_range.rolling(60, min_periods=10).rank(pct=True)
    out["range_percentile"] = range_pct.fillna(0.5)

    # 14. RSI divergence (price makes new high but RSI doesn't — bearish divergence)
    rsi_s = out["rsi_14"].values
    close_s = close
    price_20h = pd.Series(close_s, index=df.index).rolling(20).max().values
    rsi_20h = pd.Series(rsi_s, index=df.index).rolling(20).max().values
    price_at_high = np.where(close_s >= price_20h * 0.995, 1.0, 0.0)
    rsi_at_high = np.where(rsi_s >= rsi_20h * 0.97, 1.0, 0.0)
    out["rsi_divergence"] = pd.Series(price_at_high - rsi_at_high, index=df.index).clip(-1, 1)

    # 15. Multi-timeframe momentum consensus
    mom_5 = pd.Series(close).pct_change(5)
    mom_20 = pd.Series(close).pct_change(20)
    mom_60 = pd.Series(close).pct_change(60)
    consensus = (np.sign(mom_5) + np.sign(mom_20) + np.sign(mom_60)) / 3
    out["mtf_momentum"] = consensus.fillna(0)

    # Drop the raw OHLCV + derived absolute-price columns for NN
    # Total: 50 TA features (35 original + 15 new high-signal features)
    # + 5 FF5 daily factor returns = 55 features total
    feature_cols = [
        # Moving average context (4)
        "dev_sma20", "dev_sma50", "dev_sma200", "vwap_dev",
        # MA crossover (2)
        "ma_cross_20_50", "ma_cross_50_200",
        # Momentum oscillators (5)
        "rsi_14", "macd", "macd_sig", "macd_hist", "stoch_k",
        # Trend / band indicators (2)
        "bb_pct_b", "bb_bandwidth",
        # Volatility & trend strength (3)
        "atr_pct", "adx_14", "vol_5d",
        # Volume (3)
        "vol_ratio", "obv_norm", "mfi_14",
        # Price momentum (5)
        "ret_1d", "ret_5d", "ret_20d", "roc_3", "roc_10",
        # Candlestick / structure (4)
        "body_size", "upper_wick", "lower_wick", "close_pos",
        # Gap & streak (2)
        "gap_pct", "streak",
        # Academic factors (5)
        "mom_12_1", "z_score_20", "z_score_50", "sharpe_21d", "vol_60d",
        # High-signal features (15)
        "williams_r", "cci_20", "cmf_20", "keltner_pos", "vol_weighted_mom",
        "price_accel", "vol_spike", "range_ratio", "dist_52w_high", "dist_52w_low",
        "efficiency_ratio", "vol_trend", "range_percentile", "rsi_divergence", "mtf_momentum",
    ]
    # Forward-fill, backfill, then zero-fill any remaining NaN
    features = out[feature_cols].copy()
    features = features.ffill().bfill().fillna(0)

    # ── FF5 daily factor returns (5 columns, always present) ─────────────────
    # These carry current macro factor realizations into every timestep of the
    # sequence so the CNN/Transformer can relate price patterns to factor context.
    # Scaled: raw decimal returns ×10 and clipped to ±0.5.
    FF5_COLS = ["ff5_mktrf", "ff5_smb", "ff5_hml", "ff5_rmw", "ff5_cma"]

    if ff5_daily is not None and not ff5_daily.empty:
        # Align ff5_daily to the stock's date index using forward-fill
        ff5_aligned = ff5_daily.reindex(features.index, method="ffill")
        for col in FF5_COLS:
            if col in ff5_aligned.columns:
                features[col] = ff5_aligned[col].fillna(0).astype(float)
            else:
                features[col] = 0.0
    else:
        for col in FF5_COLS:
            features[col] = 0.0

    return features


# ── Signal thresholds ─────────────────────────────────────────────────────────

SIGNAL_CONFIG = {
    "rsi_oversold":      35,   # RSI below = potential buy zone (tightened per request)
    "rsi_overbought":    60,   # RSI above = potential sell zone (stricter per Martin)
    "vwap_tol_pct":      1.5,  # within ±1.5% of VWAP preferred for entries
    "vol_ratio_min":     1.2,  # need at least 1.2x avg volume for confirmation
    "atr_stop_mult":     1.5,  # stop = entry - 1.5 × ATR
    "target_mult":       2.5,  # target = entry + 2.5 × ATR (R:R = 1:1.67)
    "hold_days_max":     7,    # max hold for swing trade
    # Advanced filters
    "adv_enable":                True,
    "adv_macd_hist_rising":      True,
    "adv_atr_pct_max":           4.0,    # if ATR% > 4%, require high volume
    "adv_atr_relax_vol_ratio":   1.8,    # relax ATR filter if volume >= 1.8x avg
    "adv_require_above_50dma":   True,   # must be above 50DMA for longs
    "adv_ret5_min":             -0.02,   # 5d return band
    "adv_ret5_max":              0.02,
    # Confidence thresholds (act only on highest conviction)
    "conf_buy_min":             0.58,
    "conf_sell_min":            0.58,
    # Position sizing targets and bounds
    "pos_target_atr_pct":       2.5,
    "pos_min_mult":             0.3,
    "pos_max_mult":             1.8,
}
