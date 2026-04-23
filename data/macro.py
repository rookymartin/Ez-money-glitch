"""
macro.py — Market-wide macro indicators fetched via yfinance and cached in SQLite.

Series stored:
    VIX        — CBOE Volatility Index (^VIX)
    YIELD_10Y  — US 10-year Treasury yield (^TNX)
    YIELD_2Y   — US 2-year Treasury yield (^IRX)
    DXY        — US Dollar Index (DX-Y.NYB)
    GOLD       — Gold price proxy (GLD ETF close)

Usage:
    from data.macro import get_macro_df
    df = get_macro_df()   # returns DataFrame indexed by date, one column per series
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import sys

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_SERIES = {
    "VIX":       "^VIX",
    "YIELD_10Y": "^TNX",
    "YIELD_2Y":  "^IRX",
    "DXY":       "DX-Y.NYB",
    "GOLD":      "GLD",
}

_YEARS = 14


def _last_macro_date(series: str) -> str | None:
    from data.db import get_conn
    with get_conn() as con:
        row = con.execute(
            "SELECT MAX(date) FROM macro WHERE series = ?", (series,)
        ).fetchone()
    return row[0] if row else None


def _upsert_series(series: str, s: pd.Series) -> None:
    from data.db import get_conn
    rows = [
        (series, idx.strftime("%Y-%m-%d"), float(val))
        for idx, val in s.items()
        if pd.notna(val)
    ]
    with get_conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO macro (series, date, value) VALUES (?, ?, ?)",
            rows,
        )


def _load_series(series: str) -> pd.Series:
    from data.db import get_conn
    with get_conn() as con:
        rows = con.execute(
            "SELECT date, value FROM macro WHERE series = ? ORDER BY date", (series,)
        ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.Series([r[1] for r in rows], index=idx, name=series)


def refresh_macro(force: bool = False) -> None:
    """Download and cache all macro series."""
    import yfinance as yf

    end = datetime.today()
    start = end - timedelta(days=int(_YEARS * 365.25))

    for name, symbol in _SERIES.items():
        last = _last_macro_date(name)
        if last and not force:
            last_dt = datetime.fromisoformat(last)
            if (end - last_dt).days < 1:
                continue
            dl_start = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            dl_start = start.strftime("%Y-%m-%d")

        try:
            df = yf.download(symbol, start=dl_start,
                             end=end.strftime("%Y-%m-%d"),
                             auto_adjust=True, progress=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            _upsert_series(name, close)
            print(f"  [macro] {name} ({symbol}): {len(close)} rows")
        except Exception as exc:
            print(f"  [macro] {name} failed: {exc}")


def get_macro_df(refresh: bool = True) -> pd.DataFrame:
    """
    Return DataFrame with columns [VIX, YIELD_10Y, YIELD_2Y, DXY, GOLD,
    YIELD_CURVE, VIX_CHG] indexed by date.
    Refreshes stale data automatically.
    """
    if refresh:
        refresh_macro()

    series = {name: _load_series(name) for name in _SERIES}
    df = pd.DataFrame(series)
    df = df.sort_index().ffill().bfill()

    df["YIELD_CURVE"] = df["YIELD_10Y"] - df["YIELD_2Y"]
    df["VIX_CHG"] = df["VIX"].pct_change(5)   # 5-day VIX momentum
    df["DXY_CHG"] = df["DXY"].pct_change(5)

    return df
