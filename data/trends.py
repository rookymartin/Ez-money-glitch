"""
trends.py — Google Trends features via pytrends, cached in SQLite.

Keywords tracked (market sentiment proxies):
    "stock market", "recession", "buy stocks", "market crash", "interest rates"

Returns daily interpolated interest (0-100) aligned to trading dates.

Usage:
    from data.trends import get_trends_df
    df = get_trends_df()   # DataFrame indexed by date, one col per keyword
"""

import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import sys

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

KEYWORDS = [
    "stock market",
    "recession",
    "buy stocks",
    "market crash",
    "interest rates",
]

_YEARS = 5   # Trends history is reliable for ~5 years


def _last_trends_date(keyword: str) -> str | None:
    from data.db import get_conn
    with get_conn() as con:
        row = con.execute(
            "SELECT MAX(date) FROM google_trends WHERE keyword = ?", (keyword,)
        ).fetchone()
    return row[0] if row else None


def _upsert_trends(keyword: str, s: pd.Series) -> None:
    from data.db import get_conn
    rows = [
        (keyword, idx.strftime("%Y-%m-%d"), float(val))
        for idx, val in s.items()
        if pd.notna(val)
    ]
    with get_conn() as con:
        con.executemany(
            "INSERT OR REPLACE INTO google_trends (keyword, date, value) VALUES (?, ?, ?)",
            rows,
        )


def _load_trends(keyword: str) -> pd.Series:
    from data.db import get_conn
    with get_conn() as con:
        rows = con.execute(
            "SELECT date, value FROM google_trends WHERE keyword = ? ORDER BY date",
            (keyword,),
        ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.Series([r[1] for r in rows], index=idx, name=keyword)


def refresh_trends(force: bool = False) -> None:
    """Download Google Trends for all keywords and store in DB."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  [trends] pytrends not installed — skipping (pip install pytrends)")
        return

    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25), retries=2, backoff_factor=0.5)
    end = datetime.today()
    start = end - timedelta(days=int(_YEARS * 365.25))

    for kw in KEYWORDS:
        last = _last_trends_date(kw)
        # Trends are weekly — refresh if last update > 7 days ago
        if last and not force:
            last_dt = datetime.fromisoformat(last)
            if (end - last_dt).days < 7:
                continue

        try:
            timeframe = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
            pytrends.build_payload([kw], timeframe=timeframe, geo="US")
            raw = pytrends.interest_over_time()
            if raw is None or raw.empty or kw not in raw.columns:
                continue
            weekly = raw[kw].rename(kw)
            # Interpolate weekly → daily
            daily_idx = pd.date_range(weekly.index.min(), weekly.index.max(), freq="D")
            daily = weekly.reindex(daily_idx).interpolate("linear")
            _upsert_trends(kw, daily)
            print(f"  [trends] '{kw}': {len(daily)} daily rows")
            time.sleep(1.5)   # be polite to Google
        except Exception as exc:
            print(f"  [trends] '{kw}' failed: {exc}")


def get_trends_df(refresh: bool = True) -> pd.DataFrame:
    """
    Return DataFrame of Google Trends interest, indexed by date.
    Columns: one per keyword, normalised 0-100.
    Returns empty DataFrame if pytrends is unavailable.
    """
    if refresh:
        refresh_trends()

    series = {kw: _load_trends(kw) for kw in KEYWORDS}
    non_empty = {k: v for k, v in series.items() if not v.empty}
    if not non_empty:
        return pd.DataFrame()

    df = pd.DataFrame(non_empty)
    df = df.sort_index().ffill().bfill()
    # Rename to safe column names
    df.columns = [f"gt_{c.replace(' ', '_')}" for c in df.columns]
    return df
