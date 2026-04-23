"""
fetch.py — OHLCV price data fetching and caching.

Uses SQLite (data/ez_money.db) as the cache layer via data.db.
Falls back to downloading from yfinance when the DB has no / stale data.

Usage:
    from data.fetch import get_prices, refresh_cache, load_tickers_batch
"""

import io
import sys
import logging
import warnings
import contextlib
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from data.db import init_db, ohlcv_last_date, ohlcv_load, ohlcv_upsert

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent

# Initialise DB on first import (no-op if tables already exist)
init_db()


@contextlib.contextmanager
def _silent_stderr():
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = [c.strip().capitalize() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        if col.lower().replace(" ", "") == "adjclose":
            rename_map[col] = "Close"
    if rename_map:
        df = df.rename(columns=rename_map)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.dropna(how="all")


def _download_full(ticker: str, years: int) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        end = datetime.today()
        start = end - timedelta(days=int(years * 365.25))
        with _silent_stderr():
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
        if df is None or df.empty:
            return None
        return _normalise_columns(df)
    except Exception as exc:
        logger.warning("Failed to download %s: %s", ticker, exc)
        return None


def _download_incremental(ticker: str, from_date: date) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        start = (datetime.combine(from_date, datetime.min.time()) + timedelta(days=1)).strftime("%Y-%m-%d")
        end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        with _silent_stderr():
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        return _normalise_columns(df)
    except Exception as exc:
        logger.warning("Incremental download failed for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def get_prices(ticker: str, years: int = 13) -> pd.DataFrame:
    """
    Return OHLCV DataFrame for ticker. Serves from SQLite DB, updates incrementally.
    Default 13 years to match research backtest period.
    """
    now = datetime.now()
    last_date_str = ohlcv_last_date(ticker)

    if last_date_str:
        last_date = date.fromisoformat(last_date_str)
        if last_date >= now.date():
            return ohlcv_load(ticker)

        new_df = _download_incremental(ticker, last_date)
        if new_df is not None and not new_df.empty:
            ohlcv_upsert(ticker, new_df)
        return ohlcv_load(ticker)

    df = _download_full(ticker, years)
    if df is not None and not df.empty:
        ohlcv_upsert(ticker, df)
        return ohlcv_load(ticker)
    return pd.DataFrame()


def refresh_cache(tickers: List[str], years: int = 13, force: bool = False) -> None:
    """Download / update DB cache for all tickers in the list."""
    from data.db import get_conn
    total = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{total}] {ticker} ...", end=" ", flush=True)
        try:
            if force:
                with get_conn() as con:
                    con.execute("DELETE FROM ohlcv WHERE ticker = ?", (ticker,))
            df = get_prices(ticker, years=years)
            if df is not None and not df.empty:
                print(f"OK ({len(df)} rows)")
            else:
                print("WARNING: empty / no data")
        except Exception as exc:
            print(f"ERROR: {exc}")


def load_tickers_batch(tickers: List[str], years: int = 13,
                       verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load OHLCV for many tickers. DB-first; only downloads missing/stale tickers.
    Returns {ticker: DataFrame} — only tickers with data are included.
    """
    import yfinance as yf
    import time

    end = datetime.today()
    start = end - timedelta(days=int(years * 365.25))

    result: Dict[str, pd.DataFrame] = {}
    need_download: List[str] = []

    for ticker in tickers:
        df = get_prices(ticker, years=years)
        if df is not None and not df.empty and len(df) > 50:
            result[ticker] = df.loc[df.index >= pd.Timestamp(start)]
        else:
            need_download.append(ticker)

    if need_download:
        if verbose:
            print(f"  [db] downloading {len(need_download)} missing tickers from yfinance…")
        try:
            with _silent_stderr():
                df_batch = yf.download(need_download, start=start, end=end,
                                       auto_adjust=True, progress=False,
                                       group_by="ticker", timeout=20)
            if not df_batch.empty:
                single = len(need_download) == 1
                for t in need_download:
                    try:
                        if single:
                            df_t = df_batch
                            if isinstance(df_t.columns, pd.MultiIndex):
                                df_t.columns = df_t.columns.get_level_values(0)
                        else:
                            if t not in df_batch.columns.get_level_values(0):
                                continue
                            df_t = df_batch[t]
                        df_t = df_t.dropna(how="all")
                        if df_t.empty or "Close" not in df_t.columns:
                            continue
                        df_t = df_t[["Open", "High", "Low", "Close", "Volume"]].dropna()
                        if len(df_t) > 50:
                            ohlcv_upsert(t, df_t)
                            result[t] = ohlcv_load(t, start=start.strftime("%Y-%m-%d"))
                    except Exception:
                        pass
        except Exception as e:
            if verbose:
                print(f"  [db] batch failed ({e}), falling back one-by-one…")
            for t in need_download:
                try:
                    with _silent_stderr():
                        df_t = yf.download(t, start=start, end=end,
                                           auto_adjust=True, progress=False, timeout=10)
                    if isinstance(df_t.columns, pd.MultiIndex):
                        df_t.columns = df_t.columns.get_level_values(0)
                    if not df_t.empty and "Close" in df_t.columns:
                        df_t = df_t[["Open", "High", "Low", "Close", "Volume"]].dropna()
                        ohlcv_upsert(t, df_t)
                        result[t] = ohlcv_load(t, start=start.strftime("%Y-%m-%d"))
                    time.sleep(0.3)
                except Exception:
                    pass

    if verbose:
        print(f"  [db] {len(result)}/{len(tickers)} tickers ready")
    return result


def cache_info() -> Dict[str, Dict]:
    """Return metadata for every cached ticker from the DB."""
    from data.db import get_conn
    with get_conn() as con:
        rows = con.execute(
            """SELECT ticker, COUNT(*) as rows, MAX(date) as last_date
               FROM ohlcv GROUP BY ticker ORDER BY ticker"""
        ).fetchall()
    today = date.today().isoformat()
    return {
        r["ticker"]: {
            "rows": r["rows"],
            "last_date": r["last_date"],
            "fresh": r["last_date"] == today,
        }
        for r in rows
    }
