"""
fetch.py — OHLCV price data fetching and caching.

Stores per-ticker parquet (or CSV fallback) files in ./data/cache/
relative to the project root.

Usage:
    from data.fetch import get_prices, refresh_cache, load_tickers_batch
"""

import io
import os
import sys
import logging
import warnings
import contextlib
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data directory — local to the project
# ---------------------------------------------------------------------------

# Resolve project root as 2 levels up from this file (data/fetch.py → root)
_PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR: Path = _PROJECT_ROOT / "data" / "cache"
OHLCV_DIR: Path = CACHE_DIR / "ohlcv"


@contextlib.contextmanager
def _silent_stderr():
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# Parquet / CSV helpers
# ---------------------------------------------------------------------------

_PARQUET_AVAILABLE: Optional[bool] = None


def _parquet_available() -> bool:
    global _PARQUET_AVAILABLE
    if _PARQUET_AVAILABLE is None:
        try:
            import pyarrow  # noqa: F401
            _PARQUET_AVAILABLE = True
        except ImportError:
            try:
                import fastparquet  # noqa: F401
                _PARQUET_AVAILABLE = True
            except ImportError:
                _PARQUET_AVAILABLE = False
    return _PARQUET_AVAILABLE


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _parquet_available():
        df.to_parquet(path, index=True)
    else:
        df.to_csv(path, index=True)


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _cache_path(ticker: str) -> Path:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    ext = ".parquet" if _parquet_available() else ".csv"
    return OHLCV_DIR / f"{ticker}{ext}"


def _find_cache_file(ticker: str) -> Optional[Path]:
    for ext in (".parquet", ".csv"):
        p = OHLCV_DIR / f"{ticker}{ext}"
        if p.exists():
            return p
    return None


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
    Return OHLCV DataFrame for ticker. Serves from local cache, updates incrementally.
    Default 13 years to match research backtest period.
    """
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _find_cache_file(ticker)
    now = datetime.now()

    if cache_file is not None and cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age_hours = (now - mtime).total_seconds() / 3600

        if age_hours < 24:
            try:
                return _read_df(cache_file)
            except Exception as exc:
                logger.warning("Could not read cache for %s (%s), re-downloading.", ticker, exc)

        try:
            cached_df = _read_df(cache_file)
            last_cached_date = cached_df.index.max().date()
        except Exception:
            cached_df = None
            last_cached_date = None

        if last_cached_date is not None and last_cached_date >= now.date():
            return cached_df

        if last_cached_date is not None:
            new_df = _download_incremental(ticker, last_cached_date)
            if new_df is not None and not new_df.empty:
                combined = pd.concat([cached_df, new_df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            else:
                combined = cached_df
        else:
            combined = _download_full(ticker, years)

        if combined is not None and not combined.empty:
            _write_df(combined, _cache_path(ticker))
        return combined if combined is not None else pd.DataFrame()

    df = _download_full(ticker, years)
    if df is not None and not df.empty:
        _write_df(df, _cache_path(ticker))
        return df
    return pd.DataFrame()


def refresh_cache(tickers: List[str], years: int = 13, force: bool = False) -> None:
    """Download / update cache for all tickers in the list."""
    total = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{total}] {ticker} ...", end=" ", flush=True)
        try:
            if force:
                existing = _find_cache_file(ticker)
                if existing and existing.exists():
                    existing.unlink()
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
    Load OHLCV for many tickers. Cache-first; only downloads missing/stale tickers.
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
            print(f"  [cache] downloading {len(need_download)} missing tickers from yfinance…")
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
                            _write_df(df_t, _cache_path(t))
                            result[t] = df_t.loc[df_t.index >= pd.Timestamp(start)]
                    except Exception:
                        pass
        except Exception as e:
            if verbose:
                print(f"  [cache] batch failed ({e}), falling back one-by-one…")
            for t in need_download:
                try:
                    with _silent_stderr():
                        df_t = yf.download(t, start=start, end=end,
                                           auto_adjust=True, progress=False, timeout=10)
                    if isinstance(df_t.columns, pd.MultiIndex):
                        df_t.columns = df_t.columns.get_level_values(0)
                    if not df_t.empty and "Close" in df_t.columns:
                        df_t = df_t[["Open", "High", "Low", "Close", "Volume"]].dropna()
                        _write_df(df_t, _cache_path(t))
                        result[t] = df_t.loc[df_t.index >= pd.Timestamp(start)]
                    time.sleep(0.3)
                except Exception:
                    pass

    if verbose:
        print(f"  [cache] {len(result)}/{len(tickers)} tickers ready")
    return result


def cache_info() -> Dict[str, Dict]:
    """Return metadata for every cached ticker."""
    result = {}
    if not OHLCV_DIR.exists():
        return result
    now = datetime.now()
    for fpath in OHLCV_DIR.iterdir():
        if fpath.suffix not in (".parquet", ".csv"):
            continue
        try:
            df = _read_df(fpath)
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            age_hours = (now - mtime).total_seconds() / 3600
            last_date = df.index.max().date() if not df.empty else None
            result[fpath.stem] = {
                "rows": len(df),
                "last_date": str(last_date) if last_date else None,
                "size_kb": round(fpath.stat().st_size / 1024, 1),
                "fresh": age_hours < 24,
            }
        except Exception:
            pass
    return result
