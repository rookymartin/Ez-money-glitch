"""
migrate.py — One-time import of existing CSV/Parquet cache into SQLite.

Usage:
    python -m data.migrate
"""

from pathlib import Path
import pandas as pd
from data.db import init_db, ohlcv_upsert

_OHLCV_DIR = Path(__file__).parent / "cache" / "ohlcv"


def migrate_ohlcv() -> None:
    init_db()
    files = list(_OHLCV_DIR.glob("*.csv")) + list(_OHLCV_DIR.glob("*.parquet"))
    if not files:
        print("[migrate] No CSV/Parquet files found — nothing to import.")
        return

    for fpath in sorted(files):
        ticker = fpath.stem
        try:
            if fpath.suffix == ".parquet":
                df = pd.read_parquet(fpath)
            else:
                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            n = ohlcv_upsert(ticker, df)
            print(f"  {ticker}: {n} rows imported")
        except Exception as exc:
            print(f"  {ticker}: ERROR — {exc}")

    print("[migrate] done.")


if __name__ == "__main__":
    migrate_ohlcv()
