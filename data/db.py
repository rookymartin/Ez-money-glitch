"""
db.py — SQLite database connection and helpers.

The DB lives at data/ez_money.db relative to the project root.
Call init_db() once on startup; everything else uses get_conn().
"""

import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH: Path = _PROJECT_ROOT / "data" / "ez_money.db"
_SCHEMA_PATH: Path = _PROJECT_ROOT / "data" / "schema.sql"


def init_db() -> None:
    """Create all tables if they don't exist yet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(DB_PATH) as con:
        con.executescript(schema)
    print(f"[db] initialised → {DB_PATH}")


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ---------------------------------------------------------------------------
# OHLCV helpers  (used by data/fetch.py)
# ---------------------------------------------------------------------------

def ohlcv_last_date(ticker: str) -> str | None:
    """Return the most recent date stored for a ticker, or None."""
    with get_conn() as con:
        row = con.execute(
            "SELECT MAX(date) FROM ohlcv WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row[0] if row else None


def ohlcv_load(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Load OHLCV rows for *ticker* into a DataFrame (index = Date)."""
    sql = "SELECT date, open, high, low, close, volume FROM ohlcv WHERE ticker = ?"
    params: list = [ticker]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    sql += " ORDER BY date"
    with get_conn() as con:
        df = pd.read_sql_query(sql, con, params=params, parse_dates=["date"],
                               index_col="date")
    df.index.name = "Date"
    df.columns = [c.capitalize() for c in df.columns]
    return df


def ohlcv_upsert(ticker: str, df: pd.DataFrame) -> int:
    """
    Insert-or-replace rows from *df* into ohlcv.
    Ensures the ticker row exists in the tickers table first.
    Returns number of rows written.
    """
    if df is None or df.empty:
        return 0
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.columns = [c.capitalize() for c in df.columns]
    rows = [
        (ticker, idx.strftime("%Y-%m-%d"),
         float(row["Open"]), float(row["High"]),
         float(row["Low"]),  float(row["Close"]),
         int(row["Volume"]))
        for idx, row in df.iterrows()
        if all(c in df.columns for c in ("Open", "High", "Low", "Close", "Volume"))
    ]
    with get_conn() as con:
        con.execute(
            "INSERT OR IGNORE INTO tickers (ticker, exchange) VALUES (?, 'UNKNOWN')",
            (ticker,)
        )
        con.executemany(
            """INSERT OR REPLACE INTO ohlcv
               (ticker, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return len(rows)


# ---------------------------------------------------------------------------
# Backtest result helpers
# ---------------------------------------------------------------------------

def save_backtest_run(tickers: list[str], years: int,
                      results: dict) -> int:
    """
    Persist a full backtest run.
    *results* maps strategy_name → dict with keys:
        sharpe, cagr_pct, max_drawdown_pct, win_rate,
        total_return_pct, n_trades, equity (list), trades (list of dicts)
    Returns the new run_id.
    """
    with get_conn() as con:
        cur = con.execute(
            "INSERT INTO backtest_runs (tickers_json, years) VALUES (?, ?)",
            (json.dumps(tickers), years)
        )
        run_id = cur.lastrowid

        for strategy, data in results.items():
            con.execute(
                "INSERT OR IGNORE INTO strategies (name, category) VALUES (?, 'unknown')",
                (strategy,)
            )
            cur2 = con.execute(
                """INSERT INTO backtest_results
                   (run_id, strategy, sharpe, cagr_pct, max_drawdown_pct,
                    win_rate, total_return_pct, n_trades, years)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, strategy,
                 data.get("sharpe"), data.get("cagr_pct"),
                 data.get("max_drawdown_pct"), data.get("win_rate"),
                 data.get("total_return_pct"), data.get("n_trades"), years)
            )
            result_id = cur2.lastrowid

            for i, equity_val in enumerate(data.get("equity", [])):
                # equity is a list indexed by bar; we don't have dates here
                con.execute(
                    "INSERT OR IGNORE INTO portfolio_equity (result_id, date, equity) VALUES (?, ?, ?)",
                    (result_id, str(i), float(equity_val))
                )

            for t in data.get("trades", []):
                con.execute(
                    """INSERT INTO trades
                       (result_id, ticker, entry_date, exit_date, direction,
                        entry_price, exit_price, return_pct, pnl, stop_hit, hold_days)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (result_id,
                     t.get("ticker"), t.get("entry_date"), t.get("exit_date"),
                     t.get("direction"), t.get("entry_price"), t.get("exit_price"),
                     t.get("return_pct"), t.get("pnl"),
                     int(t.get("stop_hit", 0)), t.get("hold_days"))
                )
    return run_id


# ---------------------------------------------------------------------------
# Research experiment helper
# ---------------------------------------------------------------------------

def save_experiment(strategy: str, sig_params: dict, port_params: dict,
                    sharpe: float, cagr_pct: float,
                    n_trades: int, score: float) -> int:
    with get_conn() as con:
        cur = con.execute(
            """INSERT INTO research_experiments
               (strategy, sig_params_json, port_params_json,
                sharpe, cagr_pct, n_trades, score)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (strategy, json.dumps(sig_params), json.dumps(port_params),
             sharpe, cagr_pct, n_trades, score)
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Training log helper
# ---------------------------------------------------------------------------

def save_epoch(epoch: int, train_loss: float,
               val_loss: float, val_accuracy: float) -> None:
    with get_conn() as con:
        con.execute(
            """INSERT INTO training_log (epoch, train_loss, val_loss, val_accuracy)
               VALUES (?, ?, ?, ?)""",
            (epoch, train_loss, val_loss, val_accuracy)
        )
