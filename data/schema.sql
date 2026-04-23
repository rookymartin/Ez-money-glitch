-- Ez-money-glitch database schema (SQLite)
-- Run once via: python -c "from data.db import init_db; init_db()"

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- -------------------------------------------------------------------------
-- Ticker universe
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tickers (
    ticker   TEXT PRIMARY KEY,
    exchange TEXT NOT NULL,        -- SP500 | NASDAQ100 | OSLO
    sector   TEXT,
    added_at TEXT DEFAULT (date('now'))
);

-- -------------------------------------------------------------------------
-- OHLCV time series  (replaces data/cache/ohlcv/*.csv)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker  TEXT NOT NULL REFERENCES tickers (ticker),
    date    TEXT NOT NULL,         -- ISO-8601  YYYY-MM-DD
    open    REAL NOT NULL,
    high    REAL NOT NULL,
    low     REAL NOT NULL,
    close   REAL NOT NULL,
    volume  INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv (date);

-- -------------------------------------------------------------------------
-- Strategy registry
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS strategies (
    name        TEXT PRIMARY KEY,
    category    TEXT NOT NULL,     -- classic | advanced | nn
    params_json TEXT               -- default hyper-params as JSON
);

-- -------------------------------------------------------------------------
-- Backtest runs  (one row per execution of run_all_strategies)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at      TEXT NOT NULL DEFAULT (datetime('now')),
    tickers_json TEXT,             -- JSON array of tickers used
    years       INTEGER
);

-- -------------------------------------------------------------------------
-- Backtest results  (aggregate metrics per strategy per run)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL REFERENCES backtest_runs (id),
    strategy         TEXT    NOT NULL REFERENCES strategies (name),
    sharpe           REAL,
    cagr_pct         REAL,
    max_drawdown_pct REAL,
    win_rate         REAL,
    total_return_pct REAL,
    n_trades         INTEGER,
    years            INTEGER
);

CREATE INDEX IF NOT EXISTS idx_results_run ON backtest_results (run_id);

-- -------------------------------------------------------------------------
-- Trades  (individual entries/exits from a backtest result)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id   INTEGER NOT NULL REFERENCES backtest_results (id),
    ticker      TEXT NOT NULL,
    entry_date  TEXT,
    exit_date   TEXT,
    direction   TEXT,              -- LONG | SHORT
    entry_price REAL,
    exit_price  REAL,
    return_pct  REAL,
    pnl         REAL,
    stop_hit    INTEGER,           -- 0 | 1
    hold_days   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_trades_result ON trades (result_id);

-- -------------------------------------------------------------------------
-- Portfolio equity curve  (daily equity per backtest result)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_equity (
    result_id INTEGER NOT NULL REFERENCES backtest_results (id),
    date      TEXT    NOT NULL,
    equity    REAL    NOT NULL,
    PRIMARY KEY (result_id, date)
);

-- -------------------------------------------------------------------------
-- Per-bar signals  (optional; written by nn/signals.py)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
    strategy TEXT NOT NULL REFERENCES strategies (name),
    ticker   TEXT NOT NULL,
    date     TEXT NOT NULL,
    value    REAL NOT NULL,        -- -1.0 … +1.0
    PRIMARY KEY (strategy, ticker, date)
);

-- -------------------------------------------------------------------------
-- Researcher experiments
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_experiments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy         TEXT NOT NULL,
    sig_params_json  TEXT,
    port_params_json TEXT,
    sharpe           REAL,
    cagr_pct         REAL,
    n_trades         INTEGER,
    score            REAL,
    run_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

-- -------------------------------------------------------------------------
-- Macro time series  (VIX, yields, DXY, gold — market-wide daily data)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS macro (
    series  TEXT NOT NULL,   -- e.g. VIX, YIELD_10Y, YIELD_2Y, DXY
    date    TEXT NOT NULL,
    value   REAL NOT NULL,
    PRIMARY KEY (series, date)
);

CREATE INDEX IF NOT EXISTS idx_macro_date ON macro (date);

-- -------------------------------------------------------------------------
-- Google Trends  (weekly search interest, interpolated to daily)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS google_trends (
    keyword TEXT NOT NULL,
    date    TEXT NOT NULL,
    value   REAL NOT NULL,   -- 0-100 normalised interest
    PRIMARY KEY (keyword, date)
);

-- -------------------------------------------------------------------------
-- Neural-network training log
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS training_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch        INTEGER NOT NULL,
    train_loss   REAL,
    val_loss     REAL,
    val_accuracy REAL,
    run_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
