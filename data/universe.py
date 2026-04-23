"""
universe.py — Stock Universe Manager
Dynamically fetches S&P 500 + NASDAQ 100 from Wikipedia (cached 7 days).
Oslo Børs top 60 most liquid tickers (.OL suffix).

Usage:
    from data.universe import get_sp500, get_nasdaq100, get_oslo_top60, get_all_tickers
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Set

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_CACHE_FILE = _PROJECT_ROOT / "data" / "universe_cache.json"
_CACHE_TTL_DAYS = 7
_FETCH_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Oslo Børs top 60 (ranked by average daily trading volume)
# ---------------------------------------------------------------------------

OSLO_TOP60: List[str] = [
    "EQNR.OL", "DNB.OL", "TEL.OL", "NHY.OL", "YAR.OL",
    "MOWI.OL", "ORK.OL", "KOG.OL", "GJF.OL", "STB.OL",
    "AKRBP.OL", "NEL.OL", "NOD.OL", "SALM.OL", "FRO.OL",
    "TOM.OL", "SCATC.OL", "GOGL.OL", "HAFNI.OL", "FLEX.OL",
    "AKSO.OL", "TGS.OL", "PGS.OL", "REC.OL", "ELKEM.OL",
    "AUTO.OL", "SUBC.OL", "ENTRA.OL", "SRBNK.OL", "NONG.OL",
    "SCHA.OL", "SCHB.OL", "AVANCE.OL", "BWLG.OL", "2020BUL.OL",
    "BWO.OL", "KCC.OL", "BONHR.OL", "KAHOT.OL", "CRAYON.OL",
    "BOUVET.OL", "OPERA.OL", "NORBIT.OL", "ITERA.OL", "LSG.OL",
    "AUSS.OL", "GSF.OL", "BAKKA.OL", "MING.OL", "PROTCT.OL",
    "PARB.OL", "NSKOG.OL", "HEX.OL", "AFG.OL", "VEIDKR.OL",
    "EUROPRIS.OL", "XXL.OL", "OKEA.OL", "DOF.OL", "BORR.OL",
]


# ---------------------------------------------------------------------------
# Fallback static lists (used when Wikipedia is down)
# ---------------------------------------------------------------------------

_SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","AVGO","TSLA",
    "JPM","LLY","UNH","V","XOM","COST","MA","PG","JNJ","ABBV","HD","MRK","CVX",
    "BAC","NFLX","CRM","AMD","PEP","ORCL","KO","TMO","WMT","CSCO","ADBE","MCD",
    "ACN","ABT","DHR","TMUS","NKE","TXN","VZ","QCOM","INTC","NEE","RTX","UNP",
    "HON","LOW","AMGN","IBM","COP","CAT","GE","SPGI","BMY","BKNG","ISRG","SYK",
    "SBUX","CI","ELV","CVS","GS","BLK","AXP","MS","NOW","INTU","VRTX","MMC",
    "ETN","PLD","CME","AMAT","LRCX","ADI","PANW","TJX","REGN","GILD","ITW",
    "BSX","LMT","MDLZ","MO","SHW","EQIX","DE","AMT","KLAC","NOC","GD","PH",
    "SNPS","CDNS","MRVL","ZTS","APD","CMG","AON","ICE","ECL","MCO","FTNT","CCI",
    "NXPI","MU","PAYX","CTAS","MRNA","BIIB","ILMN","DXCM","MMM","WM","PSA",
    "WELL","HUM","MCHP","FDX","NSC","EMR","PSX","MPC","VLO","OXY","EOG","SLB",
    "HAL","DVN","FANG","BKR","HES","MRO","APA","CTRA","F","GM",
    "DIS","CHTR","CMCSA","T","WFC","SCHW","PGR","CB","USB","TFC",
    "PYPL","COIN","PLTR","APP","TTD","DDOG","ZS","CRWD","NET","SNOW","TEAM",
    "WDAY","MDB","ANET","CEG","VST","NRG","ETN","VRT","SMCI","ARM",
    "UBER","ABNB","EBAY","LULU","MELI","CPRT","ORLY","AZO","ROST",
    "LIN","FCX","NEM","NUE","VMC","ALB","MP",
]

_NASDAQ100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST",
    "NFLX","TMUS","AMD","INTU","QCOM","AMAT","ISRG","BKNG","HON","VRTX",
    "MU","PANW","LRCX","KLAC","SNPS","CDNS","ADI","NXPI","MRVL","CRWD",
    "MCHP","FTNT","WDAY","TEAM","MNST","DDOG","FAST","EA","GEHC","MDLZ",
    "CTAS","AEP","REGN","CSX","IDXX","TTD","ZS","PYPL","ON","ODFL",
    "BIIB","KDP","DLTR","ROST","APP","ILMN","VRSK","CPRT","PAYX","PCAR",
    "EXC","XEL","FANG","PDD","JD","BIDU","NTES","DXCM","ENPH","SMCI",
    "ARM","COIN","PLTR","ASML","TSM","CEG","ADSK","CMCSA","TXN","ORCL",
    "AMGN","GILD","MRNA","REGN","INTC","WBD","LULU","MELI","ABNB","EBAY",
]


# ---------------------------------------------------------------------------
# Wikipedia fetchers
# ---------------------------------------------------------------------------

def _fetch_sp500_wiki():
    try:
        import pandas as pd
        import urllib.request
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(html, attrs={"id": "constituents"})
        if not tables:
            tables = pd.read_html(html)
        df = tables[0]
        col_map = {c.lower().strip(): c for c in df.columns}
        sym_col = col_map.get("symbol") or col_map.get("ticker")
        sec_col = col_map.get("gics sector") or col_map.get("sector")
        if sym_col is None:
            return [], {}
        tickers = [str(s).strip().replace(".", "-") for s in df[sym_col].dropna()]
        sectors = {}
        if sec_col:
            for _, row in df.iterrows():
                t = str(row[sym_col]).strip().replace(".", "-")
                sectors[t] = str(row[sec_col]).strip()
        log.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers, sectors
    except Exception as e:
        log.warning(f"S&P 500 Wikipedia fetch failed: {e}")
        return [], {}


def _fetch_nasdaq100_wiki() -> List[str]:
    try:
        import pandas as pd
        import urllib.request
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(html)
        for df in tables:
            col_map = {c.lower().strip(): c for c in df.columns}
            sym_col = col_map.get("ticker") or col_map.get("symbol")
            if sym_col and len(df) >= 90:
                tickers = [str(s).strip().replace(".", "-") for s in df[sym_col].dropna()]
                log.info(f"Fetched {len(tickers)} NASDAQ 100 tickers from Wikipedia")
                return tickers
    except Exception as e:
        log.warning(f"NASDAQ 100 Wikipedia fetch failed: {e}")
    return []


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache() -> Optional[dict]:
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r") as f:
                data = json.load(f)
            age_days = (time.time() - data.get("ts", 0)) / 86400
            if age_days < _CACHE_TTL_DAYS:
                return data
    except Exception as e:
        log.warning(f"Universe cache load failed: {e}")
    return None


def _save_cache(sp500, sp500_sectors, nasdaq100) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump({"ts": time.time(), "sp500": sp500,
                       "sp500_sectors": sp500_sectors, "nasdaq100": nasdaq100}, f)
    except Exception as e:
        log.warning(f"Universe cache save failed: {e}")


_DATA: Optional[dict] = None


def _get_data(force_refresh: bool = False) -> dict:
    global _DATA
    if _DATA is not None and not force_refresh:
        return _DATA
    if not force_refresh:
        cached = _load_cache()
        if cached:
            _DATA = cached
            return _DATA
    sp500, sp500_sectors = _fetch_sp500_wiki()
    nasdaq100 = _fetch_nasdaq100_wiki()
    if len(sp500) < 400:
        log.warning("S&P 500 fetch returned too few — using fallback")
        sp500 = list(dict.fromkeys(_SP500_FALLBACK))
        sp500_sectors = {}
    if len(nasdaq100) < 90:
        log.warning("NASDAQ 100 fetch returned too few — using fallback")
        nasdaq100 = list(dict.fromkeys(_NASDAQ100_FALLBACK))
    _DATA = {"ts": time.time(), "sp500": sp500,
              "sp500_sectors": sp500_sectors, "nasdaq100": nasdaq100}
    _save_cache(sp500, sp500_sectors, nasdaq100)
    return _DATA


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sp500(force_refresh: bool = False) -> List[str]:
    """Full S&P 500 tickers (~503). Dynamically fetched, cached 7 days."""
    return list(_get_data(force_refresh)["sp500"])


def get_nasdaq100(force_refresh: bool = False) -> List[str]:
    """Full NASDAQ 100 tickers. Dynamically fetched, cached 7 days."""
    return list(_get_data(force_refresh)["nasdaq100"])


def get_oslo_top60() -> List[str]:
    """Top 60 most liquid Oslo Børs tickers (all .OL suffix)."""
    return list(OSLO_TOP60)


def get_all_tickers(force_refresh: bool = False, include_oslo: bool = True) -> List[str]:
    """Full universe: S&P 500 + NASDAQ 100 (deduped) + Oslo Børs top 60. ~700 unique."""
    data = _get_data(force_refresh)
    combined = list(data["sp500"]) + list(data["nasdaq100"])
    if include_oslo:
        combined += OSLO_TOP60
    seen: Set[str] = set()
    result: List[str] = []
    for t in combined:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def get_swing_candidates() -> List[str]:
    """~40 highest conviction tickers for quick signal scanning."""
    return [
        "NVDA", "AMD", "AVGO", "ARM", "SMCI", "ASML", "TSM",
        "MSFT", "GOOGL", "META", "AMZN",
        "CEG", "VST", "VRT", "ETN", "NRG",
        "ANET", "MRVL", "PLTR", "CRWD", "DDOG", "APP", "SNOW", "CRM", "NOW",
        "TSLA", "NFLX", "COST", "INTU", "PYPL", "TTD", "COIN",
        "AAPL", "JPM", "V", "LLY", "XOM",
        "EQNR.OL", "KOG.OL", "MOWI.OL", "DNB.OL", "FRO.OL",
    ]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    sp = get_sp500()
    ndx = get_nasdaq100()
    oslo = get_oslo_top60()
    all_t = get_all_tickers()
    print(f"S&P 500: {len(sp)} | NASDAQ 100: {len(ndx)} | Oslo Top 60: {len(oslo)} | Total: {len(all_t)}")
