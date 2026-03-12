"""
local_data_cache.py — SQLite-backed local cache for all external signal data.

Why this exists:
  KindPath doctrine requires sovereignty. Every external API call is a point
  of dependency, rate-limiting, and data-loss risk. This cache pre-populates
  from external sources on a schedule and serves that data locally — making
  the engine fully offline-capable once seeded.

Usage as a module:
  from scripts.local_data_cache import LocalDataCache
  cache = LocalDataCache()
  bars = cache.get_ohlcv("BTC-USD", "1d")

Usage as a CLI (refresh all):
  python scripts/local_data_cache.py --all
  python scripts/local_data_cache.py --prices
  python scripts/local_data_cache.py --macro
  python scripts/local_data_cache.py --wikipedia
  python scripts/local_data_cache.py --weather
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
import urllib.request
import zipfile
import io
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cache DB lives in local_data/ — gitignored, runtime data
CACHE_DIR = Path(__file__).parent.parent / "local_data"
CACHE_DB  = CACHE_DIR / "market_cache.db"

# Default symbols to keep fresh
DEFAULT_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "POL-USD",
    "OP-USD", "ARB-USD", "LINK-USD", "TSLA", "AAPL",
    "ENPH", "NEE", "SPY", "QQQ", "^VIX", "^IRX",
    "GC=F", "DX-Y.NYB",
]

# FRED series to cache (these also have a public CSV endpoint — no API key needed)
FRED_SERIES = {
    "yield_curve": "T10Y2Y",
    "inflation":   "CPIAUCSL",
    "employment":  "ICSA",
    "m2":          "M2SL",
    "dxy":         "DTWEXBGS",
}

# Wikipedia topics to prefetch
DEFAULT_WIKI_TOPICS = [
    "Bitcoin", "Ethereum", "Solana_(blockchain_platform)",
    "Cardano_(blockchain_platform)", "Tesla,_Inc.", "Apple_Inc.",
    "S&P_500", "Nasdaq-100",
]

# Northern NSW coordinates for Open-Meteo
WEATHER_LAT = -28.65
WEATHER_LON = 153.56


class LocalDataCache:
    """
    Thread-safe SQLite cache. All feeds use this as a local-first layer.
    Entries expire based on staleness — prefer fresh network data when available,
    but never fail hard: stale is better than empty.
    """

    def __init__(self, db_path: Path = CACHE_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol    TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts        TEXT NOT NULL,
                    open      REAL, high REAL, low REAL, close REAL, volume REAL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, timeframe, ts)
                );
                CREATE INDEX IF NOT EXISTS idx_ohlcv_sym ON ohlcv(symbol, timeframe, fetched_at);

                CREATE TABLE IF NOT EXISTS macro (
                    series_id TEXT NOT NULL,
                    obs_date  TEXT NOT NULL,
                    value     REAL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (series_id, obs_date)
                );

                CREATE TABLE IF NOT EXISTS wikipedia_views (
                    topic     TEXT NOT NULL,
                    date      TEXT NOT NULL,
                    views     INTEGER,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (topic, date)
                );

                CREATE TABLE IF NOT EXISTS weather (
                    lat        REAL NOT NULL,
                    lon        REAL NOT NULL,
                    ts         TEXT NOT NULL,
                    data_json  TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (lat, lon, ts)
                );

                CREATE TABLE IF NOT EXISTS search_cache (
                    query      TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    fetched_at   TEXT NOT NULL,
                    PRIMARY KEY (query)
                );

                CREATE TABLE IF NOT EXISTS cftc_positions (
                    report_date TEXT NOT NULL,
                    symbol      TEXT NOT NULL,
                    long_pct    REAL,
                    short_pct   REAL,
                    fetched_at  TEXT NOT NULL,
                    PRIMARY KEY (report_date, symbol)
                );
            """)

    # ── OHLCV ─────────────────────────────────────────────────────────────────

    def get_ohlcv(self, symbol: str, timeframe: str = "1d",
                  min_bars: int = 50, max_age_hours: int = 25) -> Optional[List[Dict]]:
        """Return cached bars if fresh enough. None if cache miss or stale."""
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT ts, open, high, low, close, volume FROM ohlcv
                   WHERE symbol=? AND timeframe=? AND fetched_at > ?
                   ORDER BY ts ASC""",
                (symbol, timeframe, cutoff)
            ).fetchall()
        if len(rows) >= min_bars:
            return [{"timestamp": r[0], "open": r[1], "high": r[2],
                     "low": r[3], "close": r[4], "volume": r[5]} for r in rows]
        return None

    def put_ohlcv(self, symbol: str, timeframe: str, bars: List[Dict]) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO ohlcv
                   (symbol, timeframe, ts, open, high, low, close, volume, fetched_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [(symbol, timeframe, str(b.get("timestamp", "")),
                  b.get("open"), b.get("high"), b.get("low"),
                  b.get("close"), b.get("volume"), now) for b in bars]
            )

    # ── Macro / FRED ──────────────────────────────────────────────────────────

    def get_macro(self, series_id: str, max_age_days: int = 3) -> Optional[float]:
        """Return most recent observation value if cache is fresh."""
        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT value FROM macro WHERE series_id=? AND fetched_at > ?
                   ORDER BY obs_date DESC LIMIT 1""",
                (series_id, cutoff)
            ).fetchone()
        return row[0] if row else None

    def put_macro(self, series_id: str, value: float, obs_date: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO macro (series_id, obs_date, value, fetched_at)
                   VALUES (?,?,?,?)""",
                (series_id, obs_date, value, datetime.utcnow().isoformat())
            )

    # ── Wikipedia ─────────────────────────────────────────────────────────────

    def get_wikipedia(self, topic: str, days: int = 7,
                      max_age_hours: int = 25) -> Optional[List[int]]:
        """Return list of recent daily view counts. None if stale/missing."""
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT views FROM wikipedia_views
                   WHERE topic=? AND fetched_at > ?
                   ORDER BY date DESC LIMIT ?""",
                (topic, cutoff, days)
            ).fetchall()
        if len(rows) >= 2:
            return [r[0] for r in reversed(rows)]
        return None

    def put_wikipedia(self, topic: str, items: List[Dict]) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO wikipedia_views (topic, date, views, fetched_at)
                   VALUES (?,?,?,?)""",
                [(topic, item.get("timestamp", item.get("date", "")),
                  item.get("views", 0), now) for item in items]
            )

    # ── Weather ───────────────────────────────────────────────────────────────

    def get_weather(self, lat: float, lon: float,
                    max_age_hours: int = 3) -> Optional[Dict]:
        """Return cached current weather data. None if stale."""
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT data_json FROM weather
                   WHERE lat=? AND lon=? AND fetched_at > ?
                   ORDER BY ts DESC LIMIT 1""",
                (lat, lon, cutoff)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_weather(self, lat: float, lon: float, data: Dict) -> None:
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO weather (lat, lon, ts, data_json, fetched_at)
                   VALUES (?,?,?,?,?)""",
                (lat, lon, now, json.dumps(data), now)
            )

    # ── Search ────────────────────────────────────────────────────────────────

    def get_search(self, query: str, max_age_hours: int = 6) -> Optional[Dict]:
        """Return cached search results. None if stale."""
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT results_json FROM search_cache
                   WHERE query=? AND fetched_at > ?""",
                (query, cutoff)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_search(self, query: str, results: Dict) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO search_cache (query, results_json, fetched_at)
                   VALUES (?,?,?)""",
                (query, json.dumps(results), datetime.utcnow().isoformat())
            )

    def stats(self) -> Dict[str, int]:
        """Return row counts per table — useful for monitoring cache health."""
        tables = ["ohlcv", "macro", "wikipedia_views", "weather", "search_cache", "cftc_positions"]
        with sqlite3.connect(self.db_path) as conn:
            return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


# ── Refreshers ────────────────────────────────────────────────────────────────

def _refresh_prices(cache: LocalDataCache, symbols: List[str]) -> None:
    """Fetch OHLCV from Yahoo Finance and store in cache."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — cannot refresh prices")
        return

    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(period="2y", interval="1d")
            if df.empty:
                continue
            bars = [{"timestamp": str(ts), "open": float(r["Open"]),
                     "high": float(r["High"]), "low": float(r["Low"]),
                     "close": float(r["Close"]), "volume": float(r["Volume"])}
                    for ts, r in df.iterrows()]
            cache.put_ohlcv(sym, "1d", bars)
            logger.info(f"  [prices] {sym}: {len(bars)} bars cached")
            time.sleep(0.3)  # be gentle with Yahoo
        except Exception as e:
            logger.warning(f"  [prices] {sym} failed: {e}")


def _refresh_macro(cache: LocalDataCache) -> None:
    """
    Fetch FRED series via the public CSV endpoint — no API key required.
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y
    """
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    for name, series_id in FRED_SERIES.items():
        try:
            url = f"{base_url}?id={series_id}"
            req = urllib.request.Request(url, headers={"User-Agent": "KindPath-Bot/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read().decode("utf-8")
            reader = csv.reader(content.splitlines())
            next(reader)  # skip header
            rows = [(date, float(val)) for date, val in reader if val and val != "."]
            if rows:
                date, val = rows[-1]  # most recent observation
                cache.put_macro(series_id, val, date)
                logger.info(f"  [macro] {name} ({series_id}): {val} @ {date}")
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"  [macro] {name} ({series_id}) failed: {e}")


def _refresh_wikipedia(cache: LocalDataCache, topics: List[str]) -> None:
    """Fetch Wikipedia daily pageviews for tracked topics."""
    today = datetime.utcnow().strftime("%Y%m%d")
    start = (datetime.utcnow() - timedelta(days=14)).strftime("%Y%m%d")
    headers = {"User-Agent": "KindPath-Bot/1.0 (sam@kindpath.org)"}
    for topic in topics:
        try:
            url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/"
                   f"per-article/en.wikipedia/all-access/all-agents/"
                   f"{topic}/daily/{start}/{today}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            items = data.get("items", [])
            if items:
                cache.put_wikipedia(topic, items)
                logger.info(f"  [wiki] {topic}: {len(items)} days cached")
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"  [wiki] {topic} failed: {e}")


def _refresh_weather(cache: LocalDataCache) -> None:
    """Fetch current weather from Open-Meteo (free, no key required)."""
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={WEATHER_LAT}&longitude={WEATHER_LON}"
           f"&current=temperature_2m,rain,cloud_cover,wind_speed_10m"
           f"&timezone=Australia%2FSydney")
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        current = data.get("current", {})
        cache.put_weather(WEATHER_LAT, WEATHER_LON, current)
        logger.info(f"  [weather] {current.get('temperature_2m')}°C @ Bundjalung")
    except Exception as e:
        logger.warning(f"  [weather] failed: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    parser = argparse.ArgumentParser(description="Refresh KindPath local data cache")
    parser.add_argument("--all",       action="store_true", help="Refresh everything")
    parser.add_argument("--prices",    action="store_true", help="Refresh OHLCV prices")
    parser.add_argument("--macro",     action="store_true", help="Refresh FRED macro data")
    parser.add_argument("--wikipedia", action="store_true", help="Refresh Wikipedia views")
    parser.add_argument("--weather",   action="store_true", help="Refresh weather")
    parser.add_argument("--stats",     action="store_true", help="Print cache stats and exit")
    args = parser.parse_args()

    cache = LocalDataCache()

    if args.stats:
        stats = cache.stats()
        print("\n  Local data cache stats:")
        for table, count in stats.items():
            print(f"    {table:<25} {count:>8} rows")
        return

    do_all = args.all or not any([args.prices, args.macro, args.wikipedia, args.weather])

    print(f"\n  KindPath local data refresh — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Cache: {cache.db_path}\n")

    if do_all or args.prices:
        print("[1/4] Refreshing prices...")
        _refresh_prices(cache, DEFAULT_SYMBOLS)

    if do_all or args.macro:
        print("[2/4] Refreshing macro data (FRED CSV — no key required)...")
        _refresh_macro(cache)

    if do_all or args.wikipedia:
        print("[3/4] Refreshing Wikipedia pageviews...")
        _refresh_wikipedia(cache, DEFAULT_WIKI_TOPICS)

    if do_all or args.weather:
        print("[4/4] Refreshing weather...")
        _refresh_weather(cache)

    stats = cache.stats()
    print(f"\n  Done. Cache: {sum(stats.values())} total rows across {len(stats)} tables.")
    for table, count in stats.items():
        print(f"    {table:<25} {count:>6} rows")


if __name__ == "__main__":
    main()
