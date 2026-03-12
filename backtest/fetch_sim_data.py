
import os
import sys
import yfinance as yf
import pickle

SYMBOLS = [
    # Core Caps & Field Proxies
    "SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD", "SOL-USD",
    # Energy-Exchange (Transition + Legacy)
    "ENPH", "ICLN", "TAN", "URA", "COPX", "LIT", "XLE", "XOM", "CVX",
    # Global Breadth (Emerging + Established)
    "EEM", "VWO", "INDA", "EWZ", "FXI", "WOOD", "DBA", "ADM", "NTR",
    # The 'Taboo' Mechanics (Defense, Tobacco, Gaming)
    "LMT", "RTX", "PM", "MO", "DKNG"
]

# FRED macro series to supplement price data
FRED_SERIES = {
    "FEDFUNDS": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Yield Spread (recession indicator)",
    "CPIAUCSL": "CPI All Urban Consumers (inflation)",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
}

FRED_CACHE_PATH = "backtest/fred_data_cache.pkl"


def fetch_fred_data() -> dict:
    """
    Fetch macro series from the St. Louis FRED public API.
    Uses FRED_API_KEY env var if set; falls back to the public CSV endpoint.
    Saves to fred_data_cache.pkl alongside the price data cache.
    Returns dict mapping series_id → pandas DataFrame with Date/Value columns.
    """
    import pandas as pd
    import io

    fred_key = os.getenv("FRED_API_KEY", "")
    data: dict = {}

    for series_id, description in FRED_SERIES.items():
        try:
            if fred_key:
                # FRED JSON API (preferred — respects filter dates)
                import requests
                resp = requests.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": fred_key,
                        "file_type": "json",
                        "observation_start": "2020-01-01",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                obs = resp.json().get("observations", [])
                df = pd.DataFrame([
                    {"Date": o["date"], "Value": float(o["value"])}
                    for o in obs if o["value"] != "."
                ])
            else:
                # Public CSV endpoint — no key required, returns full history
                import requests
                resp = requests.get(
                    f"https://fred.stlouisfed.org/graph/fredgraph.csv",
                    params={"id": series_id},
                    timeout=15,
                    headers={"User-Agent": "KindPath-DFTE/1.0"},
                )
                resp.raise_for_status()
                df = pd.read_csv(
                    io.StringIO(resp.text),
                    names=["Date", "Value"],
                    skiprows=1,
                )
                df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
                df = df.dropna(subset=["Value"])

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            data[series_id] = df
            print(f"  ✅ FRED {series_id} ({description}): {len(df)} observations")

        except Exception as e:
            print(f"  ⚠️  FRED {series_id} failed: {e}")
            continue

    if data:
        with open(FRED_CACHE_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"📦 FRED macro data cached to {FRED_CACHE_PATH}")
    else:
        print("⚠️  No FRED data fetched — macro baseline unavailable for backtest")

    return data


def fetch_data():
    cache_path = "backtest/price_data_cache.pkl"
    start_date = "2025-03-01"
    end_date = "2026-03-06"
    
    print(f"🌐 Fetching hourly data for {len(SYMBOLS)} symbols...")
    raw_data = yf.download(
        SYMBOLS, 
        start=start_date, 
        end=end_date, 
        interval="1h", 
        group_by='ticker',
        auto_adjust=True,
        threads=True
    )
    
    data_map = {}
    for sym in SYMBOLS:
        if sym in raw_data:
            df = raw_data[sym].dropna(subset=['Close'])
            if not df.empty:
                data_map[sym] = df
    
    print(f"✅ Fetched data for {len(data_map)} symbols.")
    with open(cache_path, "wb") as f:
        pickle.dump(data_map, f)
    print(f"📦 Cached to {cache_path}")

    # Fetch FRED macro data
    print(f"\n📊 Fetching FRED macro series ({len(FRED_SERIES)} indicators)...")
    fetch_fred_data()


if __name__ == "__main__":
    fetch_data()

