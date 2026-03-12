
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

if __name__ == "__main__":
    fetch_data()
