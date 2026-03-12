
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), "kindpath-bmr"))

SYMBOLS = ["SPY"]

def test():
    print("🌐 Fetching data for SPY...")
    try:
        raw_data = yf.download(
            SYMBOLS, 
            start="2025-03-01", 
            end="2026-03-01", 
            interval="1h", 
            group_by='ticker',
            auto_adjust=True
        )
        print(f"✅ Data fetched: {len(raw_data)} rows")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test()
