"""
KindPath Grand Campaign — The Total Immanence Incursion
=======================================================
- $1,000.00 Initial Seed | 20.0% Base Risk (Grand Scale).
- Cross-Sector Resonance Engine Active.
- Trap Avoidance (Coherence Decoupling) active.
- 5-Year Successional Timeline (Y1 -> Y5).
"""

from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
import time
import pickle
import json
from datetime import datetime, timedelta, timezone

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), "kindpath-bmr"))

from dfte.dfte_engine import BMRSummary, KEPESummary, synthesise_dfte_signal
from dfte.wisdom_engine import RecursiveWisdom

SYMBOLS = [
    "SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD", "SOL-USD",
    "ENPH", "ICLN", "TAN", "URA", "COPX", "LIT", "XLE", "XOM", "CVX",
    "EEM", "VWO", "INDA", "EWZ", "FXI", "WOOD", "DBA", "ADM", "NTR",
    "LMT", "RTX", "PM", "MO", "DKNG"
]

DB_PATH = "logger/high_fid_backtest.db"

class GrandSimulator:
    def __init__(self, cache_path: str):
        with open(cache_path, "rb") as f:
            self.data_map = pickle.load(f)
        self.symbols = SYMBOLS
        self.wisdom_engine = RecursiveWisdom(DB_PATH)
        self.wisdom_cache = {sym: 2.0 for sym in self.symbols}
        self.symbol_pnl = {sym: 0.0 for sym in self.symbols}
        self._prepare_data()

    def _prepare_data(self):
        all_dates = sorted(list(set().union(*(df.index for df in self.data_map.values()))))
        self.dates = pd.Index(all_dates)
        num_pulses = len(self.dates)
        num_syms = len(self.symbols)

        self.prices = np.full((num_pulses, num_syms), np.nan)
        self.nu_mat = np.zeros((num_pulses, num_syms))
        self.mom_mat = np.zeros((num_pulses, num_syms))
        self.wfs_mat = np.zeros((num_pulses, num_syms))

        for s_idx, sym in enumerate(self.symbols):
            df = self.data_map.get(sym)
            if df is None: continue
            df = df.reindex(self.dates).ffill()
            self.prices[:, s_idx] = df['Close'].values
            rets = df['Close'].pct_change()
            mom = rets.rolling(20).mean() * 100 
            std = rets.rolling(20).std() * 10
            self.mom_mat[:, s_idx] = np.clip(mom.fillna(0).values, -1, 1)
            self.nu_mat[:, s_idx] = np.clip(1.0 - std.fillna(0).values, 0.1, 0.95)
            self.wfs_mat[:, s_idx] = np.clip(0.5 + (self.mom_mat[:, s_idx] * 0.4), 0.1, 0.9)

    def run_year(self, year_num: int, start_idx: int, end_idx: int, initial_cash: float):
        t0 = time.time()
        cash = initial_cash
        positions = np.zeros(len(self.symbols))
        cost_basis = np.zeros(len(self.symbols))
        
        # Pre-calc outcomes
        outcomes_10d = np.full((len(self.dates), len(self.symbols)), np.nan)
        for s_idx in range(len(self.symbols)):
            p = self.prices[:, s_idx]
            outcomes_10d[:-10, s_idx] = (p[10:] - p[:-10]) / (p[:-10] + 1e-9)

        for i in range(start_idx, end_idx):
            dt = self.dates[i]
            
            if i % 100 == 0:
                print(f"    - Pulse {i}/{end_idx} heartbeat...", flush=True)

            # Weekly Resonance Update
            if i % 5 == 0:
                self.wisdom_engine.update_resonance(as_of=dt)

            current_prices = self.prices[i]
            equity = cash + np.nansum(positions * current_prices)
            
            for s_idx, sym in enumerate(self.symbols):
                price = current_prices[s_idx]
                if np.isnan(price) or price <= 0: continue
                
                h_edge = self.wisdom_cache[sym]
                bmr = BMRSummary(sym, 0.5+(self.mom_mat[i, s_idx]*0.5), "COHERENT", self.mom_mat[i, s_idx], self.nu_mat[i, s_idx], "DRIFT", "NANO")
                kepe_sum = KEPESummary(sym, self.wfs_mat[i, s_idx], "COHERENT", 0.5, 0.5, 0.1, 0.5, 1.0, True, False)
                sig = synthesise_dfte_signal(bmr, kepe_sum, historical_edge=h_edge, override_timestamp=dt)
                
                # GRAND EXECUTION: 20% RISK
                if sig.action == "BUY":
                    notional = equity * 0.20
                    if cash >= notional:
                        qty = notional / price
                        cash -= notional
                        positions[s_idx] += qty
                        cost_basis[s_idx] += notional
                elif sig.action == "SELL":
                    if positions[s_idx] > 0:
                        sale_value = positions[s_idx] * price
                        self.symbol_pnl[sym] += (sale_value - cost_basis[s_idx])
                        cash += sale_value
                        positions[s_idx] = 0.0
                        cost_basis[s_idx] = 0.0

                # Adaptive Wisdom
                if i > 10:
                    past_idx = i - 10
                    p_out = outcomes_10d[past_idx, s_idx]
                    if not np.isnan(p_out):
                        # Use the new discovery logic periodically
                        if i % 20 == 0:
                            metrics = self.wisdom_engine.discover_hidden_metrics(sym, as_of=dt)
                            self.wisdom_cache[sym] = metrics.get("predictive_edge", 1.0)

        final_equity = cash + np.nansum(positions * self.prices[end_idx-1])
        return final_equity

def analyze_pnl(symbol_pnl):
    print("\n📊 GRAND CAMPAIGN PnL ANALYSIS:")
    sorted_pnl = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)
    for sym, val in sorted_pnl:
        print(f"  {sym:8}: ${val:12,.2f}")

if __name__ == "__main__":
    print("\n🌟 STARTING GRAND CAMPAIGN: $1,000 Seed | 20% Risk")
    sim = GrandSimulator("backtest/price_data_5y.pkl")
    total_pulses = len(sim.dates)
    chunk_size = total_pulses // 5
    current_cash = 1000.0
    for y in range(1, 6):
        start = (y-1) * chunk_size
        end = y * chunk_size if y < 5 else total_pulses
        current_cash = sim.run_year(y, start, end, current_cash)
        print(f"  Year {y} Equity: ${current_cash:,.2f}", flush=True)
    
    analyze_pnl(sim.symbol_pnl)
    print(f"\n✨ GRAND CAMPAIGN COMPLETE. FINAL WEALTH: ${current_cash:,.2f}")
