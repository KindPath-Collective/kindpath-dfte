"""
KindPath Grand Campaign — Singularity Refactor
===============================================
Zero SQL in the inner loop. All wisdom arrays pre-computed via NumPy/Pandas
before simulation begins. Identical trade logic to recursive_simulator.py.

- $1,000.00 Initial Seed | 20.0% Base Risk (Grand Scale).
- Cross-Sector Resonance Engine Active (vectorized).
- Trap Avoidance (Coherence Decoupling) active (vectorized).
- 5-Year Successional Timeline (Y1 -> Y5).
"""

from __future__ import annotations
import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Path setup
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(_ROOT), "kindpath-bmr"))

from dfte.dfte_engine import BMRSummary, KEPESummary, synthesise_dfte_signal

SYMBOLS = [
    "SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD", "SOL-USD",
    "ENPH", "ICLN", "TAN", "URA", "COPX", "LIT", "XLE", "XOM", "CVX",
    "EEM", "VWO", "INDA", "EWZ", "FXI", "WOOD", "DBA", "ADM", "NTR",
    "LMT", "RTX", "PM", "MO", "DKNG"
]

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD", "SOL-USD"}


class SingularitySimulator:
    def __init__(self, cache_path: str):
        with open(cache_path, "rb") as f:
            self.data_map = pickle.load(f)
        self.symbols = SYMBOLS
        self.symbol_pnl = {sym: 0.0 for sym in self.symbols}
        self._prepare_data()
        self._compute_wisdom_arrays()

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
            if df is None:
                continue
            df = df.reindex(self.dates).ffill()
            self.prices[:, s_idx] = df['Close'].values
            rets = df['Close'].pct_change()
            mom = rets.rolling(20).mean() * 100
            std = rets.rolling(20).std() * 10
            self.mom_mat[:, s_idx] = np.clip(mom.fillna(0).values, -1, 1)
            self.nu_mat[:, s_idx] = np.clip(1.0 - std.fillna(0).values, 0.1, 0.95)
            self.wfs_mat[:, s_idx] = np.clip(0.5 + (self.mom_mat[:, s_idx] * 0.4), 0.1, 0.9)

    def _compute_wisdom_arrays(self):
        """
        Pre-compute all edge, resonance, and decoupling arrays in one vectorized
        pass. Called once after _prepare_data(); replaces all inner-loop SQL calls.
        """
        num_pulses, num_syms = self.mom_mat.shape

        # --- Identify sector indices ---
        crypto_indices = [
            i for i, sym in enumerate(self.symbols) if sym in CRYPTO_SYMBOLS
        ]

        # ------------------------------------------------------------------ #
        # Resonance vector — shape (num_pulses,)                              #
        # Rolling 20-pulse mean of the crypto sector mean momentum.           #
        # ------------------------------------------------------------------ #
        if crypto_indices:
            crypto_mom = self.mom_mat[:, crypto_indices].mean(axis=1)
        else:
            crypto_mom = np.zeros(num_pulses)

        resonance_series = (
            pd.Series(crypto_mom).rolling(20, min_periods=1).mean()
        )
        self.resonance_vec = resonance_series.values  # shape (num_pulses,)

        # res_bonus per (pulse, symbol): 1.2 for non-crypto when resonance > 0.02
        # else 1.0.  Shape (num_pulses, num_syms).
        is_crypto_col = np.array(
            [1 if sym in CRYPTO_SYMBOLS else 0 for sym in self.symbols],
            dtype=float,
        )  # (num_syms,)

        high_resonance = (self.resonance_vec > 0.02).astype(float)  # (num_pulses,)
        # broadcast: (num_pulses, 1) * (1, num_syms) -> (num_pulses, num_syms)
        non_crypto_mask = (1.0 - is_crypto_col)[np.newaxis, :]  # (1, num_syms)
        res_bonus_mat = (
            1.0
            + high_resonance[:, np.newaxis] * non_crypto_mask * 0.2
        )  # 1.0 or 1.2

        # ------------------------------------------------------------------ #
        # Decoupling matrix — shape (num_pulses, num_syms)                   #
        # Coherence trap: high nu AND negative momentum.                      #
        # Decoupling ratio: rolling 60-pulse mean of trap condition.          #
        # decoupling_mat = 0.4 where trap is active + low conviction,         #
        #                  else 1.0.                                           #
        # ------------------------------------------------------------------ #
        coherence_trap = (
            (self.nu_mat > 0.7) & (self.mom_mat < -0.03)
        ).astype(float)  # (num_pulses, num_syms)

        decoupling_ratio = (
            pd.DataFrame(coherence_trap)
            .rolling(60, min_periods=1)
            .mean()
            .values
        )  # (num_pulses, num_syms)

        conviction_proxy = self.wfs_mat  # (num_pulses, num_syms)

        trap_active = (
            (decoupling_ratio > 0.1) & (conviction_proxy < 0.7)
        )  # bool (num_pulses, num_syms)

        self.decoupling_mat = np.where(trap_active, 0.4, 1.0)  # (num_pulses, num_syms)

        # ------------------------------------------------------------------ #
        # Edge matrix — shape (num_pulses, num_syms)                         #
        # Vectorized equivalent of discover_hidden_metrics() for each         #
        # (pulse, symbol) pair.                                               #
        # ------------------------------------------------------------------ #
        high_conviction = self.wfs_mat > 0.4   # bool (num_pulses, num_syms)
        positive_outcome = self.mom_mat > 0.0  # bool (num_pulses, num_syms)
        success = (high_conviction & positive_outcome).astype(float)

        success_rate_mat = (
            pd.DataFrame(success)
            .rolling(60, min_periods=1)
            .mean()
            .values
        )  # (num_pulses, num_syms)

        self.edge_mat = np.clip(
            (1.0 + success_rate_mat * 2.5) * res_bonus_mat * self.decoupling_mat,
            0.3,
            10.0,
        )  # (num_pulses, num_syms)

    def run_year(
        self,
        year_num: int,
        start_idx: int,
        end_idx: int,
        initial_cash: float,
    ) -> float:
        t0 = time.time()
        cash = initial_cash
        positions = np.zeros(len(self.symbols))
        cost_basis = np.zeros(len(self.symbols))

        # Pre-calc 10-day forward outcomes (kept for structural parity; not used
        # for DB calls in this refactor).
        outcomes_10d = np.full((len(self.dates), len(self.symbols)), np.nan)
        for s_idx in range(len(self.symbols)):
            p = self.prices[:, s_idx]
            outcomes_10d[:-10, s_idx] = (p[10:] - p[:-10]) / (p[:-10] + 1e-9)

        for i in range(start_idx, end_idx):
            if i % 100 == 0:
                print(f"    - Pulse {i}/{end_idx} heartbeat...", flush=True)

            current_prices = self.prices[i]
            equity = cash + np.nansum(positions * current_prices)

            for s_idx, sym in enumerate(self.symbols):
                price = current_prices[s_idx]
                if np.isnan(price) or price <= 0:
                    continue

                # Pre-computed historical edge — no DB call.
                h_edge = self.edge_mat[i, s_idx]

                bmr = BMRSummary(
                    sym,
                    0.5 + (self.mom_mat[i, s_idx] * 0.5),
                    "COHERENT",
                    self.mom_mat[i, s_idx],
                    self.nu_mat[i, s_idx],
                    "DRIFT",
                    "NANO",
                )
                kepe_sum = KEPESummary(
                    sym,
                    self.wfs_mat[i, s_idx],
                    "COHERENT",
                    0.5,
                    0.5,
                    0.1,
                    0.5,
                    1.0,
                    True,
                    False,
                )
                sig = synthesise_dfte_signal(
                    bmr,
                    kepe_sum,
                    historical_edge=h_edge,
                    override_timestamp=self.dates[i],
                )

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
                        self.symbol_pnl[sym] += sale_value - cost_basis[s_idx]
                        cash += sale_value
                        positions[s_idx] = 0.0
                        cost_basis[s_idx] = 0.0

        final_equity = cash + np.nansum(positions * self.prices[end_idx - 1])
        elapsed = time.time() - t0
        print(f"  Year {year_num} time: {elapsed:.1f}s", flush=True)
        return final_equity


def analyze_pnl(symbol_pnl: dict):
    print("\n📊 GRAND CAMPAIGN PnL ANALYSIS:")
    sorted_pnl = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)
    for sym, val in sorted_pnl:
        print(f"  {sym:8}: ${val:12,.2f}")


if __name__ == "__main__":
    t_total = time.time()
    print("\n🌟 STARTING GRAND CAMPAIGN [SINGULARITY MODE]: $1,000 Seed | 20% Risk")
    sim = SingularitySimulator("backtest/price_data_5y.pkl")
    total_pulses = len(sim.dates)
    chunk_size = total_pulses // 5
    current_cash = 1000.0
    for y in range(1, 6):
        start = (y - 1) * chunk_size
        end = y * chunk_size if y < 5 else total_pulses
        current_cash = sim.run_year(y, start, end, current_cash)
        print(f"  Year {y} Equity: ${current_cash:,.2f}", flush=True)

    analyze_pnl(sim.symbol_pnl)
    total_elapsed = time.time() - t_total
    print(f"\n✨ GRAND CAMPAIGN COMPLETE. FINAL WEALTH: ${current_cash:,.2f}")
    print(f"Total time: {total_elapsed:.1f}s")
