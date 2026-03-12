"""
DFTE — Recursive Wisdom Engine (Resonance & Incursion Edition)
==============================================================
Advanced Non-Linear Engine for 'Total Immanence' Doctrine.
- Cross-Sector Resonance: Detects leading field alignment (e.g. Crypto -> Energy).
- Decoupling Detection: Identifies 'Coherence Traps' (High nu, Negative Outcome).
- Resonance Modifiers: Rewards symbols moving in sync with successful Vanguard sectors.
"""

from __future__ import annotations
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)

try:
    from kindai_client import KindAIClient as _KAI
    _kindai = _KAI(
        system=(
            "You are the KindPath wisdom analyst. Given trading signal metrics, "
            "respond in 2-3 sentences using KindPath terminology (ν, syntropy, ZPB, "
            "IN, coherence traps, R→B). Be direct, no filler."
        )
    )
    _KINDAI_AVAILABLE = True
except Exception:
    _kindai = None
    _KINDAI_AVAILABLE = False


class RecursiveWisdom:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.resonance_map: Dict[str, float] = {} # Sector-level resonance

    def update_resonance(self, as_of: Optional[datetime] = None):
        """Computes current resonance across sectors based on recent performance."""
        try:
            ts_filter = ""
            if as_of:
                realised_iso = (as_of - timedelta(days=5)).isoformat()
                ts_filter = f"WHERE s.timestamp <= '{realised_iso}'"

            query = f"""
                SELECT s.symbol, o.forward_10d
                FROM signals s
                JOIN outcomes o ON s.id = o.signal_id
                {ts_filter}
                AND o.forward_10d IS NOT NULL
                ORDER BY s.timestamp DESC
                LIMIT 2000
            """
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
            
            if df.empty: return

            # Group symbols into implicit sectors
            crypto = ['BTC-USD', 'ETH-USD', 'SOL-USD']
            energy = ['ENPH', 'ICLN', 'TAN', 'URA', 'XLE', 'XOM', 'CVX']
            
            # Compute average recent success per sector
            crypto_res = df[df['symbol'].isin(crypto)]['forward_10d'].mean()
            energy_res = df[df['symbol'].isin(energy)]['forward_10d'].mean()
            
            self.resonance_map = {
                "CRYPTO": float(crypto_res) if not np.isnan(crypto_res) else 0.0,
                "ENERGY": float(energy_res) if not np.isnan(energy_res) else 0.0
            }
        except Exception as e:
            logger.warning(f"Resonance update failed: {e}")

    def discover_hidden_metrics(self, symbol: str, as_of: Optional[datetime] = None) -> dict:
        try:
            ts_filter = ""
            if as_of:
                realised_iso = (as_of - timedelta(days=11)).isoformat()
                ts_filter = f"AND s.timestamp <= '{realised_iso}'"

            query = f"""
                SELECT s.timestamp, s.nu, s.wfs, s.mfs, s.conviction, o.forward_10d
                FROM signals s
                JOIN outcomes o ON s.id = o.signal_id
                WHERE s.symbol = '{symbol}'
                {ts_filter}
                AND o.forward_10d IS NOT NULL
                ORDER BY s.timestamp ASC
            """
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
            
            if df.empty or len(df) < 20:
                return {"predictive_edge": 1.5, "dominant_driver": "expansion", "anomalies": {}}

            # 1. Non-Linear Metrics
            df['nu_vel'] = df['nu'].diff()
            df['nu_accel'] = df['nu_vel'].diff()
            
            # 2. Coherence Decoupling Detection (The DKNG Trap)
            # High Nu + Negative Outcome = Decoupled Field
            decoupling = df[(df['nu'] > 0.7) & (df['forward_10d'] < -0.03)]
            decoupling_ratio = len(decoupling) / len(df) if len(df) > 0 else 0

            anomalies = {}
            if decoupling_ratio > 0.1:
                anomalies["coherence_trap"] = float(decoupling_ratio)

            # 3. Cross-Sector Resonance
            # If Crypto is booming, does it lead this symbol? (Simple heuristic)
            res_bonus = 1.0
            if "CRYPTO" in self.resonance_map and self.resonance_map["CRYPTO"] > 0.02:
                # If symbol is not crypto, give it a resonance lift
                if symbol not in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
                    res_bonus = 1.2

            # 4. Hyper-Expansion Scoring
            success_rate = (df[df['conviction'] > 0.4]['forward_10d'] > 0).mean()
            if pd.isna(success_rate): success_rate = 0.5
            
            edge = 1.0 + (success_rate * 2.5) 
            edge *= res_bonus # Apply resonance

            # 5. Trap Avoidance (DKNG Refinement)
            if "coherence_trap" in anomalies:
                # If we are in a trap sector, we require extreme conviction to expand
                if df['conviction'].iloc[-1] < 0.7:
                    edge *= 0.4 # Strategic dampening of decoupled fields

            # Non-linear Acceleration Snap
            latest_nu_accel = df['nu_accel'].iloc[-1] if not pd.isna(df['nu_accel'].iloc[-1]) else 0
            if latest_nu_accel > 0.04:
                edge *= 2.0 

            # HYPER-EXPANSION CAP: 10.0x
            edge = float(np.clip(edge, 0.3, 10.0))
            
            return {
                "predictive_edge": edge,
                "dominant_driver": "resonance_incursion",
                "anomalies": anomalies,
                "success_rate": float(success_rate)
            }
        except Exception as e:
            logger.warning(f"Wisdom discovery failed for {symbol}: {e}")
            return {"predictive_edge": 1.5, "dominant_driver": "expansion"}

    def get_lateral_consensus_modifier(self, symbol: str, as_of: Optional[datetime] = None) -> float:
        wisdom = self.discover_hidden_metrics(symbol, as_of=as_of)
        return wisdom.get("predictive_edge", 1.0)

    def kindai_explain(self, symbol: str, metrics: Optional[dict] = None, as_of: Optional[datetime] = None) -> str:
        """
        Ask KindAI for a narrative interpretation of this symbol's field state.
        Falls back to a plain summary if KindAI is unavailable.
        """
        if metrics is None:
            metrics = self.discover_hidden_metrics(symbol, as_of=as_of)

        prompt = (
            f"Symbol: {symbol}\n"
            f"Predictive edge: {metrics.get('predictive_edge', '?'):.2f}\n"
            f"Success rate: {metrics.get('success_rate', '?')}\n"
            f"Dominant driver: {metrics.get('dominant_driver', '?')}\n"
            f"Anomalies: {metrics.get('anomalies', {})}\n"
            f"Resonance map: {self.resonance_map}\n"
            f"\nInterpret this field state for {symbol}."
        )

        if not _KINDAI_AVAILABLE or _kindai is None:
            return (
                f"{symbol}: edge={metrics.get('predictive_edge', 1.0):.2f}, "
                f"driver={metrics.get('dominant_driver', 'unknown')}, "
                f"anomalies={metrics.get('anomalies', {})}"
            )

        try:
            return _kindai.ask(prompt)
        except Exception as e:
            logger.warning(f"KindAI explain failed for {symbol}: {e}")
            return f"{symbol}: edge={metrics.get('predictive_edge', 1.0):.2f}"


class BasketWisdom:
    """
    Multi-symbol Wisdom Engine — cross-basket resonance analysis.

    Takes a basket of symbols and computes:
    - Per-symbol predictive edge (via RecursiveWisdom)
    - Cross-basket resonance: are the symbols moving in harmony or diverging?
    - Basket conviction: aggregate signal strength
    - Synthetic elder reading for the portfolio field
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.recursive = RecursiveWisdom(db_path)

    def analyse_basket(
        self,
        symbols: list,
        as_of: Optional[datetime] = None,
        update_resonance: bool = True,
    ) -> dict:
        """
        Full basket analysis.

        Returns:
          {
            "symbols": {symbol: {predictive_edge, success_rate, dominant_driver, anomalies}},
            "basket_edge": float,           # geometric mean of individual edges
            "basket_resonance": float,      # 0-1: how aligned are the edges?
            "basket_conviction": float,     # average success_rate
            "divergent_symbols": list,      # symbols pulling against the basket
            "lead_symbol": str | None,      # symbol with highest edge
            "resonance_map": dict,          # sector resonance from RecursiveWisdom
            "narrative": str,               # plain language basket reading
          }
        """
        if update_resonance:
            self.recursive.update_resonance(as_of=as_of)

        results = {}
        for sym in symbols:
            try:
                m = self.recursive.discover_hidden_metrics(sym, as_of=as_of)
                results[sym] = m
            except Exception as e:
                logger.warning(f"BasketWisdom: failed for {sym}: {e}")
                results[sym] = {"predictive_edge": 1.0, "dominant_driver": "unknown", "anomalies": {}}

        edges = [r.get("predictive_edge", 1.0) for r in results.values()]
        success_rates = [r.get("success_rate", 0.5) for r in results.values() if "success_rate" in r]

        # Basket edge: geometric mean (preserves scale, penalises outliers)
        basket_edge = float(np.exp(np.mean(np.log(np.clip(edges, 0.1, 10.0))))) if edges else 1.0

        # Resonance: 1 - (std / mean) clamped to [0, 1]
        # High resonance = symbols are all pulling in a consistent direction
        mean_edge = float(np.mean(edges)) if edges else 1.0
        std_edge = float(np.std(edges)) if edges else 0.0
        cv = std_edge / max(abs(mean_edge), 0.1)  # coefficient of variation
        basket_resonance = float(np.clip(1.0 - cv, 0.0, 1.0))

        basket_conviction = float(np.mean(success_rates)) if success_rates else 0.5

        # Divergent symbols: those pulling more than 1.5x below basket median
        median_edge = float(np.median(edges)) if edges else 1.0
        divergent = [sym for sym, r in results.items()
                     if r.get("predictive_edge", 1.0) < median_edge * 0.5]

        lead_symbol = max(results, key=lambda s: results[s].get("predictive_edge", 0)) if results else None

        narrative = _build_basket_narrative(
            symbols, basket_edge, basket_resonance, basket_conviction,
            divergent, lead_symbol, self.recursive.resonance_map
        )

        return {
            "symbols": results,
            "basket_edge": round(basket_edge, 4),
            "basket_resonance": round(basket_resonance, 4),
            "basket_conviction": round(basket_conviction, 4),
            "divergent_symbols": divergent,
            "lead_symbol": lead_symbol,
            "resonance_map": self.recursive.resonance_map,
            "narrative": narrative,
        }


def _build_basket_narrative(
    symbols: list,
    basket_edge: float,
    basket_resonance: float,
    basket_conviction: float,
    divergent: list,
    lead_symbol: Optional[str],
    resonance_map: dict,
) -> str:
    """
    Generate a plain-language basket reading.
    This is the synthetic elder's voice for a portfolio of symbols.
    """
    parts = []

    n = len(symbols)
    parts.append(f"Basket of {n} symbol{'s' if n != 1 else ''}: {', '.join(symbols)}.")

    if basket_edge > 4.0:
        parts.append(f"Collective predictive edge is high ({basket_edge:.1f}x) — strong field alignment.")
    elif basket_edge > 2.0:
        parts.append(f"Predictive edge is moderate ({basket_edge:.1f}x).")
    else:
        parts.append(f"Predictive edge is low ({basket_edge:.1f}x) — proceed with caution.")

    if basket_resonance > 0.75:
        parts.append("Symbols are resonating coherently — the field is unified.")
    elif basket_resonance > 0.4:
        parts.append("Partial resonance — some divergence in the basket.")
    else:
        parts.append("Low resonance — symbols are pulling in different directions.")

    if divergent:
        parts.append(f"Divergent signals: {', '.join(divergent)} — contra-field. Reduce exposure.")

    if lead_symbol:
        parts.append(f"Lead signal: {lead_symbol}.")

    if basket_conviction > 0.65:
        parts.append("Historical conviction is strong.")
    elif basket_conviction < 0.45:
        parts.append("Historical conviction is weak — field history shows mixed outcomes.")

    # Sector resonance context
    for sector, val in resonance_map.items():
        if abs(val) > 0.03:
            direction = "positive" if val > 0 else "negative"
            parts.append(f"{sector.capitalize()} sector resonance is {direction} ({val:+.3f}).")

    return " ".join(parts)

