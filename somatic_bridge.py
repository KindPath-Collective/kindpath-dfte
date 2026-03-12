"""
DFTE — Somatic Synapse Bridge
==============================
Translates music divergence logic (LSII) into market somatic logic (LMII).
Links kindpath-analyser core to kindpath-dfte indicators.
"""

from __future__ import annotations
import numpy as np
import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass

# Add kindpath-analyser to path for core logic access
ANALYSER_PATH = "/Users/sam/kindpath-collective/kindpath-analyser"
if ANALYSER_PATH not in sys.path:
    sys.path.insert(0, ANALYSER_PATH)

try:
    from core.divergence import compute_lsii, DivergenceVector
except ImportError:
    # Fallback if repo is moved
    compute_lsii = None

@dataclass
class MarketSomaticState:
    lmii: float             # Late-Move Inversion Index
    tension_delta: float    # Change in field tension
    energy_flux: float      # Change in market energy flux
    dominant_axis: str      # Where the divergence is strongest
    is_protest: bool        # Does it match a 'protest' signature?

class SomaticSynapse:
    """
    Translates intra-timeframe price/volume segments into 'musical' features
    to detect somatic inversions in the market field.
    """
    
    @staticmethod
    def calculate_lmii(ohlcv_data: list) -> MarketSomaticState:
        """
        Segments a timeframe into 4 quarters. 
        Detects if the 4th quarter 'breaks the frame' of the first 3.
        """
        if len(ohlcv_data) < 20:
            return MarketSomaticState(0.0, 0.0, 0.0, "none", False)
            
        # Segment into 4 chunks
        chunk_size = len(ohlcv_data) // 4
        q1 = ohlcv_data[0:chunk_size]
        q2 = ohlcv_data[chunk_size:chunk_size*2]
        q3 = ohlcv_data[chunk_size*2:chunk_size*3]
        q4 = ohlcv_data[chunk_size*3:]
        
        # Translate to 'Sonic' features
        # Tension = Spread volatility / Volume
        # Energy = Volume acceleration
        # Centroid = Price trend slope
        
        def get_features(chunk):
            prices = [x['close'] for x in chunk]
            vols = [x['volume'] for x in chunk]
            return {
                'tension': np.std(prices) / (np.mean(vols) + 1e-10),
                'energy': np.mean(vols),
                'centroid': (prices[-1] - prices[0]) / (prices[0] + 1e-10)
            }
            
        f1, f2, f3, f4 = get_features(q1), get_features(q2), get_features(q3), get_features(q4)
        
        # Simple Divergence (Iceberg tip)
        baseline_tension = np.mean([f1['tension'], f2['tension'], f3['tension']])
        lmii = abs(f4['tension'] - baseline_tension) / (baseline_tension + 1e-10)
        
        # If LMII > 0.6, it's a high-fidelity Inversion Signature
        is_protest = lmii > 0.6
        
        return MarketSomaticState(
            lmii=float(np.clip(lmii, 0, 1)),
            tension_delta=f4['tension'] - baseline_tension,
            energy_flux=f4['energy'],
            dominant_axis="tension",
            is_protest=is_protest
        )
