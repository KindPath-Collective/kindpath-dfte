"""
Vicarious Field Layer
=====================
Implements proximity-based transmission and field impact dynamics.
Based on "Vicarious Nature" canon document.

Core Principles:
1. Inverse Square Law: Intensity drops with square of distance from source.
2. dB Exponent: Perceived impact is logarithmic against energy.
3. Lifeguard Posture: Identifying "drowning swimmers" (high-tension nodes) 
   and maintaining field coherence (shore team).

Location: Northern NSW, Bundjalung Country (-28.65, 153.56)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Primary location (The Shore / Observer)
SHORE_LAT = -28.6500
SHORE_LON = 153.5600

# Major Exchange Coordinates
EXCHANGE_COORDS = {
    "NYSE": (40.7069, -74.0113),  # New York
    "LSE":  (51.5151, -0.0984),   # London
    "TSE":  (35.6812, 139.7671),  # Tokyo
    "ASX":  (-33.8678, 151.2073), # Sydney
}

@dataclass
class VicariousSignal:
    symbol: str
    impact_magnitude: float  # Weighted by distance
    perceived_intensity_db: float
    tension_load: float      # Potential "drowning" signal
    is_drowning: bool        # True if tension exceeds threshold

class VicariousFieldEngine:
    """
    Calculates field transmission from source signals to the shore.
    """

    def __init__(self, threshold_db: float = 60.0):
        self.threshold_db = threshold_db

    def _haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
        """Calculate the great-circle distance between two points on Earth in km."""
        R = 6371.0  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * \
            math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def calculate_impact(self, symbol: str, source_energy: float, exchange: str = "NYSE") -> VicariousSignal:
        """
        Calculate the vicarious impact of a signal.
        source_energy: normalized energy (e.g. KEPE or BMR magnitude) 0.0 to 1.0.
        """
        dest_coords = EXCHANGE_COORDS.get(exchange, EXCHANGE_COORDS["NYSE"])
        dist_km = self._haversine_distance(SHORE_LAT, SHORE_LON, dest_coords[0], dest_coords[1])
        
        # Avoid division by zero, min distance 1km
        dist_km = max(dist_km, 1.0)
        
        # 1. Inverse Square Law
        # impact = energy / d^2. We use (dist/1000) to keep numbers manageable (per 1000km).
        dist_unit = dist_km / 1000.0
        impact = source_energy / (dist_unit ** 2)
        
        # 2. dB Exponent (Logarithmic perceived impact)
        # reference energy at 1.0. 
        # intensity_db = 10 * log10(impact / ref)
        # We use a floor to avoid log(0)
        impact_clamped = max(impact, 1e-10)
        intensity_db = 10 * math.log10(impact_clamped / 1e-6) # 1e-6 as baseline "noise" floor
        
        # 3. Tension Load (High intensity at source)
        # Symbols expressing deep damage (e.g. high volatility/panic)
        tension = source_energy * (1.0 / math.sqrt(dist_unit)) # tension felt closer is higher
        
        is_drowning = intensity_db > self.threshold_db
        
        return VicariousSignal(
            symbol=symbol,
            impact_magnitude=round(impact, 6),
            perceived_intensity_db=round(intensity_db, 2),
            tension_load=round(tension, 4),
            is_drowning=is_drowning
        )

    def synthesize_field(self, signals: List[VicariousSignal]) -> float:
        """
        Synthesizes the aggregate vicarious field intensity at the shore.
        """
        if not signals:
            return 0.0
        
        # Shore team coherence: sum of impacts
        total_impact = sum(s.impact_magnitude for s in signals)
        return round(total_impact, 6)
