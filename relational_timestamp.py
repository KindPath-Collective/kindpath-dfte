"""
Relational Timestamp Layer
===========================
Country time alongside UTC.

UTC stays as the database primary key. These are signal VARIABLES —
the relational context in which a signal is generated. Boundaries
are zones of field intensification, not walls. Market open/close
are gradients, not on/off switches.

Location: Northern NSW, Bundjalung Country (-28.65°, 153.56°)

Fields
------
solar_elevation      float  — Sun degrees above horizon at signal time
solar_arc_phase      float  — 0=midnight, 0.5=solar noon, 1=next midnight
lunar_phase          float  — 0=new moon, 0.5=full moon, 1=next new moon
season_southern      str    — SUMMER|AUTUMN|WINTER|SPRING (southern hemisphere)
market_phase         dict   — {symbol: PREOPEN|OPEN|MID|CLOSE|AFTERHOURS}
cross_market_overlap bool   — True when 2+ major markets simultaneously OPEN
session_arc          dict   — {symbol: float 0→1} fraction through primary session
boundary_proximity   float  — 0→1, 1.0 at open/close boundary, 0.0 at midpoint

Evidence posture: [TESTABLE]
Relational fields as signal variables requires outcome validation.
The physical measurements (solar, lunar) are [ESTABLISHED].
Their predictive relevance to field signals is [TESTABLE].
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ─── Location ─────────────────────────────────────────────────────────────────

LATITUDE  = "-28.6500"   # Northern NSW, Bundjalung Country
LONGITUDE = "153.5600"

# ─── Major market sessions (UTC hour ranges, approximate) ──────────────────
# Boundaries are gradients — ±1h DST drift doesn't change the field logic.
#
# format: (open_utc_decimal, close_utc_decimal)
# decimal hours: 14.5 = 14:30 UTC

_SESSIONS: Dict[str, tuple] = {
    "ASX":     (23.0, 30.0),   # Sydney 10:00-16:00 AEST → 23:00-06:00 UTC (wrap +24h)
    "TSE":     (0.0,  6.5),    # Tokyo 09:00-15:30 JST → 00:00-06:30 UTC
    "LSE":     (8.0,  16.5),   # London 09:00-17:30 BST → 08:00-16:30 UTC
    "NYSE":    (14.5, 21.0),   # New York 09:30-16:00 EST → 14:30-21:00 UTC
}

# Symbols routed to their primary exchange for phase calculation
SYMBOL_EXCHANGE: Dict[str, str] = {
    "ICLN": "NYSE", "NEE": "NYSE", "ENPH": "NYSE", "FSLR": "NYSE",
    "BEP": "NYSE", "TSLA": "NYSE", "SPY": "NYSE", "QQQ": "NYSE",
    "GLD": "NYSE", "BTC-USD": "NYSE", "ETH-USD": "NYSE",
    "BHP": "ASX", "CBA": "ASX", "WBC": "ASX",
}

_PREOPEN_BUFFER_H  = 0.5   # 30 min before open = PREOPEN zone
_SESSION_EDGE_FRAC = 0.15  # first/last 15% of session = OPEN/CLOSE phase


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class RelationalTimestamp:
    """Relational context for one signal generation event."""
    utc_time:             datetime
    solar_elevation:      float           # degrees
    solar_arc_phase:      float           # 0→1
    lunar_phase:          float           # 0→1
    season_southern:      str             # SUMMER|AUTUMN|WINTER|SPRING
    market_phase:         Dict[str, str]  # {symbol: phase_label}
    cross_market_overlap: bool
    session_arc:          Dict[str, float]  # {symbol: 0→1}
    boundary_proximity:   float           # 0→1


# ─── Helper: decimal UTC hour ─────────────────────────────────────────────────

def _utc_hour(dt: datetime) -> float:
    """Current UTC hour as decimal (e.g. 14:30 → 14.5)."""
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


# ─── Solar ────────────────────────────────────────────────────────────────────

def _compute_solar(dt: datetime) -> tuple[float, float]:
    """
    Returns (solar_elevation_degrees, solar_arc_phase).
    solar_arc_phase: 0=midnight, 0.5=solar noon, 1=next midnight.
    Uses ephem for accurate ephemeris calculation.
    Falls back to simple approximation if ephem unavailable.
    """
    try:
        import ephem  # type: ignore

        obs        = ephem.Observer()
        obs.lat    = LATITUDE
        obs.lon    = LONGITUDE
        obs.date   = dt.strftime("%Y/%m/%d %H:%M:%S")
        obs.pressure = 0  # no atmospheric refraction correction

        sun = ephem.Sun(obs)
        elevation = float(math.degrees(sun.alt))

        # Solar arc phase: fraction of 24h day centred on solar noon
        # 0 = local midnight, 0.5 = solar noon, 1 = next midnight
        # Compute solar noon for this date
        obs_noon = ephem.Observer()
        obs_noon.lat  = LATITUDE
        obs_noon.lon  = LONGITUDE
        obs_noon.date = dt.strftime("%Y/%m/%d 12:00:00")
        transit_utc   = obs_noon.next_transit(sun)
        transit_dt    = ephem.Date(transit_utc).datetime().replace(tzinfo=timezone.utc)

        # solar_arc_phase from midnight to midnight centred on solar noon
        # Treat solar noon as 0.5; map the rest linearly.
        delta_h    = (dt - transit_dt).total_seconds() / 3600.0
        arc_phase  = (delta_h / 24.0 + 0.5) % 1.0

        return elevation, arc_phase

    except Exception as e:
        logger.debug(f"ephem solar calc failed: {e} — using approximation")
        # Fallback: day-fraction approximation
        utc_h     = _utc_hour(dt)
        local_h   = (utc_h + 10.0) % 24.0   # AEST offset approx
        arc_phase = local_h / 24.0
        # Elevation: simple sine approximation
        elev_angle = math.pi * (arc_phase - 0.25) * 2  # peaks at noon
        elevation  = math.degrees(math.sin(elev_angle)) * 50.0 - 15.0
        return elevation, arc_phase


# ─── Lunar ────────────────────────────────────────────────────────────────────

def _compute_lunar_phase(dt: datetime) -> float:
    """
    Returns lunar phase 0→1: 0=new moon, 0.5=full moon, 1=back to new.
    Uses ephem for accuracy; falls back to synodic approximation.
    """
    try:
        import ephem  # type: ignore
        moon      = ephem.Moon()
        moon.compute(dt.strftime("%Y/%m/%d %H:%M:%S"))
        # moon.phase is illumination 0-100%; convert to 0→1 phase cycle
        # (not the same as cycle position, but sufficient for field work)
        # Use moon.elong instead for proper phase cycle
        elong_deg = float(math.degrees(moon.elong))  # -180 to +180
        phase     = (elong_deg + 180.0) / 360.0      # 0=new, 0.5=full, 1=new
        return round(phase, 4)
    except Exception as e:
        logger.debug(f"ephem lunar calc failed: {e} — using synodic approximation")
        # Known new moon reference: 2024-01-11 UTC
        ref_new_moon = datetime(2024, 1, 11, tzinfo=timezone.utc)
        synodic_days = 29.53059
        elapsed      = (dt - ref_new_moon).total_seconds() / 86400.0
        return (elapsed % synodic_days) / synodic_days


# ─── Season ───────────────────────────────────────────────────────────────────

def _southern_season(dt: datetime) -> str:
    """Southern hemisphere calendar season by month."""
    m = dt.month
    if m in (12, 1, 2):
        return "SUMMER"
    if m in (3, 4, 5):
        return "AUTUMN"
    if m in (6, 7, 8):
        return "WINTER"
    return "SPRING"  # Sep, Oct, Nov


# ─── Market phase ─────────────────────────────────────────────────────────────

def _session_open_close(exchange: str, utc_h: float) -> tuple[float, float]:
    """
    Return (open_utc, close_utc) normalised to a 0–48h window
    so ASX's wrap-around (23:00–06:00 UTC) works cleanly.
    """
    open_h, close_h = _SESSIONS[exchange]
    # Wrap ASX: if current hour < 6 (within the wrapped session), shift it
    if exchange == "ASX":
        if utc_h < 7.0:
            utc_h += 24.0
    return open_h, close_h


def _market_phase_for_exchange(exchange: str, utc_h: float) -> tuple[str, float, float]:
    """
    Returns (phase_label, session_arc, boundary_proximity) for one exchange.
    session_arc: -1 if market not open/pre-open; 0→1 through session.
    boundary_proximity: 0→1, 1 at open or close.
    """
    open_h, close_h = _SESSIONS[exchange]
    dur = close_h - open_h

    # Normalise ASX wrap
    adj_h = utc_h
    if exchange == "ASX" and utc_h < 7.0:
        adj_h = utc_h + 24.0

    preopen_start = open_h - _PREOPEN_BUFFER_H

    if adj_h < preopen_start or adj_h >= close_h + _PREOPEN_BUFFER_H:
        return "AFTERHOURS", -1.0, 0.0

    if preopen_start <= adj_h < open_h:
        return "PREOPEN", 0.0, 1.0

    # Within session
    arc = (adj_h - open_h) / dur
    arc = max(0.0, min(1.0, arc))

    if arc <= _SESSION_EDGE_FRAC:
        phase = "OPEN"
    elif arc >= (1.0 - _SESSION_EDGE_FRAC):
        phase = "CLOSE"
    else:
        phase = "MID"

    # boundary_proximity: 1 at edges (arc=0 or 1), 0 at midpoint (arc=0.5)
    boundary_proximity = 1.0 - 2.0 * abs(arc - 0.5)

    return phase, arc, boundary_proximity


# ─── Public interface ─────────────────────────────────────────────────────────

def compute_relational_timestamp(
    symbols: list,
    dt: Optional[datetime] = None,
) -> RelationalTimestamp:
    """
    Compute all relational fields for the given symbols at the given time.
    dt defaults to now (UTC). Returns a RelationalTimestamp.

    Safe — never raises. Falls back to defaults on any error.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    utc_h = _utc_hour(dt)

    # ── Solar ────────────────────────────────────────────────────────────────
    try:
        solar_elevation, solar_arc_phase = _compute_solar(dt)
    except Exception as e:
        logger.warning(f"Solar computation failed: {e}")
        solar_elevation, solar_arc_phase = 0.0, 0.5

    # ── Lunar ────────────────────────────────────────────────────────────────
    try:
        lunar_phase = _compute_lunar_phase(dt)
    except Exception as e:
        logger.warning(f"Lunar computation failed: {e}")
        lunar_phase = 0.0

    # ── Season ───────────────────────────────────────────────────────────────
    season_southern = _southern_season(dt)

    # ── Market phases ────────────────────────────────────────────────────────
    market_phase:  Dict[str, str]   = {}
    session_arc:   Dict[str, float] = {}
    open_exchanges = []
    boundary_proximities = []

    for symbol in symbols:
        exchange = SYMBOL_EXCHANGE.get(symbol.upper(), "NYSE")
        try:
            phase, arc, bp = _market_phase_for_exchange(exchange, utc_h)
        except Exception:
            phase, arc, bp = "AFTERHOURS", -1.0, 0.0

        market_phase[symbol] = phase
        session_arc[symbol]  = round(arc, 4)
        if phase not in ("AFTERHOURS",):
            open_exchanges.append(exchange)
            boundary_proximities.append(bp)

    # ── Cross-market overlap ─────────────────────────────────────────────────
    cross_market_overlap = len(set(open_exchanges)) >= 2

    # ── Aggregate boundary proximity (max across open markets) ──────────────
    boundary_proximity = float(max(boundary_proximities)) if boundary_proximities else 0.0

    return RelationalTimestamp(
        utc_time             = dt,
        solar_elevation      = round(solar_elevation, 3),
        solar_arc_phase      = round(solar_arc_phase, 4),
        lunar_phase          = round(lunar_phase, 4),
        season_southern      = season_southern,
        market_phase         = market_phase,
        cross_market_overlap = cross_market_overlap,
        session_arc          = session_arc,
        boundary_proximity   = round(boundary_proximity, 4),
    )
