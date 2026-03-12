"""
global_circadian_trading_state.py — Field-Time Market Session Model

The binary open/closed session framework (London open 08:00 UTC, NYC close 22:00 UTC)
is a forgazi construct: it describes when extraction institutions operate,
not when human biological capacity peaks and dips.

This module replaces that binary with a continuous circadian arc model.
At any UTC moment, multiple population centres are at different circadian phases.
The aggregate of those phases is the actual participant biological state of the market.

A market where London is in Q2 midday-consolidation and NYC is in Q1 morning-activation
is a different entity from a market where Tokyo is in Q3 and Sydney is in Q4.
The price action is the same. The human substrate producing it is different.

See kindpath-canon/TEMPORAL_SOVEREIGNTY.md for the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from dfte.field_time_bridge import (
    TemporalContext,
    build_temporal_context,
    CircadianQuarter,
)

# ── Region definitions ──────────────────────────────────────────────────────
# Each region: (utc_offset_hours, latitude_band, hemisphere, label)
# Latitude bands are 10-degree bands — sovereignty-preserving, no precise GPS.

MARKET_REGIONS: dict[str, tuple[float, int, str, str]] = {
    # (utc_offset, latitude_band, hemisphere, label)
    "London":   (0.0,  51, "north", "UTC+00:00"),    # Southern England: lat ~51N
    "Frankfurt":(1.0,  50, "north", "UTC+01:00"),    # Central Europe winter
    "Dubai":    (4.0,  25, "north", "UTC+04:00"),    # Gulf
    "Mumbai":   (5.5,  20, "north", "UTC+05:30"),    # Indian subcontinent
    "Singapore":(8.0,   1, "equatorial", "UTC+08:00"),
    "Tokyo":    (9.0,  35, "north", "UTC+09:00"),
    "Sydney":   (10.0,-35, "south",  "UTC+10:00"),   # Northern Rivers ref point
    "NYC":      (-5.0, 41, "north", "UTC-05:00"),    # Eastern seaboard winter
    "Chicago":  (-6.0, 42, "north", "UTC-06:00"),
    "LA":       (-8.0, 34, "north", "UTC-08:00"),
    "SaoPaulo": (-3.0,-23, "south",  "UTC-03:00"),
}

# ── Named market states ─────────────────────────────────────────────────────
# These are the trading-relevant combinations that appear in literature as
# "London open", "NY open" etc. We keep them as labels on the forgazi
# coordinate system — never as field-time primary coordinates.

EXTRACTION_FRAME_LABELS = {
    "london_Q1": "London morning-activation (08:00-10:00 UTC)",
    "london_Q2": "London midday-consolidation (12:00-14:00 UTC)",
    "nyc_Q1":   "NYC morning-activation (13:00-15:00 UTC)",
    "overlap_Q3": "London-NYC Q3 overlap (14:00-17:00 UTC)",
    "nyc_Q4":   "NYC evening-repair (20:00-22:00 UTC)",
    "asia_Q1":  "Asia-Pacific Q1 activation (23:00-02:00 UTC)",
    "asia_Q2":  "Asia-Pacific midday (04:00-07:00 UTC)",
}


@dataclass
class RegionCircadianState:
    """
    The circadian state of one market region at a given moment.

    participant_weight: relative volume weighting for this region in
    typical FX/equity markets. Used only for aggregate state computation —
    not for position sizing.
    """
    region_name: str
    temporal_context: TemporalContext
    circadian_quarter: CircadianQuarter
    circadian_state: str
    is_institutional_hours: bool             # Within typical institutional operating hours
    participant_weight: float                # 0-1: relative activity weight


@dataclass
class GlobalCircadianTradingState:
    """
    The aggregate circadian state of global market participants at one moment.

    This replaces the binary open/closed session logic entirely.
    At every UTC moment, multiple population centres are active at
    different circadian phases. The aggregate is the actual human substrate.

    Key derived metrics:
    - dominant_phase: which circadian phase has the most weighted participants
    - activation_score: 0-1, weighted average of morning/afternoon activation
    - consolidation_pressure: weight of participants in Q2 (historically: consolidation/chop)
    - repair_pressure: weight of participants in Q4+ (late-day, post-session fatigue)
    - phase_transition_proximity: minutes to next major region Q1 onset
    """
    utc_reference: str

    # Per-region states
    regions: dict[str, RegionCircadianState]

    # Aggregate field-time metrics
    dominant_phase: str                      # The most-weighted circadian state description
    dominant_quarter: CircadianQuarter
    activation_score: float                  # 0-1: Q1+Q3 weighted activation
    consolidation_pressure: float            # 0-1: Q2 weight (chop signal)
    repair_pressure: float                   # 0-1: Q4+Q5 weight (fatigue signal)
    phase_transition_proximity_minutes: int  # Minutes to next major Q1 onset

    # Forgazi reference labels (kept as archive, not as reasoning)
    extraction_frame_label: str              # e.g. "London Q3 / NYC Q1 overlap"


def compute_global_state(utc_dt: Optional[datetime] = None) -> GlobalCircadianTradingState:
    """
    Compute the full global circadian trading state for a given UTC moment.

    This runs continuously in the DFTE engine pipeline — not as a preprocessing
    step, but as a real-time coordinate for every prediction cycle.
    """
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)

    # Participant weights: adjusted for typical FX volume contribution
    weights = {
        "London":    0.35,
        "Frankfurt": 0.08,
        "Dubai":     0.04,
        "Mumbai":    0.04,
        "Singapore": 0.08,
        "Tokyo":     0.09,
        "Sydney":    0.05,
        "NYC":       0.25,
        "Chicago":   0.05,
        "LA":        0.04,
        "SaoPaulo":  0.03,
    }

    # Typical institutional hours by region (UTC hour window, not mandatory)
    institutional_windows = {
        "London":    (7, 17),
        "Frankfurt": (7, 17),
        "Dubai":     (5, 15),
        "Mumbai":    (3, 12),
        "Singapore": (1, 10),
        "Tokyo":     (0,  9),
        "Sydney":    (22, 7),   # Wraps midnight
        "NYC":       (13, 22),
        "Chicago":   (14, 23),
        "LA":        (15, 22),
        "SaoPaulo":  (12, 21),
    }

    regions: dict[str, RegionCircadianState] = {}
    quarter_weights = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

    for region_name, (utc_offset, lat_band, hemisphere, frame_label) in MARKET_REGIONS.items():
        tc = build_temporal_context(
            utc_dt=utc_dt,
            latitude_band=lat_band,
            hemisphere=hemisphere,  # type: ignore[arg-type]
            utc_offset_hours=utc_offset,
        )

        # Institutional hours check
        utc_hour = utc_dt.hour
        inst_start, inst_end = institutional_windows[region_name]
        if inst_start < inst_end:
            inst_active = inst_start <= utc_hour < inst_end
        else:  # Wraps midnight (Sydney)
            inst_active = utc_hour >= inst_start or utc_hour < inst_end

        w = weights[region_name]
        quarter_weights[tc.circadian_quarter] += w

        regions[region_name] = RegionCircadianState(
            region_name=region_name,
            temporal_context=tc,
            circadian_quarter=tc.circadian_quarter,
            circadian_state=tc.circadian_state,
            is_institutional_hours=inst_active,
            participant_weight=w,
        )

    # Dominant quarter = highest weighted
    dominant_q = max(quarter_weights, key=quarter_weights.get)   # type: ignore[arg-type]

    # Find region with most weight in dominant quarter to get its state description
    dominant_state = "unknown"
    max_w = 0.0
    for r in regions.values():
        if r.circadian_quarter == dominant_q and r.participant_weight > max_w:
            dominant_state = r.circadian_state
            max_w = r.participant_weight

    # Aggregate scores
    activation_score = quarter_weights[1] + quarter_weights[3]   # Q1 + Q3
    consolidation_pressure = quarter_weights[2]
    repair_pressure = quarter_weights[4] + quarter_weights[5]

    # Phase transition: next Q1 onset — simplified: find first region about to
    # enter Q1 (solar minutes since dawn < 60) as a proxy
    phase_transition_minutes = 240  # Default fallback: 4 hours
    for r in regions.values():
        tc = r.temporal_context
        if tc.solar_minutes_since_dawn < 60:
            # This region is in early Q1 — transition just happened
            phase_transition_minutes = max(0, 60 - tc.solar_minutes_since_dawn)
            break

    # Extraction frame label: describe the UTC moment in forgazi terms
    utc_h = utc_dt.hour
    if 7 <= utc_h < 10:
        frame_label = "London Q1 (08:00-10:00 UTC)"
    elif 12 <= utc_h < 14:
        frame_label = "London Q2 / Pre-NYC"
    elif 13 <= utc_h < 17:
        frame_label = "London Q3 / NYC Q1 overlap"
    elif 17 <= utc_h < 22:
        frame_label = "Post-London / NYC Q2-Q3"
    elif 22 <= utc_h or utc_h < 2:
        frame_label = "Asia-Pacific Q1 activation"
    elif 2 <= utc_h < 7:
        frame_label = "Asia-Pacific Q2-Q3"
    else:
        frame_label = f"UTC {utc_h:02d}:00"

    return GlobalCircadianTradingState(
        utc_reference=utc_dt.isoformat(),
        regions=regions,
        dominant_phase=dominant_state,
        dominant_quarter=dominant_q,
        activation_score=round(activation_score, 3),
        consolidation_pressure=round(consolidation_pressure, 3),
        repair_pressure=round(repair_pressure, 3),
        phase_transition_proximity_minutes=phase_transition_minutes,
        extraction_frame_label=frame_label,
    )


def get_dominant_session_label(state: GlobalCircadianTradingState) -> str:
    """
    Return a plain-language description of the current global field-time state.

    This replaces terminology like "London session" or "NY open"
    with a description of actual aggregate biological state.
    """
    if state.activation_score > 0.55:
        if state.phase_transition_proximity_minutes < 90:
            return "High activation — phase transition approaching"
        return "High activation — multiple Q1/Q3 populations active"
    elif state.consolidation_pressure > 0.40:
        return "Consolidation dominant — major populations in Q2"
    elif state.repair_pressure > 0.45:
        return "Repair phase — most major populations in Q4/Q5"
    else:
        return f"Mixed field — dominant {state.dominant_phase} ({state.dominant_quarter=})"
