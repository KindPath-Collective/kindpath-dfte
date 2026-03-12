"""
field_time_bridge.py — Temporal Sovereignty Bridge for DFTE

Translates between UTC timestamps (the forgazi coordinate) and field-time
(the biological/ecological temporal coordinate).

UTC is kept. It is the forgazi reference — the extractive world's record
of when events occur. It is never discarded. But it is never the primary
reasoning coordinate. Field-time is the primary coordinate.

See kindpath-canon/TEMPORAL_SOVEREIGNTY.md for the full framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal, Optional

# ── Types ──────────────────────────────────────────────────────────────────

CircadianQuarter = Literal[1, 2, 3, 4, 5]
# 1 = activating-rise (dawn–late-morning)
# 2 = consolidation-dip (midday)
# 3 = second-activation (afternoon)
# 4 = evening-repair (dusk)
# 5 = deep-repair (sleep)

UltradianPhase = Literal["peak", "trough", "transition"]

EightPointSeason = Literal[
    "early-spring", "mid-spring", "late-spring",
    "early-summer", "midsummer",
    "early-autumn", "mid-autumn", "late-autumn",
    "early-winter", "deep-winter",
]

LunarPhase = Literal[
    "new", "waxing-crescent", "first-quarter", "waxing-gibbous",
    "full", "waning-gibbous", "last-quarter", "waning-crescent",
]

FieldPhase = Literal["seedling", "establishing", "productive", "senescent", "dormant"]


@dataclass
class TemporalContext:
    """
    The primary temporal coordinate of a DFTE event.

    The UTC timestamp is retained as a forgazi reference label.
    All reasoning about market-participant biological state,
    session timing, and circadian market structure uses this object.

    Primary use: `global_circadian_trading_state.py` uses TemporalContext
    to compute the biological state of active market participant populations.
    """

    # The actual coordinates
    circadian_quarter: CircadianQuarter
    circadian_state: str                        # e.g. 'morning-activation'
    ultradian_phase: UltradianPhase
    solar_minutes_since_dawn: int               # Minutes since local sunrise
    solar_minutes_to_dusk: int                  # Minutes until local sunset

    # Ecological position
    season: EightPointSeason
    hemisphere: Literal["north", "south", "equatorial"]
    lunar_phase: LunarPhase
    solar_event_proximity_days: Optional[int]   # None if >21 days from solstice/equinox

    # Latitude (sovereignty-preserving: 10-degree band, never precise GPS)
    latitude_band: int                          # e.g. -35 for central NSW, +51 for southern England

    # The forgazi reference — kept, never the reasoning ground
    utc_reference: str                          # ISO 8601 UTC string
    extraction_frame_label: str                 # e.g. 'UTC+11' — the forgazi coordinate label

    def to_dict(self) -> dict:
        return {
            "circadian_quarter": self.circadian_quarter,
            "circadian_state": self.circadian_state,
            "ultradian_phase": self.ultradian_phase,
            "solar_minutes_since_dawn": self.solar_minutes_since_dawn,
            "solar_minutes_to_dusk": self.solar_minutes_to_dusk,
            "season": self.season,
            "hemisphere": self.hemisphere,
            "lunar_phase": self.lunar_phase,
            "solar_event_proximity_days": self.solar_event_proximity_days,
            "latitude_band": self.latitude_band,
            "utc_reference": self.utc_reference,
            "extraction_frame_label": self.extraction_frame_label,
        }


# ── Solar calculations ─────────────────────────────────────────────────────


def _solar_noon_and_day_length(
    latitude_deg: float, dt: date
) -> tuple[float, float]:
    """
    Approximate solar noon (fractional hours, local solar time)
    and day length (hours) for a given latitude and date.

    Uses the Spencer formula for solar declination.
    Accurate to within ~2 minutes — sufficient for circadian quarter assignment.
    """
    day_of_year = dt.timetuple().tm_yday
    # Solar declination (degrees) via Spencer formula
    B = math.radians((360 / 365) * (day_of_year - 81))
    declination = 23.45 * math.sin(B)
    lat_rad = math.radians(latitude_deg)
    dec_rad = math.radians(declination)

    # Hour angle at sunrise/sunset
    cos_ha = -math.tan(lat_rad) * math.tan(dec_rad)
    cos_ha = max(-1.0, min(1.0, cos_ha))  # Clamp for polar regions
    ha = math.degrees(math.acos(cos_ha))  # Degrees

    # Day length in hours; solar noon is always at 12:00 local solar time
    day_length = (2 * ha) / 15.0
    sunrise = 12.0 - day_length / 2.0
    sunset = 12.0 + day_length / 2.0
    return 12.0, day_length, sunrise, sunset


def _assign_circadian_quarter(
    minutes_since_dawn: int, total_daylight_minutes: int
) -> tuple[CircadianQuarter, str]:
    """
    Assign a circadian quarter from position in the solar day.

    The quarters are solar-relative, not clock-relative.
    Q1 begins at dawn. Q4 ends near sleep onset (typically ~4h after dusk).
    """
    if total_daylight_minutes <= 0:
        return 5, "deep-repair"

    # Express as fraction of the daylight period
    fraction = minutes_since_dawn / total_daylight_minutes

    if minutes_since_dawn < 0:
        return 5, "deep-repair"
    elif fraction < 0.35:
        return 1, "morning-activation"
    elif fraction < 0.50:
        return 2, "midday-consolidation"
    elif fraction < 0.75:
        return 3, "afternoon-activation"
    elif fraction < 1.0:
        return 4, "evening-repair"
    else:
        # Post-dusk — could be Q4 tail or approaching deep repair
        post_dusk_minutes = minutes_since_dawn - total_daylight_minutes
        if post_dusk_minutes < 180:      # Up to 3 hours after dusk: still Q4
            return 4, "evening-repair"
        else:
            return 5, "deep-repair"


def _lunar_phase(dt: date) -> LunarPhase:
    """
    Approximate lunar phase from date.

    Uses a well-known reference new moon (Jan 6 2000) and the mean synodic month.
    Error margin: ±0.5 days, sufficient for ecological phase assignment.
    """
    reference_new_moon = date(2000, 1, 6)
    synodic_month = 29.53058867
    days_since_ref = (dt - reference_new_moon).days
    cycle_position = (days_since_ref % synodic_month) / synodic_month

    if cycle_position < 0.0625:
        return "new"
    elif cycle_position < 0.1875:
        return "waxing-crescent"
    elif cycle_position < 0.3125:
        return "first-quarter"
    elif cycle_position < 0.4375:
        return "waxing-gibbous"
    elif cycle_position < 0.5625:
        return "full"
    elif cycle_position < 0.6875:
        return "waning-gibbous"
    elif cycle_position < 0.8125:
        return "last-quarter"
    else:
        return "waning-crescent"


def _ecological_season(dt: date, hemisphere: str) -> EightPointSeason:
    """
    8-point ecological season from date and hemisphere.

    Southern hemisphere seasons are inverted. The split is hard —
    March in the south is early autumn, not early spring.
    """
    month = dt.month

    # Northern hemisphere base mapping
    if month in (3, 4):
        north_season: EightPointSeason = "early-spring" if month == 3 else "mid-spring"
    elif month == 5:
        north_season = "late-spring"
    elif month == 6:
        north_season = "early-summer"
    elif month == 7:
        north_season = "midsummer"
    elif month == 8:
        north_season = "early-autumn"
    elif month in (9, 10):
        north_season = "mid-autumn" if month == 9 else "late-autumn"
    elif month in (11, 12):
        north_season = "early-winter" if month == 11 else "deep-winter"
    else:  # Jan, Feb
        north_season = "deep-winter" if month == 1 else "early-spring"

    if hemisphere == "north" or hemisphere == "equatorial":
        return north_season

    # Southern inversion map
    inversion: dict[EightPointSeason, EightPointSeason] = {
        "early-spring": "early-autumn",
        "mid-spring": "mid-autumn",
        "late-spring": "late-autumn",
        "early-summer": "early-winter",
        "midsummer": "deep-winter",
        "early-autumn": "early-spring",
        "mid-autumn": "mid-spring",
        "late-autumn": "late-spring",
        "early-winter": "early-summer",
        "deep-winter": "midsummer",
    }
    return inversion[north_season]


def _solar_event_proximity(dt: date) -> Optional[int]:
    """
    Days until/since nearest solstice or equinox.
    Returns None if more than 21 days away.
    """
    year = dt.year
    events = [
        date(year, 3, 20),   # Vernal equinox (approx)
        date(year, 6, 21),   # Summer solstice (approx)
        date(year, 9, 22),   # Autumnal equinox (approx)
        date(year, 12, 21),  # Winter solstice (approx)
    ]
    min_days = min(abs((dt - ev).days) for ev in events)
    return min_days if min_days <= 21 else None


# ── Main builder ────────────────────────────────────────────────────────────

def build_temporal_context(
    utc_dt: Optional[datetime] = None,
    latitude_band: int = -35,       # Default: approximate Northern Rivers, NSW
    hemisphere: Literal["north", "south", "equatorial"] = "south",
    utc_offset_hours: float = 10.0, # Default: AEST
    ultradian_phase: UltradianPhase = "peak",  # Caller provides or estimates
) -> TemporalContext:
    """
    Build a TemporalContext from a UTC datetime and location parameters.

    latitude_band: 10-degree band (e.g. -35 for NSW, +51 for London). Not precise GPS.
    hemisphere: hard split for season inversion.
    utc_offset_hours: used only to derive local solar time for circadian quarter.
    ultradian_phase: the caller tracks this from session start time + individual baseline.

    The UTC timestamp is stored as the forgazi reference but does not govern
    the field-time coordinates.
    """
    if utc_dt is None:
        utc_dt = datetime.now(timezone.utc)

    # Convert UTC to approximate local solar time
    local_hour = (utc_dt.hour + utc_offset_hours) % 24
    local_dt_date = utc_dt.date()

    # Solar calculations for this latitude and date
    _, day_length_hours, sunrise_hour, sunset_hour = _solar_noon_and_day_length(
        latitude_band, local_dt_date
    )

    sunrise_minutes = sunrise_hour * 60
    sunset_minutes = sunset_hour * 60
    local_minutes = local_hour * 60 + utc_dt.minute

    solar_minutes_since_dawn = int(local_minutes - sunrise_minutes)
    solar_minutes_to_dusk = max(0, int(sunset_minutes - local_minutes))
    total_daylight_minutes = int(day_length_hours * 60)

    circadian_q, circadian_state = _assign_circadian_quarter(
        solar_minutes_since_dawn, total_daylight_minutes
    )

    # Derive extraction frame label from utc_offset
    sign = "+" if utc_offset_hours >= 0 else ""
    offset_str = f"{sign}{int(utc_offset_hours):02d}:00"
    extraction_frame_label = f"UTC{offset_str}"

    return TemporalContext(
        circadian_quarter=circadian_q,
        circadian_state=circadian_state,
        ultradian_phase=ultradian_phase,
        solar_minutes_since_dawn=max(0, solar_minutes_since_dawn),
        solar_minutes_to_dusk=solar_minutes_to_dusk,
        season=_ecological_season(local_dt_date, hemisphere),
        hemisphere=hemisphere,
        lunar_phase=_lunar_phase(local_dt_date),
        solar_event_proximity_days=_solar_event_proximity(local_dt_date),
        latitude_band=latitude_band,
        utc_reference=utc_dt.isoformat(),
        extraction_frame_label=extraction_frame_label,
    )


def get_default_field_time() -> TemporalContext:
    """
    Convenience: build a TemporalContext for the current moment with
    default Northern Rivers, NSW coordinates.
    """
    return build_temporal_context()
