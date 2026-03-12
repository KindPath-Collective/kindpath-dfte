"""
DFTE — Dual Field Trading Engine
===================================
The integration layer consuming both field readings:

  BMR → Market Field Score (MFS)  — how the price field looks
  KEPE → World Field Score (WFS)  — how the world field looks

A trade is only executed when BOTH fields are favourable.
MFS without WFS = technical without ethics.
WFS without MFS = impact without timing.
Together = the first syntropy-governed, benevolence-first trading engine.
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class BMRSummary:
    """Minimal BMR signal consumed by DFTE (from BMR server response)."""
    symbol: str
    mfs: float
    mfs_label: str
    direction: float
    nu: float
    field_state: str
    trade_tier: str
    lsii: Optional[float] = None
    lsii_flag: Optional[str] = None
    k: Optional[float] = None
    curvature_state: Optional[str] = None


@dataclass
class KEPESummary:
    """Minimal KEPE signal consumed by DFTE (from KEPEProfile)."""
    symbol: str
    wfs: float
    wfs_label: str
    spi: float
    opc: float
    interference_load: float
    unified_curvature: float
    equity_weight: float
    is_syntropic: bool
    is_extractive: bool
    sts: str = "STABLE"
    sts_position: str = "RANGE"


@dataclass
class DFTESignal:
    """
    Unified trade signal from DFTE.
    The synthesised output of both field readings.
    """
    symbol: str
    timestamp: datetime

    # Decision
    action: str           # BUY | SELL | HOLD | BLOCKED
    tier: str             # NANO | MID | LARGE | WAIT
    conviction: float     # 0.0 → 1.0 combined field conviction
    direction: float      # -1.0 (short) → +1.0 (long)

    # Sizing
    position_size_pct: float    # % of available capital to allocate
    risk_pct: float             # max risk per trade as % of portfolio

    # Field readings
    mfs: float
    wfs: float
    nu: float
    ucs: float

    # Gate results
    mfs_gate: bool
    wfs_gate: bool
    governance_gate: bool
    all_gates_passed: bool

    # STS
    sts: str = "STABLE"
    sts_position: str = "RANGE"

    # Rationale
    rationale: str = ""
    warnings: List[str] = field(default_factory=list)
    evidence_level: str = "TESTABLE"


# ─── Gate functions ───────────────────────────────────────────────────────────

def mfs_gate(bmr: BMRSummary, tier: str) -> tuple[bool, str]:
    """Market Field Score gate per tier."""
    if tier == "NANO":
        if bmr.field_state == "SIC":
            return False, "SIC event — no nano trades during coherence collapse"
        if bmr.lsii_flag in ("high", "very_high") and abs(bmr.direction) < 0.3:
            return False, "LSII-Price high + weak direction — arc integrity failed"
        return True, "MFS gate: NANO passed"

    elif tier == "MID":
        if bmr.nu < 0.35:
            return False, f"ν={bmr.nu:.3f} below MID threshold (0.35)"
        if bmr.field_state in ("SIC", "IN_LOADING") and bmr.nu < 0.45:
            return False, f"IN-Loading with low ν — wait for field recovery"
        return True, f"MFS gate: MID passed (ν={bmr.nu:.3f})"

    elif tier == "LARGE":
        if bmr.nu < 0.55:
            return False, f"ν={bmr.nu:.3f} below LARGE threshold (0.55)"
        if bmr.field_state not in ("ZPB", "DRIFT"):
            return False, f"LARGE requires ZPB or DRIFT field, got {bmr.field_state}"
        if bmr.lsii_flag in ("high", "very_high"):
            return False, f"LSII-Price {bmr.lsii_flag} — structural break, no large position"
        return True, f"MFS gate: LARGE passed (ν={bmr.nu:.3f}, {bmr.field_state})"

    return False, f"Unknown tier: {tier}"


def wfs_gate(kepe: KEPESummary, tier: str) -> tuple[bool, str]:
    """World Field Score gate per tier."""
    if kepe.is_extractive:
        return False, f"BLOCKED: extractive asset — conflicts with KindPath sovereignty principles"

    if tier == "NANO":
        return True, "WFS gate: NANO passed (world field is background filter)"

    elif tier == "MID":
        if kepe.wfs < 0.35:
            return False, f"WFS={kepe.wfs:.2f} below MID threshold (0.35)"
        if kepe.interference_load > 0.65:
            return False, f"Interference load {kepe.interference_load:.2f} too high for MID"
        return True, f"WFS gate: MID passed (WFS={kepe.wfs:.2f})"

    elif tier == "LARGE":
        if kepe.wfs < 0.55:
            return False, f"WFS={kepe.wfs:.2f} below LARGE threshold (0.55)"
        if kepe.spi < 0.45:
            return False, f"SPI={kepe.spi:.2f} — world syntropy insufficient for impact position"
        if kepe.interference_load > 0.50:
            return False, f"Interference load {kepe.interference_load:.2f} — contradictory world field"
        return True, f"WFS gate: LARGE passed (WFS={kepe.wfs:.2f}, SPI={kepe.spi:.2f})"

    return False, f"Unknown tier: {tier}"


def determine_tier(bmr: BMRSummary, kepe: KEPESummary) -> str:
    """Determine appropriate tier from both field readings."""
    bmr_tier = bmr.trade_tier
    if bmr_tier == "LARGE":
        if kepe.wfs < 0.55 or kepe.is_extractive:
            bmr_tier = "MID"
        if kepe.is_syntropic and kepe.wfs >= 0.60:
            return "LARGE"
    if bmr_tier == "MID":
        if kepe.wfs < 0.35:
            bmr_tier = "NANO"
    if bmr_tier == "WAIT":
        return "WAIT"
    return bmr_tier


def compute_position_size(
    bmr: BMRSummary,
    kepe: KEPESummary,
    tier: str,
    base_risk_pct: float = 1.0,
    maturity_score: float = 0.0,
) -> tuple[float, float]:
    """Returns (position_size_pct, risk_pct)."""
    tier_scales = {"NANO": 0.15, "MID": 0.5, "LARGE": 1.0, "WAIT": 0.0}
    tier_scale = tier_scales.get(tier, 0.0)
    if tier_scale == 0.0: return 0.0, 0.0

    nu = bmr.nu
    opc = kepe.opc
    opc_mod = float(0.5 + opc * 0.5)
    equity_boost = 1.0 + (maturity_score * (kepe.equity_weight - 1.0))
    equity_w = float(np.clip(equity_boost, 0.5, 2.5))

    mfs_weight = 0.7 - (maturity_score * 0.5) 
    wfs_weight = 0.3 + (maturity_score * 0.5) 
    conviction = float(np.clip((bmr.mfs * mfs_weight) + (kepe.wfs * wfs_weight), 0, 1))

    pos_size = base_risk_pct * nu * opc_mod * equity_w * tier_scale * conviction
    if tier == "NANO":
        pos_size = float(np.clip(pos_size, 0.1, 1.0))
    else:
        pos_size = float(np.clip(pos_size, 0, 10.0))

    risk_pct = float(np.clip(base_risk_pct * (0.5 + conviction * 0.5), 0.1, 2.0))
    return pos_size, risk_pct


def compute_lateral_wisdom(
    bmr: BMRSummary,
    kepe: KEPESummary,
    somatic_value: float = 0.0,
    static_value: float = 0.0,
    historical_edge: float = 1.0
) -> tuple[float, str]:
    """Consensus of lateral vantage points + Field Memory."""
    wisdom_mod = 1.0 * historical_edge
    notes = []
    if historical_edge > 1.0: notes.append("Field Memory (Edge)")
    elif historical_edge < 1.0: notes.append("Field Memory (Dampen)")

    if bmr.direction > 0 and somatic_value < -0.3:
        wisdom_mod *= 0.7
        notes.append("Somatic Shadow")
    if static_value < -0.5:
        wisdom_mod *= 0.8
        notes.append("High Static")
    if (bmr.direction > 0 and kepe.wfs < 0.45) or (bmr.direction < 0 and kepe.wfs > 0.55):
        wisdom_mod *= 0.5
        notes.append("Field Dissonance")

    wisdom_note = ", ".join(notes) if notes else "Lateral Harmony"
    return float(np.clip(wisdom_mod, 0.1, 1.25)), wisdom_note


# ─── Main DFTE Engine ─────────────────────────────────────────────────────────

def synthesise_dfte_signal(
    bmr: BMRSummary,
    kepe: KEPESummary,
    base_risk_pct: float = 1.0,
    maturity_score: float = 0.0,
    somatic_value: float = 0.0,
    static_value: float = 0.0,
    historical_edge: float = 1.0,
    override_timestamp: Optional[datetime] = None
) -> DFTESignal:
    """Synthesise both field readings + Lateral Wisdom into a unified signal."""
    symbol = bmr.symbol
    ts = override_timestamp if override_timestamp else datetime.now(timezone.utc)

    # 0. Clean NaN values
    mfs_val = float(bmr.mfs) if not np.isnan(bmr.mfs) else 0.5
    wfs_val = float(kepe.wfs) if not np.isnan(kepe.wfs) else 0.5
    nu_val  = float(bmr.nu)  if not np.isnan(bmr.nu)  else 0.0
    somatic_val = float(somatic_value) if not np.isnan(somatic_value) else 0.0
    static_val  = float(static_value)  if not np.isnan(static_value)  else 0.0

    # 1. Lateral Wisdom Consensus (LWL)
    wisdom_mod, wisdom_note = compute_lateral_wisdom(
        bmr, kepe, somatic_val, static_val, historical_edge
    )

    # 2. Determine tier
    tier = determine_tier(bmr, kepe)

    # Run gates
    mfs_pass, mfs_reason = mfs_gate(bmr, tier)
    wfs_pass, wfs_reason = wfs_gate(kepe, tier)
    governance_pass = not kepe.is_extractive
    governance_reason = "Governance gate: passed" if governance_pass else f"BLOCK: {symbol} extractive"
    all_gates = mfs_pass and wfs_pass and governance_pass

    # Conviction Shift (Maturity-linked) + Wisdom Modulation
    mfs_w = 0.7 - (maturity_score * 0.5)
    wfs_w = 0.3 + (maturity_score * 0.5)
    conviction = float(np.clip(((mfs_val * mfs_w) + (wfs_val * wfs_w)) * wisdom_mod, 0, 1))

    # Action logic with ZPC
    zpc_threshold = 0.5 if wisdom_mod > 0.9 else 0.65
    zpc_aligned = (bmr.direction > 0 and wfs_val > zpc_threshold) or \
                  (bmr.direction < 0 and wfs_val < (1.0 - zpc_threshold))

    if kepe.is_extractive:
        action = "BLOCKED"
        pos_size, risk_pct = 0.0, 0.0
    elif not all_gates or tier == "WAIT" or wisdom_mod < 0.4:
        action = "HOLD"
        pos_size, risk_pct = 0.0, 0.0
    elif bmr.direction > 0.15 and zpc_aligned:
        action = "BUY"
        pos_size, risk_pct = compute_position_size(bmr, kepe, tier, base_risk_pct, maturity_score)
        pos_size *= wisdom_mod
    elif bmr.direction < -0.15 and zpc_aligned:
        action = "SELL"
        pos_size, risk_pct = compute_position_size(bmr, kepe, tier, base_risk_pct, maturity_score)
        pos_size = -(pos_size * wisdom_mod)
    else:
        action = "HOLD"
        pos_size, risk_pct = 0.0, 0.0

    # Rationale
    rationale_parts = [
        f"Tier={tier} | MFS={mfs_val:.2f} | WFS={wfs_val:.2f} | ν={nu_val:.3f} | M={maturity_score:.2f}",
        f"Wisdom={wisdom_mod:.2f} [{wisdom_note}]",
        f"Gates: M={'✓' if mfs_pass else '✗'} W={'✓' if wfs_pass else '✗'} G={'✓' if governance_pass else '✗'}",
    ]
    if kepe.is_syntropic: rationale_parts.append(f"Syntropic (x{kepe.equity_weight})")
    rationale = " | ".join(rationale_parts)

    return DFTESignal(
        symbol=symbol, timestamp=ts, action=action, tier=tier,
        conviction=conviction, direction=bmr.direction, position_size_pct=abs(pos_size),
        risk_pct=risk_pct, mfs=mfs_val, wfs=wfs_val, nu=nu_val, ucs=kepe.unified_curvature,
        mfs_gate=mfs_pass, wfs_gate=wfs_pass, governance_gate=governance_pass,
        all_gates_passed=all_gates, sts=kepe.sts, sts_position=kepe.sts_position,
        rationale=rationale, warnings=[], evidence_level="TESTABLE",
    )
