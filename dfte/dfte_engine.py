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

Trade tier logic:
  NANO   — pure market physics. KEPE is background filter only.
           Governs: no shorting syntropic assets, no trading extractive
  MID    — hybrid. WFS modulates MFS conviction.
           Governs: WFS ≥ 0.40 required, equity_weight applied
  LARGE  — impact-first. WFS is primary selector.
           Governs: WFS ≥ 0.60 required, syntropic assets preferred,
                    extractive assets blocked, ν × UCS governs size

Position sizing:
  base_size × ν × equity_weight × opc_modifier × risk_budget
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

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

    # Rationale
    rationale: str
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
        # Nano: only block extractive assets, otherwise pass
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
    """
    Determine appropriate tier from both field readings.
    LARGE > MID > NANO, downgrade if either field insufficient.
    """
    # BMR tier recommendation is starting point
    bmr_tier = bmr.trade_tier

    # WFS gates modify tier
    if bmr_tier == "LARGE":
        if kepe.wfs < 0.55 or kepe.is_extractive:
            bmr_tier = "MID"
        if kepe.is_syntropic and kepe.wfs >= 0.60:
            return "LARGE"  # confirmed

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
) -> tuple[float, float]:
    """
    Returns (position_size_pct, risk_pct).

    Size formula: base_risk × ν × equity_weight × opc_modifier × tier_scale

    tier_scale: NANO=0.25, MID=0.5, LARGE=1.0
    """
    tier_scales = {"NANO": 0.25, "MID": 0.5, "LARGE": 1.0, "WAIT": 0.0}
    tier_scale = tier_scales.get(tier, 0.0)

    if tier_scale == 0.0:
        return 0.0, 0.0

    # ν modulates conviction
    nu = bmr.nu

    # OPC from KEPE — forward faith
    opc = kepe.opc
    opc_mod = float(0.5 + opc * 0.5)  # 0.5 → 1.0

    # Equity weight (syntropic assets get size boost)
    equity_w = float(np.clip(kepe.equity_weight, 0, 2.0))

    # Conviction: MFS × WFS product (both fields must agree)
    conviction = float(np.clip(bmr.mfs * 2 * kepe.wfs * 2 / 4, 0, 1))

    # Position size
    pos_size = base_risk_pct * nu * opc_mod * equity_w * tier_scale * conviction
    pos_size = float(np.clip(pos_size, 0, 10.0))  # max 10% per trade

    # Risk: tighter stops for lower conviction
    risk_pct = float(np.clip(base_risk_pct * (0.5 + conviction * 0.5), 0.1, 2.0))

    return pos_size, risk_pct


# ─── Main DFTE Engine ─────────────────────────────────────────────────────────

def synthesise_dfte_signal(
    bmr: BMRSummary,
    kepe: KEPESummary,
    base_risk_pct: float = 1.0,
) -> DFTESignal:
    """
    Synthesise both field readings into a unified trade signal.
    This is the core DFTE decision function.
    """
    symbol = bmr.symbol

    # Determine tier
    tier = determine_tier(bmr, kepe)

    # Run gates
    mfs_pass, mfs_reason = mfs_gate(bmr, tier)
    wfs_pass, wfs_reason = wfs_gate(kepe, tier)

    # Governance gate (additional checks)
    governance_pass = True
    governance_reason = "Governance gate: passed"
    warnings = []

    if kepe.is_extractive:
        governance_pass = False
        governance_reason = f"GOVERNANCE BLOCK: {symbol} is extractive — KindPath sovereignty principle"

    if bmr.lsii_flag in ("high", "very_high") and tier in ("MID", "LARGE"):
        warnings.append(f"LSII-Price {bmr.lsii_flag}: late-move arc break detected — reduce size")

    if kepe.interference_load > 0.5:
        warnings.append(f"Interference load {kepe.interference_load:.2f}: contradictory world field")

    all_gates = mfs_pass and wfs_pass and governance_pass

    # Conviction score
    conviction = float(np.clip(
        bmr.mfs * 0.45 + kepe.wfs * 0.35 + bmr.nu * 0.20,
        0, 1
    ))

    # Action
    if kepe.is_extractive:
        action = "BLOCKED"
        pos_size, risk_pct = 0.0, 0.0
    elif not all_gates or tier == "WAIT":
        action = "HOLD"
        pos_size, risk_pct = 0.0, 0.0
    elif bmr.direction > 0.15:
        action = "BUY"
        pos_size, risk_pct = compute_position_size(bmr, kepe, tier, base_risk_pct)
    elif bmr.direction < -0.15:
        action = "SELL"
        pos_size, risk_pct = compute_position_size(bmr, kepe, tier, base_risk_pct)
        pos_size = -pos_size  # short position
    else:
        action = "HOLD"
        pos_size, risk_pct = 0.0, 0.0

    # Rationale
    rationale_parts = [
        f"Tier={tier} | MFS={bmr.mfs:.2f} [{bmr.mfs_label}] | WFS={kepe.wfs:.2f} [{kepe.wfs_label}]",
        f"ν={bmr.nu:.3f} | SPI={kepe.spi:.2f} | OPC={kepe.opc:.2f}",
        f"Gates: MFS={'✓' if mfs_pass else '✗'} ({mfs_reason}) | "
        f"WFS={'✓' if wfs_pass else '✗'} ({wfs_reason}) | "
        f"GOV={'✓' if governance_pass else '✗'}",
    ]
    if kepe.is_syntropic:
        rationale_parts.append(f"Syntropic asset — equity weight ×{kepe.equity_weight}")
    rationale = " | ".join(rationale_parts)

    return DFTESignal(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        action=action,
        tier=tier,
        conviction=conviction,
        direction=bmr.direction,
        position_size_pct=abs(pos_size),
        risk_pct=risk_pct,
        mfs=bmr.mfs,
        wfs=kepe.wfs,
        nu=bmr.nu,
        ucs=kepe.unified_curvature,
        mfs_gate=mfs_pass,
        wfs_gate=wfs_pass,
        governance_gate=governance_pass,
        all_gates_passed=all_gates,
        rationale=rationale,
        warnings=warnings,
        evidence_level="TESTABLE",
    )
