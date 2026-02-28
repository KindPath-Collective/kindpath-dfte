"""
DFTE — Governance & Benevolence Layer
=======================================
Gates every trade through KindPath ethical architecture.

Three governance functions:

1. BENEVOLENCE SCORING
   Every instrument scored against KindPath equity principles.
   Syntropic assets amplified. Extractive assets blocked from LARGE.
   Scoring integrates: sector ethics, community impact, ecological footprint.

2. CONTRADICTION DETECTION
   Interference Load (IL) computation per portfolio.
   High IL = portfolio is internally contradictory
   (e.g. holding clean energy long AND fossil fuels long = IN-Loading state).
   Contradiction resolution: flag for review, suggest coherent rebalance.

3. INFLUENCE TRACKING
   Every executed trade is a participant-scale signal in BMR.
   At scale, DFTE positions contribute to the field they read.
   Influence log tracks the system's own contribution to field coherence.
   This closes the feedback loop: trades → curvature shifts → updated ν.

   The system is not just reading the field.
   It is contributing to the field it reads.
   This is the KindPath benevolence propagation mechanism
   operating through capital rather than community intervention.
"""

from __future__ import annotations
import json
import os
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

GOVERNANCE_LOG = "/tmp/dfte_governance_log.json"
INFLUENCE_LOG  = "/tmp/dfte_influence_log.json"


# ─── Sector ethics taxonomy ───────────────────────────────────────────────────

SECTOR_SCORES = {
    # Syntropic sectors (+)
    "clean_energy":    +0.90,
    "healthcare":      +0.70,
    "education":       +0.75,
    "sustainable_ag":  +0.80,
    "water":           +0.85,
    "housing_access":  +0.70,
    "community_bank":  +0.65,

    # Neutral sectors (0)
    "technology":      +0.10,
    "consumer":        +0.00,
    "finance":         -0.10,
    "industrials":     -0.05,
    "real_estate":     +0.05,

    # Extractive sectors (-)
    "fossil_fuel":     -0.80,
    "weapons":         -0.90,
    "tobacco":         -0.85,
    "gambling":        -0.60,
    "private_prison":  -0.95,
    "predatory_lending": -0.80,
}

SYMBOL_SECTOR_MAP = {
    # Clean energy
    "ICLN": "clean_energy", "ENPH": "clean_energy", "FSLR": "clean_energy",
    "NEE": "clean_energy", "BEP": "clean_energy", "TSLA": "clean_energy",

    # Healthcare
    "XLV": "healthcare", "JNJ": "healthcare", "UNH": "healthcare",
    "ISRG": "healthcare", "MRNA": "healthcare",

    # Sustainable
    "ESGV": "sustainable_ag", "ESGU": "sustainable_ag",
    "MOO": "sustainable_ag", "SOIL": "sustainable_ag",

    # Tech (neutral)
    "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
    "NVDA": "technology", "META": "technology", "AMZN": "technology",

    # Finance
    "JPM": "finance", "BAC": "finance", "GS": "finance", "MS": "finance",

    # Fossil fuel
    "XOM": "fossil_fuel", "CVX": "fossil_fuel", "BP": "fossil_fuel",
    "SHEL": "fossil_fuel", "COP": "fossil_fuel", "OXY": "fossil_fuel",
    "XLE": "fossil_fuel",

    # Weapons
    "LMT": "weapons", "RTX": "weapons", "NOC": "weapons",
    "BA": "weapons", "GD": "weapons", "ITA": "weapons",

    # Tobacco
    "MO": "tobacco", "PM": "tobacco", "BTI": "tobacco",

    # Crypto / digital assets (neutral-positive)
    "BTC-USD": "technology", "ETH-USD": "technology",

    # Indices / broad market
    "SPY": "technology", "QQQ": "technology", "IWM": "technology",
    "DIA": "technology", "GLD": "water",  # Gold in water/resource category

    # Commodities
    "GC=F": "water", "CL=F": "fossil_fuel", "SI=F": "water",
    "ZC=F": "sustainable_ag",

    # Private prison
    "GEO": "private_prison", "CXW": "private_prison",

    # Predatory lending
    "SLM": "predatory_lending",
}


@dataclass
class BenevolenceScore:
    """Benevolence assessment for one instrument."""
    symbol: str
    sector: str
    score: float          # -1.0 (extractive) → +1.0 (syntropic)
    is_blocked: bool      # True if blocked from LARGE tier
    tier_cap: str         # Maximum allowable tier: NANO | MID | LARGE
    rationale: str


@dataclass
class ContradictionReport:
    """Portfolio-level contradiction analysis."""
    interference_load: float   # 0→1 (0 = no contradictions)
    contradictions: List[str]  # human-readable contradictions
    coherence_score: float     # portfolio-level ν equivalent
    recommendation: str


@dataclass
class InfluenceRecord:
    """Records DFTE's own contribution to the field."""
    symbol: str
    timestamp: str
    action: str
    tier: str
    size_pct: float
    mfs_at_trade: float
    wfs_at_trade: float
    nu_at_trade: float
    expected_field_contribution: str


# ─── Benevolence scoring ──────────────────────────────────────────────────────

def score_benevolence(symbol: str) -> BenevolenceScore:
    """Score an instrument's benevolence alignment."""
    s = symbol.upper()
    sector = SYMBOL_SECTOR_MAP.get(s, "technology")
    score = SECTOR_SCORES.get(sector, 0.0)

    is_blocked = score <= -0.60  # strong extractive = blocked from LARGE

    if score >= 0.70:
        tier_cap = "LARGE"
        rationale = f"High benevolence ({sector}) — LARGE tier eligible"
    elif score >= 0.0:
        tier_cap = "MID"
        rationale = f"Neutral-positive ({sector}) — MID tier maximum"
    elif score > -0.60:
        tier_cap = "NANO"
        rationale = f"Negative benevolence ({sector}) — NANO tier only"
    else:
        tier_cap = "BLOCKED"
        rationale = f"BLOCKED: extractive sector ({sector}) — conflicts with KindPath values"

    return BenevolenceScore(
        symbol=symbol, sector=sector,
        score=score, is_blocked=is_blocked,
        tier_cap=tier_cap, rationale=rationale
    )


def apply_governance_tier(
    requested_tier: str,
    benevolence: BenevolenceScore,
) -> tuple[str, str]:
    """
    Apply governance ceiling to requested tier.
    Returns (approved_tier, reason).
    """
    tier_order = {"NANO": 0, "MID": 1, "LARGE": 2, "WAIT": -1, "BLOCKED": -2}

    if benevolence.is_blocked:
        return "BLOCKED", benevolence.rationale

    req_level = tier_order.get(requested_tier, -1)
    cap_level = tier_order.get(benevolence.tier_cap, 2)

    if req_level <= cap_level:
        return requested_tier, f"Governance approved: {benevolence.rationale}"
    else:
        approved = benevolence.tier_cap
        return approved, f"Governance downgrade {requested_tier}→{approved}: {benevolence.rationale}"


# ─── Contradiction detection ──────────────────────────────────────────────────

def detect_contradictions(portfolio: Dict[str, float]) -> ContradictionReport:
    """
    Detect ethical + strategic contradictions in a portfolio.
    portfolio: {symbol: position_pct} (positive = long, negative = short)
    """
    contradictions = []
    scores = {}

    for symbol, pos in portfolio.items():
        b = score_benevolence(symbol)
        scores[symbol] = (b.score, b.sector, pos)

    # Check for direct contradictions
    # 1. Long syntropic + long extractive in same theme
    has_clean_long = any(
        pos > 0 and SYMBOL_SECTOR_MAP.get(sym.upper(), "") == "clean_energy"
        for sym, (s, sec, pos) in scores.items()
    )
    has_fossil_long = any(
        pos > 0 and SYMBOL_SECTOR_MAP.get(sym.upper(), "") == "fossil_fuel"
        for sym, (s, sec, pos) in scores.items()
    )
    if has_clean_long and has_fossil_long:
        contradictions.append(
            "Portfolio holds both clean energy (long) and fossil fuel (long) — "
            "IN-Loading contradiction: cancel each other's syntropy signal"
        )

    # 2. Extractive assets in portfolio
    for sym, (score, sector, pos) in scores.items():
        if score <= -0.60 and abs(pos) > 0:
            contradictions.append(
                f"{sym} ({sector}) is extractive but present in portfolio — "
                f"governance review required"
            )

    # 3. Excessive concentration in neutral/low-benevolence
    low_ben_weight = sum(
        abs(pos) for sym, (score, sec, pos) in scores.items()
        if score < 0.1
    )
    total_weight = sum(abs(pos) for _, (_, _, pos) in scores.items()) + 1e-10
    if low_ben_weight / total_weight > 0.60:
        contradictions.append(
            f"{low_ben_weight/total_weight*100:.0f}% of portfolio in neutral/negative "
            "benevolence assets — consider rebalancing toward syntropic sectors"
        )

    # Interference load = proportion of contradictory weight
    il = min(len(contradictions) * 0.25, 1.0)

    # Coherence score = inverse
    coherence = 1.0 - il

    if not contradictions:
        recommendation = "Portfolio coherent — no governance contradictions detected"
    elif il < 0.5:
        recommendation = "Minor contradictions — review but not blocking"
    else:
        recommendation = (
            "Significant interference load — portfolio requires governance review "
            "before LARGE tier trades. Resolve contradictions first."
        )

    return ContradictionReport(
        interference_load=il,
        contradictions=contradictions,
        coherence_score=coherence,
        recommendation=recommendation,
    )


# ─── Influence tracking ───────────────────────────────────────────────────────

def log_influence(
    symbol: str,
    action: str,
    tier: str,
    size_pct: float,
    mfs: float,
    wfs: float,
    nu: float,
) -> InfluenceRecord:
    """
    Record DFTE's own field contribution.
    Every trade is a participant-scale signal.
    At scale, this closes the benevolence propagation loop.
    """
    # Estimate field contribution
    if size_pct > 0 and action == "BUY":
        contrib = f"Positive participant signal on {symbol} — reinforcing field direction"
    elif size_pct > 0 and action == "SELL":
        contrib = f"Corrective participant signal on {symbol} — counterweight to overextension"
    else:
        contrib = "No field contribution (HOLD/BLOCKED)"

    if tier == "LARGE":
        contrib += " [LARGE impact position — field contribution material at scale]"

    record = InfluenceRecord(
        symbol=symbol,
        timestamp=datetime.utcnow().isoformat(),
        action=action,
        tier=tier,
        size_pct=size_pct,
        mfs_at_trade=mfs,
        wfs_at_trade=wfs,
        nu_at_trade=nu,
        expected_field_contribution=contrib,
    )

    # Persist to log
    try:
        log = []
        if os.path.exists(INFLUENCE_LOG):
            with open(INFLUENCE_LOG) as f:
                log = json.load(f)
        log.append({
            "symbol": record.symbol,
            "timestamp": record.timestamp,
            "action": record.action,
            "tier": record.tier,
            "size_pct": record.size_pct,
            "mfs": record.mfs_at_trade,
            "wfs": record.wfs_at_trade,
            "nu": record.nu_at_trade,
            "contribution": record.expected_field_contribution,
        })
        # Keep last 1000 records
        log = log[-1000:]
        with open(INFLUENCE_LOG, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logger.warning(f"Influence log write failed: {e}")

    return record


def get_influence_summary(n: int = 20) -> List[dict]:
    """Retrieve recent influence records."""
    try:
        if os.path.exists(INFLUENCE_LOG):
            with open(INFLUENCE_LOG) as f:
                log = json.load(f)
            return log[-n:]
    except Exception:
        pass
    return []
