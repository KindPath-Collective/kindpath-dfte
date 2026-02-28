"""
KEPE — Syntropy Engine + World Field Score
============================================
Synthesises all world field indicators into the
Syntropy Potential Index (SPI) and World Field Score (WFS).

SPI = ZPB equivalent in the world field
  High SPI (→1.0) = world conditions supporting syntropic growth
  Low SPI (→0.0)  = world conditions under entropic loading

WFS = the KEPE output consumed by DFTE
  Used alongside BMR's MFS for unified trade selection.

Doctrine-to-computation mappings (from Copilot brief):
  IN (Insecure Neutrality)    → Entropy Indicator (EI)
  ZPB (Zero-Point Benevolence)→ Syntropy Potential Index (SPI)
  Curvature                   → Field Gradient Tensor (FGT)
  Contradiction               → Interference Load (IL)
  Placebo of kindness         → Optimism Propagation Coefficient (OPC)
  System coherence            → Unified Curvature Score (UCS)

E = [(Me × Community × Country) · ν]² governs at every scale.
The KEPE applies this to the world field — Country is the primary layer.
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from kepe.indicators import WorldSignal

logger = logging.getLogger(__name__)

# Domain weights in WFS computation
DOMAIN_WEIGHTS = {
    "SOCIAL":      0.25,  # equity, wellbeing — most direct KindPath relevance
    "ECOLOGICAL":  0.20,  # Country layer — land as living system
    "MACRO":       0.20,  # sovereign economic conditions
    "NARRATIVE":   0.15,  # collective story field
    "OPTIMISM":    0.12,  # faith engine direction
    "CONFLICT":    0.08,  # pressure/interference load
}

# WFS thresholds → World Field State
WFS_THRESHOLDS = {
    "ZPB":        0.65,   # High syntropy — world field supporting growth
    "COHERENT":   0.45,   # Moderate coherence
    "LOADED":     0.25,   # Entropic loading
    "DISRUPTED":  0.00,   # World field under significant distress
}

# Benevolence filter — assets in these sectors get positive equity weighting
SYNTROPIC_SECTORS = {
    "clean_energy":   ["ICLN", "ENPH", "FSLR", "NEE", "BEP"],
    "healthcare":     ["XLV", "JNJ", "UNH", "ISRG"],
    "education_tech": ["ARKG", "EDTK"],
    "sustainable":    ["ESGV", "ESGU", "SUSL"],
    "infrastructure": ["PAVE", "GII"],
    "food_security":  ["MOO", "SOIL"],
}

# Contradiction filter — these are structurally extractive (against KindPath values)
EXTRACTIVE_INDICATORS = {
    "tobacco":   ["MO", "PM", "BTI"],
    "weapons":   ["LMT", "RTX", "NOC", "BA"],
    "predatory_finance": ["SLM"],  # student loan servicers
    "private_prison":    ["GEO", "CXW"],
}


@dataclass
class KEPEProfile:
    """
    Full KEPE world field reading for a given instrument/sector.
    The world-side complement to BMRProfile.
    """
    symbol: str
    timestamp: datetime

    # Core scores
    wfs: float              # World Field Score 0.0 → 1.0
    wfs_label: str          # ZPB | COHERENT | LOADED | DISRUPTED
    spi: float              # Syntropy Potential Index 0.0 → 1.0
    opc: float              # Optimism Propagation Coefficient 0.0 → 1.0

    # Doctrine computations
    entropy_indicator: float     # EI: 0 = low entropy, 1 = high entropy
    field_gradient: float        # FGT: rate of change in world field
    interference_load: float     # IL: contradiction/conflict loading
    unified_curvature: float     # UCS: combined world + market curvature

    # Domain readings
    domain_scores: Dict[str, float] = field(default_factory=dict)

    # Equity / benevolence
    is_syntropic_asset: bool = False
    is_extractive_asset: bool = False
    equity_weight: float = 1.0      # multiplier: syntropic > 1, extractive = 0

    # Evidence
    evidence_notes: List[str] = field(default_factory=list)
    interpretation: str = ""
    field_note: str = ""


def compute_entropy_indicator(signals: List[WorldSignal]) -> float:
    """
    EI = weighted mean of signal deviations from syntropy (positive pole).
    High EI = world field under entropic loading = IN-Loading condition.
    """
    if not signals:
        return 0.5
    weighted = []
    weights = []
    for sig in signals:
        # Entropy is inverse of syntropy
        entropy_val = (1.0 - sig.value) / 2  # map -1→+1 to 1→0
        w = sig.confidence * DOMAIN_WEIGHTS.get(sig.domain, 0.1)
        weighted.append(entropy_val * w)
        weights.append(w)
    if not weights or sum(weights) < 1e-10:
        return 0.5
    return float(np.clip(sum(weighted) / sum(weights), 0, 1))


def compute_field_gradient(signals: List[WorldSignal],
                           prior_wfs: Optional[float] = None) -> float:
    """
    FGT = rate of change in world field.
    Computed from signal value spread (divergence across domains).
    High gradient = field in transition. Low gradient = stable field.
    """
    if len(signals) < 2:
        return 0.0
    values = [s.value for s in signals if s.confidence > 0.2]
    if len(values) < 2:
        return 0.0
    gradient = float(np.std(values))  # spread across domains = field tension
    return float(np.clip(gradient, 0, 1))


def compute_interference_load(signals: List[WorldSignal]) -> float:
    """
    IL = contradiction load in the world field.
    When domains strongly contradict each other, interference is high.
    e.g. Optimism high but Conflict high = contradictory field.
    """
    if len(signals) < 2:
        return 0.0
    values = [s.value for s in signals if s.confidence > 0.2]
    if len(values) < 2:
        return 0.0
    # Pairwise contradiction
    contradictions = []
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            if values[i] * values[j] < 0:  # opposite signs
                contradictions.append(abs(values[i] - values[j]) / 2)
    if not contradictions:
        return 0.0
    return float(np.clip(np.mean(contradictions), 0, 1))


def compute_opc(optimism_signal: Optional[WorldSignal],
                social_signal: Optional[WorldSignal]) -> float:
    """
    OPC = Optimism Propagation Coefficient.
    How much is the faith engine aimed forward?
    Combines optimism signal with social equity base.
    High OPC = community faith is active and directed = ZPB potential.
    """
    base = 0.5
    if optimism_signal and optimism_signal.confidence > 0.2:
        base = (optimism_signal.value + 1) / 2  # map -1→+1 to 0→1
    if social_signal and social_signal.confidence > 0.2:
        equity_adj = (social_signal.value + 1) / 4  # 0→0.5
        base = float(np.clip(base + equity_adj, 0, 1))
    return float(base)


def classify_symbol(symbol: str) -> tuple[bool, bool, float]:
    """
    Returns (is_syntropic, is_extractive, equity_weight).
    Syntropic assets get boosted equity weight.
    Extractive assets are blocked from LARGE tier.
    """
    s = symbol.upper()
    for sector, tickers in SYNTROPIC_SECTORS.items():
        if s in tickers:
            return True, False, 1.5
    for sector, tickers in EXTRACTIVE_INDICATORS.items():
        if s in tickers:
            return False, True, 0.0  # blocked
    return False, False, 1.0


def synthesise_kepe_profile(
    symbol: str,
    signals: List[WorldSignal],
    prior_wfs: Optional[float] = None,
    market_curvature_k: Optional[float] = None,
) -> KEPEProfile:
    """
    Synthesise all world field signals into a KEPEProfile.
    """
    # Domain score aggregation
    domain_scores: Dict[str, float] = {}
    domain_confidence: Dict[str, float] = {}

    for sig in signals:
        if sig.confidence > 0.1:
            if sig.domain not in domain_scores:
                domain_scores[sig.domain] = 0.0
                domain_confidence[sig.domain] = 0.0
            domain_scores[sig.domain] += sig.value * sig.confidence
            domain_confidence[sig.domain] += sig.confidence

    # Normalise
    for domain in list(domain_scores.keys()):
        conf = domain_confidence[domain]
        if conf > 1e-10:
            domain_scores[domain] = float(domain_scores[domain] / conf)
        else:
            del domain_scores[domain]

    # WFS: confidence-weighted, domain-weighted mean
    wfs_num = 0.0
    wfs_den = 0.0
    for domain, score in domain_scores.items():
        w = DOMAIN_WEIGHTS.get(domain, 0.1)
        conf = domain_confidence.get(domain, 0.0)
        eff_w = w * conf
        wfs_num += ((score + 1) / 2) * eff_w  # map -1→+1 to 0→1
        wfs_den += eff_w

    wfs = float(np.clip(wfs_num / (wfs_den + 1e-10), 0, 1))

    # WFS label
    wfs_label = "DISRUPTED"
    for state, thresh in sorted(WFS_THRESHOLDS.items(), key=lambda x: -x[1]):
        if wfs >= thresh:
            wfs_label = state
            break

    # SPI — syntropy potential (WFS skewed toward ecological + social)
    eco  = domain_scores.get("ECOLOGICAL", 0.0)
    soc  = domain_scores.get("SOCIAL", 0.0)
    mac  = domain_scores.get("MACRO", 0.0)
    spi = float(np.clip((eco * 0.4 + soc * 0.4 + mac * 0.2 + 1) / 2, 0, 1))

    # Doctrine computations
    ei = compute_entropy_indicator(signals)
    fgt = compute_field_gradient(signals, prior_wfs)
    il = compute_interference_load(signals)

    # OPC
    opt_sig = next((s for s in signals if s.domain == "OPTIMISM"), None)
    soc_sig = next((s for s in signals if s.domain == "SOCIAL"), None)
    opc = compute_opc(opt_sig, soc_sig)

    # Unified Curvature Score
    # UCS combines world field curvature (WFS distance from neutral) + market curvature
    world_curve = abs(wfs - 0.5) * 2  # 0 = balanced, 1 = extreme
    market_k = abs(market_curvature_k) if market_curvature_k is not None else 0.0
    ucs = float(np.clip(world_curve * 0.6 + market_k * 0.4, 0, 1))

    # Symbol classification
    is_syntropic, is_extractive, equity_weight = classify_symbol(symbol)

    # Interpretation
    dir_word = "syntropic" if wfs >= 0.55 else "entropic" if wfs < 0.40 else "neutral"
    interpretation = (
        f"World field reading {dir_word} (WFS={wfs:.2f}). "
        f"SPI={spi:.2f} | OPC={opc:.2f} | EI={ei:.2f} | IL={il:.2f}. "
    )
    if is_syntropic:
        interpretation += f"{symbol} is a syntropic asset — equity weight boosted (×{equity_weight}). "
    elif is_extractive:
        interpretation += f"{symbol} is flagged as extractive — blocked from LARGE tier. "
    if il > 0.4:
        interpretation += "High interference load — contradictory world field signals. "

    field_note = (
        f"WFS={wfs:.2f} [{wfs_label}] | SPI={spi:.2f} | OPC={opc:.2f} | "
        f"EI={ei:.2f} | IL={il:.2f} | UCS={ucs:.2f}"
    )

    evidence_notes = [
        "World Bank social/equity data: [ESTABLISHED] — authoritative annual indicators.",
        "Commodity ecological proxy: [TESTABLE] — indirect measure requiring calibration.",
        "GDELT narrative sentiment: [TESTABLE] — directional, not predictive alone.",
        "Optimism/conflict proxies: [TESTABLE] — market-derived, not direct social measures.",
        "WFS as trade input: [SPECULATIVE] — world field → price action correlation unproven without backtesting.",
    ]

    return KEPEProfile(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        wfs=wfs,
        wfs_label=wfs_label,
        spi=spi,
        opc=opc,
        entropy_indicator=ei,
        field_gradient=fgt,
        interference_load=il,
        unified_curvature=ucs,
        domain_scores=domain_scores,
        is_syntropic_asset=is_syntropic,
        is_extractive_asset=is_extractive,
        equity_weight=equity_weight,
        evidence_notes=evidence_notes,
        interpretation=interpretation,
        field_note=field_note,
    )
