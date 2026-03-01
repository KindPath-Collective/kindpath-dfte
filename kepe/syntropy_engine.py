"""
KEPE — Syntropy Engine + World Field Score
============================================
Synthesises all world field indicators into the
Syntropy Potential Index (SPI) and World Field Score (WFS).

WFS is now instrument-specific and temporally layered:
  Structural  (annual/quarterly) weight=0.40 — World Bank, yield curve
  Medium      (monthly/weekly)   weight=0.35 — sector flows, credit, real yield
  Surface     (daily)            weight=0.25 — sentiment, VIX, narrative

SPI = ZPB equivalent in the world field
  High SPI (→1.0) = world conditions supporting syntropic growth
  Low SPI (→0.0)  = world conditions under entropic loading

Doctrine-to-computation mappings:
  IN (Insecure Neutrality)    → Entropy Indicator (EI)
  ZPB (Zero-Point Benevolence)→ Syntropy Potential Index (SPI)
  Curvature                   → Field Gradient Tensor (FGT)
  Contradiction               → Interference Load (IL)
  Placebo of kindness         → Optimism Propagation Coefficient (OPC)
  System coherence            → Unified Curvature Score (UCS)
"""

from __future__ import annotations
import os
import json
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from kepe.indicators import WorldSignal

logger = logging.getLogger(__name__)

# Domain weights — used within each temporal layer
# Higher weight = more world-field relevance per unit of domain signal
DOMAIN_WEIGHTS: Dict[str, float] = {
    "SOCIAL":        0.20,  # equity, wellbeing — KindPath primary layer
    "ECOLOGICAL":    0.18,  # land as living system
    "MACRO":         0.22,  # sovereign economic conditions (yield, credit)
    "NARRATIVE":     0.12,  # collective story field
    "OPTIMISM":      0.10,  # faith engine direction
    "CONFLICT":      0.08,  # pressure/interference loading
    "SECTOR_FLOW":   0.25,  # instrument-specific sector capital flows
    "RISK_APPETITE": 0.22,  # risk-on/off (primarily for crypto)
    "KPRE":          0.35,  # physical flow — real-economy generative field
}

# Temporal layer weights
TEMPORAL_WEIGHTS: Dict[str, float] = {
    "STRUCTURAL": 0.40,   # annual/quarterly — background macro/social regime
    "MEDIUM":     0.35,   # monthly/weekly  — sector flow, credit, real yield
    "SURFACE":    0.25,   # daily           — sentiment, volatility, narrative
}

# WFS thresholds → World Field State
WFS_THRESHOLDS = {
    "ZPB":       0.65,
    "COHERENT":  0.45,
    "LOADED":    0.25,
    "DISRUPTED": 0.00,
}

# Benevolence filter
SYNTROPIC_SECTORS = {
    "clean_energy":   ["ICLN", "ENPH", "FSLR", "NEE", "BEP", "PLUG"],
    "healthcare":     ["XLV", "JNJ", "UNH", "ISRG"],
    "education_tech": ["ARKG", "EDTK"],
    "sustainable":    ["ESGV", "ESGU", "SUSL"],
    "infrastructure": ["PAVE", "GII"],
    "food_security":  ["MOO", "SOIL"],
}

EXTRACTIVE_INDICATORS = {
    "tobacco":          ["MO", "PM", "BTI"],
    "weapons":          ["LMT", "RTX", "NOC", "BA"],
    "predatory_finance": ["SLM"],
    "private_prison":   ["GEO", "CXW"],
}

# ─── WFS history persistence ──────────────────────────────────────────────────

_WFS_HISTORY_DIR = "/tmp/kepe_cache"
os.makedirs(_WFS_HISTORY_DIR, exist_ok=True)


def _history_path(symbol: str) -> str:
    safe = symbol.replace("/", "_").replace("=", "_").replace("-", "_")
    return f"{_WFS_HISTORY_DIR}/wfs_history_{safe}.json"


def _load_wfs_history(symbol: str) -> List[float]:
    try:
        with open(_history_path(symbol)) as f:
            data = json.load(f)
        return [float(r["wfs"]) for r in data.get("readings", [])][-5:]
    except Exception:
        return []


def _save_wfs_history(symbol: str, history: List[float]):
    readings = [{"wfs": v, "ts": datetime.utcnow().isoformat()} for v in history[-5:]]
    try:
        with open(_history_path(symbol), "w") as f:
            json.dump({"symbol": symbol, "readings": readings}, f)
    except Exception:
        pass


# ─── STS computation ──────────────────────────────────────────────────────────

def compute_sts(wfs_history: List[float]) -> Tuple[str, float]:
    """
    Compute Syntropy Trajectory Score from WFS history (last 3 readings used).

    LOADING      — net change > +0.03  (world field gaining syntropy)
    DETERIORATING— net change < -0.03  (world field losing coherence)
    STABLE       — net change within ±0.03

    Returns (state, delta).
    [TESTABLE] — threshold and window require calibration against outcomes.
    """
    if len(wfs_history) < 2:
        return "STABLE", 0.0
    recent = wfs_history[-3:]
    delta  = recent[-1] - recent[0]
    if delta > 0.03:
        return "LOADING", delta
    if delta < -0.03:
        return "DETERIORATING", delta
    return "STABLE", delta


def compute_sts_position(sts: str, is_syntropic: bool, is_extractive: bool) -> str:
    """
    2×2 positioning matrix: governance category × STS trajectory.

    Syntropic + LOADING      → ZPB_LOADING   (early position, size up)
    Syntropic + STABLE       → COMPRESSION   (watch for ν recovery)
    Syntropic + DETERIORATING→ REVIEW        (category vs field tension)
    Neutral   + LOADING      → EMERGING      (monitor)
    Neutral   + STABLE       → RANGE         (BMR/ν primary)
    Neutral   + DETERIORATING→ FADING        (reduce/avoid)
    Extractive (any)         → BLOCKED
    """
    if is_extractive:
        return "BLOCKED"
    if is_syntropic:
        return {
            "LOADING":       "ZPB_LOADING",
            "STABLE":        "COMPRESSION",
            "DETERIORATING": "REVIEW",
        }.get(sts, "COMPRESSION")
    return {
        "LOADING":       "EMERGING",
        "STABLE":        "RANGE",
        "DETERIORATING": "FADING",
    }.get(sts, "RANGE")


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
    entropy_indicator: float
    field_gradient: float
    interference_load: float
    unified_curvature: float

    # Syntropy Trajectory Score
    sts: str = "STABLE"           # LOADING | DETERIORATING | STABLE
    sts_position: str = "RANGE"   # ZPB_LOADING | COMPRESSION | REVIEW | EMERGING | FADING | RANGE | BLOCKED
    sts_delta: float = 0.0        # net WFS change over last 3 readings
    wfs_history: List[float] = field(default_factory=list)   # last 5 WFS readings

    # Temporal layer WFS breakdown
    layer_wfs: Dict[str, float] = field(default_factory=dict)

    # Domain readings
    domain_scores: Dict[str, float] = field(default_factory=dict)

    # Equity / benevolence
    is_syntropic_asset: bool = False
    is_extractive_asset: bool = False
    equity_weight: float = 1.0

    # Evidence
    evidence_notes: List[str] = field(default_factory=list)
    interpretation: str = ""
    field_note: str = ""


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _compute_wfs_from_signals(signals: List[WorldSignal]) -> float:
    """
    Compute WFS from a list of signals using domain-weighted mean.
    Normalises by effective weight (domain_weight × confidence) so that
    missing domains don't pull the score toward zero.
    """
    num = 0.0
    den = 0.0
    for sig in signals:
        if sig.confidence > 0.1:
            w    = DOMAIN_WEIGHTS.get(sig.domain, 0.10) * sig.confidence
            num += ((sig.value + 1) / 2) * w   # map -1→+1 to 0→1
            den += w
    if den < 1e-10:
        return 0.5   # neutral if no confident signals
    return float(np.clip(num / den, 0, 1))


def _compute_temporal_wfs(signals: List[WorldSignal]) -> tuple[float, Dict[str, float]]:
    """
    Compute WFS with temporal layering.

    Returns (final_wfs, {layer: layer_wfs}) so per-layer scores are available
    for interpretation/debugging.
    """
    by_layer: Dict[str, List[WorldSignal]] = {
        "STRUCTURAL": [], "MEDIUM": [], "SURFACE": []
    }
    for sig in signals:
        layer = getattr(sig, "temporal_layer", "SURFACE")
        if layer in by_layer:
            by_layer[layer].append(sig)

    layer_wfs: Dict[str, float] = {}
    for layer, sigs in by_layer.items():
        if sigs:
            layer_wfs[layer] = _compute_wfs_from_signals(sigs)

    if not layer_wfs:
        return 0.5, {}

    # Weighted combination — renormalise if some layers are absent
    num = sum(TEMPORAL_WEIGHTS[l] * layer_wfs[l] for l in layer_wfs)
    den = sum(TEMPORAL_WEIGHTS[l] for l in layer_wfs)
    wfs = float(np.clip(num / den, 0, 1))
    return wfs, layer_wfs


def compute_entropy_indicator(signals: List[WorldSignal]) -> float:
    """
    EI = weighted mean of signal deviations from syntropy (positive pole).
    High EI = world field under entropic loading = IN-Loading condition.
    """
    if not signals:
        return 0.5
    weighted = []
    weights  = []
    for sig in signals:
        entropy_val = (1.0 - sig.value) / 2   # map -1→+1 to 1→0
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
    return float(np.clip(float(np.std(values)), 0, 1))


def compute_interference_load(signals: List[WorldSignal]) -> float:
    """
    IL = contradiction load in the world field.
    When domains strongly contradict each other, interference is high.
    """
    if len(signals) < 2:
        return 0.0
    values = [s.value for s in signals if s.confidence > 0.2]
    if len(values) < 2:
        return 0.0
    contradictions = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] * values[j] < 0:
                contradictions.append(abs(values[i] - values[j]) / 2)
    if not contradictions:
        return 0.0
    return float(np.clip(float(np.mean(contradictions)), 0, 1))


def compute_opc(optimism_signal: Optional[WorldSignal],
                social_signal: Optional[WorldSignal]) -> float:
    """
    OPC = Optimism Propagation Coefficient.
    High OPC = community faith active and directed = ZPB potential.
    """
    base = 0.5
    if optimism_signal and optimism_signal.confidence > 0.2:
        base = (optimism_signal.value + 1) / 2
    if social_signal and social_signal.confidence > 0.2:
        equity_adj = (social_signal.value + 1) / 4
        base = float(np.clip(base + equity_adj, 0, 1))
    return float(base)


def classify_symbol(symbol: str) -> tuple[bool, bool, float]:
    """
    Returns (is_syntropic, is_extractive, equity_weight).
    """
    s = symbol.upper()
    for sector, tickers in SYNTROPIC_SECTORS.items():
        if s in tickers:
            return True, False, 1.5
    for sector, tickers in EXTRACTIVE_INDICATORS.items():
        if s in tickers:
            return False, True, 0.0
    return False, False, 1.0


# ─── Main synthesis ───────────────────────────────────────────────────────────

def synthesise_kepe_profile(
    symbol: str,
    signals: List[WorldSignal],
    prior_wfs: Optional[float] = None,
    market_curvature_k: Optional[float] = None,
) -> KEPEProfile:
    """
    Synthesise instrument-specific world field signals into a KEPEProfile.
    WFS is computed with temporal layering (STRUCTURAL 0.40 / MEDIUM 0.35 / SURFACE 0.25).
    """
    # ── Domain score aggregation (used for SPI, EI, OPC) ──────────────────────
    domain_scores: Dict[str, float] = {}
    domain_confidence: Dict[str, float] = {}

    for sig in signals:
        if sig.confidence > 0.1:
            if sig.domain not in domain_scores:
                domain_scores[sig.domain]     = 0.0
                domain_confidence[sig.domain] = 0.0
            domain_scores[sig.domain]     += sig.value * sig.confidence
            domain_confidence[sig.domain] += sig.confidence

    for domain in list(domain_scores.keys()):
        conf = domain_confidence[domain]
        if conf > 1e-10:
            domain_scores[domain] = float(domain_scores[domain] / conf)
        else:
            del domain_scores[domain]

    # ── Temporal WFS ──────────────────────────────────────────────────────────
    wfs, layer_wfs = _compute_temporal_wfs(signals)

    # WFS label
    wfs_label = "DISRUPTED"
    for state, thresh in sorted(WFS_THRESHOLDS.items(), key=lambda x: -x[1]):
        if wfs >= thresh:
            wfs_label = state
            break

    # ── SPI — syntropy potential (ecological + social + macro) ────────────────
    eco = domain_scores.get("ECOLOGICAL", 0.0)
    soc = domain_scores.get("SOCIAL", 0.0)
    mac = domain_scores.get("MACRO", 0.0)
    spi = float(np.clip((eco * 0.4 + soc * 0.4 + mac * 0.2 + 1) / 2, 0, 1))

    # ── Doctrine computations ─────────────────────────────────────────────────
    ei  = compute_entropy_indicator(signals)
    fgt = compute_field_gradient(signals, prior_wfs)
    il  = compute_interference_load(signals)

    opt_sig = next((s for s in signals if s.domain == "OPTIMISM"), None)
    soc_sig = next((s for s in signals if s.domain == "SOCIAL"), None)
    opc     = compute_opc(opt_sig, soc_sig)

    world_curve = abs(wfs - 0.5) * 2
    market_k    = abs(market_curvature_k) if market_curvature_k is not None else 0.0
    ucs         = float(np.clip(world_curve * 0.6 + market_k * 0.4, 0, 1))

    # ── Symbol classification ─────────────────────────────────────────────────
    is_syntropic, is_extractive, equity_weight = classify_symbol(symbol)

    # ── STS — Syntropy Trajectory Score ───────────────────────────────────────
    wfs_history = _load_wfs_history(symbol)
    wfs_history.append(wfs)
    wfs_history = wfs_history[-5:]
    _save_wfs_history(symbol, wfs_history)
    sts, sts_delta       = compute_sts(wfs_history)
    sts_position         = compute_sts_position(sts, is_syntropic, is_extractive)

    # ── Interpretation ────────────────────────────────────────────────────────
    dir_word = "syntropic" if wfs >= 0.55 else "entropic" if wfs < 0.40 else "neutral"
    layer_str = " | ".join(
        f"{l}={v:.2f}" for l, v in sorted(layer_wfs.items())
    )
    interpretation = (
        f"World field reading {dir_word} (WFS={wfs:.2f}). "
        f"Layers: [{layer_str}]. "
        f"SPI={spi:.2f} | OPC={opc:.2f} | EI={ei:.2f} | IL={il:.2f}."
    )
    if is_syntropic:
        interpretation += f" {symbol} is a syntropic asset — equity weight ×{equity_weight}."
    elif is_extractive:
        interpretation += f" {symbol} is flagged extractive — blocked from LARGE tier."
    if il > 0.4:
        interpretation += " High interference load — contradictory world field signals."

    field_note = (
        f"WFS={wfs:.2f} [{wfs_label}] | "
        + " | ".join(f"{l}={v:.2f}" for l, v in sorted(layer_wfs.items()))
        + f" | SPI={spi:.2f} | OPC={opc:.2f} | EI={ei:.2f} | IL={il:.2f} | UCS={ucs:.2f}"
    )

    evidence_notes = [
        "World Bank social/equity data: [ESTABLISHED] — authoritative annual indicators.",
        "Yield curve (10Y-3M): [ESTABLISHED] — validated recession/expansion regime signal.",
        "Credit spread proxy (HYG/LQD): [TESTABLE] — ETF ratio as spread directional proxy.",
        "Equity breadth (RSP/SPY): [TESTABLE] — equal-weight ratio as participation measure.",
        "Clean energy flow (ICLN/XLE): [TESTABLE] — sector rotation as world-field proxy.",
        "Grid parity (XLE/ICLN ratio): [TESTABLE] — relative competitiveness proxy.",
        "Real yield (TIP/IEF): [TESTABLE] — TIPS vs nominal as real yield directional.",
        "Crypto risk appetite (BTC + ARKK): [TESTABLE] — momentum and risk-premium proxy.",
        "Regulatory proxy (COIN vs BTC): [TESTABLE] — Coinbase spread as regulatory sentiment.",
        "VIX narrative proxy: [TESTABLE] — volatility as fear/narrative stress measure.",
        "Optimism/conflict proxies: [TESTABLE] — market-derived, not direct social measures.",
        "WFS as trade input: [SPECULATIVE] — world field → price action correlation requires backtesting.",
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
        sts=sts,
        sts_position=sts_position,
        sts_delta=sts_delta,
        wfs_history=wfs_history,
        layer_wfs=layer_wfs,
        domain_scores=domain_scores,
        is_syntropic_asset=is_syntropic,
        is_extractive_asset=is_extractive,
        equity_weight=equity_weight,
        evidence_notes=evidence_notes,
        interpretation=interpretation,
        field_note=field_note,
    )
