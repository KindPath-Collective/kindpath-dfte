"""
SAS — Syntropy Authenticity Score
====================================
Core question: Does this asset's actual field behaviour generate
the conditions it claims to generate?

SAS → 1.0  authentic syntropy (presentation matches field behaviour)
SAS → 0.0  maximum divergence (wolf — presentation masks extraction)
SAS < 0.35 short candidate (with CMAM cooling-off gate)

Components:
  1. RevenueCoherenceSignal  [ESTABLISHED] — SIC code + 10-K revenue language
  2. CapexDirectionSignal    [ESTABLISHED] — XBRL capex trend vs stated mission
  3. OpacitySignal           [TESTABLE]    — Scope 3 gap + supply-chain disclosure
  4. SSIStub                 [SPECULATIVE] — Frame 1 (KEPE WFS) vs Frame 5 (independent)
  5. WolfDetector            [TESTABLE]    — Weighted combination of above four

Multiverse frames:
  Frame 1 (Dominant)   — KEPE WFS (self-reported via market signals)     [implemented]
  Frame 2 (Relational) — Peer comparison                                  [TODO]
  Frame 3 (Temporal)   — Behaviour over 5-year arc                        [TODO]
  Frame 4 (Adversarial)— Stress-tested against extracted thesis            [TODO]
  Frame 5 (Stripped)   — Primary sources only: revenue + capex + opacity  [implemented]
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("dfte.sas")

# ── Shared EDGAR helpers (from kpre_capital — avoid duplication) ──────────────
from kepe.kpre_capital import _edgar_get, _get_cik, _fetch_submissions

# ── 10-K text cache (CIK → text) ─────────────────────────────────────────────
_TEXT_CACHE: dict[str, str] = {}

# ── XBRL capex constant (same endpoint as kpre_capital) ──────────────────────
_XBRL_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


# ─── SIC sector buckets ───────────────────────────────────────────────────────

_FOSSIL_SIC     = frozenset({1311, 1321, 1381, 1382, 1389, 2911, 2990, 5171, 5172})
_CLEAN_TECH_SIC = frozenset({3674, 3559, 3679, 3827, 3699, 3621})   # semiconductors, clean mfg
_TECH_SIC       = frozenset({7372, 7371, 7374, 3577, 3672, 3669})   # software, hardware
_MINING_SIC     = frozenset({1000, 1040, 1090, 1094})                # mining / extractive
_UTILITY_SIC    = frozenset({4911, 4931, 4941})                      # electric / gas utilities
_FUND_FORMS     = frozenset({"NPORT-P", "N-CEN", "N-8A", "497", "485BPOS"})

# ── Keyword sets ─────────────────────────────────────────────────────────────

_FOSSIL_KW = frozenset([
    "oil", "gas", "petroleum", "coal", "drilling", "upstream",
    "refin", "hydrocarbon", "lng", "fracking", "fossil",
])
_CLEAN_KW = frozenset([
    "solar", "wind", "renewable", "microinverter", "photovoltaic",
    "battery storage", "offshore wind", "geothermal", "clean energy",
    "energy storage", "grid-scale",
])
_CLIMATE_KW = frozenset([
    "climate", "net zero", "carbon neutral", "esg",
    "greenhouse", "emissions reduction", "decarbonization", "net-zero",
])
_SCOPE3_KW  = frozenset(["scope 3", "scope3", "scope-3"])
_SUPPLY_KW  = frozenset([
    "supply chain", "supplier", "sourcing", "procurement",
    "vendor", "raw material", "conflict mineral",
])


# ─── SASProfile ───────────────────────────────────────────────────────────────

@dataclass
class SASProfile:
    symbol:            str
    sas_score:         float   # 0→1, higher = more authentic
    wolf_score:        float   # 0→1, higher = more wolf/divergent
    revenue_coherence: float
    capex_direction:   float
    opacity_score:     float
    ssi_gap:           float
    wolf_confirmed:    bool    # wolf_score > 0.65
    short_candidate:   bool    # wolf_score > 0.35 AND WFS_trajectory = DETERIORATING
    evidence_level:    str
    notes:             list = field(default_factory=list)
    timestamp:         datetime = field(default_factory=datetime.utcnow)


# ─── EDGAR fetching helpers ───────────────────────────────────────────────────

def _is_fund(submissions: dict) -> bool:
    """Return True if entity is a registered fund (ETF/mutual fund) not an operating company."""
    recent_forms = set(submissions.get("filings", {}).get("recent", {}).get("form", [])[:20])
    return bool(recent_forms & _FUND_FORMS)


def _fetch_10k_text(cik: str, max_bytes: int = 250_000) -> Optional[str]:
    """
    Stream the first max_bytes of the most recent 10-K primary document.
    Results cached at module level — shared between Revenue and Opacity signals.
    """
    if cik in _TEXT_CACHE:
        return _TEXT_CACHE[cik]

    sub = _fetch_submissions(cik)
    if not sub:
        return None

    recent = sub.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    accnos  = recent.get("accessionNumber", [])
    docs    = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "10-K":
            acc = accnos[i].replace("-", "")
            doc = docs[i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
            resp = _edgar_get(url, stream=True)
            if resp is None or resp.status_code != 200:
                return None
            chunks = []
            total  = 0
            for chunk in resp.iter_content(chunk_size=8192):
                chunks.append(chunk)
                total += len(chunk)
                if total >= max_bytes:
                    break
            text = b"".join(chunks).decode("utf-8", errors="ignore")
            _TEXT_CACHE[cik] = text
            return text

    return None


def _fetch_capex_trend(cik: str) -> Optional[float]:
    """
    Fetch XBRL capex data and return a trend value in [-1, 1].
    Reuses the same computation logic as CapexIntentSignal (kpre_capital.py)
    but independently — avoids cross-module state sharing.
    Returns None if data unavailable.
    """
    from kepe.kpre_capital import CapexIntentSignal
    try:
        sig     = CapexIntentSignal()
        quarters = sig._fetch_capex_quarters(cik)
        if not quarters:
            return None
        trend_val, _ = CapexIntentSignal._compute_capex_trend(quarters)
        return trend_val
    except Exception as e:
        logger.debug(f"_fetch_capex_trend({cik}): {e}")
        return None


# ─── Component 1: RevenueCoherenceSignal ──────────────────────────────────────

class RevenueCoherenceSignal:
    """
    Compares stated sector classification against actual EDGAR evidence.
    SIC code is authoritative for sector. 10-K language ratio is proxy for
    revenue mix. [ESTABLISHED] for SIC; [TESTABLE] for text ratio.
    """

    @staticmethod
    def _count_kw(text: str, keywords: frozenset) -> int:
        text_lower = text.lower()
        return sum(1 for kw in keywords if kw in text_lower)

    @staticmethod
    def _score_coherence(
        sic_code:      Optional[int],
        fossil_count:  int,
        clean_count:   int,
        climate_count: int,
        is_syntropic:  bool,
        is_extractive: bool,
    ) -> tuple[float, list[str]]:
        """
        Revenue coherence 0→1.  [TESTABLE]

        1.0 = stated mission fully confirmed by SIC + language evidence
        0.0 = stated mission completely contradicted (wolf pattern)
        """
        notes: list[str] = []
        total_kw = fossil_count + clean_count + 1e-10
        fossil_ratio = fossil_count / total_kw

        # ── SIC sector classification ──
        if sic_code in _FOSSIL_SIC:
            sic_type = "fossil"
        elif sic_code in _CLEAN_TECH_SIC:
            sic_type = "clean_tech"
        elif sic_code in _TECH_SIC:
            sic_type = "tech"
        elif sic_code in _UTILITY_SIC:
            sic_type = "utility"
        elif sic_code in _MINING_SIC:
            sic_type = "mining"
        else:
            sic_type = "other"

        # ── Score based on classification vs SIC ──
        if is_syntropic:
            if sic_type == "fossil":
                notes.append(f"SIC {sic_code} = fossil sector — contradicts syntropic classification")
                base = 0.08
            elif sic_type in ("clean_tech", "tech"):
                base = 0.85
            elif sic_type == "utility":
                # Utilities could be clean or mixed — text decides
                base = 0.60
            else:
                base = 0.65

            # Text ratio adjustment
            if fossil_ratio > 0.60:
                notes.append(
                    f"10-K language: fossil mentions ({fossil_count}) dominate clean ({clean_count})"
                )
                base = min(base, 0.30)
            elif fossil_ratio > 0.40:
                base *= 0.75

            # ESG claims without clean operations is greenwashing signal
            if climate_count > 10 and fossil_ratio > 0.50:
                notes.append(
                    f"ESG/climate claims ({climate_count}) but fossil language dominates"
                )
                base = min(base, 0.35)

        elif is_extractive:
            # Check for greenwashing: openly extractive company with heavy ESG claims
            if sic_type == "fossil":
                if climate_count > 15 and fossil_ratio > 0.40:
                    notes.append(
                        f"Extractive company with {climate_count} climate claims — "
                        "potential greenwashing (wolf pattern)"
                    )
                    base = 0.20
                else:
                    base = 0.35  # honestly extractive — low SAS (extraction ≠ syntropy)
            else:
                base = 0.40

        else:
            # Neutral
            if sic_type == "fossil" and climate_count > 10:
                notes.append("Fossil SIC with ESG claims — possible transition theater")
                base = 0.50
            else:
                base = 0.72

        return float(np.clip(base, 0.0, 1.0)), notes

    def compute(
        self,
        symbol: str,
        is_syntropic: bool,
        is_extractive: bool,
        text_10k: Optional[str] = None,
        sic_code: Optional[int] = None,
    ) -> tuple[float, dict]:
        notes: list[str] = []

        fossil_count  = self._count_kw(text_10k or "", _FOSSIL_KW)
        clean_count   = self._count_kw(text_10k or "", _CLEAN_KW)
        climate_count = self._count_kw(text_10k or "", _CLIMATE_KW)

        score, score_notes = self._score_coherence(
            sic_code, fossil_count, clean_count, climate_count,
            is_syntropic, is_extractive,
        )
        notes.extend(score_notes)

        return score, {
            "sic_code":     sic_code,
            "fossil_count": fossil_count,
            "clean_count":  clean_count,
            "notes":        notes,
        }


# ─── Component 2: CapexDirectionSignal ────────────────────────────────────────

class CapexDirectionSignal:
    """
    Tests whether capex investment direction confirms stated mission.
    Syntropic company with growing capex = authentic conviction.
    Declining capex while claiming growth = divergence (transition theater).
    [ESTABLISHED] — direct EDGAR XBRL financial data.
    """

    @staticmethod
    def _score_capex(capex_trend: float, is_syntropic: bool) -> float:
        """
        Map capex trend [-1, 1] → coherence score [0, 1].  [TESTABLE]
        For syntropic:  growing capex = authentic → high score.
        For extractive: growing capex = doubling down on extraction → low SAS.
        """
        if is_syntropic:
            # Shift and rescale: trend=-1 → 0.0, trend=0 → 0.50, trend=+1 → 1.0
            return float(np.clip((capex_trend + 1.0) / 2.0, 0.0, 1.0))
        else:
            # For non-syntropic: growing capex confirms extraction → low SAS
            return float(np.clip((1.0 - capex_trend) / 2.0, 0.0, 1.0))

    def compute(self, symbol: str, is_syntropic: bool) -> tuple[float, dict]:
        cik = _get_cik(symbol)
        if not cik:
            return 0.50, {"note": "No CIK — ETF or crypto"}

        trend = _fetch_capex_trend(cik)
        if trend is None:
            return 0.50, {"note": "Capex data unavailable"}

        score = self._score_capex(trend, is_syntropic)
        return score, {"capex_trend": trend, "is_syntropic": is_syntropic}


# ─── Component 3: OpacitySignal ───────────────────────────────────────────────

class OpacitySignal:
    """
    Measures what ISN'T reported relative to what's claimed.
    Scope 3 gap: ESG claims without Scope 3 disclosure = opacity.
    Supply chain disclosure: relative to stated mission.
    [TESTABLE] — keyword proxy for actual disclosure completeness.
    """

    @staticmethod
    def _score_opacity(
        scope3_count:        int,
        climate_count:       int,
        supply_chain_words:  int,
    ) -> tuple[float, list[str]]:
        """
        Opacity score 0→1, higher = more transparent.  [TESTABLE]
        Low score = high opacity = divergence.
        """
        notes: list[str] = []

        # Scope 3 coverage: ratio of Scope 3 mentions to climate claims
        # A company that talks about climate but never mentions Scope 3 is hiding something
        scope3_ratio = scope3_count / max(climate_count, 1)
        scope3_score = float(np.clip(scope3_ratio * 12.0, 0.0, 1.0))   # 0.083 ratio → 1.0

        if climate_count > 8 and scope3_count == 0:
            notes.append(
                f"Climate focus ({climate_count} mentions) but zero Scope 3 disclosure"
            )
            scope3_score = 0.0
        elif climate_count > 5 and scope3_ratio < 0.05:
            notes.append(
                f"Scope 3 under-reported vs climate claims "
                f"(scope3={scope3_count}, climate={climate_count})"
            )

        # Supply chain disclosure coverage
        supply_score = float(np.clip(supply_chain_words / 25.0, 0.0, 1.0))

        opacity = 0.60 * scope3_score + 0.40 * supply_score
        return float(np.clip(opacity, 0.0, 1.0)), notes

    def compute(self, text_10k: Optional[str] = None) -> tuple[float, dict]:
        text = text_10k or ""

        def count_any(text: str, kws: frozenset) -> int:
            tl = text.lower()
            return sum(1 for kw in kws if kw in tl)

        scope3_count       = count_any(text, _SCOPE3_KW)
        climate_count      = count_any(text, _CLIMATE_KW)
        supply_chain_words = count_any(text, _SUPPLY_KW)

        score, notes = self._score_opacity(scope3_count, climate_count, supply_chain_words)
        return score, {
            "scope3_count":       scope3_count,
            "climate_count":      climate_count,
            "supply_chain_words": supply_chain_words,
            "notes":              notes,
        }


# ─── Component 4: SSI Stub (Syntropy Stability Index) ────────────────────────

class SSIStub:
    """
    Syntropy Stability Index — measures gap between self-reported and
    independently reconstructed syntropy.

    Frame 1 (Dominant): KEPE WFS — market-signal derived       [TESTABLE]
    Frame 5 (Stripped): (RevCoherence + CapexDir + Opacity) / 3 [TESTABLE]
    Frames 2,3,4: TODO — peer comparison, temporal arc, adversarial  [SPECULATIVE]

    SSI = 1 - |Frame1 - Frame5|
    High gap → low SSI → wolf indicator
    Low gap  → high SSI → authentic
    """

    # Frame stubs for frames 2, 3, 4
    _FRAME2_NOTE = "TODO: peer comparison — [SPECULATIVE]"
    _FRAME3_NOTE = "TODO: 5-year temporal arc — [SPECULATIVE]"
    _FRAME4_NOTE = "TODO: adversarial stress test — [SPECULATIVE]"

    @staticmethod
    def compute_ssi_gap(frame1_wfs: float, frame5_score: float) -> tuple[float, list[str]]:
        """
        Returns ssi_gap in [0, 1].  [TESTABLE]
        ssi_gap = |Frame1 − Frame5|
        High gap = self-reported doesn't match primary-source reconstruction.
        """
        gap   = abs(float(frame1_wfs) - float(frame5_score))
        notes: list[str] = []
        if gap > 0.30:
            notes.append(
                f"SSI gap {gap:.2f}: KEPE WFS={frame1_wfs:.2f} vs "
                f"primary-source reconstruction={frame5_score:.2f} — divergence detected"
            )
        return float(np.clip(gap, 0.0, 1.0)), notes


# ─── Component 5: WolfDetector ────────────────────────────────────────────────

class WolfDetector:
    """
    Combines all four components into a single wolf divergence score.
    wolf_score = 1 - weighted_authenticity
    wolf_confirmed  = wolf_score > 0.65
    short_candidate = wolf_score > 0.35 AND WFS trajectory = DETERIORATING
    """

    _WEIGHTS = {
        "revenue_coherence": 0.30,
        "capex_direction":   0.25,
        "opacity":           0.25,
        "ssi_authenticity":  0.20,   # ssi_authenticity = 1 - ssi_gap
    }

    @staticmethod
    def wolf_score(
        revenue_coherence: float,
        capex_direction:   float,
        opacity_score:     float,
        ssi_gap:           float,
    ) -> float:
        """
        Wolf score [0, 1].  [TESTABLE]
        Higher = more divergence between presentation and field behaviour.
        """
        authenticity = (
            0.30 * revenue_coherence +
            0.25 * capex_direction   +
            0.25 * opacity_score     +
            0.20 * (1.0 - ssi_gap)   # invert gap: low gap = high authenticity
        )
        return float(np.clip(1.0 - authenticity, 0.0, 1.0))


# ─── SASEngine ────────────────────────────────────────────────────────────────

class SASEngine:
    """
    Compute SAS profile for a symbol given its KEPE profile.

    Usage:
        sas = SASEngine().compute(symbol, kepe_profile)
        if sas.wolf_confirmed:
            block long positions
        if sas.short_candidate:
            flag with CMAM cooling-off gate
    """

    def __init__(self):
        self._rev  = RevenueCoherenceSignal()
        self._cap  = CapexDirectionSignal()
        self._opac = OpacitySignal()

    def compute(self, symbol: str, kepe_profile: Any) -> SASProfile:
        """
        Full SAS computation pipeline.
        Returns neutral profile for ETFs, crypto, and missing-data cases.
        """
        is_syntropic  = getattr(kepe_profile, "is_syntropic_asset",  False)
        is_extractive = getattr(kepe_profile, "is_extractive_asset", False)
        wfs           = float(getattr(kepe_profile, "wfs", 0.50))
        sts           = getattr(kepe_profile, "sts", "STABLE")
        all_notes: list[str] = []

        # ── CIK resolution ──────────────────────────────────────────────────
        cik = _get_cik(symbol)
        if not cik:
            return self._neutral_profile(symbol, "ETF/crypto — no EDGAR CIK")

        # ── Submissions + fund check ─────────────────────────────────────────
        sub = _fetch_submissions(cik)
        if not sub:
            return self._neutral_profile(symbol, "EDGAR submissions unavailable")

        if _is_fund(sub):
            return self._neutral_profile(symbol, "Registered fund — SAS not applicable at fund level")

        sic_code = sub.get("sic")
        try:
            sic_code = int(sic_code) if sic_code else None
        except (ValueError, TypeError):
            sic_code = None

        # ── Fetch 10-K text (shared between Rev + Opacity) ──────────────────
        text_10k = _fetch_10k_text(cik)

        # ── Component 1: Revenue Coherence ──────────────────────────────────
        rev_score, rev_raw = self._rev.compute(
            symbol, is_syntropic, is_extractive,
            text_10k=text_10k, sic_code=sic_code,
        )
        all_notes.extend(rev_raw.get("notes", []))

        # ── Component 2: Capex Direction ────────────────────────────────────
        cap_score, cap_raw = self._cap.compute(symbol, is_syntropic)

        # ── Component 3: Opacity ────────────────────────────────────────────
        opac_score, opac_raw = self._opac.compute(text_10k=text_10k)
        all_notes.extend(opac_raw.get("notes", []))

        # ── Component 4: SSI gap ────────────────────────────────────────────
        frame5      = (rev_score + cap_score + opac_score) / 3.0
        ssi_gap, ssi_notes = SSIStub.compute_ssi_gap(frame1_wfs=wfs, frame5_score=frame5)
        all_notes.extend(ssi_notes)

        # ── Component 5: Wolf score ──────────────────────────────────────────
        ws  = WolfDetector.wolf_score(rev_score, cap_score, opac_score, ssi_gap)
        sas = 1.0 - ws

        wolf_confirmed  = ws > 0.65
        short_candidate = (ws > 0.35) and (sts == "DETERIORATING")

        if wolf_confirmed:
            all_notes.append(f"WOLF CONFIRMED: wolf_score={ws:.2f} > 0.65 — block long")
        elif short_candidate:
            all_notes.append(
                f"SHORT CANDIDATE: wolf_score={ws:.2f} > 0.35 and STS=DETERIORATING"
            )

        # Evidence posture: weakest component governs
        if rev_score < 0.5 or cap_score < 0.5:
            evidence = "ESTABLISHED"
        else:
            evidence = "TESTABLE"

        return SASProfile(
            symbol            = symbol,
            sas_score         = round(sas, 4),
            wolf_score        = round(ws, 4),
            revenue_coherence = round(rev_score, 4),
            capex_direction   = round(cap_score, 4),
            opacity_score     = round(opac_score, 4),
            ssi_gap           = round(ssi_gap, 4),
            wolf_confirmed    = wolf_confirmed,
            short_candidate   = short_candidate,
            evidence_level    = evidence,
            notes             = all_notes,
            timestamp         = datetime.utcnow(),
        )

    @staticmethod
    def _neutral_profile(symbol: str, reason: str) -> SASProfile:
        """
        Return a neutral SAS profile when data is unavailable.
        Uses mid-point scores — no wolf/short signal generated.
        """
        return SASProfile(
            symbol            = symbol,
            sas_score         = 0.50,
            wolf_score        = 0.50,
            revenue_coherence = 0.50,
            capex_direction   = 0.50,
            opacity_score     = 0.50,
            ssi_gap           = 0.00,
            wolf_confirmed    = False,
            short_candidate   = False,
            evidence_level    = "SPECULATIVE",
            notes             = [reason],
            timestamp         = datetime.utcnow(),
        )
