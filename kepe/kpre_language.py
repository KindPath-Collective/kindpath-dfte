"""
KPRE Language Field Layer
==========================
Language-layer signals that precede price by weeks to months.

Three sources:
1. FedLanguageSignal      — Federal Reserve speech NLP (macro, instrument-agnostic)
2. SECRiskDriftSignal     — 10-K risk factor keyword drift (symbol-specific, annual)
3. EarningsLanguageSignal — 8-K earnings release language quality (symbol-specific)

Theory:
  Language is the earliest observable form of field shift. When the Fed begins
  using "cautious" rather than "committed", when risk factor sections gain
  "climate" and "regulatory" keywords, when CEOs shift from "strong demand"
  to "navigating uncertainty" — these are field movements before they manifest
  in price.

  In KindPath terms: the narrative layer (NARRATIVE domain) captures collective
  story field. The language layer reads individual authoritative narratives —
  central bank, corporate disclosure, executive communication.

No ML models. All signals are count-based NLP (word frequency analysis).
Count-based NLP keeps evidence posture at [TESTABLE] and makes all scoring
functions independently testable without network access.

Evidence posture:
  [TESTABLE] — Fed language shifts precede policy; documented in academic
               literature (Gürkaynak et al. 2005, Bernanke et al. 2004).
               Instrument-specific forward return calibration: [TESTABLE].
  [TESTABLE] — 10-K risk factor drift as leading indicator. Keyword changes
               precede material events directionally; requires outcome validation.
  [TESTABLE] — Earnings release hedging density as guidance quality proxy.
               Known qualitative signal; lacks systematic calibration.

Domain: "LANGUAGE" (weight 0.15 in DOMAIN_WEIGHTS, see syntropy_engine.py).
"""

from __future__ import annotations
import re
import time
import logging
import requests
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from kepe.indicators import WorldSignal
from kepe.kpre_capital import _get_cik, _fetch_submissions, _EDGAR_UA, _EDGAR_DELAY

logger = logging.getLogger(__name__)

# ─── Word lists ───────────────────────────────────────────────────────────────

_FED_UNCERTAINTY = frozenset({
    "uncertain", "uncertainty", "unclear", "depends", "monitor", "monitoring",
    "cautious", "caution", "vigilant", "watch", "watching", "evaluate",
    "assess", "assessing", "variable", "unknown", "unpredictable",
})

_FED_COMMITMENT = frozenset({
    "will", "committed", "commit", "expect", "expected", "target", "objective",
    "goal", "intend", "mandate", "resolve", "decisive", "determined", "ensure",
})

_FED_TIGHTENING = frozenset({
    "inflation", "inflationary", "restrictive", "elevated", "hike", "hiking",
    "tighten", "tightening", "overheating", "above-target", "price stability",
    "above target", "supply constraints", "labour market tightness",
})

_FED_EASING = frozenset({
    "soften", "softening", "cooling", "normalize", "normalise", "cut", "cutting",
    "ease", "easing", "accommodative", "below-target", "below target",
    "disinflation", "slowdown", "slowdowns", "recessionary", "weakness",
})

_EARNINGS_CONFIDENT = frozenset({
    "strong", "robust", "exceeded", "record", "growth", "accelerating", "outperform",
    "delivering", "momentum", "expanding", "raised", "guidance", "increased", "ahead",
    "positive", "beat", "upbeat", "confidence", "confident", "healthy",
})

_EARNINGS_HEDGING = frozenset({
    "may", "might", "could", "potentially", "challenging", "uncertain",
    "difficult", "headwinds", "unforeseen", "subject to", "risk of", "concerns",
    "below expectations", "disappointed", "softer", "cautious",
})

_EARNINGS_INVEST = frozenset({
    "investing", "capex", "capital expenditure", "research", "innovation",
    "build", "expand", "hire", "capacity", "acquisition", "infrastructure",
})

_EARNINGS_RETURN = frozenset({
    "buyback", "dividend", "repurchase", "returning capital", "return to shareholders",
    "share repurchase", "yield",
})

# Risk factor keywords with sentiment polarity
# +ve = country-layer field awakening (ESG/regulatory awareness = field emerging)
# −ve = interference loading (operational risk increasing)
_RISK_POLARITY: Dict[str, float] = {
    "climate":        +1.0,
    "environmental":  +0.8,
    "sustainability": +0.8,
    "carbon":         +0.7,
    "emissions":      +0.7,
    "esg":            +1.0,
    "regulatory":     +0.5,
    "regulation":     +0.5,
    "compliance":     +0.3,
    "litigation":     -0.8,
    "lawsuit":        -0.8,
    "damages":        -0.6,
    "cybersecurity":  -0.7,
    "breach":         -0.6,
    "ransomware":     -0.9,
    "supply chain":   -0.4,
    "geopolitical":   -0.4,
    "tariff":         -0.3,
    "sanctions":      -0.5,
}

# ─── Fed Language Signal ──────────────────────────────────────────────────────

_FED_CACHE: Dict[str, Any] = {}   # {"signal": WorldSignal, "ts": float}
_FED_CACHE_TTL = 6 * 3600         # 6 hours — speeches don't change intraday


class FedLanguageSignal:
    """
    Federal Reserve speech language signal.

    Fetches last 5 Fed speeches from the RSS feed, scores for:
      - Uncertainty density  (uncertain, monitor, cautious, …)
      - Policy direction     (easing vs tightening language)
      - Commitment strength  (committed, will, expect, …)

    High certainty + easing = positive sovereign signal.
    High certainty + tightening = negative sovereign signal.
    High uncertainty always dampens the signal regardless of direction.

    This signal is INSTRUMENT-AGNOSTIC — same value applied to all symbols.
    Cached for 6 hours.

    Source: https://www.federalreserve.gov/feeds/speeches.xml (free, no auth)

    [TESTABLE] — directional validity documented; calibration against
                 instrument-specific forward returns: [TESTABLE].
    Temporal layer: SURFACE — language changes weekly with each speech.
    """

    _RSS_URL = "https://www.federalreserve.gov/feeds/speeches.xml"
    _MAX_SPEECHES = 5

    def compute(self) -> WorldSignal:
        global _FED_CACHE
        cached = _FED_CACHE.get("signal")
        if cached and (time.time() - _FED_CACHE.get("ts", 0)) < _FED_CACHE_TTL:
            return cached

        texts = self._fetch_speeches()
        if not texts:
            return WorldSignal(
                domain="LANGUAGE", source="fed_speeches",
                region="US", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
                notes="FedLanguageSignal: no speeches fetched"
            )

        combined = " ".join(texts)
        scores = self._score_speech_text(combined)

        sig = WorldSignal(
            domain="LANGUAGE", source="fed_speeches",
            region="US",
            value=float(np.clip(scores["signal"], -1, 1)),
            confidence=0.55,
            evidence_level="TESTABLE",
            timestamp=datetime.utcnow(),
            temporal_layer="SURFACE",
            raw={
                "n_speeches":        len(texts),
                "uncertainty_pct":   round(scores["uncertainty_density"] * 100, 2),
                "tightening_pct":    round(scores["tightening_density"] * 100, 2),
                "easing_pct":        round(scores["easing_density"] * 100, 2),
                "policy_direction":  round(scores["policy_direction"], 3),
            },
            notes=(
                f"Fed speeches NLP ({len(texts)} speeches): "
                f"policy={scores['policy_direction']:+.3f}, "
                f"uncertainty={scores['uncertainty_density']:.3f} [TESTABLE]"
            )
        )

        _FED_CACHE["signal"] = sig
        _FED_CACHE["ts"] = time.time()
        return sig

    def _fetch_speeches(self) -> List[str]:
        """Fetch RSS and return list of description texts (network)."""
        try:
            time.sleep(_EDGAR_DELAY)
            resp = requests.get(self._RSS_URL, headers=_EDGAR_UA, timeout=12)
            if resp.status_code != 200:
                return []
            root = ET.fromstring(resp.text)
            texts = []
            for item in root.findall(".//item")[:self._MAX_SPEECHES]:
                parts = []
                for tag in ("title", "description"):
                    el = item.find(tag)
                    if el is not None and el.text:
                        parts.append(el.text)
                if parts:
                    texts.append(" ".join(parts))
            return texts
        except Exception as e:
            logger.debug(f"FedLanguageSignal fetch failed: {e}")
            return []

    @staticmethod
    def _score_speech_text(text: str) -> Dict[str, float]:
        """
        Score Fed speech text for policy direction and certainty.
        Pure function — testable with synthetic text.

        Returns dict with:
          uncertainty_density: float  [0, 1]
          tightening_density:  float  [0, 1]
          easing_density:      float  [0, 1]
          policy_direction:    float  [-1, +1]  easing=+1, tightening=-1
          certainty:           float  [-1, +1]  committed=+1, uncertain=-1
          signal:              float  [-1, +1]  final signal
        """
        words = re.findall(r'\b\w+\b', text.lower())
        n = max(len(words), 1)

        unc_count  = sum(1 for w in words if w in _FED_UNCERTAINTY)
        comm_count = sum(1 for w in words if w in _FED_COMMITMENT)
        tight_count = sum(1 for w in words if w in _FED_TIGHTENING)
        ease_count  = sum(1 for w in words if w in _FED_EASING)

        unc_density   = unc_count  / n
        comm_density  = comm_count / n
        tight_density = tight_count / n
        ease_density  = ease_count  / n

        # Policy direction: positive = easing, negative = tightening
        # Scale: 0.01 density difference → ~0.5 signal (tanh)
        policy = float(np.tanh((ease_density - tight_density) * 50))

        # Certainty: positive = committed, negative = uncertain
        certainty = float(np.tanh((comm_density - unc_density) * 50))

        # Uncertainty dampens signal — high uncertainty = Fed itself unsure
        uncertainty_damper = max(0.25, 1.0 - unc_density * 30)

        # Final signal: certain Fed easing = positive, certain tightening = negative
        signal = float(np.clip(policy * uncertainty_damper, -1, 1))

        return {
            "uncertainty_density": unc_density,
            "tightening_density":  tight_density,
            "easing_density":      ease_density,
            "policy_direction":    policy,
            "certainty":           certainty,
            "signal":              signal,
        }


# ─── SEC Risk Drift Signal ────────────────────────────────────────────────────

class SECRiskDriftSignal:
    """
    10-K risk factor keyword drift signal.

    Fetches the last two annual 10-K filings for a symbol and compares
    keyword frequency in the risk factors section.

    Rising ESG/climate/regulatory language = Country-layer field awakening
    in the company's own disclosure = positive LANGUAGE signal.

    Rising litigation/cyber/geopolitical language = interference load
    increasing in the company's operating environment = negative.

    Source: SEC EDGAR submissions API + filing documents.
    Fetches max 200KB of each 10-K primary document (stream=True).

    [TESTABLE] — risk factor drift as leading indicator. Language changes
                 precede material events directionally; calibration required.
    Temporal layer: STRUCTURAL — 10-K annual cycle, slow-moving.
    """

    _RISK_KEYWORDS = list(_RISK_POLARITY.keys())

    def compute(self, symbol: str) -> WorldSignal:
        cik = _get_cik(symbol)
        if cik is None:
            return self._no_data(symbol, "CIK not found")

        subs = _fetch_submissions(cik)
        if subs is None:
            return self._no_data(symbol, "Submissions fetch failed")

        ten_k_filings = self._extract_10k_filings(subs, n=2)
        if len(ten_k_filings) < 2:
            return self._no_data(symbol, f"Need 2 10-K filings, found {len(ten_k_filings)}")

        old_filing, new_filing = ten_k_filings[0], ten_k_filings[1]

        old_text = self._fetch_risk_text(cik, old_filing["accno"], old_filing["doc"])
        new_text = self._fetch_risk_text(cik, new_filing["accno"], new_filing["doc"])

        if not old_text or not new_text:
            return self._no_data(symbol, "Could not fetch filing text")

        old_counts = self._count_keywords(old_text)
        new_counts = self._count_keywords(new_text)

        value, raw = self._score_risk_drift(old_counts, new_counts,
                                            len(old_text), len(new_text))
        raw.update({
            "old_filing_date": old_filing["date"],
            "new_filing_date": new_filing["date"],
            "old_text_chars":  len(old_text),
            "new_text_chars":  len(new_text),
        })

        return WorldSignal(
            domain="LANGUAGE", source=f"sec_risk_drift_{symbol.lower()}",
            region="US",
            value=float(np.clip(value, -1, 1)),
            confidence=0.50,
            evidence_level="TESTABLE",
            timestamp=datetime.utcnow(),
            temporal_layer="STRUCTURAL",
            raw=raw,
            notes=(
                f"10-K risk drift [{symbol}]: {old_filing['date']} → {new_filing['date']} "
                f"signal={value:+.3f} [TESTABLE]"
            )
        )

    def _extract_10k_filings(self, subs: dict, n: int = 2) -> List[Dict]:
        """Extract last n 10-K filings from submissions JSON."""
        filings = subs.get("filings", {}).get("recent", {})
        forms   = filings.get("form", [])
        dates   = filings.get("filingDate", [])
        accnos  = filings.get("accessionNumber", [])
        pdocs   = filings.get("primaryDocument", [])

        results = []
        for form, date, accno, pdoc in zip(forms, dates, accnos, pdocs):
            if form in ("10-K", "10-K405", "10-KSB"):
                results.append({"date": date, "accno": accno, "doc": pdoc})
            if len(results) >= n:
                break

        return list(reversed(results))   # chronological order (oldest first)

    def _fetch_risk_text(self, cik: str, accno: str, primary_doc: str,
                         max_bytes: int = 200_000) -> str:
        """
        Stream the first max_bytes of a 10-K primary document (network).
        Extract and return the risk factors section text.
        """
        acc_nodash = accno.replace("-", "")
        cik_int    = int(cik)
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"

        time.sleep(_EDGAR_DELAY)
        try:
            resp = requests.get(url, headers=_EDGAR_UA, stream=True, timeout=15)
            if resp.status_code != 200:
                return ""
            content = b""
            for chunk in resp.iter_content(chunk_size=8192):
                content += chunk
                if len(content) >= max_bytes:
                    break
            resp.close()
            raw_text = content.decode("utf-8", errors="ignore")
            return self._extract_risk_section(raw_text)
        except Exception as e:
            logger.debug(f"Risk text fetch failed ({url}): {e}")
            return ""

    @staticmethod
    def _extract_risk_section(html_text: str, max_chars: int = 40_000) -> str:
        """
        Extract risk factors section from 10-K HTML text.
        Pure function — testable with synthetic HTML.

        Looks for 'Item 1A' or 'Risk Factors' marker.
        Returns up to max_chars of text from that point.
        """
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_text)
        text = re.sub(r'&#\d+;|&\w+;', ' ', text)   # HTML entities
        text = re.sub(r'\s+', ' ', text)

        # Find risk factors section
        pattern = re.compile(
            r'item\s+1\s*[a.,]?\s*\.?\s*risk\s+factor', re.IGNORECASE
        )
        match = pattern.search(text)

        if not match:
            # Broader fallback
            match = re.search(r'risk\s+factors?', text, re.IGNORECASE)

        if match:
            start = match.start()
            return text[start: start + max_chars]

        return text[:10_000]   # no section found — use beginning

    @staticmethod
    def _count_keywords(text: str) -> Dict[str, int]:
        """
        Count risk keyword occurrences in text.
        Pure function — testable with synthetic text.
        """
        text_lower = text.lower()
        return {kw: text_lower.count(kw) for kw in _RISK_POLARITY}

    @staticmethod
    def _score_risk_drift(
        old_counts: Dict[str, int],
        new_counts: Dict[str, int],
        old_total: int = 1000,
        new_total: int = 1000,
    ) -> Tuple[float, dict]:
        """
        Score keyword density change between two 10-K risk sections.
        Pure function — testable with synthetic count dicts.

        Positive polarity keywords (climate, ESG): new > old = positive
        Negative polarity keywords (litigation, cyber): new > old = negative

        Returns (signal_value in [-1, 1], raw_metadata).
        """
        old_n = max(old_total, 1)
        new_n = max(new_total, 1)

        drift_score = 0.0
        details: Dict[str, dict] = {}

        for keyword, polarity in _RISK_POLARITY.items():
            old_density = old_counts.get(keyword, 0) / old_n
            new_density = new_counts.get(keyword, 0) / new_n
            delta       = new_density - old_density
            contrib     = delta * polarity

            drift_score += contrib

            if old_counts.get(keyword, 0) != new_counts.get(keyword, 0):
                details[keyword] = {
                    "old":   old_counts.get(keyword, 0),
                    "new":   new_counts.get(keyword, 0),
                    "polarity": polarity,
                }

        # Scale: ±0.01 density change × polarity → ±0.5 signal
        value = float(np.tanh(drift_score * 20))

        raw = {
            "drift_score":   round(drift_score, 5),
            "n_drifted":     len(details),
            "drift_details": {k: v for k, v in list(details.items())[:8]},
        }

        return value, raw

    def _no_data(self, symbol: str, reason: str) -> WorldSignal:
        return WorldSignal(
            domain="LANGUAGE", source=f"sec_risk_drift_{symbol.lower()}",
            region="US", value=0.0, confidence=0.0,
            evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            temporal_layer="STRUCTURAL",
            notes=f"SECRiskDrift [{symbol}]: no data — {reason}"
        )


# ─── Earnings Language Signal ─────────────────────────────────────────────────

class EarningsLanguageSignal:
    """
    Management language quality from 8-K earnings releases.

    Fetches the most recent 8-K filing with Item 2.02 (Results of Operations)
    from EDGAR and scores the press release text for:
      - Confidence markers (strong, robust, exceeded, record, …)
      - Hedging density (may, might, could, potentially, challenging, …)
      - Forward investment language (capex, investing, expand, innovate, …)
      - Return-capital language (buyback, dividend, repurchase, …)

    Confident, forward-investing language = healthy narrative field.
    Hedging-dominant language = uncertainty entering the narrative.

    [TESTABLE] — earnings language as guidance quality proxy.
                 Known qualitative signal; systematic calibration required.
    Temporal layer: MEDIUM — quarterly earnings cycle.
    """

    def compute(self, symbol: str) -> WorldSignal:
        cik = _get_cik(symbol)
        if cik is None:
            return self._no_data(symbol, "CIK not found")

        text = self._fetch_earnings_text(cik)
        if not text:
            return self._no_data(symbol, "No 8-K earnings release in last 90 days")

        scores = self._score_earnings_text(text)

        return WorldSignal(
            domain="LANGUAGE", source=f"earnings_language_{symbol.lower()}",
            region="US",
            value=float(np.clip(scores["signal"], -1, 1)),
            confidence=0.45,
            evidence_level="TESTABLE",
            timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
            raw={
                "confidence_density": round(scores["confidence_density"] * 100, 2),
                "hedging_density":    round(scores["hedging_density"] * 100, 2),
                "invest_signal":      scores["invest_signal"],
                "signal":             round(scores["signal"], 3),
            },
            notes=(
                f"Earnings language [{symbol}]: "
                f"conf={scores['confidence_density']:.3f}, "
                f"hedge={scores['hedging_density']:.3f} [TESTABLE]"
            )
        )

    def _fetch_earnings_text(self, cik: str, max_days: int = 90) -> str:
        """
        Find most recent 8-K with Item 2.02, fetch and return press release text.
        """
        subs = _fetch_submissions(cik)
        if subs is None:
            return ""

        filings = subs.get("filings", {}).get("recent", {})
        forms   = filings.get("form", [])
        dates   = filings.get("filingDate", [])
        accnos  = filings.get("accessionNumber", [])
        pdocs   = filings.get("primaryDocument", [])
        items   = filings.get("items", [""] * len(forms))

        cutoff = (datetime.utcnow() - timedelta(days=max_days)).strftime("%Y-%m-%d")

        for form, date, accno, pdoc, item_str in zip(forms, dates, accnos, pdocs, items):
            if form != "8-K":
                continue
            if date < cutoff:
                break
            # Item 2.02 = Results of Operations and Financial Condition
            if "2.02" not in (item_str or ""):
                continue

            acc_nodash = accno.replace("-", "")
            cik_int    = int(cik)
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{pdoc}"

            time.sleep(_EDGAR_DELAY)
            try:
                resp = requests.get(url, headers=_EDGAR_UA, stream=True, timeout=12)
                if resp.status_code != 200:
                    continue
                content = b""
                for chunk in resp.iter_content(chunk_size=8192):
                    content += chunk
                    if len(content) >= 150_000:
                        break
                resp.close()
                html = content.decode("utf-8", errors="ignore")
                # Strip tags
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\s+', ' ', text)
                if len(text) > 200:
                    return text
            except Exception as e:
                logger.debug(f"EarningsLanguage fetch failed: {e}")
                continue

        return ""

    @staticmethod
    def _score_earnings_text(text: str) -> Dict[str, float]:
        """
        Score earnings release text for management language quality.
        Pure function — testable with synthetic text.

        Returns dict with:
          confidence_density: float [0, 1]
          hedging_density:    float [0, 1]
          invest_signal:      float [-1, +1]  investing vs returning capital
          signal:             float [-1, +1]  final signal
        """
        words = re.findall(r'\b\w+\b', text.lower())
        n = max(len(words), 1)

        conf_count   = sum(1 for w in words if w in _EARNINGS_CONFIDENT)
        hedge_count  = sum(1 for w in words if w in _EARNINGS_HEDGING)
        invest_count = sum(1 for w in words if w in _EARNINGS_INVEST)
        return_count = sum(1 for w in words if w in _EARNINGS_RETURN)

        conf_density  = conf_count  / n
        hedge_density = hedge_count / n

        # Confidence − hedging
        net_conf = float(np.tanh((conf_density - hedge_density) * 50))

        # Investment signal: forward-investing vs returning-capital
        inv_sig = float(np.tanh((invest_count - return_count) * 5))

        # Combine: language quality (70%) + investment signal (30%)
        signal = float(np.clip(net_conf * 0.70 + inv_sig * 0.30, -1, 1))

        return {
            "confidence_density": conf_density,
            "hedging_density":    hedge_density,
            "invest_signal":      round(inv_sig, 3),
            "signal":             signal,
        }

    def _no_data(self, symbol: str, reason: str) -> WorldSignal:
        return WorldSignal(
            domain="LANGUAGE", source=f"earnings_language_{symbol.lower()}",
            region="US", value=0.0, confidence=0.0,
            evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
            notes=f"EarningsLanguage [{symbol}]: no data — {reason}"
        )


# ─── KPRE Language Layer ──────────────────────────────────────────────────────

class KPRELanguageLayer:
    """
    KPRE Language Field Layer composite.

    Aggregates:
      - FedLanguageSignal (instrument-agnostic, cached 6h)
      - SECRiskDriftSignal (symbol-specific, annual 10-K drift)
      - EarningsLanguageSignal (symbol-specific, quarterly 8-K)

    Domain: "LANGUAGE" (weight 0.15 in DOMAIN_WEIGHTS).
    Confidence ceiling: 0.60 (all sub-signals are [TESTABLE]).

    Usage:
        sig = KPRELanguageLayer().compute("AAPL")    # full pipeline
        sig = KPRELanguageLayer._aggregate([...])    # testable aggregation
    """

    def compute(self, symbol: str) -> WorldSignal:
        """Run all language sub-signals for symbol."""
        sub_signals: List[WorldSignal] = []

        for cls, kwargs in [
            (FedLanguageSignal,      {}),
            (SECRiskDriftSignal,     {"symbol": symbol}),
            (EarningsLanguageSignal, {"symbol": symbol}),
        ]:
            try:
                instance = cls()
                sig = instance.compute(**kwargs) if kwargs else instance.compute()
                if sig.confidence > 0:
                    sub_signals.append(sig)
            except Exception as e:
                logger.warning(f"KPRELanguage {cls.__name__} failed for {symbol}: {e}")

        result = self._aggregate(sub_signals)
        logger.debug(
            f"KPRELanguage [{symbol}]: {result.value:+.3f} "
            f"(conf={result.confidence:.2f}, {len(sub_signals)}/3 sub-signals)"
        )
        return result

    @staticmethod
    def _aggregate(sub_signals: List[WorldSignal]) -> WorldSignal:
        """
        Confidence-weighted aggregation of language sub-signals.
        Testable independently of network.

        Confidence ceiling: 0.60 — language signals are [TESTABLE],
        a lower ceiling than capital formation signals (0.65).
        """
        if not sub_signals:
            return WorldSignal(
                domain="LANGUAGE", source="language_field_composite",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                notes="No KPRE Language sub-signals available"
            )

        total_w = sum(s.confidence for s in sub_signals)
        if total_w < 1e-10:
            composite_value = 0.0
            composite_conf  = 0.0
        else:
            composite_value = sum(s.value * s.confidence for s in sub_signals) / total_w
            completeness    = len(sub_signals) / 3.0
            avg_conf        = total_w / len(sub_signals)
            composite_conf  = min(0.60, avg_conf * completeness)

        # Evidence: worst of sub-signals
        ev_order = {"ESTABLISHED": 0, "TESTABLE": 1, "SPECULATIVE": 2}
        worst_ev = max(sub_signals, key=lambda s: ev_order.get(s.evidence_level, 1))

        return WorldSignal(
            domain="LANGUAGE", source="language_field_composite",
            region="GLOBAL",
            value=float(np.clip(composite_value, -1, 1)),
            confidence=float(np.clip(composite_conf, 0, 0.60)),
            evidence_level=worst_ev.evidence_level,
            timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
            raw={
                "n_signals":    len(sub_signals),
                "n_attempted":  3,
                "completeness": round(len(sub_signals) / 3.0, 2),
                "sources":      [s.source for s in sub_signals],
                "sub_values":   {s.source: round(s.value, 3) for s in sub_signals},
            },
            notes=(
                f"Language composite: {len(sub_signals)}/3 signals "
                f"(value={composite_value:.3f}, conf={composite_conf:.2f}) "
                f"[{worst_ev.evidence_level}]"
            )
        )
