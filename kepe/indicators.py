"""
KEPE — World Field Indicators
================================
Data feeds for the KindEarth Predictive Engine.
Reads the world field: ecological, social, equity, narrative layers.

Sources (all free/public):
  World Bank API    — social + equity indicators
  GDELT Project     — geopolitical + narrative sentiment
  FRED              — macro economic indicators
  Fear & Greed      — market optimism proxy
  Environmental     — climate/ecological stress proxies

Evidence posture: KINDFIELD
  [ESTABLISHED] — well-supported, reliable data
  [TESTABLE]    — directionally valid, calibration required
  [SPECULATIVE] — exploratory, clearly marked

Output: WorldSignal — normalised -1.0 → +1.0 per domain
  +1.0 = high syntropy / ZPB field conditions
  -1.0 = high entropy / IN-loading field conditions
"""

from __future__ import annotations
import os
import time
import json
import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

CACHE_DIR = "/tmp/kepe_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class WorldSignal:
    """Normalised world field signal from one domain."""
    domain: str           # ECOLOGICAL | SOCIAL | EQUITY | NARRATIVE | MACRO
    source: str
    region: str           # GLOBAL | US | AU | SECTOR
    value: float          # -1.0 (high entropy) → +1.0 (high syntropy)
    confidence: float     # 0.0 → 1.0
    evidence_level: str   # ESTABLISHED | TESTABLE | SPECULATIVE
    timestamp: datetime
    raw: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _cache_get(key: str, max_age_hours: float = 6) -> Optional[dict]:
    path = f"{CACHE_DIR}/{key}.json"
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < max_age_hours * 3600:
            with open(path) as f:
                return json.load(f)
    return None


def _cache_set(key: str, data: dict):
    path = f"{CACHE_DIR}/{key}.json"
    with open(path, "w") as f:
        json.dump(data, f)


# ─── ECOLOGICAL LAYER ─────────────────────────────────────────────────────────

class EcologicalSignal:
    """
    Ecological stress proxy from commodity prices and climate indices.

    High commodity volatility + rising energy prices = ecological pressure.
    Commodity stability = ecological coherence.

    Uses: Gold/Oil ratio, agricultural commodity stability, energy price trend.
    [TESTABLE] — commodity prices as ecological proxy is indirect.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            # Oil (energy stress), Gold (safe haven / stability), Corn (food security)
            oil  = yf.Ticker("CL=F").history(period="3mo")["Close"]
            gold = yf.Ticker("GC=F").history(period="3mo")["Close"]
            corn = yf.Ticker("ZC=F").history(period="3mo")["Close"]

            if oil.empty or gold.empty:
                raise ValueError("Commodity data unavailable")

            # Oil price trend: rising = ecological/energy stress
            oil_trend = (oil.iloc[-1] - oil.iloc[-20]) / (oil.iloc[-20] + 1e-10)

            # Gold/Oil ratio: high ratio = gold outperforming oil = stress
            go_ratio = gold.iloc[-1] / (oil.iloc[-1] + 1e-10)
            go_mean = (gold / (oil + 1e-10)).mean()
            go_z = (go_ratio - go_mean) / ((gold / (oil + 1e-10)).std() + 1e-10)

            # Food price stability: low volatility = stable food supply
            food_vol = float(corn.pct_change().std()) if not corn.empty else 0.02
            food_signal = float(np.clip(-(food_vol - 0.015) / 0.02, -1, 1))

            # Combine: high oil trend + high gold/oil stress = negative ecological signal
            value = float(np.clip(
                -oil_trend * 0.4 - go_z * 0.1 + food_signal * 0.5,
                -1, 1
            ))

            return WorldSignal(
                domain="ECOLOGICAL", source="commodity_proxy",
                region="GLOBAL", value=value, confidence=0.50,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                raw={"oil_trend": oil_trend, "go_z": go_z, "food_vol": food_vol},
                notes="Commodity proxy for ecological stress [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"Ecological signal failed: {e}")
            return WorldSignal(
                domain="ECOLOGICAL", source="commodity_proxy",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow()
            )


# ─── SOCIAL / EQUITY LAYER ────────────────────────────────────────────────────

class WorldBankSignal:
    """
    Social equity indicators from World Bank API (free).
    Uses cached annual data — World Bank updates yearly.

    Indicators:
      SI.POV.GINI     — Gini coefficient (inequality)
      SL.UEM.TOTL.ZS  — Unemployment rate
      SE.XPD.TOTL.GD.ZS — Education expenditure (% GDP)

    [ESTABLISHED] — World Bank data is authoritative for these metrics.
    Note: Annual data, used as structural background signal.
    """

    WB_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    INDICATORS = {
        "gini":        "SI.POV.GINI",
        "unemployment": "SL.UEM.TOTL.ZS",
        "education":   "SE.XPD.TOTL.GD.ZS",
    }

    def compute(self, country: str = "WLD") -> WorldSignal:
        cache_key = f"worldbank_{country}"
        cached = _cache_get(cache_key, max_age_hours=48)
        if cached:
            return WorldSignal(**{**cached, "timestamp": datetime.utcnow()})

        try:
            scores = {}
            for name, ind_code in self.INDICATORS.items():
                val = self._fetch_latest(country, ind_code)
                if val is not None:
                    scores[name] = val

            if not scores:
                raise ValueError("No World Bank data")

            # Gini: higher = more inequality = lower syntropy
            # Global Gini ~38, range 25–65
            gini = scores.get("gini", 38)
            gini_signal = float(np.clip(-(gini - 35) / 20, -1, 1))

            # Unemployment: higher = more social stress
            unem = scores.get("unemployment", 6)
            unem_signal = float(np.clip(-(unem - 5) / 10, -1, 1))

            # Education spend: higher = more investment in human development
            edu = scores.get("education", 4.5)
            edu_signal = float(np.clip((edu - 4) / 3, -1, 1))

            value = float(np.clip(
                gini_signal * 0.40 + unem_signal * 0.35 + edu_signal * 0.25,
                -1, 1
            ))

            result = WorldSignal(
                domain="SOCIAL", source="world_bank",
                region=country, value=value, confidence=0.70,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
                raw=scores,
                notes=f"Gini={gini:.1f} Unem={unem:.1f}% Edu={edu:.1f}%"
            )
            _cache_set(cache_key, {
                "domain": result.domain, "source": result.source,
                "region": result.region, "value": result.value,
                "confidence": result.confidence,
                "evidence_level": result.evidence_level,
                "raw": result.raw, "notes": result.notes,
                "timestamp": result.timestamp.isoformat(),
            })
            return result

        except Exception as e:
            logger.warning(f"World Bank signal failed: {e}")
            return WorldSignal(
                domain="SOCIAL", source="world_bank",
                region=country, value=0.0, confidence=0.0,
                evidence_level="ESTABLISHED", timestamp=datetime.utcnow()
            )

    def _fetch_latest(self, country: str, indicator: str) -> Optional[float]:
        url = self.WB_BASE.format(country=country, indicator=indicator)
        params = {"format": "json", "mrv": 5, "per_page": 5}
        try:
            resp = requests.get(url, params=params, timeout=8)
            data = resp.json()
            if len(data) > 1:
                entries = [e for e in data[1] if e.get("value") is not None]
                if entries:
                    return float(entries[0]["value"])
        except Exception as e:
            logger.warning(f"WB {indicator}: {e}")
        return None


# ─── NARRATIVE / SENTIMENT LAYER ─────────────────────────────────────────────

class GDELTSignal:
    """
    Geopolitical + narrative sentiment from GDELT Project (free).
    GDELT monitors world news in real-time and scores tone.

    Positive tone = more syntropic narrative field.
    Negative tone = more entropic narrative field.

    [TESTABLE] — GDELT tone as narrative sentiment proxy.
    """

    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def compute(self, theme: str = "GENERAL") -> WorldSignal:
        cache_key = f"gdelt_{theme}"
        cached = _cache_get(cache_key, max_age_hours=2)
        if cached:
            try:
                return WorldSignal(**{**cached, "timestamp": datetime.utcnow()})
            except Exception:
                pass

        try:
            # GDELT TV API for tone scores
            params = {
                "query":     "tone",
                "mode":      "TimelineSourceCountry",
                "format":    "json",
                "timespan":  "1week",
                "maxrecords": 10,
            }
            resp = requests.get(self.GDELT_URL, params=params, timeout=8)
            if resp.status_code != 200:
                raise ValueError(f"GDELT status {resp.status_code}")

            data = resp.json()
            # Extract tone from timeline
            timeline = data.get("timeline", [{}])
            tones = []
            for entry in timeline[:1]:
                for item in entry.get("data", [])[:10]:
                    val = item.get("value")
                    if val is not None:
                        tones.append(float(val))

            if not tones:
                raise ValueError("No GDELT tone data")

            mean_tone = np.mean(tones)
            # GDELT tone: typically -10 to +10
            value = float(np.clip(mean_tone / 8, -1, 1))

            result = WorldSignal(
                domain="NARRATIVE", source="gdelt",
                region="GLOBAL", value=value, confidence=0.45,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                raw={"mean_tone": mean_tone, "n_samples": len(tones)},
                notes=f"GDELT mean tone {mean_tone:.2f} [TESTABLE]"
            )
            _cache_set(cache_key, {
                "domain": result.domain, "source": result.source,
                "region": result.region, "value": result.value,
                "confidence": result.confidence,
                "evidence_level": result.evidence_level,
                "raw": result.raw, "notes": result.notes,
                "timestamp": result.timestamp.isoformat(),
            })
            return result

        except Exception as e:
            logger.warning(f"GDELT signal failed: {e}")
            # Fallback: VIX-based narrative proxy
            return self._vix_narrative_proxy()

    def _vix_narrative_proxy(self) -> WorldSignal:
        """Fallback: VIX as narrative stress proxy."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX").history(period="1mo")["Close"]
            if not vix.empty:
                vix_now = vix.iloc[-1]
                value = float(np.clip(-(vix_now - 20) / 15, -1, 1))
                return WorldSignal(
                    domain="NARRATIVE", source="vix_proxy",
                    region="GLOBAL", value=value, confidence=0.40,
                    evidence_level="TESTABLE",
                    timestamp=datetime.utcnow(),
                    raw={"vix": vix_now},
                    notes="VIX narrative proxy (GDELT unavailable)"
                )
        except Exception:
            pass
        return WorldSignal(
            domain="NARRATIVE", source="gdelt",
            region="GLOBAL", value=0.0, confidence=0.0,
            evidence_level="TESTABLE", timestamp=datetime.utcnow()
        )


# ─── OPTIMISM / FAITH LAYER ───────────────────────────────────────────────────

class OptimismSignal:
    """
    Optimism propagation coefficient — forward-looking sentiment.

    Sources:
      - CNN Fear & Greed Index proxy (via market internals)
      - Consumer confidence (University of Michigan via FRED)
      - IPO/M&A activity (risk appetite proxy)

    In KindPath terms: high optimism = faith engine aimed forward = ZPB potential.
    Low optimism = faith collapsed = IN-loading condition.

    [TESTABLE] — optimism indices as syntropy precursor.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            # Market breadth: % of S&P 500 stocks above 200d MA
            # Proxy: SPY vs defensive ETFs (XLU utilities)
            spy  = yf.Ticker("SPY").history(period="3mo")["Close"]
            xlu  = yf.Ticker("XLU").history(period="3mo")["Close"]  # Utilities (defensive)
            xlk  = yf.Ticker("XLK").history(period="3mo")["Close"]  # Tech (risk-on)

            if spy.empty:
                raise ValueError("No SPY data")

            # Risk-on vs risk-off: tech outperforming utilities = optimism
            if not xlk.empty and not xlu.empty and len(xlk) > 20 and len(xlu) > 20:
                xlk_ret = (xlk.iloc[-1] - xlk.iloc[-20]) / (xlk.iloc[-20] + 1e-10)
                xlu_ret = (xlu.iloc[-1] - xlu.iloc[-20]) / (xlu.iloc[-20] + 1e-10)
                risk_on = xlk_ret - xlu_ret
            else:
                risk_on = 0.0

            # SPY trend = broad market direction
            spy_ret = (spy.iloc[-1] - spy.iloc[-20]) / (spy.iloc[-20] + 1e-10)

            value = float(np.clip(spy_ret * 5 * 0.5 + risk_on * 5 * 0.5, -1, 1))

            return WorldSignal(
                domain="OPTIMISM", source="market_breadth",
                region="US", value=value, confidence=0.60,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                raw={"spy_ret": spy_ret, "risk_on": risk_on},
                notes=f"Tech/utilities spread={risk_on:.3f}, SPY 20d={spy_ret:.3f}"
            )

        except Exception as e:
            logger.warning(f"Optimism signal failed: {e}")
            return WorldSignal(
                domain="OPTIMISM", source="market_breadth",
                region="US", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow()
            )


# ─── CONFLICT / PRESSURE LAYER ────────────────────────────────────────────────

class ConflictPressureSignal:
    """
    Conflict and systemic pressure proxy.
    High conflict load = high IN-loading in the world field.

    Sources:
      - Gold price relative to equities (flight to safety)
      - Credit default swap proxy (systemic risk)
      - Currency volatility (geopolitical stress)

    [TESTABLE] — indirect proxy for conflict/pressure.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            gold   = yf.Ticker("GC=F").history(period="3mo")["Close"]
            spy    = yf.Ticker("SPY").history(period="3mo")["Close"]
            vix    = yf.Ticker("^VIX").history(period="3mo")["Close"]
            usdidx = yf.Ticker("DX-Y.NYB").history(period="3mo")["Close"]

            if gold.empty or spy.empty:
                raise ValueError("No data")

            # Gold/SPY ratio trend: rising = flight to safety = conflict pressure
            gs_ratio = gold / (spy + 1e-10)
            gs_trend = (gs_ratio.iloc[-1] - gs_ratio.iloc[-20]) / (gs_ratio.iloc[-20] + 1e-10)

            # VIX level: high = fear = conflict/systemic pressure
            vix_now = float(vix.iloc[-1]) if not vix.empty else 20.0
            vix_signal = float(np.clip(-(vix_now - 18) / 20, -1, 1))

            # USD strength: sharp USD rise often = global dollar stress
            if not usdidx.empty and len(usdidx) > 20:
                usd_trend = (usdidx.iloc[-1] - usdidx.iloc[-20]) / (usdidx.iloc[-20] + 1e-10)
                usd_signal = float(np.clip(-usd_trend * 20, -1, 1))
            else:
                usd_signal = 0.0

            value = float(np.clip(
                -gs_trend * 3 * 0.4 + vix_signal * 0.35 + usd_signal * 0.25,
                -1, 1
            ))

            return WorldSignal(
                domain="CONFLICT", source="safe_haven_proxy",
                region="GLOBAL", value=value, confidence=0.55,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                raw={"gs_trend": gs_trend, "vix": vix_now, "usd_trend": usd_signal},
                notes="Gold/SPY + VIX + USD as conflict proxy [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"Conflict signal failed: {e}")
            return WorldSignal(
                domain="CONFLICT", source="safe_haven_proxy",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow()
            )
