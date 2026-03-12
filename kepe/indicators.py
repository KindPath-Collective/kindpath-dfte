"""
KEPE — World Field Indicators
================================
Data feeds for the KindEarth Predictive Engine.
Reads the world field: ecological, social, equity, narrative layers.

Signals are now instrument-specific and temporally layered:
  STRUCTURAL  (annual/quarterly)  — World Bank, yield curve, macro regime
  MEDIUM      (monthly/weekly)    — sector flows, credit, physical proxies
  SURFACE     (daily)             — sentiment, volatility, narrative

Sources (all free/public):
  World Bank API    — social + equity indicators
  GDELT Project     — geopolitical + narrative sentiment
  FRED              — macro economic indicators
  yfinance          — ETF flows, yield proxies, sector signals

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
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

CACHE_DIR = "/tmp/kepe_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class WorldSignal:
    """Normalised world field signal from one domain."""
    domain: str           # ECOLOGICAL | SOCIAL | MACRO | NARRATIVE | OPTIMISM | CONFLICT
                          # SECTOR_FLOW | RISK_APPETITE
    source: str
    region: str           # GLOBAL | US | AU | SECTOR
    value: float          # -1.0 (high entropy) → +1.0 (high syntropy)
    confidence: float     # 0.0 → 1.0
    evidence_level: str   # ESTABLISHED | TESTABLE | SPECULATIVE
    timestamp: datetime
    raw: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    temporal_layer: str = "SURFACE"   # STRUCTURAL | MEDIUM | SURFACE


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


# ─── STRUCTURAL LAYER (annual / quarterly) ────────────────────────────────────

class PsychosomaticFieldSignal:
    """
    Measures the 'somatics' of the market field per region.
    """
    REGIONAL_INDICES = {
        "US": "^VIX",
        "EU": "^V2TX",    # Euro Stoxx 50 Volatility
        "APAC": "^N225",  # Nikkei (proxy)
        "LATAM": "^BVSP", # Bovespa (proxy)
        "AFRICA": "^JTOPI", # JSE Top 40 (proxy)
        "GLOBAL": "^VIX"
    }

    def compute(self, region: str = "GLOBAL") -> WorldSignal:
        try:
            import yfinance as yf
            idx = self.REGIONAL_INDICES.get(region, "^VIX")
            data = yf.Ticker(idx).history(period="10d")["Close"]
            if data.empty:
                return self._empty(region)
            
            adrenaline = (data.iloc[-1] - data.iloc[-5]) / (data.iloc[-5] + 1e-10)
            # Normalize: High volatility acceleration = high tension (-1.0)
            value = float(np.clip(-adrenaline * 0.5, -1, 1))

            return WorldSignal(
                domain="PSYCHOSOMATIC", source="field_tension",
                region=region, value=value, confidence=0.70,
                evidence_level="SPECULATIVE",
                timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
                raw={"adrenaline": adrenaline},
                notes=f"Psychosomatic tension for {region}."
            )
        except Exception:
            return self._empty(region)

    def _empty(self, region):
        return WorldSignal(domain="PSYCHOSOMATIC", source="field_tension", region=region, value=0.0, confidence=0.0, evidence_level="SPECULATIVE", timestamp=datetime.utcnow(), temporal_layer="SURFACE")

class FredIndicator:
    """
    Macro Field Aggregator using FRED (Federal Reserve Economic Data).
    Synthesizes multiple critical series into a single WorldSignal.
    """
    FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
    
    # Series ID -> (Weight, Direction [1=positive-is-good, -1=negative-is-good])
    MACRO_BASKET = {
        "UNRATE":   (0.15, -1), # Unemployment (Lower is better)
        "GDPC1":    (0.15,  1), # Real GDP (Higher is better)
        "CPIAUCSL": (0.10, -1), # Inflation (Lower is better)
        "FEDFUNDS": (0.10, -1), # Fed Funds Rate (Lower is better for growth)
        "INDPRO":   (0.10,  1), # Industrial Production
        "PAYEMS":   (0.10,  1), # Nonfarm Payrolls
        "UMCSENT":  (0.10,  1), # Consumer Sentiment
        "T10Y2Y":   (0.10,  1), # Yield Spread (Inversion is bad)
        "M2SL":     (0.05,  1), # Money Supply
        "WALCL":    (0.05,  1), # Fed Balance Sheet (Liquidity)
    }

    def _load_cached_pickle(self) -> dict:
        """
        Read FRED data from the backtest pickle cache (created by
        backtest/fetch_sim_data.py).  Returns dict of series_id → DataFrame,
        or {} if the cache file is absent or stale (>7 days old).
        """
        import pickle
        import time
        # Look for the cache relative to this file's location (kepe/ → project root)
        here = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(here, "..", "backtest", "fred_data_cache.pkl")
        cache_path = os.path.normpath(cache_path)
        if not os.path.exists(cache_path):
            return {}
        # Treat cache as stale after 7 days
        if time.time() - os.path.getmtime(cache_path) > 7 * 86400:
            return {}
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}

    def _signal_from_cache(self, series_id: str, cache: dict) -> Optional["WorldSignal"]:
        """
        Convert cached DataFrame for series_id into a WorldSignal.
        Returns None if series is not present or has insufficient data.
        """
        df = cache.get(series_id)
        if df is None or len(df) < 2:
            return None
        try:
            df = df.sort_values("Date")
            curr = float(df["Value"].iloc[-1])
            prev = float(df["Value"].iloc[-2])
            delta = (curr - prev) / (prev + 1e-10)
            _, direction = self.MACRO_BASKET.get(series_id, (1, 1))
            value = float(np.clip(delta * 20 * direction, -1, 1))
            return WorldSignal(
                domain="MACRO", source="fred_cache", region="US", value=value,
                confidence=0.60,  # Lower than live: data may be days old
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(), temporal_layer="STRUCTURAL",
                raw={"val": curr, "delta": delta},
                notes=f"{series_id}={curr} (from backtest cache)",
            )
        except Exception:
            return None

    def compute(self, series_id: Optional[str] = None) -> WorldSignal:
        """
        If series_id is provided, returns that specific signal.
        Otherwise, returns a weighted synthesis of the entire MACRO_BASKET.
        Priority: live FRED API → backtest pickle cache → zero/silent fallback.
        """
        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            # Try backtest pickle cache before going silent
            cache = self._load_cached_pickle()
            if cache:
                return self._synthesize_from_cache(cache, series_id)
            return WorldSignal(domain="MACRO", source="fred", region="US", value=0.0,
                               confidence=0.0, evidence_level="ESTABLISHED",
                               timestamp=datetime.utcnow(), temporal_layer="STRUCTURAL")

        if series_id:
            return self._fetch_single(series_id, api_key)

        # Synthesize entire basket from live API
        total_value = 0.0
        total_weight = 0.0
        details = {}

        for sid, (weight, direction) in self.MACRO_BASKET.items():
            sig = self._fetch_single(sid, api_key)
            if sig.confidence > 0:
                total_value += sig.value * weight
                total_weight += weight
                details[sid] = sig.value

        final_value = total_value / total_weight if total_weight > 0 else 0.0
        
        return WorldSignal(
            domain="MACRO", source="fred_synthesizer",
            region="US", value=float(np.clip(final_value, -1, 1)),
            confidence=0.85 if total_weight > 0.5 else 0.0,
            evidence_level="ESTABLISHED",
            timestamp=datetime.utcnow(),
            temporal_layer="STRUCTURAL",
            raw=details,
            notes=f"Synthesized {len(details)} FRED series into Macro Field"
        )

    def _fetch_single(self, series_id: str, api_key: str) -> WorldSignal:
        cache_key = f"fred_{series_id}"
        cached = _cache_get(cache_key, max_age_hours=24)
        if cached:
            return WorldSignal(**{**cached, "timestamp": datetime.utcnow()})

        try:
            params = {"series_id": series_id, "api_key": api_key, "file_type": "json", 
                      "sort_order": "desc", "limit": 2}
            resp = requests.get(self.FRED_BASE, params=params, timeout=10)
            if resp.status_code != 200:
                logger.error(f"FRED error {resp.status_code} for {series_id}: {resp.text}")
                return self._empty(series_id)
            
            data = resp.json()
            obs = data.get("observations", [])
            if len(obs) < 2:
                logger.warning(f"FRED insufficient data for {series_id}")
                return self._empty(series_id)

            # Some values are "." or empty strings in FRED
            try:
                curr = float(obs[0]["value"])
                prev = float(obs[1]["value"])
            except (ValueError, TypeError):
                logger.warning(f"FRED non-numeric data for {series_id}")
                return self._empty(series_id)

            delta = (curr - prev) / (prev + 1e-10)
            
            # Use basket logic if available, default to positive-is-good
            _, direction = self.MACRO_BASKET.get(series_id, (1, 1))
            value = float(np.clip(delta * 20 * direction, -1, 1))

            result = WorldSignal(
                domain="MACRO", source="fred", region="US", value=value, 
                confidence=0.80, evidence_level="ESTABLISHED", 
                timestamp=datetime.utcnow(), temporal_layer="STRUCTURAL",
                raw={"val": curr, "delta": delta}
            )
            _cache_set(cache_key, {
                "domain": "MACRO", "source": "fred", "region": "US", "value": value,
                "confidence": 0.80, "evidence_level": "ESTABLISHED",
                "raw": result.raw, "notes": f"{series_id}={curr}",
                "temporal_layer": "STRUCTURAL", "timestamp": result.timestamp.isoformat()
            })
            return result
        except Exception as e:
            logger.error(f"FRED exception for {series_id}: {e}")
            return self._empty(series_id)

    def _synthesize_from_cache(self, cache: dict, series_id: Optional[str] = None) -> "WorldSignal":
        """Synthesize a WorldSignal from the backtest pickle cache instead of the live API."""
        if series_id:
            sig = self._signal_from_cache(series_id, cache)
            return sig if sig is not None else self._empty(series_id)

        total_value = 0.0
        total_weight = 0.0
        details = {}
        for sid, (weight, _) in self.MACRO_BASKET.items():
            sig = self._signal_from_cache(sid, cache)
            if sig is not None and sig.confidence > 0:
                total_value += sig.value * weight
                total_weight += weight
                details[sid] = sig.value

        if total_weight == 0:
            return self._empty("macro_basket")

        return WorldSignal(
            domain="MACRO", source="fred_cache_synthesizer",
            region="US", value=float(np.clip(total_value / total_weight, -1, 1)),
            confidence=0.60,
            evidence_level="ESTABLISHED",
            timestamp=datetime.utcnow(), temporal_layer="STRUCTURAL",
            raw=details,
            notes=f"Synthesized {len(details)} FRED series from backtest cache",
        )

    def _empty(self, sid):
        return WorldSignal(domain="MACRO", source="fred", region="US", value=0.0, 
                           confidence=0.0, evidence_level="ESTABLISHED", 
                           timestamp=datetime.utcnow(), temporal_layer="STRUCTURAL")


from somatic_bridge import SomaticSynapse

class SomaticFieldSignal:
    """
    Reads the 'Somatic Field' using the Synapse Bridge (Late-Move Inversion Index).
    Detects if the market is 'protesting' its own price action.
    """
    def compute(self, symbol: str) -> WorldSignal:
        try:
            import yfinance as yf
            # Fetch intraday or short-term history for 4-quarter segmentation
            data = yf.Ticker(symbol).history(period="5d", interval="60m")
            if data.empty or len(data) < 20:
                return self._empty(symbol)
            
            ohlcv = []
            for i, row in data.iterrows():
                ohlcv.append({'close': row['Close'], 'volume': row['Volume']})
            
            state = SomaticSynapse.calculate_lmii(ohlcv)
            
            # LMII (Divergence) as signal: 
            # High divergence (+LMII) when ZPC is aligned = High Syntropy (+1.0)
            # High divergence (+LMII) when ZPC is broken = High Entropy (-1.0)
            value = float(np.clip(state.lmii * (1.0 if not state.is_protest else -1.0), -1, 1))

            return WorldSignal(
                domain="SOMATIC", source="synapse_bridge",
                region="GLOBAL", value=value, confidence=0.75,
                evidence_level="SPECULATIVE",
                timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
                raw={"lmii": state.lmii, "is_protest": state.is_protest},
                notes=f"Somatic field for {symbol}: LMII={state.lmii:.2f}"
            )
        except Exception as e:
            logger.warning(f"SomaticFieldSignal failed for {symbol}: {e}")
            return self._empty(symbol)

    def _empty(self, symbol):
        return WorldSignal(domain="SOMATIC", source="synapse_bridge", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="SPECULATIVE", timestamp=datetime.utcnow(), temporal_layer="SURFACE")

class FieldStaticSignal:
    """
    Measures 'Field Static' — communication disruption, censorship, and 
    digital friction as a proxy for suppressed mass-scale expression.
    [UNCONVENTIONAL] — Digital flow = Field Breath. Friction = Entropy.
    """
    GDELT_THEMES = ["COMMUNICATION_CENSORSHIP", "INTERNET_SHUTDOWN", "CYBER_ATTACK"]

    def compute(self, region: str = "GLOBAL") -> WorldSignal:
        try:
            # Proxying 'Static' via GDELT volume for censorship/shutdown themes
            # High volume of these themes = High Static (-1.0)
            # We'll use a placeholder logic that simulates a 0.80 confidence 
            # based on recent narrative drift if GDELT is slow.
            vix_accel = GDELTSignal()._vix_narrative_proxy().value
            
            # If VIX is accelerating, it usually correlates with digital friction/panic
            value = float(np.clip(vix_accel * 0.8, -1, 1))

            return WorldSignal(
                domain="STATIC", source="digital_friction",
                region=region, value=value, confidence=0.60,
                evidence_level="SPECULATIVE",
                timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
                notes=f"Field static (digital friction) for {region}."
            )
        except Exception:
            return WorldSignal(domain="STATIC", source="digital_friction", region=region, value=0.0, confidence=0.0, evidence_level="SPECULATIVE", timestamp=datetime.utcnow(), temporal_layer="SURFACE")

class EcologicalSignal:
    """
    Ecological stress proxy from commodity prices and climate indices.

    High commodity volatility + rising energy prices = ecological pressure.
    Commodity stability = ecological coherence.

    Uses: Gold/Oil ratio, agricultural commodity stability, energy price trend.
    [TESTABLE] — commodity prices as ecological proxy is indirect.
    Temporal layer: STRUCTURAL — commodity regimes shift slowly.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            oil  = yf.Ticker("CL=F").history(period="3mo")["Close"]
            gold = yf.Ticker("GC=F").history(period="3mo")["Close"]
            corn = yf.Ticker("ZC=F").history(period="3mo")["Close"]

            if oil.empty or gold.empty:
                raise ValueError("Commodity data unavailable")

            oil_trend = (oil.iloc[-1] - oil.iloc[-20]) / (oil.iloc[-20] + 1e-10)

            go_ratio  = gold.iloc[-1] / (oil.iloc[-1] + 1e-10)
            go_series = gold / (oil + 1e-10)
            go_z      = (go_ratio - go_series.mean()) / (go_series.std() + 1e-10)

            food_vol    = float(corn.pct_change().std()) if not corn.empty else 0.02
            food_signal = float(np.clip(-(food_vol - 0.015) / 0.02, -1, 1))

            value = float(np.clip(
                -oil_trend * 0.4 - go_z * 0.1 + food_signal * 0.5,
                -1, 1
            ))

            return WorldSignal(
                domain="ECOLOGICAL", source="commodity_proxy",
                region="GLOBAL", value=value, confidence=0.50,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="STRUCTURAL",
                raw={"oil_trend": oil_trend, "go_z": float(go_z), "food_vol": food_vol},
                notes="Commodity proxy for ecological stress [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"Ecological signal failed: {e}")
            return WorldSignal(
                domain="ECOLOGICAL", source="commodity_proxy",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="STRUCTURAL",
            )


class WorldBankSignal:
    """
    Social equity indicators from World Bank API (free).
    Uses cached annual data — World Bank updates yearly.

    Indicators:
      SI.POV.GINI     — Gini coefficient (inequality)
      SL.UEM.TOTL.ZS  — Unemployment rate
      SE.XPD.TOTL.GD.ZS — Education expenditure (% GDP)

    [ESTABLISHED] — World Bank data is authoritative for these metrics.
    Temporal layer: STRUCTURAL — annual indicators, background social field.
    """

    WB_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    INDICATORS = {
        "gini":         "SI.POV.GINI",
        "unemployment": "SL.UEM.TOTL.ZS",
        "education":    "SE.XPD.TOTL.GD.ZS",
    }

    def compute(self, country: str = "WLD") -> WorldSignal:
        cache_key = f"worldbank_{country}"
        cached = _cache_get(cache_key, max_age_hours=48)
        if cached:
            sig = WorldSignal(**{**cached, "timestamp": datetime.utcnow()})
            sig.temporal_layer = "STRUCTURAL"
            return sig

        try:
            scores = {}
            for name, ind_code in self.INDICATORS.items():
                val = self._fetch_latest(country, ind_code)
                if val is not None:
                    scores[name] = val

            if not scores:
                raise ValueError("No World Bank data")

            gini = scores.get("gini", 38)
            gini_signal = float(np.clip(-(gini - 35) / 20, -1, 1))

            unem = scores.get("unemployment", 6)
            unem_signal = float(np.clip(-(unem - 5) / 10, -1, 1))

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
                temporal_layer="STRUCTURAL",
                raw=scores,
                notes=f"Gini={gini:.1f} Unem={unem:.1f}% Edu={edu:.1f}%"
            )
            _cache_set(cache_key, {
                "domain": result.domain, "source": result.source,
                "region": result.region, "value": result.value,
                "confidence": result.confidence,
                "evidence_level": result.evidence_level,
                "raw": result.raw, "notes": result.notes,
                "temporal_layer": "STRUCTURAL",
                "timestamp": result.timestamp.isoformat(),
            })
            return result

        except Exception as e:
            logger.warning(f"World Bank signal failed: {e}")
            return WorldSignal(
                domain="SOCIAL", source="world_bank",
                region=country, value=0.0, confidence=0.0,
                evidence_level="ESTABLISHED", timestamp=datetime.utcnow(),
                temporal_layer="STRUCTURAL",
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


class YieldCurveSignal:
    """
    US Treasury yield curve slope as macro regime indicator.

    Uses 10Y (^TNX) minus 3-month (^IRX) spread.
      Positive spread = normal curve = expansionary = positive for equity.
      Negative spread = inverted = recessionary signal = negative for equity.

    [ESTABLISHED] — yield curve inversion is a well-validated recession predictor.
    Temporal layer: STRUCTURAL — yield curve regime persists for months.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            ten_y   = yf.Ticker("^TNX").history(period="3mo")["Close"]
            three_m = yf.Ticker("^IRX").history(period="3mo")["Close"]

            if ten_y.empty or three_m.empty:
                raise ValueError("No yield data")

            spread = float(ten_y.iloc[-1]) - float(three_m.iloc[-1])
            # +2.5 → normal/steep (+1.0), −2.5 → deeply inverted (−1.0)
            value = float(np.clip(spread / 2.5, -1, 1))

            return WorldSignal(
                domain="MACRO", source="yield_curve",
                region="US", value=value, confidence=0.75,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
                temporal_layer="STRUCTURAL",
                raw={"ten_y": float(ten_y.iloc[-1]), "three_m": float(three_m.iloc[-1]),
                     "spread": spread},
                notes=f"10Y-3M spread={spread:.2f}% [ESTABLISHED]"
            )
        except Exception as e:
            logger.warning(f"Yield curve signal failed: {e}")
            return WorldSignal(
                domain="MACRO", source="yield_curve",
                region="US", value=0.0, confidence=0.0,
                evidence_level="ESTABLISHED", timestamp=datetime.utcnow(),
                temporal_layer="STRUCTURAL",
            )


# ─── MEDIUM LAYER (monthly / weekly) ──────────────────────────────────────────

class CleanEnergyFlowSignal:
    """
    Sector flow proxy for clean energy assets.

    ICLN vs XLE (traditional energy) relative performance.
    ICLN outperforming XLE = capital rotating into clean energy = positive.

    [TESTABLE] — ETF relative flow as sector world-field proxy.
    Temporal layer: MEDIUM — sector rotation plays out over weeks/months.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            icln = yf.Ticker("ICLN").history(period="3mo")["Close"]
            xle  = yf.Ticker("XLE").history(period="3mo")["Close"]

            if icln.empty or xle.empty or len(icln) < 20:
                raise ValueError("Clean energy ETF data unavailable")

            icln_ret = (icln.iloc[-1] - icln.iloc[-20]) / (icln.iloc[-20] + 1e-10)
            xle_ret  = (xle.iloc[-1]  - xle.iloc[-20])  / (xle.iloc[-20]  + 1e-10)

            rel_perf = icln_ret - xle_ret   # positive = clean energy outperforming
            # ±10% relative outperformance → ±1.0
            value = float(np.clip(rel_perf * 10, -1, 1))

            return WorldSignal(
                domain="SECTOR_FLOW", source="clean_energy_flow",
                region="GLOBAL", value=value, confidence=0.55,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                raw={"icln_ret": icln_ret, "xle_ret": xle_ret, "rel_perf": rel_perf},
                notes=f"ICLN vs XLE 20d relative={rel_perf:.3f} [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"CleanEnergyFlow signal failed: {e}")
            return WorldSignal(
                domain="SECTOR_FLOW", source="clean_energy_flow",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
            )


class GridParitySignal:
    """
    Grid parity proxy: XLE/ICLN ratio trend + ENPH as solar cost proxy.

    Falling XLE/ICLN ratio = traditional energy losing ground to clean = positive.
    ENPH rising = solar module economics improving (cost curve proxy).

    [TESTABLE] — ETF ratio as grid parity directional proxy.
    Temporal layer: MEDIUM — structural competitiveness shifts over months.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            xle  = yf.Ticker("XLE").history(period="6mo")["Close"]
            icln = yf.Ticker("ICLN").history(period="6mo")["Close"]

            if xle.empty or icln.empty or len(xle) < 40:
                raise ValueError("ETF data unavailable")

            ratio       = xle / (icln + 1e-10)
            ratio_trend = (ratio.iloc[-1] - ratio.iloc[-40]) / (ratio.iloc[-40] + 1e-10)

            # Negative ratio trend = clean energy gaining vs fossil = positive
            value = float(np.clip(-ratio_trend * 10, -1, 1))

            # Blend with ENPH solar proxy
            try:
                enph = yf.Ticker("ENPH").history(period="3mo")["Close"]
                if not enph.empty and len(enph) > 20:
                    enph_trend = (enph.iloc[-1] - enph.iloc[-20]) / (enph.iloc[-20] + 1e-10)
                    value = float(np.clip(value * 0.6 + enph_trend * 3 * 0.4, -1, 1))
            except Exception:
                pass

            return WorldSignal(
                domain="SECTOR_FLOW", source="grid_parity",
                region="GLOBAL", value=value, confidence=0.50,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                raw={"xle_icln_ratio_trend": ratio_trend},
                notes=f"XLE/ICLN ratio 40d trend={ratio_trend:.3f} [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"GridParity signal failed: {e}")
            return WorldSignal(
                domain="SECTOR_FLOW", source="grid_parity",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
            )


class CreditSpreadSignal:
    """
    US credit spread proxy via HYG/LQD ratio.

    HYG = iShares HY Corp Bond. LQD = iShares IG Corp Bond.
    HYG outperforming LQD = spreads tightening = credit risk appetite = positive.
    HYG underperforming   = spreads widening = systemic stress = negative.

    [TESTABLE] — ETF ratio as credit spread directional proxy.
    Temporal layer: MEDIUM — credit conditions shift over weeks.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            hyg = yf.Ticker("HYG").history(period="3mo")["Close"]
            lqd = yf.Ticker("LQD").history(period="3mo")["Close"]

            if hyg.empty or lqd.empty or len(hyg) < 20:
                raise ValueError("Credit ETF data unavailable")

            ratio      = hyg / (lqd + 1e-10)
            ratio_mean = ratio.mean()
            ratio_std  = ratio.std() + 1e-10
            z          = (ratio.iloc[-1] - ratio_mean) / ratio_std

            # +2σ = very tight spreads (+1.0), -2σ = very wide (-1.0)
            value = float(np.clip(z / 2, -1, 1))

            return WorldSignal(
                domain="MACRO", source="credit_spread",
                region="US", value=value, confidence=0.65,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                raw={"hyg_lqd_z": float(z), "ratio_now": float(ratio.iloc[-1]),
                     "ratio_mean": float(ratio_mean)},
                notes=f"HYG/LQD z-score={z:.2f} [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"CreditSpread signal failed: {e}")
            return WorldSignal(
                domain="MACRO", source="credit_spread",
                region="US", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
            )


class EquityBreadthSignal:
    """
    Market breadth via RSP/SPY ratio (equal-weight vs cap-weight S&P 500).

    Rising RSP/SPY = broad participation across constituents = healthy equity field.
    Falling ratio = leadership narrowing = fragility signal.

    [TESTABLE] — equal-weight vs cap-weight as market breadth proxy.
    Temporal layer: MEDIUM — breadth conditions persist over weeks.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            rsp = yf.Ticker("RSP").history(period="3mo")["Close"]
            spy = yf.Ticker("SPY").history(period="3mo")["Close"]

            if rsp.empty or spy.empty or len(rsp) < 20:
                raise ValueError("Breadth ETF data unavailable")

            ratio       = rsp / (spy + 1e-10)
            ratio_trend = (ratio.iloc[-1] - ratio.iloc[-20]) / (ratio.iloc[-20] + 1e-10)

            # ±5% ratio trend → ±1.0
            value = float(np.clip(ratio_trend * 20, -1, 1))

            return WorldSignal(
                domain="MACRO", source="equity_breadth",
                region="US", value=value, confidence=0.60,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                raw={"rsp_spy_trend": ratio_trend},
                notes=f"RSP/SPY 20d trend={ratio_trend:.4f} [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"EquityBreadth signal failed: {e}")
            return WorldSignal(
                domain="MACRO", source="equity_breadth",
                region="US", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
            )


class RealYieldSignal:
    """
    Real yield proxy for gold world field via TIP/IEF relative performance.

    TIP = iShares TIPS Bond ETF (inflation-protected).
    IEF = iShares 7-10Y Treasury (nominal).

    Rising TIP = TIPS prices rising = real yields falling = gold-positive.
    TIP outperforming IEF = inflation premium rising = also gold-positive.

    [TESTABLE] — TIPS/nominal Treasury spread as real yield proxy.
    Temporal layer: MEDIUM — real yield regime shifts over weeks.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            tip = yf.Ticker("TIP").history(period="3mo")["Close"]
            ief = yf.Ticker("IEF").history(period="3mo")["Close"]

            if tip.empty or len(tip) < 20:
                raise ValueError("TIPS data unavailable")

            tip_trend = (tip.iloc[-1] - tip.iloc[-20]) / (tip.iloc[-20] + 1e-10)

            # TIP vs IEF: TIP outperforming = inflation expectations rising = gold-positive
            if not ief.empty and len(ief) >= 20:
                ief_trend = (ief.iloc[-1] - ief.iloc[-20]) / (ief.iloc[-20] + 1e-10)
                tip_vs_ief = tip_trend - ief_trend
            else:
                tip_vs_ief = 0.0

            # Combine: TIP trend (primary) + TIP vs IEF (inflation premium signal)
            value = float(np.clip(tip_trend * 15 * 0.6 + tip_vs_ief * 10 * 0.4, -1, 1))

            return WorldSignal(
                domain="MACRO", source="real_yield",
                region="US", value=value, confidence=0.65,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                raw={"tip_trend": tip_trend, "tip_vs_ief": tip_vs_ief},
                notes=f"TIP 20d trend={tip_trend:.4f}, vs IEF={tip_vs_ief:.4f} [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"RealYield signal failed: {e}")
            return WorldSignal(
                domain="MACRO", source="real_yield",
                region="US", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
            )


class CryptoRegulatorySignal:
    """
    Regulatory sentiment proxy for crypto via Coinbase (COIN) relative performance.

    COIN outperforming BTC = regulatory environment perceived as constructive.
    COIN underperforming BTC = regulatory headwind narrative dominant.

    [TESTABLE] — COIN vs BTC spread as regulatory sentiment proxy.
    Temporal layer: MEDIUM — regulatory narratives shift over weeks.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            coin = yf.Ticker("COIN").history(period="3mo")["Close"]
            btc  = yf.Ticker("BTC-USD").history(period="3mo")["Close"]

            if coin.empty or btc.empty or len(coin) < 20:
                raise ValueError("COIN/BTC data unavailable")

            coin_ret = (coin.iloc[-1] - coin.iloc[-20]) / (coin.iloc[-20] + 1e-10)
            btc_ret  = (btc.iloc[-1]  - btc.iloc[-20])  / (btc.iloc[-20]  + 1e-10)

            coin_vs_btc = coin_ret - btc_ret
            # ±20% relative → ±1.0
            value = float(np.clip(coin_vs_btc * 5, -1, 1))

            return WorldSignal(
                domain="NARRATIVE", source="regulatory_proxy",
                region="GLOBAL", value=value, confidence=0.40,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                raw={"coin_ret": coin_ret, "btc_ret": btc_ret, "spread": coin_vs_btc},
                notes=f"COIN vs BTC 20d spread={coin_vs_btc:.3f} [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"CryptoRegulatory signal failed: {e}")
            return WorldSignal(
                domain="NARRATIVE", source="regulatory_proxy",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
            )


# ─── SURFACE LAYER (daily) ────────────────────────────────────────────────────

class GDELTSignal:
    """
    Geopolitical + narrative sentiment.
    VIX proxy runs first (reliable, no external API dependency).
    GDELT is the fallback if VIX data is unavailable.

    [TESTABLE] — VIX as narrative stress proxy; GDELT as narrative sentiment.
    Temporal layer: SURFACE — daily narrative field.
    """

    GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def compute(self, theme: str = "GENERAL") -> WorldSignal:
        cache_key = f"gdelt_{theme}"
        cached = _cache_get(cache_key, max_age_hours=2)
        if cached:
            try:
                sig = WorldSignal(**{**cached, "timestamp": datetime.utcnow()})
                sig.temporal_layer = "SURFACE"
                return sig
            except Exception:
                pass

        # VIX proxy runs first — reliable, no external API dependency
        vix_result = self._vix_narrative_proxy()
        if vix_result.confidence > 0:
            return vix_result

        # Fallback: GDELT
        try:
            params = {
                "query":      "tone",
                "mode":       "TimelineSourceCountry",
                "format":     "json",
                "timespan":   "1week",
                "maxrecords": 10,
            }
            resp = requests.get(self.GDELT_URL, params=params, timeout=8)
            if resp.status_code != 200:
                raise ValueError(f"GDELT status {resp.status_code}")

            data     = resp.json()
            timeline = data.get("timeline", [{}])
            tones    = []
            for entry in timeline[:1]:
                for item in entry.get("data", [])[:10]:
                    val = item.get("value")
                    if val is not None:
                        tones.append(float(val))

            if not tones:
                raise ValueError("No GDELT tone data")

            mean_tone = float(np.mean(tones))
            value     = float(np.clip(mean_tone / 8, -1, 1))

            result = WorldSignal(
                domain="NARRATIVE", source="gdelt",
                region="GLOBAL", value=value, confidence=0.45,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
                raw={"mean_tone": mean_tone, "n_samples": len(tones)},
                notes=f"GDELT mean tone {mean_tone:.2f} [TESTABLE]"
            )
            _cache_set(cache_key, {
                "domain": result.domain, "source": result.source,
                "region": result.region, "value": result.value,
                "confidence": result.confidence,
                "evidence_level": result.evidence_level,
                "raw": result.raw, "notes": result.notes,
                "temporal_layer": "SURFACE",
                "timestamp": result.timestamp.isoformat(),
            })
            return result

        except Exception as e:
            logger.warning(f"GDELT signal failed: {e}")
            return WorldSignal(
                domain="NARRATIVE", source="gdelt",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
            )

    def _vix_narrative_proxy(self) -> WorldSignal:
        """VIX as narrative stress proxy (primary path)."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX").history(period="1mo")["Close"]
            if not vix.empty:
                vix_now = float(vix.iloc[-1])
                value   = float(np.clip(-(vix_now - 20) / 15, -1, 1))
                return WorldSignal(
                    domain="NARRATIVE", source="vix_proxy",
                    region="GLOBAL", value=value, confidence=0.40,
                    evidence_level="TESTABLE",
                    timestamp=datetime.utcnow(),
                    temporal_layer="SURFACE",
                    raw={"vix": vix_now},
                    notes=f"VIX={vix_now:.1f} narrative proxy"
                )
        except Exception:
            pass
        return WorldSignal(
            domain="NARRATIVE", source="vix_proxy",
            region="GLOBAL", value=0.0, confidence=0.0,
            evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            temporal_layer="SURFACE",
        )


class OptimismSignal:
    """
    Optimism propagation coefficient — forward-looking sentiment.

    Tech (XLK) outperforming utilities (XLU) = risk-on = forward faith active.
    SPY trend = broad market direction.

    [TESTABLE] — optimism indices as syntropy precursor.
    Temporal layer: SURFACE — daily risk appetite.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            spy = yf.Ticker("SPY").history(period="3mo")["Close"]
            xlu = yf.Ticker("XLU").history(period="3mo")["Close"]
            xlk = yf.Ticker("XLK").history(period="3mo")["Close"]

            if spy.empty or len(spy) < 20:
                raise ValueError("No SPY data")

            if not xlk.empty and not xlu.empty and len(xlk) > 20 and len(xlu) > 20:
                xlk_ret  = (xlk.iloc[-1] - xlk.iloc[-20]) / (xlk.iloc[-20] + 1e-10)
                xlu_ret  = (xlu.iloc[-1] - xlu.iloc[-20]) / (xlu.iloc[-20] + 1e-10)
                risk_on  = xlk_ret - xlu_ret
            else:
                risk_on = 0.0

            spy_ret = (spy.iloc[-1] - spy.iloc[-20]) / (spy.iloc[-20] + 1e-10)
            value   = float(np.clip(spy_ret * 5 * 0.5 + risk_on * 5 * 0.5, -1, 1))

            return WorldSignal(
                domain="OPTIMISM", source="market_breadth",
                region="US", value=value, confidence=0.60,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
                raw={"spy_ret": spy_ret, "risk_on": risk_on},
                notes=f"Tech/utilities spread={risk_on:.3f}, SPY 20d={spy_ret:.3f}"
            )
        except Exception as e:
            logger.warning(f"Optimism signal failed: {e}")
            return WorldSignal(
                domain="OPTIMISM", source="market_breadth",
                region="US", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
            )


class ConflictPressureSignal:
    """
    Conflict and systemic pressure proxy.

    Sources: Gold/SPY ratio, VIX level, USD index trend.
    [TESTABLE] — indirect proxy for conflict/pressure.
    Temporal layer: SURFACE — daily fear/stress readings.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            gold   = yf.Ticker("GC=F").history(period="3mo")["Close"]
            spy    = yf.Ticker("SPY").history(period="3mo")["Close"]
            vix    = yf.Ticker("^VIX").history(period="3mo")["Close"]
            usdidx = yf.Ticker("DX-Y.NYB").history(period="3mo")["Close"]

            if gold.empty or spy.empty or len(gold) < 20:
                raise ValueError("No data")

            gs_ratio = gold / (spy + 1e-10)
            gs_trend = (gs_ratio.iloc[-1] - gs_ratio.iloc[-20]) / (gs_ratio.iloc[-20] + 1e-10)

            vix_now    = float(vix.iloc[-1]) if not vix.empty else 20.0
            vix_signal = float(np.clip(-(vix_now - 18) / 20, -1, 1))

            if not usdidx.empty and len(usdidx) > 20:
                usd_trend  = (usdidx.iloc[-1] - usdidx.iloc[-20]) / (usdidx.iloc[-20] + 1e-10)
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
                temporal_layer="SURFACE",
                raw={"gs_trend": float(gs_trend), "vix": vix_now, "usd_signal": usd_signal},
                notes="Gold/SPY + VIX + USD as conflict proxy [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"Conflict signal failed: {e}")
            return WorldSignal(
                domain="CONFLICT", source="safe_haven_proxy",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
            )


class CryptoRiskAppetiteSignal:
    """
    Risk appetite specific to crypto world field.

    BTC 30-day momentum + ARKK vs SPY spread (innovation/risk premium).
    High ARKK outperformance = high risk appetite = positive crypto world field.

    [TESTABLE] — BTC momentum and ARKK spread as crypto risk proxy.
    Temporal layer: SURFACE — daily risk appetite signal.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            btc  = yf.Ticker("BTC-USD").history(period="1mo")["Close"]
            arkk = yf.Ticker("ARKK").history(period="1mo")["Close"]
            spy  = yf.Ticker("SPY").history(period="1mo")["Close"]

            if btc.empty or len(btc) < 5:
                raise ValueError("No BTC data")

            btc_ret = (btc.iloc[-1] - btc.iloc[0]) / (btc.iloc[0] + 1e-10)

            if not arkk.empty and not spy.empty and len(arkk) > 5 and len(spy) > 5:
                arkk_ret      = (arkk.iloc[-1] - arkk.iloc[0]) / (arkk.iloc[0] + 1e-10)
                spy_ret       = (spy.iloc[-1]  - spy.iloc[0])  / (spy.iloc[0]  + 1e-10)
                risk_premium  = arkk_ret - spy_ret
            else:
                risk_premium  = 0.0

            # BTC momentum weighted higher; ARKK spread as corroborating signal
            value = float(np.clip(btc_ret * 3 * 0.6 + risk_premium * 3 * 0.4, -1, 1))

            return WorldSignal(
                domain="RISK_APPETITE", source="crypto_risk",
                region="GLOBAL", value=value, confidence=0.50,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
                raw={"btc_ret": btc_ret, "risk_premium": risk_premium},
                notes=f"BTC 30d={btc_ret:.3f}, ARKK spread={risk_premium:.3f} [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"CryptoRiskAppetite signal failed: {e}")
            return WorldSignal(
                domain="RISK_APPETITE", source="crypto_risk",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="SURFACE",
            )
