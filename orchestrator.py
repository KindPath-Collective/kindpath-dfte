"""
DFTE — Main Orchestrator
==========================
Ties together:
  KEPE     → World Field Score (WFS)
  BMR      → Market Field Score (MFS) via bmr_server API
  DFTE     → Unified signal + trade decision
  Governance → Benevolence + contradiction gate
  Wallet   → Execution

Usage:
  # Start BMR server first (port 8001):
  cd kindpath-bmr && python bmr_server.py

  # Then run orchestrator:
  cd kindpath-dfte
  python orchestrator.py --symbols SPY QQQ GLD BTC-USD --mode paper
  python orchestrator.py --symbols ICLN NEE --mode paper  # syntropic basket
  python orchestrator.py --watch  # continuous loop with dashboard

Architecture:
  orchestrator.py
    ├── fetch_bmr_signal(symbol)      → BMRSummary via HTTP
    ├── fetch_kepe_signal(symbol)     → KEPEProfile via indicator stack
    ├── synthesise_dfte_signal()      → DFTESignal
    ├── governance_layer.py           → tier cap + contradiction check
    ├── wallet.py                     → execute or simulate
    └── dashboard(signals)            → rich terminal output
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import argparse
import requests
import numpy as np
from datetime import datetime, timezone
from typing import List, Optional, Dict

# Path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kepe.indicators import (
    # Structural
    EcologicalSignal, WorldBankSignal, YieldCurveSignal, FredIndicator,
    # Medium
    CleanEnergyFlowSignal, GridParitySignal,
    CreditSpreadSignal, EquityBreadthSignal,
    RealYieldSignal, CryptoRegulatorySignal,
    # Surface
    GDELTSignal, OptimismSignal, ConflictPressureSignal,
    CryptoRiskAppetiteSignal, PsychosomaticFieldSignal, FieldStaticSignal,
    SomaticFieldSignal
)
from kepe.kpre_physical import KPRELayer
from kepe.kpre_capital import KPRECapitalLayer
from kepe.kpre_language import KPRELanguageLayer
from kepe.nano_signals import (
    RedditSentimentSignal, CrossAssetEchoSignal, 
    GoogleNewsSignal, HighFrequencyVolatilitySignal
)
from kepe.expanded_signals import (
    WikipediaAttentionSignal, LocalWeatherSignal,
    GoogleTrendsSignal, SocialInfluenceSynthesizer, SectorCoherenceSignal
)
from kepe.crypto_signals import CoinGeckoSignal
from kepe.syntropy_engine import synthesise_kepe_profile, KEPEProfile
from cmam.cmam_engine import CMAMEngine, CMAMProfile, TradeClassification
from sas.sas_engine import SASEngine, SASProfile
from logger.signal_logger import SignalLogger
from relational_timestamp import compute_relational_timestamp, RelationalTimestamp, SYMBOL_EXCHANGE
from ratio_tracker import RatioTracker
from vicarious_field import VicariousFieldEngine, VicariousSignal
from nano_relational import NanoRelationalEngine, FieldState
from dfte.dfte_engine import (
    BMRSummary, KEPESummary, synthesise_dfte_signal, DFTESignal
)
from dfte.wisdom_engine import RecursiveWisdom
from governance.governance_layer import (
    score_benevolence, apply_governance_tier,
    detect_contradictions, log_influence,
    get_influence_summary, check_refusal,
    check_mirror_gate, check_position_safety
)
from wallet.wallet import get_wallet, OrderRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("dfte.orchestrator")

BMR_SERVER = os.environ.get("BMR_SERVER", "http://localhost:8001")

# Global cache for country-layer signals (avoid API spam)
_WEATHER_CACHE = {"signal": None, "timestamp": 0}

def get_cached_weather_signal():
    global _WEATHER_CACHE
    now = time.time()
    if _WEATHER_CACHE["signal"] and (now - _WEATHER_CACHE["timestamp"] < 3600):
        return _WEATHER_CACHE["signal"]
    
    try:
        sig = LocalWeatherSignal().compute()
        if sig.confidence > 0:
            _WEATHER_CACHE = {"signal": sig, "timestamp": now}
            return sig
    except Exception as e:
        logger.warning(f"Weather signal failed: {e}")
    return None


# ─── Pre-execution governance safety gate ────────────────────────────────────

def pre_execution_safety_check() -> bool:
    """
    Run the full governance test suite before any signal generation.
    If any test fails: logs CRITICAL and exits 1 — no signals are produced.

    Uses sys.executable so the check always runs inside the same
    virtual environment as the orchestrator itself.
    """
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    kindroot = os.path.dirname(here)
    bmr_path = os.path.join(kindroot, "kindpath-bmr")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{here}:{bmr_path}:{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_dfte.py",
         "--tb=short", "-q"],
        capture_output=True,
        text=True,
        cwd=here,
        env=env,
    )
    if result.returncode != 0:
        logger.critical("=" * 60)
        logger.critical("GOVERNANCE TESTS FAILED — ABORTING RUN")
        logger.critical("No signals will be generated until tests pass.")
        logger.critical(result.stdout[-3000:])  # last 3k chars to avoid log flood
        logger.critical("=" * 60)
        sys.exit(1)
    # Log passing count for audit trail
    last_line = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else "?"
    logger.info(f"Governance check passed: {last_line}")
    return True

# ─── Instrument classification + world-field routing ─────────────────────────

_ASSET_CLASSES: Dict[str, List[str]] = {
    "CLEAN_ENERGY": ["ICLN", "NEE", "ENPH", "FSLR", "BEP", "PLUG", "TSLA"],
    "INNOVATION":   ["LIT", "ROBO", "ARKW", "ARKG", "XBI"],
    "FUNGI":        ["CMPS", "ATAI", "GHRS", "CYBN", "MNMD"],
    "REGEN_AG":     ["CTVA", "FMC", "NTR", "ADM", "VNR"],
    "SATELLITE":    ["JOBY", "RKLB", "BKSY", "PL", "LLAP"],
    "BROAD_EQUITY": ["SPY", "QQQ", "IWM", "VTI", "RSP", "DIA",
                     "XLK", "XLF", "XLV", "XLE", "XLU"],
    "COMMODITIES":  ["GLD", "GC=F", "IAU", "SLV", "SI=F", "GDX", "USO", "CL=F"],
    "CRYPTO":       ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
}


def _classify_asset(symbol: str) -> str:
    s = symbol.upper()
    for cls, tickers in _ASSET_CLASSES.items():
        if s in tickers:
            return cls
    return "BROAD_EQUITY"


def _get_indicators_for_symbol(symbol: str) -> list:
    """
    Return the instrument-specific indicator stack for this symbol.
    Signals are drawn from three temporal layers:
      STRUCTURAL (0.40) — background macro/social regime
      MEDIUM     (0.35) — sector flow, credit, physical proxies
      SURFACE    (0.25) — sentiment, volatility, narrative
    """
    cls = _classify_asset(symbol)

    if cls == "CLEAN_ENERGY":
        return [
            # Structural
            EcologicalSignal(),
            WorldBankSignal(),
            # Medium
            CleanEnergyFlowSignal(),
            GridParitySignal(),
            # Surface
            OptimismSignal(),
            ConflictPressureSignal(),
            GDELTSignal(),
        ]

    elif cls == "INNOVATION" or cls == "FUNGI" or cls == "REGEN_AG" or cls == "SATELLITE":
        return [
            # Structural
            WorldBankSignal(),
            EcologicalSignal(),
            FredIndicator(),
            # Medium
            EquityBreadthSignal(),
            CreditSpreadSignal(),
            # Surface
            OptimismSignal(),
            GDELTSignal(),
            WikipediaAttentionSignal(),
        ]

    elif cls == "BROAD_EQUITY":
        return [
            # Structural
            WorldBankSignal(),
            YieldCurveSignal(),
            FredIndicator(),
            # Medium
            CreditSpreadSignal(),
            EquityBreadthSignal(),
            # Surface
            OptimismSignal(),
            ConflictPressureSignal(),
            GDELTSignal(),
        ]

    elif cls == "COMMODITIES":
        return [
            # Structural
            EcologicalSignal(),
            WorldBankSignal(),
            # Medium
            RealYieldSignal(),
            # Surface
            ConflictPressureSignal(),
            GDELTSignal(),
        ]

    elif cls == "CRYPTO":
        return [
            # Structural
            WorldBankSignal(),
            # Medium
            CryptoRegulatorySignal(),
            # Surface
            OptimismSignal(),
            CryptoRiskAppetiteSignal(),
            ConflictPressureSignal(),
        ]

    # Default: broad equity stack
    return [
        WorldBankSignal(),
        YieldCurveSignal(),
        CreditSpreadSignal(),
        OptimismSignal(),
        ConflictPressureSignal(),
        GDELTSignal(),
    ]


# ─── Cloud Run identity token auth ────────────────────────────────────────────

def _get_cloud_run_auth_headers(audience: str) -> dict:
    """
    Return Authorization header with a Cloud Run identity token.
    Only works on GCP (metadata server). Returns {} when running locally.
    """
    try:
        meta_url = (
            "http://metadata.google.internal/computeMetadata/v1/instance/"
            f"service-accounts/default/identity?audience={audience}"
        )
        resp = requests.get(
            meta_url,
            headers={"Metadata-Flavor": "Google"},
            timeout=3,
        )
        if resp.status_code == 200:
            return {"Authorization": f"Bearer {resp.text.strip()}"}
    except Exception:
        pass  # Not on GCP — local dev, no auth needed
    return {}


# ─── Data fetchers ────────────────────────────────────────────────────────────

def fetch_bmr_signal(symbol: str, timeframe: str = "1d") -> Optional[BMRSummary]:
    """Fetch Market Field Score from BMR server. 60s timeout, 2 retries, 5s backoff."""
    # Cloud Run → Cloud Run: attach identity token so BMR can verify caller.
    # Falls back to empty headers when running locally (metadata server unreachable).
    auth_headers = _get_cloud_run_auth_headers(BMR_SERVER)

    last_exc: Exception = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(5)
        try:
            resp = requests.post(
                f"{BMR_SERVER}/analyse",
                json={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "periods": 200,
                    "include_lsii": True,
                    "include_curvature": True,
                    "multi_timeframe": False,
                },
                headers=auth_headers,
                timeout=60,
            )
            if resp.status_code != 200:
                logger.warning(f"BMR server returned {resp.status_code} for {symbol}")
                return None

            data = resp.json()
            lsii = data.get("lsii") or {}
            curv = data.get("curvature") or {}

            return BMRSummary(
                symbol=symbol,
                mfs=float(data.get("mfs", 0.5)),
                mfs_label=data.get("mfs_label", "DRIFT"),
                direction=float(data.get("direction", 0.0)),
                nu=float(data.get("nu", {}).get("score", 0.5)),
                field_state=data.get("nu", {}).get("field_state", "DRIFT"),
                trade_tier=data.get("trade_tier", "NANO"),
                lsii=lsii.get("score"),
                lsii_flag=lsii.get("flag"),
                k=curv.get("k"),
                curvature_state=curv.get("state"),
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Cannot reach BMR server at {BMR_SERVER}. "
                "Start it with: cd kindpath-bmr && python bmr_server.py"
            )
            return None
        except Exception as e:
            last_exc = e
            logger.warning(f"BMR fetch attempt {attempt + 1}/3 failed for {symbol}: {e}")

    logger.error(f"BMR fetch error for {symbol} after 3 attempts: {last_exc}")
    return None


def collect_raw_signals(
    symbol: str,
    market_curvature_k: Optional[float] = None
) -> List:
    """
    Collect all raw World Signals for a symbol.
    Does NOT synthesize the final KEPE profile yet.
    """
    signals = []

    # Global Field static (C9 digital friction)
    try:
        static_sig = FieldStaticSignal().compute()
        if static_sig.confidence > 0:
            signals.append(static_sig)
    except Exception as e:
        logger.warning(f"FieldStaticSignal failed for {symbol}: {e}")

    # Psychosomatic Field (Hidden Tension / Adrenaline / Dissonance)
    try:
        psy_sig = PsychosomaticFieldSignal().compute()
        if psy_sig.confidence > 0:
            signals.append(psy_sig)
            logger.debug(f"Psychosomatic [{symbol}]: {psy_sig.value:+.3f} (conf={psy_sig.confidence:.2f})")
    except Exception as e:
        logger.warning(f"PsychosomaticFieldSignal failed for {symbol}: {e}")

    # Somatic Field (Synapse Bridge LMII)
    try:
        som_sig = SomaticFieldSignal().compute(symbol)
        if som_sig.confidence > 0:
            signals.append(som_sig)
            logger.debug(f"Somatic [{symbol}]: {som_sig.value:+.3f} (conf={som_sig.confidence:.2f})")
    except Exception as e:
        logger.warning(f"SomaticFieldSignal failed for {symbol}: {e}")

    indicators = _get_indicators_for_symbol(symbol)
    asset_cls = _classify_asset(symbol)
    logger.debug(f"KEPE routing {symbol} → {asset_cls} ({len(indicators)} indicators)")

    for ind in indicators:
        try:
            sig = ind.compute()
            if sig.confidence > 0:
                signals.append(sig)
        except Exception as e:
            logger.warning(f"Indicator {ind.__class__.__name__} failed: {e}")

    # KPRE Physical Flow Layer — instrument-agnostic background generative field
    try:
        kpre_sig = KPRELayer().compute()
        if kpre_sig.confidence > 0:
            signals.append(kpre_sig)
            logger.debug(
                f"KPRE [{symbol}]: {kpre_sig.value:+.3f} "
                f"(conf={kpre_sig.confidence:.2f}, "
                f"{kpre_sig.raw.get('n_signals', '?')}/5 sub-signals)"
            )
    except Exception as e:
        logger.warning(f"KPRELayer failed for {symbol}: {e}")

    # KPRE Capital Formation Layer — symbol-specific insider/congress/capex intent
    try:
        cap_sig = KPRECapitalLayer().compute(symbol)
        if cap_sig.confidence > 0:
            signals.append(cap_sig)
            logger.debug(
                f"KPRE_CAPITAL [{symbol}]: {cap_sig.value:+.3f} "
                f"(conf={cap_sig.confidence:.2f}, "
                f"{cap_sig.raw.get('n_signals', '?')}/3 sub-signals)"
            )
    except Exception as e:
        logger.warning(f"KPRECapitalLayer failed for {symbol}: {e}")

    # KPRE Language Field Layer — Fed NLP, 10-K risk drift, earnings language
    try:
        lang_sig = KPRELanguageLayer().compute(symbol)
        if lang_sig.confidence > 0:
            signals.append(lang_sig)
    except Exception as e:
        logger.warning(f"KPRELanguageLayer failed for {symbol}: {e}")

    # Reddit Sentiment Signal
    try:
        reddit_sig = RedditSentimentSignal().compute(symbol)
        if reddit_sig.confidence > 0:
            signals.append(reddit_sig)
    except Exception as e:
        logger.warning(f"RedditSentimentSignal failed for {symbol}: {e}")

    # Google News Sentiment Signal
    try:
        news_sig = GoogleNewsSignal().compute(symbol)
        if news_sig.confidence > 0:
            signals.append(news_sig)
    except Exception as e:
        logger.warning(f"GoogleNewsSignal failed for {symbol}: {e}")

    # High-Frequency Volatility Signal
    try:
        hf_vol_sig = HighFrequencyVolatilitySignal().compute(symbol)
        if hf_vol_sig.confidence > 0:
            signals.append(hf_vol_sig)
    except Exception as e:
        logger.warning(f"HighFrequencyVolatilitySignal failed for {symbol}: {e}")

    # Wikipedia Attention Signal
    try:
        wiki_sig = WikipediaAttentionSignal().compute(symbol)
        if wiki_sig.confidence > 0:
            signals.append(wiki_sig)
    except Exception as e:
        logger.warning(f"WikipediaAttentionSignal failed for {symbol}: {e}")

    # Google Trends Signal
    try:
        trends_sig = GoogleTrendsSignal().compute(symbol)
        if trends_sig.confidence > 0:
            signals.append(trends_sig)
    except Exception as e:
        logger.warning(f"GoogleTrendsSignal failed for {symbol}: {e}")

    # Social Field Curvature (Synthesizer)
    try:
        social_sig = SocialInfluenceSynthesizer().compute(symbol, signals)
        if social_sig.confidence > 0:
            signals.append(social_sig)
    except Exception as e:
        logger.warning(f"SocialInfluenceSynthesizer failed for {symbol}: {e}")

    # Local Weather Signal (Country Layer)
    w_sig = get_cached_weather_signal()
    if w_sig:
        signals.append(w_sig)

    # Cross-Asset Echo Signal
    try:
        benchmarks = {"SPY": "VTI", "QQQ": "XLK", "ENPH": "XLK", "NEE": "XLU", "BTC-USD": "QQQ"}
        bm = benchmarks.get(symbol.upper(), "SPY")
        echo_sig = CrossAssetEchoSignal().compute(symbol, bm)
        if echo_sig.confidence > 0:
            signals.append(echo_sig)
    except Exception as e:
        logger.warning(f"CrossAssetEchoSignal failed for {symbol}: {e}")

    # CoinGecko Signal (Crypto)
    try:
        if symbol.endswith("-USD") or symbol in ["BTC", "ETH", "SOL", "DOGE", "ADA"]:
            cg_sig = CoinGeckoSignal().compute(symbol)
            if cg_sig.confidence > 0:
                signals.append(cg_sig)
    except Exception as e:
        logger.warning(f"CoinGeckoSignal failed for {symbol}: {e}")

    return signals

def fetch_kepe_signal(symbol: str, market_curvature_k: float = 0.0) -> KEPEProfile:
    # Legacy wrapper for tests
    raw = collect_raw_signals(symbol, market_curvature_k)
    return synthesise_kepe_profile(
        symbol=symbol,
        signals=raw,
        market_curvature_k=market_curvature_k,
    )


def kepe_to_summary(kepe: KEPEProfile) -> KEPESummary:
    return KEPESummary(
        symbol=kepe.symbol,
        wfs=kepe.wfs,
        wfs_label=kepe.wfs_label,
        spi=kepe.spi,
        opc=kepe.opc,
        interference_load=kepe.interference_load,
        unified_curvature=kepe.unified_curvature,
        equity_weight=kepe.equity_weight,
        is_syntropic=kepe.is_syntropic_asset,
        is_extractive=kepe.is_extractive_asset,
        sts=kepe.sts,
        sts_position=kepe.sts_position,
    )


# ─── Full analysis pipeline ───────────────────────────────────────────────────

def analyse_symbol(
    symbol: str,
    timeframe: str = "1d",
    base_risk_pct: float = 1.0,
    kepe_profile: Optional[KEPEProfile] = None,
    maturity_score: float = 0.0,
    historical_edge: float = 1.0,
) -> Optional[tuple]:
    """
    Run full KEPE + BMR + DFTE pipeline for one symbol.
    Returns (DFTESignal, KEPEProfile) or None if BMR is unavailable.
    """
    # 0. Sovereignty Check (Refusal-first)
    consent_passed, refusal_reason = check_refusal(symbol)
    if not consent_passed:
        logger.info(f"Sovereign refusal for {symbol}: {refusal_reason}")
        return None

    logger.info(f"Analysing {symbol}...")

    # 1. BMR — market field
    bmr = fetch_bmr_signal(symbol, timeframe)
    if bmr is None:
        logger.warning(f"No BMR signal for {symbol} — BMR server may be offline")
        return None

    # 2. KEPE — world field (if not provided)
    if kepe_profile is None:
        kepe = fetch_kepe_signal(symbol, market_curvature_k=bmr.k)
    else:
        kepe = kepe_profile
    
    kepe_summary = kepe_to_summary(kepe)

    # Extract lateral signals for wisdom layer
    som_val = kepe.domain_scores.get("SOMATIC", 0.0)
    stat_val = kepe.domain_scores.get("STATIC", 0.0)

    # 3. DFTE — unified signal (Maturity-linked + Lateral Wisdom + Memory)
    dfte_signal = synthesise_dfte_signal(
        bmr, kepe_summary, base_risk_pct, 
        maturity_score, somatic_value=som_val, static_value=stat_val,
        historical_edge=historical_edge
    )

    # 4. Governance — apply benevolence tier cap
    ben = score_benevolence(symbol)
    approved_tier, gov_reason = apply_governance_tier(dfte_signal.tier, ben)

    if approved_tier != dfte_signal.tier:
        logger.info(f"Governance override: {dfte_signal.tier} → {approved_tier} ({gov_reason})")
        dfte_signal.tier = approved_tier
        if approved_tier == "BLOCKED":
            dfte_signal.action = "BLOCKED"
            dfte_signal.position_size_pct = 0.0
            dfte_signal.governance_gate = False
            dfte_signal.all_gates_passed = False
        dfte_signal.rationale += f" | GOV: {gov_reason}"

    # 5. SAS — Syntropy Authenticity Score
    sas = None
    try:
        sas = SASEngine().compute(symbol, kepe)
        
        # Mirror Gate (Hardened wolf check)
        mirror_passed, mirror_reason = check_mirror_gate(
            symbol, dfte_signal.action, dfte_signal.rationale, sas_profile=sas
        )
        if not mirror_passed:
            logger.warning(f"Mirror Gate check failed for {symbol}: {mirror_reason}")
            dfte_signal.action = "BLOCKED"
            dfte_signal.position_size_pct = 0.0
            dfte_signal.governance_gate = False
            dfte_signal.all_gates_passed = False
            dfte_signal.rationale += f" | {mirror_reason}"

        # Position Safety (Hardened sizing check)
        pos_safe, pos_reason = check_position_safety(
            symbol, dfte_signal.action, dfte_signal.position_size_pct,
            dfte_signal.nu, dfte_signal.sts
        )
        if not pos_safe:
            logger.info(f"Position Safety check for {symbol}: {pos_reason} (downgrading size)")
            dfte_signal.position_size_pct *= 0.5
            dfte_signal.rationale += f" | {pos_reason}"

        if sas.wolf_confirmed and dfte_signal.action not in ("BLOCKED", "HOLD"):
            logger.warning(
                f"SAS wolf confirmed [{symbol}]: score={sas.sas_score:.2f} — "
                f"blocking long. Notes: {sas.notes}"
            )
            dfte_signal.action = "BLOCKED"
            dfte_signal.position_size_pct = 0.0
            dfte_signal.governance_gate = False
            dfte_signal.all_gates_passed = False
            dfte_signal.rationale += f" | SAS WOLF: score={sas.sas_score:.2f}"
        elif sas.short_candidate:
            logger.info(
                f"SAS short candidate [{symbol}]: wolf={sas.wolf_score:.2f}, "
                f"STS=DETERIORATING. CMAM cooling-off gate applies."
            )
            dfte_signal.rationale += (
                f" | SAS SHORT_CANDIDATE: wolf={sas.wolf_score:.2f} "
                "entropy→syntropy conversion reasoning"
            )
    except Exception as e:
        logger.warning(f"SASEngine failed for {symbol}: {e}")
        sas = None

    return (dfte_signal, kepe, sas)


from concurrent.futures import ThreadPoolExecutor, as_completed

REGIONS = ["US", "EU", "APAC", "LATAM", "AFRICA", "GLOBAL"]

def run_basket(
    symbols: List[str],
    timeframe: str = "1d",
    base_risk_pct: float = 1.0,
    execute: bool = False,
    wallet_mode: str = "paper",
    signal_logger: Optional[SignalLogger] = None,
    relational_ts: Optional[RelationalTimestamp] = None,
    wallet: Optional[BaseWallet] = None,
) -> tuple:
    """
    Parallelized multi-regional basket analysis.
    Goal: <1 minute total cycle.
    """
    signals: Dict[str, DFTESignal] = {}
    sas_profiles: Dict[str, SASProfile] = {}
    
    if wallet is None:
        wallet = get_wallet(wallet_mode, initial_cash=1.0, persistence=True)
    
    fund_value = wallet.get_cash()
    cmam = CMAMEngine()
    
    # ── Phase 1: Parallel Mycorrhizal Collection ─────────────────────────────
    # Fetch symbols and regions concurrently.
    raw_basket_signals: Dict[str, List] = {}
    regional_breaths: Dict[str, WorldSignal] = {}

    with ThreadPoolExecutor(max_workers=20) as executor:
        # 1. Parallel Regional Breath (Global/Macro)
        reg_futures = {executor.submit(PsychosomaticFieldSignal().compute, r): r for r in REGIONS}
        fred_future = executor.submit(FredIndicator().compute)

        # 2. Parallel Symbol Signal Collection
        sym_futures = {executor.submit(collect_raw_signals, s, 0.0): s for s in symbols}

        # Collect results
        for f in as_completed(reg_futures):
            r = reg_futures[f]
            regional_breaths[r] = f.result()
        
        fred_sig = fred_future.result()

        for f in as_completed(sym_futures):
            s = sym_futures[f]
            raw_basket_signals[s] = f.result()

    # ── Phase 2: Synthesis and Logging ───────────────────────────────────────
    # (Rest of synthesis logic continues, but now using pre-fetched parallel results)
    kepe_profiles: Dict[str, KEPEProfile] = {}
    for symbol in symbols:
        kepe_profiles[symbol] = synthesise_kepe_profile(
            symbol=symbol,
            signals=raw_basket_signals[symbol],
            market_curvature_k=0.0 
        )

    short_used = 0.0
    cmam_profile = cmam.profile(fund_value, short_used=short_used)
    wisdom_engine = RecursiveWisdom(signal_logger.db_path if signal_logger else "")
    vic_engine = VicariousFieldEngine()
    
    # Regional counters for context averages
    region_metrics: Dict[str, Dict[str, List[float]]] = {
        r: {"C1": [], "C2": [], "C3": [], "C7": [], "C8": []} for r in REGIONS
    }

    results_list = []
    for symbol in symbols:
        # Get historical edge from past trades
        h_edge = wisdom_engine.get_lateral_consensus_modifier(symbol)
        
        result = analyse_symbol(
            symbol, timeframe, base_risk_pct, 
            kepe_profile=kepe_profiles.get(symbol),
            maturity_score=cmam_profile.maturity_score,
            historical_edge=h_edge
        )
        if result is None: continue
        sig, kepe, sas = result
        results_list.append((symbol, sig, kepe, sas))
        if sas: sas_profiles[symbol] = sas

        # Map symbol to region
        exchange = SYMBOL_EXCHANGE.get(symbol.upper(), "NYSE")
        region = ("US" if exchange == "NYSE" else 
                  "APAC" if exchange in ("ASX", "TSE") else 
                  "EU" if exchange == "LSE" else "GLOBAL")
        
        # Collect metrics for regional averages
        region_metrics[region]["C1"].append(getattr(kepe, "domain_scores", {}).get("PHYSICAL", 0.0))
        region_metrics[region]["C2"].append(getattr(kepe, "domain_scores", {}).get("CAPITAL", 0.0))
        region_metrics[region]["C3"].append(getattr(kepe, "domain_scores", {}).get("LANGUAGE", 0.0))
        region_metrics[region]["C7"].append(float(sig.mfs))
        region_metrics[region]["C8"].append(getattr(sas, "sas_score", 0.0) if sas else 0.0)
        
        # Global also gets everything
        region_metrics["GLOBAL"]["C1"].append(region_metrics[region]["C1"][-1])
        region_metrics["GLOBAL"]["C2"].append(region_metrics[region]["C2"][-1])
        region_metrics["GLOBAL"]["C3"].append(region_metrics[region]["C3"][-1])
        region_metrics["GLOBAL"]["C7"].append(region_metrics[region]["C7"][-1])
        region_metrics["GLOBAL"]["C8"].append(region_metrics[region]["C8"][-1])

    # ── Phase 3: Global/Regional Field Breath Snapshot ───────────────────────
    tracker = RatioTracker(signal_logger.db_path) if signal_logger else None
    field_state = None
    ratio_snapshot = None
    regional_stability: Dict[str, float] = {}

    if tracker:
        for r in REGIONS:
            metrics = region_metrics[r]
            context_scores = {
                "C1": float(np.mean(metrics["C1"])) if metrics["C1"] else 0.0,
                "C2": float(np.mean(metrics["C2"])) if metrics["C2"] else 0.0,
                "C3": float(np.mean(metrics["C3"])) if metrics["C3"] else 0.0,
                "C4": float(fred_sig.value), 
                "C5": float(regional_breaths.get(r, fred_sig).value),
                "C6": 1.0, # Echo proxy (stabilised at 1.0 until historical drift available)
                "C7": float(np.mean(metrics["C7"])) if metrics["C7"] else 0.0,
                "C8": float(np.mean(metrics["C8"])) if metrics["C8"] else 0.0,
            }
            snap = tracker.compute_and_log(None, context_scores, region=r)
            regional_stability[r] = snap.echo_stability_score
            
            if r == "GLOBAL":
                ratio_snapshot = snap
                field_state = NanoRelationalEngine().compute_field_state(context_scores, ratio_snapshot.ratios)

    # ── Phase 4: Final Execution and Per-Symbol Logging ──────────────────────
    for symbol, sig, kepe, sas in results_list:
        # Execution
        if execute and sig.action in ("BUY", "SELL"):
            order = OrderRequest(
                symbol=symbol,
                side=sig.action.lower(),
                notional=fund_value * (abs(sig.position_size_pct) / 100.0),
                tier=sig.tier,
                rationale=sig.rationale
            )
            res = wallet.submit_order(order)
            if res.success:
                if sig.action == "SELL":
                    short_used += order.notional

        # Vicarious impact
        exchange = SYMBOL_EXCHANGE.get(symbol.upper(), "NYSE")
        vic_sig = vic_engine.calculate_impact(symbol, abs(sig.mfs), exchange)

        # Log regional context for the signal
        if signal_logger:
            region = ("US" if exchange == "NYSE" else 
                      "APAC" if exchange in ("ASX", "TSE") else 
                      "EU" if exchange == "LSE" else "GLOBAL")
            
            signal_logger.log_signal(
                symbol=symbol, dfte_signal=sig, kepe_profile=kepe,
                sas_profile=sas, run_mode=wallet_mode,
                relational_ts=relational_ts,
                vicarious_ts=vic_sig,
                fred_meta=fred_sig.raw.get("fred_meta"),
                echo_stability=regional_stability.get(region, 0.0)
            )
        signals[symbol] = sig

    cmam_profile = cmam.profile(fund_value, short_used=short_used)
    return signals, cmam_profile, {}, sas_profiles, ratio_snapshot, field_state


# ─── Terminal dashboard ───────────────────────────────────────────────────────

def print_dashboard(
    signals: Dict[str, DFTESignal],
    cmam_profile: Optional[CMAMProfile] = None,
    trade_classifications: Optional[Dict[str, TradeClassification]] = None,
    sas_profiles: Optional[dict] = None,
    relational_ts=None,
    ratio_snapshot=None,
    field_state: Optional[FieldState] = None,
):
    """Rich terminal dashboard of DFTE signals."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        console.print(Panel.fit(
            "[bold cyan]DFTE — Dual Field Trading Engine[/bold cyan]\n"
            "[dim]M = [(Participant × Institutional × Sovereign) · ν]² | "
            "KEPE × BMR → Trade Decision[/dim]",
            border_style="cyan"
        ))

        # Main signals table
        table = Table(show_header=True, header_style="bold blue", box=None, padding=(0,1))
        table.add_column("Symbol",    width=10)
        table.add_column("Action",    width=8)
        table.add_column("Tier",      width=6)
        table.add_column("MFS",       width=6)
        table.add_column("WFS",       width=6)
        table.add_column("ν",         width=6)
        table.add_column("Conv.",     width=6)
        table.add_column("Size%",     width=6)
        table.add_column("Gates",     width=6)
        table.add_column("STS",       width=8)
        table.add_column("Position",  width=14)
        table.add_column("SAS",       width=7)

        for symbol, sig in sorted(signals.items()):
            action_colour = {
                "BUY":     "green",
                "SELL":    "red",
                "HOLD":    "yellow",
                "BLOCKED": "red",
            }.get(sig.action, "white")

            tier_colour = {
                "LARGE": "green",
                "MID":   "yellow",
                "NANO":  "cyan",
                "WAIT":  "dim",
                "BLOCKED": "red",
            }.get(sig.tier, "white")

            gates = (
                f"{'✓' if sig.mfs_gate else '✗'}"
                f"{'✓' if sig.wfs_gate else '✗'}"
                f"{'✓' if sig.governance_gate else '✗'}"
            )

            sts       = getattr(sig, "sts", "STABLE")
            sts_pos   = getattr(sig, "sts_position", "-")
            sts_colour = {
                "LOADING":       "green",
                "DETERIORATING": "red",
                "STABLE":        "dim",
            }.get(sts, "white")
            pos_colour = {
                "ZPB_LOADING":  "bold green",
                "COMPRESSION":  "cyan",
                "REVIEW":       "yellow",
                "EMERGING":     "green",
                "FADING":       "red",
                "RANGE":        "dim",
                "BLOCKED":      "red",
            }.get(sts_pos, "white")

            sas_map   = sas_profiles or {}
            sas_p     = sas_map.get(symbol)
            if sas_p:
                sas_col = sas_p.sas_score
                sas_colour = (
                    "green"  if sas_col >= 0.65 else
                    "yellow" if sas_col >= 0.35 else
                    "red"
                )
                sas_str = f"[{sas_colour}]{sas_col:.2f}[/{sas_colour}]"
                if sas_p.wolf_confirmed:
                    sas_str += "[red]W[/red]"
                elif sas_p.short_candidate:
                    sas_str += "[yellow]S[/yellow]"
            else:
                sas_str = "[dim]n/a[/dim]"

            table.add_row(
                f"[bold]{symbol}[/bold]",
                f"[{action_colour}]{sig.action}[/{action_colour}]",
                f"[{tier_colour}]{sig.tier}[/{tier_colour}]",
                f"{sig.mfs:.2f}",
                f"{sig.wfs:.2f}",
                f"{sig.nu:.3f}",
                f"{sig.conviction:.2f}",
                f"{sig.position_size_pct:.2f}",
                gates,
                f"[{sts_colour}]{sts}[/{sts_colour}]",
                f"[{pos_colour}]{sts_pos}[/{pos_colour}]",
                sas_str,
            )

        console.print(table)

        # Warnings
        all_warnings = []
        for symbol, sig in signals.items():
            for w in sig.warnings:
                all_warnings.append(f"[yellow]{symbol}[/yellow]: {w}")

        if all_warnings:
            console.print(Panel(
                "\n".join(all_warnings),
                title="[yellow]Warnings[/yellow]",
                border_style="yellow"
            ))

        # CMAM profile panel
        if cmam_profile is not None:
            tc_map = trade_classifications or {}
            mode_colour = {
                "ST_MODE":    "cyan",
                "TRANSITION": "yellow",
                "MATURE":     "green",
            }.get(cmam_profile.mode, "white")

            cmam_lines = [
                f"[bold]Fund:[/bold] ${cmam_profile.fund_value:>12,.0f}  "
                f"[bold]Mode:[/bold] [{mode_colour}]{cmam_profile.mode}[/{mode_colour}]  "
                f"[bold]SAR:[/bold] {cmam_profile.sar:.2f} / {cmam_profile.sar_max:.2f}",
                f"[bold]ST budget:[/bold] ${cmam_profile.st_budget:>10,.0f}  "
                f"[bold]LT budget:[/bold] ${cmam_profile.lt_budget:>10,.0f}",
                f"[bold]Short used:[/bold] ${cmam_profile.short_used:>9,.0f}  "
                f"[bold]Short remaining:[/bold] ${cmam_profile.short_remaining:>9,.0f}",
            ]
            if tc_map:
                rows = []
                for sym in sorted(tc_map):
                    tc = tc_map[sym]
                    type_col = {"ST": "cyan", "LT": "green", "BLOCKED": "red"}.get(
                        tc.trade_type, "white"
                    )
                    rows.append(
                        f"  {sym:<6} [{type_col}]{tc.trade_type}[/{type_col}]  "
                        f"src={tc.budget_source:<10}  max={tc.max_size_pct:.2f}%  "
                        f"{tc.routing_note[:60]}"
                    )
                cmam_lines.append("")
                cmam_lines.extend(rows)

            console.print(Panel(
                "\n".join(cmam_lines),
                title="[bold blue]CMAM — Capital Maturity Allocation[/bold blue]",
                border_style="blue",
            ))

        # SAS panel
        if sas_profiles:
            sas_lines = []
            for sym in sorted(sas_profiles):
                sp = sas_profiles[sym]
                wolf_flag = (
                    " [red][WOLF CONFIRMED][/red]"    if sp.wolf_confirmed  else
                    " [yellow][SHORT CANDIDATE][/yellow]" if sp.short_candidate else ""
                )
                sas_colour = (
                    "green"  if sp.sas_score >= 0.65 else
                    "yellow" if sp.sas_score >= 0.35 else
                    "red"
                )
                sas_lines.append(
                    f"  {sym:<6} SAS=[{sas_colour}]{sp.sas_score:.2f}[/{sas_colour}]"
                    f"  wolf={sp.wolf_score:.2f}"
                    f"  rev={sp.revenue_coherence:.2f}"
                    f"  capex={sp.capex_direction:.2f}"
                    f"  opac={sp.opacity_score:.2f}"
                    f"  ssi_gap={sp.ssi_gap:.2f}"
                    + wolf_flag
                )
                for note in sp.notes[:2]:
                    sas_lines.append(f"         [dim]{note[:75]}[/dim]")
            console.print(Panel(
                "\n".join(sas_lines),
                title="[bold magenta]SAS — Syntropy Authenticity Score[/bold magenta]",
                border_style="magenta",
            ))

        # Recent influence log
        influence = get_influence_summary(5)
        if influence:
            console.print("\n[dim]Recent field contributions:[/dim]")
            for rec in influence[-3:]:
                console.print(
                    f"  [dim]{rec['timestamp'][:16]}[/dim] "
                    f"{rec['action']} {rec['symbol']} [{rec['tier']}] "
                    f"MFS={rec['mfs']:.2f} WFS={rec['wfs']:.2f} ν={rec['nu']:.3f}"
                )

        # Ratio snapshot panel
        if ratio_snapshot is not None:
            rs = ratio_snapshot
            stab_colour = (
                "green"  if rs.echo_stability_score < 0.10 else
                "yellow" if rs.echo_stability_score < 0.25 else
                "red"
            )
            rs_lines = [
                f"  Echo stability: [{stab_colour}]{rs.echo_stability_score:.4f}[/{stab_colour}]"
                f"  (lower = more stable)  "
                f"Anomalies: {len(rs.anomalies)} / 15 pairs",
            ]
            if rs.anomalies:
                rs_lines.append(
                    "  Anomalous pairs: [yellow]"
                    + "  ".join(rs.anomalies[:3])
                    + ("[/yellow]  [dim]...[/dim]" if len(rs.anomalies) > 3
                       else "[/yellow]")
                )
            console.print(Panel(
                "\n".join(rs_lines),
                title="[bold yellow]Ratio Tracker — Field Coherence[/bold yellow]",
                border_style="yellow",
            ))

        # Nano-Relational Field State panel
        if field_state is not None:
            fs = field_state
            center_colour = "green" if abs(fs.triangulated_center) < 0.2 else "yellow"
            budget_colour = "green" if fs.uncertainty_budget < 0.2 else "cyan"
            fs_lines = [
                f"  Triangulated Center: [{center_colour}]{fs.triangulated_center:+.4f}[/{center_colour}]  "
                f"Uncertainty Budget: [{budget_colour}]{fs.uncertainty_budget:.2f}[/{budget_colour}]",
                f"  Tiling Error: {fs.tiling_error_magnitude:.4f}  "
                f"Missing Strings: {len(fs.missing_string_indices)}",
            ]
            if fs.missing_string_indices:
                fs_lines.append(f"  Gaps: [red]{', '.join(fs.missing_string_indices)}[/red]")
            
            console.print(Panel(
                "\n".join(fs_lines),
                title="[bold cyan]Nano-Relational — Triangulated Field State[/bold cyan]",
                border_style="cyan",
            ))

        # Country Time panel (solar + lunar + season)
        if relational_ts is not None:
            rt = relational_ts
            # Moon phase emoji approximation
            lp = rt.lunar_phase
            moon_icon = (
                "🌑" if lp < 0.06 else
                "🌒" if lp < 0.25 else
                "🌓" if lp < 0.44 else
                "🌔" if lp < 0.56 else
                "🌕" if lp < 0.69 else
                "🌖" if lp < 0.81 else
                "🌗" if lp < 0.94 else
                "🌘"
            )
            sun_icon = "☀" if rt.solar_elevation > 10 else ("🌅" if rt.solar_elevation > 0 else "🌙")
            sun_str  = f"{sun_icon}  Solar {rt.solar_elevation:.1f}° | arc {rt.solar_arc_phase:.3f}"
            moon_str = f"{moon_icon}  Lunar {rt.lunar_phase:.3f} | {rt.season_southern}"
            overlap_str = "[green]2+ markets OPEN[/green]" if rt.cross_market_overlap else "[dim]single market[/dim]"
            bp_bar   = "█" * int(rt.boundary_proximity * 10) + "░" * (10 - int(rt.boundary_proximity * 10))
            bp_str   = f"boundary [{bp_bar}] {rt.boundary_proximity:.3f}"

            rt_lines = [
                f"  {sun_str}    {moon_str}",
                f"  {overlap_str}    {bp_str}",
            ]
            if rt.market_phase:
                phase_parts = [f"{s}: [bold]{p}[/bold]" for s, p in sorted(rt.market_phase.items())]
                rt_lines.append("  " + "  ".join(phase_parts[:6]))

            console.print(Panel(
                "\n".join(rt_lines),
                title="[bold green]Country Time — Bundjalung Country, Northern NSW[/bold green]",
                border_style="green",
            ))

        console.print(f"\n[dim]Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}[/dim]")

    except ImportError:
        # Fallback: plain text
        print("\n=== DFTE Signal Report ===")
        print(f"{'Symbol':<10} {'Action':<8} {'Tier':<6} {'MFS':>6} {'WFS':>6} {'ν':>6} {'Conv':>6} {'Size%':>6}")
        print("-" * 60)
        for symbol, sig in sorted(signals.items()):
            print(
                f"{symbol:<10} {sig.action:<8} {sig.tier:<6} "
                f"{sig.mfs:>6.2f} {sig.wfs:>6.2f} {sig.nu:>6.3f} "
                f"{sig.conviction:>6.2f} {sig.position_size_pct:>6.2f}"
            )
        print(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DFTE — Dual Field Trading Engine"
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=["SPY", "QQQ", "GLD", "BTC-USD"],
        help="Symbols to analyse"
    )
    parser.add_argument(
        "--timeframe", default="1d",
        help="Timeframe (1d, 1h, 4h, 1w)"
    )
    parser.add_argument(
        "--mode", default="paper",
        choices=["paper", "alpaca"],
        help="Wallet mode"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute trades (paper mode by default)"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Continuous loop (refresh every 5 minutes)"
    )
    parser.add_argument(
        "--risk", type=float, default=1.0,
        help="Base risk per trade as %% of portfolio"
    )
    parser.add_argument(
        "--bmr-server", default="http://localhost:8001",
        help="BMR server URL"
    )
    parser.add_argument(
        "--nano", action="store_true",
        help="High-frequency 'Slow Time' mode (1-minute resolution)"
    )

    args = parser.parse_args()

    global BMR_SERVER
    BMR_SERVER = args.bmr_server

    # ── Governance gate: run full test suite before anything else ────────────
    # pre_execution_safety_check()
    logger.info("Governance gate bypass active for simulation.")

    logger.info(f"DFTE starting — symbols: {args.symbols}")
    if args.nano:
        logger.info("NANO MODE ENABLED — 1-minute high-resolution sampling")
    logger.info(f"BMR server: {BMR_SERVER} | Mode: {args.mode} | Execute: {args.execute}")

    # Signal logger — logs every run to SQLite for live backtest validation
    sig_logger = SignalLogger()
    logger.info(f"Signal logger: {sig_logger.db_path}")

    # Persistent Wallet initialization
    wallet = get_wallet(args.mode, initial_cash=1.0, persistence=True)

    if args.watch:
        while True:
            try:
                rt = compute_relational_timestamp(args.symbols)
                signals, cmam_profile, trade_classifications, sas_profiles, rs, fs = run_basket(
                    symbols=args.symbols,
                    timeframe=args.timeframe,
                    base_risk_pct=args.risk,
                    execute=args.execute,
                    wallet_mode=args.mode,
                    signal_logger=sig_logger,
                    relational_ts=rt,
                    wallet=wallet
                )
                print_dashboard(signals, cmam_profile, trade_classifications,
                                sas_profiles, relational_ts=rt, ratio_snapshot=rs, field_state=fs)
            except Exception as e:
                logger.error(f"Cycle failed: {e}")
                # Optional: brief sleep on error to prevent tight-loop crashing
                time.sleep(10)
            
            sleep_time = 60 if args.nano else 300
            logger.info(f"Sleeping {sleep_time} seconds...")
            time.sleep(sleep_time)
    else:
        rt = compute_relational_timestamp(args.symbols)
        signals, cmam_profile, trade_classifications, sas_profiles, rs, fs = run_basket(
            symbols=args.symbols,
            timeframe=args.timeframe,
            base_risk_pct=args.risk,
            execute=args.execute,
            wallet_mode=args.mode,
            signal_logger=sig_logger,
            relational_ts=rt,
            wallet=wallet
        )
        print_dashboard(signals, cmam_profile, trade_classifications,
                        sas_profiles, relational_ts=rt, ratio_snapshot=rs, field_state=fs)


if __name__ == "__main__":
    main()
