"""
Backtest Engine — Phase 7
==========================
Empirical validation of ν, LSII-Price, and STS theoretical claims.

KINDFIELD principle: we do not prove, we record.
Refuted results are as valuable as confirmed ones.
Mystery pile populated from inconclusive evidence.

Verdicts: CONFIRMED | PARTIAL | INCONCLUSIVE | REFUTED

ν proxy note:
  Live ν uses multi-source scale readings (World Bank, FRED, COT, etc.).
  For historical backtesting we use price-derived proxies:
    Participant:   short-term  momentum (5-day  return, normalised)
    Institutional: medium-term momentum (20-day return, normalised)
    Sovereign:     long-term   momentum (60-day return, normalised)
  This tests the price-embedded ν signal, not the full live signal.
  Evidence level: TESTABLE for ν proxy, SPECULATIVE for full-signal claims.
"""

from __future__ import annotations

import sys
import os
import json
import logging
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import yfinance as yf

# ── path setup so we can import BMR modules ─────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BMR  = os.path.join(_ROOT, "bmr")
for _p in [_ROOT, _BMR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.nu_engine       import compute_nu, NuResult, NU_THRESHOLDS
from core.lsii_price      import compute_lsii_price, LSII_THRESHOLDS
from core.normaliser      import ScaleReading
from feeds.feeds          import OHLCV

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Verdict constants ────────────────────────────────────────────────────────

CONFIRMED    = "CONFIRMED"
PARTIAL      = "PARTIAL"
INCONCLUSIVE = "INCONCLUSIVE"
REFUTED      = "REFUTED"

# Minimum observations needed to say anything useful
MIN_OBSERVATIONS = 15


# ─── Report dataclasses ───────────────────────────────────────────────────────

@dataclass
class NuValidationReport:
    """Backtest 1 — does ν predict directional magnitude?"""
    symbol: str
    n_observations: int
    # mean forward return per ν quartile across each forward window
    per_quartile_mean_return: Dict[str, Dict[str, float]]   # {window: {quartile: return}}
    correlation_nu_vs_return: Dict[str, float]               # {window: r}
    correlation_nu_vs_magnitude: Dict[str, float]            # {window: r} — magnitude only
    evidence_verdict: str
    evidence_level: str
    notes: str
    quartile_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class LSIIValidationReport:
    """Backtest 2 — does LSII high/very_high precede mean-reversion?"""
    symbol: str
    n_flagged: int
    n_baseline: int
    flagged_mean_return:   Dict[str, float]   # {window: return}
    baseline_mean_return:  Dict[str, float]
    flag_predictive_value: Dict[str, float]   # flagged − baseline (expect negative)
    optimal_threshold: str
    current_threshold_ok:  bool
    evidence_verdict: str
    evidence_level: str
    notes: str


@dataclass
class STSValidationReport:
    """Backtest 3 — does STS trajectory predict medium-term returns?"""
    symbol: str
    n_loading:      int
    n_stable:       int
    n_deteriorating: int
    loading_mean_return:      Dict[str, float]   # {window: return}
    stable_mean_return:       Dict[str, float]
    deteriorating_mean_return: Dict[str, float]
    loading_vs_deteriorating:  Dict[str, float]  # spread
    evidence_verdict: str
    evidence_level: str
    notes: str


@dataclass
class SASValidationReport:
    """
    Backtest 4 — does opacity/incoherence predict underperformance?
    [SPECULATIVE] — limited historical data available.
    """
    symbol: str
    evidence_verdict: str   = INCONCLUSIVE
    evidence_level:   str   = "SPECULATIVE"
    notes:            str   = (
        "SAS historical proxy unavailable — requires multi-year EDGAR NLP corpus. "
        "Backtest 4 intentionally sparse: [SPECULATIVE] per Phase 7 scope. "
        "Goes in mystery pile pending data sourcing."
    )
    mystery_pile_reason: str = "No reliable historical SAS proxy available"


@dataclass
class SymbolBacktestResult:
    """All four backtests for a single symbol."""
    symbol: str
    nu_report:   NuValidationReport
    lsii_report: LSIIValidationReport
    sts_report:  STSValidationReport
    sas_report:  SASValidationReport


@dataclass
class BacktestReport:
    """Full multi-symbol backtest report."""
    run_date: str
    symbols: List[str]
    per_symbol: List[SymbolBacktestResult]

    # Aggregated across all symbols
    nu_aggregate:   NuValidationReport
    lsii_aggregate: LSIIValidationReport
    sts_aggregate:  STSValidationReport

    overall_verdict:              str
    mystery_pile_items:           List[str]
    calibration_recommendations:  List[str]

    # Meta
    price_period_days: int    = 365
    wts_period_days:   int    = 504   # 2-year weekly for STS
    nu_proxy_note:     str    = (
        "ν computed from price-derived scale proxies only. "
        "Full live signal includes World Bank, FRED, COT, etc. "
        "Evidence level: TESTABLE for proxy, SPECULATIVE for full-signal claims."
    )


# ─── Price data fetcher ───────────────────────────────────────────────────────

def fetch_price_history(symbol: str, period: str = "2y",
                        interval: str = "1d") -> Optional[np.ndarray]:
    """
    Returns (N, 6) array: [open, high, low, close, volume, timestamp_ordinal]
    sorted oldest → newest, or None on failure.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for {symbol}")
            return None
        closes  = df["Close"].values.astype(float)
        highs   = df["High"].values.astype(float)
        lows    = df["Low"].values.astype(float)
        opens   = df["Open"].values.astype(float)
        vols    = df["Volume"].values.astype(float)
        ts      = np.array([t.timestamp() for t in df.index])
        return np.column_stack([opens, highs, lows, closes, vols, ts])
    except Exception as e:
        logger.warning(f"Failed to fetch {symbol}: {e}")
        return None


def _bars_from_array(arr: np.ndarray, symbol: str,
                     timeframe: str = "1d") -> List[OHLCV]:
    bars = []
    for row in arr:
        bars.append(OHLCV(
            timestamp=datetime.fromtimestamp(row[5], tz=timezone.utc),
            open=row[0], high=row[1], low=row[2],
            close=row[3], volume=row[4],
            symbol=symbol, timeframe=timeframe,
        ))
    return bars


# ─── ν proxy computation ──────────────────────────────────────────────────────

def _tanh_norm(x: float, scale: float = 0.10) -> float:
    """Normalise a return to [-1, +1] via tanh. scale=10% return → ~0.76."""
    return float(np.tanh(x / scale))


def _compute_nu_proxy(
    closes: np.ndarray,
    idx: int,
    short_w: int  = 5,
    mid_w:   int  = 20,
    long_w:  int  = 60,
) -> Optional[NuResult]:
    """
    Compute ν from price-only proxies at position idx.
    Returns None if insufficient history.
    """
    if idx < long_w:
        return None

    c = closes

    # Returns
    r_short = (c[idx] - c[idx - short_w]) / (c[idx - short_w] + 1e-10)
    r_mid   = (c[idx] - c[idx - mid_w])   / (c[idx - mid_w]   + 1e-10)
    r_long  = (c[idx] - c[idx - long_w])  / (c[idx - long_w]  + 1e-10)

    # Participant: short-term momentum + local volatility-adjusted confidence
    std_short = float(np.std(np.diff(c[idx - short_w:idx + 1]))) + 1e-10
    conf_short = float(np.clip(abs(r_short) / (3 * std_short / c[idx]), 0.1, 0.9))

    # Institutional: medium-term momentum
    std_mid = float(np.std(np.diff(c[idx - mid_w:idx + 1]))) + 1e-10
    conf_mid = float(np.clip(abs(r_mid) / (3 * std_mid / c[idx]), 0.1, 0.9))

    # Sovereign: long-term momentum
    std_long = float(np.std(np.diff(c[idx - long_w:idx + 1]))) + 1e-10
    conf_long = float(np.clip(abs(r_long) / (3 * std_long / c[idx]), 0.1, 0.9))

    participant   = ScaleReading("PARTICIPANT",   _tanh_norm(r_short), conf_short, 1)
    institutional = ScaleReading("INSTITUTIONAL", _tanh_norm(r_mid),   conf_mid,   1)
    sovereign     = ScaleReading("SOVEREIGN",     _tanh_norm(r_long),  conf_long,  1)

    return compute_nu(participant, institutional, sovereign)


# ─── STS proxy computation ────────────────────────────────────────────────────

def _compute_sts_proxy(
    closes: np.ndarray,
    idx: int,
    wfs_window: int = 20,
    slope_window: int = 10,
) -> str:
    """
    Derive STS (LOADING / STABLE / DETERIORATING) from rolling WFS proxy.
    WFS proxy = rank-normalised 20-day return relative to all days seen so far.
    STS = slope of WFS proxy over last `slope_window` days.
    """
    if idx < wfs_window + slope_window:
        return "STABLE"

    # Rolling WFS proxy: 20-day return at each point
    wfs_series = []
    for i in range(idx - slope_window, idx + 1):
        if i < wfs_window:
            wfs_series.append(0.5)
            continue
        r = (closes[i] - closes[i - wfs_window]) / (closes[i - wfs_window] + 1e-10)
        wfs_series.append(float(np.clip(0.5 + r * 2, 0.0, 1.0)))

    # Slope of WFS proxy over window
    xs = np.arange(len(wfs_series))
    slope = float(np.polyfit(xs, wfs_series, 1)[0])

    threshold = 0.003   # ~0.3% per day WFS change → state boundary
    if slope > threshold:
        return "LOADING"
    elif slope < -threshold:
        return "DETERIORATING"
    return "STABLE"


# ─── Forward return computation ───────────────────────────────────────────────

FORWARD_WINDOWS = [5, 10, 20, 60]


def _forward_return(closes: np.ndarray, idx: int, window: int) -> Optional[float]:
    """Return forward return at `window` bars, or None if out of bounds."""
    future_idx = idx + window
    if future_idx >= len(closes):
        return None
    return float((closes[future_idx] - closes[idx]) / (closes[idx] + 1e-10))


# ─── Statistical helpers ──────────────────────────────────────────────────────

def _pearson_r(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient. Returns 0 if insufficient data."""
    if len(xs) < 5:
        return 0.0
    xa, ya = np.array(xs), np.array(ys)
    if np.std(xa) < 1e-10 or np.std(ya) < 1e-10:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _t_stat(r: float, n: int) -> float:
    """t-statistic for Pearson r."""
    if n < 3 or abs(r) >= 1.0:
        return 0.0
    return r * math.sqrt(n - 2) / math.sqrt(1 - r**2 + 1e-12)


def _verdict_from_r_and_direction(
    r: float,
    n: int,
    expected_positive: bool,
) -> str:
    """
    Generate an evidence verdict from correlation coefficient.

    expected_positive: True if hypothesis predicts positive correlation.
    We never suppress a REFUTED result.
    """
    if n < MIN_OBSERVATIONS:
        return INCONCLUSIVE

    t = abs(_t_stat(r, n))
    direction_ok = (r > 0) == expected_positive

    if not direction_ok:
        # Hypothesis goes the wrong way
        if abs(r) > 0.15 and t > 1.5:
            return REFUTED
        return INCONCLUSIVE

    # Hypothesis direction correct — how strong?
    if abs(r) >= 0.25 and t >= 2.0:
        return CONFIRMED
    elif abs(r) >= 0.12 or t >= 1.5:
        return PARTIAL
    return INCONCLUSIVE


def _combine_verdicts(verdicts: List[str]) -> str:
    """Combine multiple verdicts — most conservative wins."""
    if not verdicts:
        return INCONCLUSIVE
    order = {CONFIRMED: 3, PARTIAL: 2, INCONCLUSIVE: 1, REFUTED: 0}
    return min(verdicts, key=lambda v: order.get(v, 1))


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


# ─── Backtest 1: ν predictive validity ───────────────────────────────────────

NU_QUARTILES = {
    "LOW":  (0.00, 0.35),
    "MID":  (0.35, 0.55),
    "HIGH": (0.55, 0.75),
    "ZPB":  (0.75, 1.01),
}


def _nu_quartile(nu: float) -> str:
    for label, (lo, hi) in NU_QUARTILES.items():
        if lo <= nu < hi:
            return label
    return "ZPB"


def run_nu_backtest(
    symbol: str,
    closes: np.ndarray,
) -> NuValidationReport:
    """
    Validate: does higher ν predict stronger return magnitude?
    (ν is coherence, not direction — so HIGH ν + direction → stronger move)
    """
    nu_values: List[float] = []
    directions: List[float] = []
    forward: Dict[int, List[float]] = {w: [] for w in FORWARD_WINDOWS}
    magnitudes: Dict[int, List[float]] = {w: [] for w in FORWARD_WINDOWS}
    quartile_returns: Dict[str, Dict[int, List[float]]] = {
        q: {w: [] for w in FORWARD_WINDOWS} for q in NU_QUARTILES
    }
    quartile_counts: Dict[str, int] = {q: 0 for q in NU_QUARTILES}

    for idx in range(60, len(closes)):
        nu_result = _compute_nu_proxy(closes, idx)
        if nu_result is None:
            continue

        nu  = nu_result.nu
        dir_ = nu_result.direction
        q   = _nu_quartile(nu)

        # ν-signed return: ν × direction gives a directional signal
        nu_signal = nu * dir_   # positive = bullish with coherence

        all_windows_valid = True
        for w in FORWARD_WINDOWS:
            fr = _forward_return(closes, idx, w)
            if fr is None:
                all_windows_valid = False
                break
            forward[w].append(fr)
            magnitudes[w].append(abs(fr))
            quartile_returns[q][w].append(fr)

        if all_windows_valid:
            nu_values.append(nu_signal)
            directions.append(dir_)
            quartile_counts[q] = quartile_counts.get(q, 0) + 1

    # Per-quartile mean returns
    per_quartile: Dict[str, Dict[str, float]] = {}
    for w in FORWARD_WINDOWS:
        per_quartile[f"{w}d"] = {}
        for q in NU_QUARTILES:
            vals = quartile_returns[q][w]
            per_quartile[f"{w}d"][q] = _mean(vals) if vals else 0.0

    # Correlations: ν-signal vs forward return
    corr_return: Dict[str, float] = {}
    corr_mag:    Dict[str, float] = {}
    for w in FORWARD_WINDOWS:
        fr_list = forward[w]
        mg_list = magnitudes[w]
        min_len = min(len(nu_values), len(fr_list), len(mg_list))
        if min_len < 5:
            corr_return[f"{w}d"] = 0.0
            corr_mag[f"{w}d"]    = 0.0
        else:
            nu_arr = nu_values[:min_len]
            corr_return[f"{w}d"] = _pearson_r(nu_arr, fr_list[:min_len])
            # ν (no direction) vs magnitude
            nu_mag = [abs(v) for v in nu_arr]
            corr_mag[f"{w}d"]    = _pearson_r(nu_mag, mg_list[:min_len])

    n = len(nu_values)

    # Verdict: hypothesis = ν × direction positively correlates with return
    # We test the 10-day and 20-day windows as primary
    primary_verdicts = [
        _verdict_from_r_and_direction(corr_return.get("10d", 0.0), n, True),
        _verdict_from_r_and_direction(corr_return.get("20d", 0.0), n, True),
    ]
    magnitude_verdicts = [
        _verdict_from_r_and_direction(corr_mag.get("10d", 0.0), n, True),
        _verdict_from_r_and_direction(corr_mag.get("20d", 0.0), n, True),
    ]
    all_verdicts = primary_verdicts + magnitude_verdicts
    verdict = _combine_verdicts(all_verdicts)

    notes_parts = [
        f"n={n} observations.",
        f"10d directional corr: r={corr_return.get('10d', 0):.3f} ({primary_verdicts[0]})",
        f"20d directional corr: r={corr_return.get('20d', 0):.3f} ({primary_verdicts[1]})",
        f"10d magnitude  corr:  r={corr_mag.get('10d', 0):.3f} ({magnitude_verdicts[0]})",
        f"20d magnitude  corr:  r={corr_mag.get('20d', 0):.3f} ({magnitude_verdicts[1]})",
    ]
    if verdict == REFUTED:
        notes_parts.append("REFUTED: ν price-proxy shows negative or zero directional predictive value at tested horizons.")
    elif verdict == INCONCLUSIVE:
        notes_parts.append("INCONCLUSIVE: signal present but too weak to confirm. Candidate for mystery pile.")

    return NuValidationReport(
        symbol=symbol,
        n_observations=n,
        per_quartile_mean_return=per_quartile,
        correlation_nu_vs_return=corr_return,
        correlation_nu_vs_magnitude=corr_mag,
        evidence_verdict=verdict,
        evidence_level="TESTABLE",
        notes=" | ".join(notes_parts),
        quartile_counts=quartile_counts,
    )


# ─── Backtest 2: LSII arc break prediction ───────────────────────────────────

LSII_FLAG_WINDOW = 40    # bars per LSII computation window


def run_lsii_backtest(
    symbol: str,
    closes:  np.ndarray,
    highs:   np.ndarray,
    lows:    np.ndarray,
    opens:   np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
) -> LSIIValidationReport:
    """
    Validate: do LSII high/very_high flags precede mean-reversion?
    Hypothesis: flagged periods should have negative forward returns
    relative to unflagged baseline.
    """
    flagged_returns:  Dict[int, List[float]] = {w: [] for w in FORWARD_WINDOWS}
    baseline_returns: Dict[int, List[float]] = {w: [] for w in FORWARD_WINDOWS}

    lsii_values: List[float] = []

    step = max(1, LSII_FLAG_WINDOW // 4)   # stride between evaluations

    for idx in range(LSII_FLAG_WINDOW, len(closes) - max(FORWARD_WINDOWS), step):
        # Build OHLCV bars for this window
        start = idx - LSII_FLAG_WINDOW
        bars = []
        for i in range(start, idx):
            bars.append(OHLCV(
                timestamp=datetime.fromtimestamp(float(timestamps[i]), tz=timezone.utc),
                open=float(opens[i]), high=float(highs[i]),
                low=float(lows[i]),   close=float(closes[i]),
                volume=float(volumes[i]), symbol=symbol, timeframe="1d",
            ))

        lsii_result = compute_lsii_price(bars, min_bars=20)
        lsii_values.append(lsii_result.lsii)
        is_flagged = lsii_result.flag_level in ("high", "very_high")

        for w in FORWARD_WINDOWS:
            fr = _forward_return(closes, idx, w)
            if fr is None:
                continue
            if is_flagged:
                flagged_returns[w].append(fr)
            else:
                baseline_returns[w].append(fr)

    flagged_mean:  Dict[str, float] = {}
    baseline_mean: Dict[str, float] = {}
    pred_value:    Dict[str, float] = {}

    for w in FORWARD_WINDOWS:
        key = f"{w}d"
        flagged_mean[key]  = _mean(flagged_returns[w])
        baseline_mean[key] = _mean(baseline_returns[w])
        pred_value[key]    = flagged_mean[key] - baseline_mean[key]

    n_flagged  = len(flagged_returns[10])
    n_baseline = len(baseline_returns[10])

    # Hypothesis: flagged mean return < baseline (negative spread)
    spreads_5_10 = [pred_value.get("5d", 0), pred_value.get("10d", 0)]
    spreads_negative = sum(1 for s in spreads_5_10 if s < 0)

    # Optimal threshold: find which LSII value maximises negative spread
    if lsii_values:
        median_lsii = float(np.median(lsii_values))
        # Map to nearest threshold label
        threshold_candidates = list(LSII_THRESHOLDS.keys())
        optimal = "high"
        best_spread = pred_value.get("10d", 0)
        # Simple heuristic: if median LSII is high, current thresholds may be too loose
        if median_lsii > LSII_THRESHOLDS["very_high"]:
            optimal = "very_high"
        elif median_lsii < LSII_THRESHOLDS["moderate"]:
            optimal = "moderate"
        current_ok = (optimal == "high")
    else:
        optimal = "unknown"
        current_ok = False

    # Verdict
    if n_flagged < MIN_OBSERVATIONS:
        verdict = INCONCLUSIVE
        notes_str = f"Insufficient flagged observations (n={n_flagged}). Need ≥{MIN_OBSERVATIONS}."
    elif spreads_negative == 2 and abs(pred_value.get("10d", 0)) > 0.005:
        verdict = CONFIRMED if abs(pred_value.get("10d", 0)) > 0.015 else PARTIAL
        notes_str = (
            f"n_flagged={n_flagged}, n_baseline={n_baseline}. "
            f"Flagged 10d mean={flagged_mean.get('10d', 0):.4f} vs "
            f"baseline={baseline_mean.get('10d', 0):.4f}. "
            f"Spread={pred_value.get('10d', 0):.4f} "
            f"({'negative — hypothesis supported' if pred_value.get('10d', 0) < 0 else 'positive — hypothesis not supported'})."
        )
    elif spreads_negative == 1:
        verdict = PARTIAL
        notes_str = (
            f"Mixed: 5d and 10d spreads differ in sign. "
            f"5d={pred_value.get('5d', 0):.4f}, 10d={pred_value.get('10d', 0):.4f}. "
            f"PARTIAL — consistent at one window only."
        )
    else:
        # Hypothesis predicts negative spread; if positive, REFUTED
        if pred_value.get("10d", 0) > 0.010:
            verdict = REFUTED
            notes_str = (
                f"REFUTED: LSII flags followed by POSITIVE returns "
                f"(10d spread={pred_value.get('10d', 0):.4f}). "
                f"Arc-break may not indicate reversal for {symbol}."
            )
        else:
            verdict = INCONCLUSIVE
            notes_str = (
                f"No clear signal. Flagged 10d={flagged_mean.get('10d', 0):.4f}, "
                f"baseline={baseline_mean.get('10d', 0):.4f}. "
                f"Insufficient evidence to confirm or refute."
            )

    return LSIIValidationReport(
        symbol=symbol,
        n_flagged=n_flagged,
        n_baseline=n_baseline,
        flagged_mean_return=flagged_mean,
        baseline_mean_return=baseline_mean,
        flag_predictive_value=pred_value,
        optimal_threshold=optimal,
        current_threshold_ok=current_ok,
        evidence_verdict=verdict,
        evidence_level="TESTABLE",
        notes=notes_str,
    )


# ─── Backtest 3: STS trajectory prediction ───────────────────────────────────

STS_WINDOWS = [20, 60]   # STS is a slow signal


def run_sts_backtest(
    symbol: str,
    closes: np.ndarray,
) -> STSValidationReport:
    """
    Validate: does LOADING STS → positive returns at 20/60 days?
    Does DETERIORATING STS → negative returns?
    """
    sts_states: Dict[str, Dict[int, List[float]]] = {
        "LOADING":       {w: [] for w in STS_WINDOWS},
        "STABLE":        {w: [] for w in STS_WINDOWS},
        "DETERIORATING": {w: [] for w in STS_WINDOWS},
    }

    for idx in range(80, len(closes) - max(STS_WINDOWS)):
        sts = _compute_sts_proxy(closes, idx)

        for w in STS_WINDOWS:
            fr = _forward_return(closes, idx, w)
            if fr is not None:
                sts_states[sts][w].append(fr)

    def mean_dict(state: str) -> Dict[str, float]:
        return {
            f"{w}d": _mean(sts_states[state][w])
            for w in STS_WINDOWS
        }

    loading_means = mean_dict("LOADING")
    stable_means  = mean_dict("STABLE")
    det_means     = mean_dict("DETERIORATING")

    n_loading = len(sts_states["LOADING"][20])
    n_stable  = len(sts_states["STABLE"][20])
    n_det     = len(sts_states["DETERIORATING"][20])

    # Spread: LOADING mean - DETERIORATING mean (expect positive)
    spreads = {
        f"{w}d": loading_means.get(f"{w}d", 0) - det_means.get(f"{w}d", 0)
        for w in STS_WINDOWS
    }

    # Build correlation: sts_score (+1 LOADING, 0 STABLE, -1 DETERIORATING) vs return
    sts_score_all: List[float] = []
    return_all_20: List[float] = []
    return_all_60: List[float] = []

    sts_numeric = {"LOADING": 1.0, "STABLE": 0.0, "DETERIORATING": -1.0}

    for idx in range(80, len(closes) - 60):
        sts = _compute_sts_proxy(closes, idx)
        s   = sts_numeric[sts]
        fr20 = _forward_return(closes, idx, 20)
        fr60 = _forward_return(closes, idx, 60)
        if fr20 is not None and fr60 is not None:
            sts_score_all.append(s)
            return_all_20.append(fr20)
            return_all_60.append(fr60)

    r20 = _pearson_r(sts_score_all, return_all_20)
    r60 = _pearson_r(sts_score_all, return_all_60)

    n_total = len(sts_score_all)

    # Verdict
    v20 = _verdict_from_r_and_direction(r20, n_total, True)
    v60 = _verdict_from_r_and_direction(r60, n_total, True)

    spread_positive_count = sum(1 for w in STS_WINDOWS if spreads.get(f"{w}d", 0) > 0)
    verdict = _combine_verdicts([v20, v60])

    # Fine-tune: if spread direction is inconsistent with correlation verdict, downgrade
    if spread_positive_count == 0 and verdict in (CONFIRMED, PARTIAL):
        verdict = PARTIAL

    notes_parts = [
        f"n={n_total} (loading={n_loading}, stable={n_stable}, deteriorating={n_det}).",
        f"20d: LOAD={loading_means.get('20d', 0):.4f} DETM={det_means.get('20d', 0):.4f} "
        f"spread={spreads.get('20d', 0):.4f} r={r20:.3f} ({v20})",
        f"60d: LOAD={loading_means.get('60d', 0):.4f} DETM={det_means.get('60d', 0):.4f} "
        f"spread={spreads.get('60d', 0):.4f} r={r60:.3f} ({v60})",
    ]
    if verdict == REFUTED:
        notes_parts.append(
            "REFUTED: STS price-proxy does not predict return direction at tested horizons. "
            "STS may require full multi-source WFS (not price-only) for validity."
        )

    return STSValidationReport(
        symbol=symbol,
        n_loading=n_loading,
        n_stable=n_stable,
        n_deteriorating=n_det,
        loading_mean_return=loading_means,
        stable_mean_return=stable_means,
        deteriorating_mean_return=det_means,
        loading_vs_deteriorating=spreads,
        evidence_verdict=verdict,
        evidence_level="TESTABLE",
        notes=" | ".join(notes_parts),
    )


# ─── Aggregate across symbols ─────────────────────────────────────────────────

def _aggregate_nu(reports: List[NuValidationReport]) -> NuValidationReport:
    if not reports:
        return NuValidationReport(
            symbol="AGGREGATE", n_observations=0,
            per_quartile_mean_return={}, correlation_nu_vs_return={},
            correlation_nu_vs_magnitude={}, evidence_verdict=INCONCLUSIVE,
            evidence_level="TESTABLE", notes="No data."
        )
    all_verdicts = [r.evidence_verdict for r in reports]
    # Aggregate correlations: mean across symbols
    windows = [f"{w}d" for w in FORWARD_WINDOWS]
    agg_corr_ret = {}
    agg_corr_mag = {}
    for w in windows:
        vals_ret = [r.correlation_nu_vs_return.get(w, 0) for r in reports]
        vals_mag = [r.correlation_nu_vs_magnitude.get(w, 0) for r in reports]
        agg_corr_ret[w] = _mean(vals_ret)
        agg_corr_mag[w] = _mean(vals_mag)
    total_n = sum(r.n_observations for r in reports)

    agg_verdict = _combine_verdicts(all_verdicts)
    notes = (
        f"Aggregate across {len(reports)} symbols, n={total_n}. "
        f"Mean 10d corr={agg_corr_ret.get('10d', 0):.3f}, "
        f"20d corr={agg_corr_ret.get('20d', 0):.3f}. "
        f"Per-symbol: {', '.join(f'{r.symbol}:{r.evidence_verdict}' for r in reports)}"
    )
    return NuValidationReport(
        symbol="AGGREGATE", n_observations=total_n,
        per_quartile_mean_return={},
        correlation_nu_vs_return=agg_corr_ret,
        correlation_nu_vs_magnitude=agg_corr_mag,
        evidence_verdict=agg_verdict,
        evidence_level="TESTABLE",
        notes=notes,
        quartile_counts={},
    )


def _aggregate_lsii(reports: List[LSIIValidationReport]) -> LSIIValidationReport:
    if not reports:
        return LSIIValidationReport(
            symbol="AGGREGATE", n_flagged=0, n_baseline=0,
            flagged_mean_return={}, baseline_mean_return={},
            flag_predictive_value={}, optimal_threshold="unknown",
            current_threshold_ok=False, evidence_verdict=INCONCLUSIVE,
            evidence_level="TESTABLE", notes="No data."
        )
    all_verdicts = [r.evidence_verdict for r in reports]
    windows = [f"{w}d" for w in FORWARD_WINDOWS]
    agg_flag = {}
    agg_base = {}
    agg_pred = {}
    for w in windows:
        agg_flag[w] = _mean([r.flagged_mean_return.get(w, 0) for r in reports])
        agg_base[w] = _mean([r.baseline_mean_return.get(w, 0) for r in reports])
        agg_pred[w] = _mean([r.flag_predictive_value.get(w, 0) for r in reports])
    n_f = sum(r.n_flagged for r in reports)
    n_b = sum(r.n_baseline for r in reports)
    agg_verdict = _combine_verdicts(all_verdicts)
    # Most common optimal threshold
    threshold_votes = {}
    for r in reports:
        threshold_votes[r.optimal_threshold] = threshold_votes.get(r.optimal_threshold, 0) + 1
    optimal = max(threshold_votes, key=threshold_votes.get) if threshold_votes else "high"
    notes = (
        f"Aggregate across {len(reports)} symbols. "
        f"n_flagged={n_f}, n_baseline={n_b}. "
        f"10d mean spread={agg_pred.get('10d', 0):.4f}. "
        f"Per-symbol: {', '.join(f'{r.symbol}:{r.evidence_verdict}' for r in reports)}"
    )
    return LSIIValidationReport(
        symbol="AGGREGATE", n_flagged=n_f, n_baseline=n_b,
        flagged_mean_return=agg_flag, baseline_mean_return=agg_base,
        flag_predictive_value=agg_pred, optimal_threshold=optimal,
        current_threshold_ok=sum(1 for r in reports if r.current_threshold_ok) > len(reports) // 2,
        evidence_verdict=agg_verdict,
        evidence_level="TESTABLE",
        notes=notes,
    )


def _aggregate_sts(reports: List[STSValidationReport]) -> STSValidationReport:
    if not reports:
        return STSValidationReport(
            symbol="AGGREGATE", n_loading=0, n_stable=0, n_deteriorating=0,
            loading_mean_return={}, stable_mean_return={},
            deteriorating_mean_return={}, loading_vs_deteriorating={},
            evidence_verdict=INCONCLUSIVE, evidence_level="TESTABLE", notes="No data."
        )
    all_verdicts = [r.evidence_verdict for r in reports]
    windows = [f"{w}d" for w in STS_WINDOWS]
    agg_load = {w: _mean([r.loading_mean_return.get(w, 0) for r in reports]) for w in windows}
    agg_det  = {w: _mean([r.deteriorating_mean_return.get(w, 0) for r in reports]) for w in windows}
    agg_stab = {w: _mean([r.stable_mean_return.get(w, 0) for r in reports]) for w in windows}
    agg_spr  = {w: _mean([r.loading_vs_deteriorating.get(w, 0) for r in reports]) for w in windows}
    agg_verdict = _combine_verdicts(all_verdicts)
    notes = (
        f"Aggregate across {len(reports)} symbols. "
        f"20d spread (LOAD-DETM)={agg_spr.get('20d', 0):.4f}, "
        f"60d spread={agg_spr.get('60d', 0):.4f}. "
        f"Per-symbol: {', '.join(f'{r.symbol}:{r.evidence_verdict}' for r in reports)}"
    )
    return STSValidationReport(
        symbol="AGGREGATE",
        n_loading=sum(r.n_loading for r in reports),
        n_stable=sum(r.n_stable for r in reports),
        n_deteriorating=sum(r.n_deteriorating for r in reports),
        loading_mean_return=agg_load,
        stable_mean_return=agg_stab,
        deteriorating_mean_return=agg_det,
        loading_vs_deteriorating=agg_spr,
        evidence_verdict=agg_verdict,
        evidence_level="TESTABLE",
        notes=notes,
    )


# ─── Overall verdict + calibration ───────────────────────────────────────────

def _overall_verdict_and_recommendations(
    nu:   NuValidationReport,
    lsii: LSIIValidationReport,
    sts:  STSValidationReport,
) -> Tuple[str, List[str], List[str]]:
    """
    Derive overall verdict, mystery pile items, and calibration recommendations.
    All verdicts are recorded honestly — no suppression.
    """
    verdicts = [nu.evidence_verdict, lsii.evidence_verdict, sts.evidence_verdict]
    order = {CONFIRMED: 3, PARTIAL: 2, INCONCLUSIVE: 1, REFUTED: 0}
    overall = min(verdicts, key=lambda v: order.get(v, 1))

    mystery: List[str] = []
    calibration: List[str] = []

    # ν
    if nu.evidence_verdict == REFUTED:
        mystery.append(
            "ν price-proxy: refuted at tested horizons. "
            "Full multi-source ν (including World Bank, FRED, COT) may behave differently. "
            "Price-only proxy may not capture cross-scale coherence adequately."
        )
        calibration.append(
            "ν: Run live ν (full signal) vs forward return separately. "
            "Price proxy is a lower bound — full signal test required before final verdict."
        )
    elif nu.evidence_verdict == INCONCLUSIVE:
        mystery.append(
            "ν price-proxy: inconclusive. Signal too weak to confirm or refute with price data alone."
        )
    elif nu.evidence_verdict == PARTIAL:
        calibration.append(
            "ν: Confirmed at some windows. Investigate which time horizon (5/10/20d) "
            "best captures the coherence effect. Hypothesis may be window-specific."
        )

    # LSII
    if lsii.evidence_verdict == REFUTED:
        mystery.append(
            "LSII-Price: refuted — arc breaks NOT consistently followed by reversal. "
            "Hypothesis requires revision or threshold recalibration."
        )
        calibration.append(
            f"LSII: Optimal threshold appears to be '{lsii.optimal_threshold}' "
            f"(current: 'high'). Review threshold calibration with more data."
        )
    elif lsii.evidence_verdict == INCONCLUSIVE:
        mystery.append(
            "LSII-Price: inconclusive. Flagged periods show weak or mixed signal. "
            "May require longer price series or different window sizes."
        )
    elif not lsii.current_threshold_ok:
        calibration.append(
            f"LSII: Current 'high' threshold may not be optimal. "
            f"Aggregate data suggests '{lsii.optimal_threshold}' is better calibrated."
        )

    # STS
    if sts.evidence_verdict == REFUTED:
        mystery.append(
            "STS price-proxy: refuted. Price-derived STS does not predict direction. "
            "Full WFS (multi-source) required — price-only may be insufficient proxy."
        )
        calibration.append(
            "STS: Full live WFS (not price proxy) required for honest STS validation. "
            "Consider building a WFS time-series logger and running backtest after 6 months."
        )
    elif sts.evidence_verdict in (PARTIAL, CONFIRMED):
        spread_60 = sts.loading_vs_deteriorating.get("60d", 0)
        spread_20 = sts.loading_vs_deteriorating.get("20d", 0)
        if spread_60 > spread_20:
            calibration.append(
                "STS: 60-day horizon shows stronger signal than 20-day. "
                "STS is a slow signal — use 45-60d as primary validation window."
            )

    # SAS
    mystery.append(
        "SAS historical proxy: [SPECULATIVE] — no reliable multi-year EDGAR NLP corpus. "
        "Backtest 4 cannot be honestly run without 3-5 years of annual report NLP data. "
        "Requires building a EDGAR corpus pipeline first."
    )
    calibration.append(
        "SAS: Build EDGAR 10-K NLP corpus pipeline (3-5 years per symbol) before "
        "attempting full SAS historical validation. Current proxy is too noisy."
    )

    return overall, mystery, calibration


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_backtest(
    symbols: List[str],
    period: str = "2y",
) -> BacktestReport:
    """
    Run all four backtests for each symbol and aggregate results.
    Epistemically honest — all verdicts recorded, none suppressed.
    """
    per_symbol: List[SymbolBacktestResult] = []
    nu_reports:   List[NuValidationReport]   = []
    lsii_reports: List[LSIIValidationReport] = []
    sts_reports:  List[STSValidationReport]  = []

    for sym in symbols:
        logger.info(f"Running backtest for {sym}...")
        arr = fetch_price_history(sym, period=period)
        if arr is None:
            logger.warning(f"Skipping {sym} — no price data.")
            continue

        closes    = arr[:, 3]
        highs     = arr[:, 1]
        lows      = arr[:, 2]
        opens     = arr[:, 0]
        volumes   = arr[:, 4]
        timestamps = arr[:, 5]

        nu_r   = run_nu_backtest(sym, closes)
        lsii_r = run_lsii_backtest(sym, closes, highs, lows, opens, volumes, timestamps)
        sts_r  = run_sts_backtest(sym, closes)
        sas_r  = SASValidationReport(symbol=sym)

        logger.info(
            f"  {sym}: ν={nu_r.evidence_verdict} "
            f"LSII={lsii_r.evidence_verdict} "
            f"STS={sts_r.evidence_verdict}"
        )

        per_symbol.append(SymbolBacktestResult(
            symbol=sym,
            nu_report=nu_r, lsii_report=lsii_r,
            sts_report=sts_r, sas_report=sas_r,
        ))
        nu_reports.append(nu_r)
        lsii_reports.append(lsii_r)
        sts_reports.append(sts_r)

    nu_agg   = _aggregate_nu(nu_reports)
    lsii_agg = _aggregate_lsii(lsii_reports)
    sts_agg  = _aggregate_sts(sts_reports)

    overall, mystery, calibration = _overall_verdict_and_recommendations(
        nu_agg, lsii_agg, sts_agg
    )

    return BacktestReport(
        run_date=datetime.now(timezone.utc).isoformat(),
        symbols=[r.symbol for r in per_symbol],
        per_symbol=per_symbol,
        nu_aggregate=nu_agg,
        lsii_aggregate=lsii_agg,
        sts_aggregate=sts_agg,
        overall_verdict=overall,
        mystery_pile_items=mystery,
        calibration_recommendations=calibration,
        price_period_days=730 if period == "2y" else 365,
    )


def save_report(report: BacktestReport, path: Optional[str] = None) -> str:
    """Serialise BacktestReport to JSON. Returns file path."""
    if path is None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"backtest_{date_str}.json")

    def _serialise(obj):
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        raise TypeError(f"Not serialisable: {type(obj)}")

    data = asdict(report)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_serialise)
    logger.info(f"Report saved to {path}")
    return path
