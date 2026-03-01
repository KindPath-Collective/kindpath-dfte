"""
KPRE — Physical Flow Layer
============================
Real-world generative field signals from physical infrastructure:
trade flows, freight, energy grids, agriculture, and port activity.

These are instrument-agnostic background signals — physical flow
represents the generative capacity of the real economy underneath
the financial layer. When physical flows are expanding, the real
generative field is active. When they are contracting, the world
field is losing generative momentum regardless of financial signals.

High KPRE = physical economy expanding, real-world generative conditions good.
Low KPRE  = physical bottlenecks, trade contraction, infrastructure stress.

Evidence posture: KINDFIELD
  [ESTABLISHED] — Baltic Dry Index as global trade expansion indicator
                  (well-validated: BDI leads economic cycles by 1-3 months)
  [TESTABLE]    — ETF proxies as physical flow indicators
                  (directionally valid; requires calibration against real data)
  [SPECULATIVE] — Port congestion proxy via shipping ETF momentum
                  (low resolution; ETF reflects demand, not congestion directly)

API note on EIA crude inventory:
  EIA v2 API requires a free API key (register at https://www.eia.gov/opendata).
  Set EIA_API_KEY env var to enable direct inventory data.
  Without it, falls back to USO/XLE momentum proxy.
  EIA v1 was retired March 2023.

API note on USDA export sales:
  USDA FAS public endpoints have variable availability.
  Primary proxy uses MOO/GLD ratio (agricultural vs safe-haven).
"""

from __future__ import annotations
import os
import logging
import requests
import numpy as np
from datetime import datetime
from typing import List, Optional

from kepe.indicators import WorldSignal

logger = logging.getLogger(__name__)


# ─── 1. Baltic Dry Index proxy ────────────────────────────────────────────────

class BalticDrySignal:
    """
    Baltic Dry Index proxy via BDRY (Breakwave Dry Bulk Shipping ETF).

    BDI measures the cost of transporting raw materials globally.
    Rising BDI = shipping demand outpacing supply = expanding global trade.
    Falling BDI = trade contraction or shipping oversupply.

    Primary:  BDRY (Breakwave Dry Bulk Shipping ETF — designed to track BDI)
    Fallback: SEA (Global X Shipping & Ports ETF)

    Metric: 20-day price momentum relative to 60-day rolling mean.
    Positive z-score = BDI above trend = trade expansion signal.

    [ESTABLISHED] — BDI as leading global trade indicator is well-validated.
    [TESTABLE]    — BDRY as BDI proxy introduces basis risk and roll decay.
    Temporal layer: MEDIUM — BDI regime shifts over weeks to months.
    """

    def compute(self) -> WorldSignal:
        import yfinance as yf

        for ticker in ("BDRY", "SEA"):
            try:
                prices = yf.Ticker(ticker).history(period="6mo")["Close"]
                if prices.empty or len(prices) < 40:
                    continue

                trend_20  = (prices.iloc[-1] - prices.iloc[-20]) / (prices.iloc[-20] + 1e-10)
                mean_60   = prices.iloc[-60:].mean() if len(prices) >= 60 else prices.mean()
                std_60    = prices.iloc[-60:].std()  if len(prices) >= 60 else prices.std()
                z         = (prices.iloc[-1] - mean_60) / (std_60 + 1e-10)

                # Blend: momentum (40%) + z-score vs mean (60%)
                raw   = trend_20 * 5 * 0.40 + z * 0.25 * 0.60
                value = float(np.clip(raw, -1, 1))

                return WorldSignal(
                    domain="KPRE_FLOW", source=f"bdi_proxy_{ticker.lower()}",
                    region="GLOBAL", value=value, confidence=0.60,
                    evidence_level="TESTABLE",
                    timestamp=datetime.utcnow(),
                    temporal_layer="MEDIUM",
                    raw={"ticker": ticker, "trend_20": trend_20,
                         "z_vs_mean": float(z), "price": float(prices.iloc[-1])},
                    notes=f"Baltic Dry proxy via {ticker}: trend={trend_20:.3f}, z={z:.2f} [TESTABLE]"
                )
            except Exception as e:
                logger.debug(f"BalticDry {ticker} failed: {e}")

        logger.warning("BalticDrySignal: all tickers failed")
        return WorldSignal(
            domain="KPRE_FLOW", source="bdi_proxy",
            region="GLOBAL", value=0.0, confidence=0.0,
            evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
        )


# ─── 2. Freight Signal ────────────────────────────────────────────────────────

class FreightSignal:
    """
    US freight activity proxy via IYT (iShares Transportation ETF).

    IYT tracks trucking, rail, air freight, and delivery companies.
    Rising IYT above its medium-term trend = freight demand expanding.

    Cass Freight Index publishes monthly (not machine-readable).
    IYT provides a weekly-resolution proxy.

    Primary:  IYT (iShares Transportation Average ETF)
    Fallback: FTXL (First Trust Nasdaq Transportation ETF)

    Metric: 20-day momentum z-scored against 3-month rolling mean.

    [TESTABLE] — IYT as freight proxy; transport ETF includes non-freight
                 components. Directionally valid, not precise.
    Temporal layer: MEDIUM — freight cycles play out over weeks.
    """

    def compute(self) -> WorldSignal:
        import yfinance as yf

        for ticker in ("IYT", "FTXL"):
            try:
                prices = yf.Ticker(ticker).history(period="4mo")["Close"]
                if prices.empty or len(prices) < 25:
                    continue

                momentum  = (prices.iloc[-1] - prices.iloc[-20]) / (prices.iloc[-20] + 1e-10)
                mean_60   = prices.mean()
                std_60    = prices.std() + 1e-10
                z         = (prices.iloc[-1] - mean_60) / std_60

                # Rising momentum + above mean = freight expanding
                value = float(np.clip(momentum * 8 * 0.5 + z * 0.20 * 0.5, -1, 1))

                return WorldSignal(
                    domain="KPRE_FLOW", source=f"freight_{ticker.lower()}",
                    region="US", value=value, confidence=0.50,
                    evidence_level="TESTABLE",
                    timestamp=datetime.utcnow(),
                    temporal_layer="MEDIUM",
                    raw={"ticker": ticker, "momentum_20d": momentum,
                         "z_vs_mean": float(z)},
                    notes=f"Freight proxy via {ticker}: momentum={momentum:.3f} [TESTABLE]"
                )
            except Exception as e:
                logger.debug(f"FreightSignal {ticker} failed: {e}")

        logger.warning("FreightSignal: all tickers failed")
        return WorldSignal(
            domain="KPRE_FLOW", source="freight_proxy",
            region="US", value=0.0, confidence=0.0,
            evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
        )


# ─── 3. Energy Grid Signal ────────────────────────────────────────────────────

class EnergyGridSignal:
    """
    Energy system health proxy from US crude inventory dynamics.

    Falling crude inventory = energy system drawing down supply = tighter grid.
    Stable or rising inventory = adequate supply = grid generative capacity intact.

    Primary: EIA v2 API with EIA_API_KEY env var (free registration at eia.gov).
             Series: crude oil weekly stocks (PET.WCESTUS1.W equivalent in v2).
    Fallback: USO (United States Oil Fund) momentum + XLE/XLU ratio as
              energy grid tightness proxy.

    Interpretation:
      USO rising moderately = demand active, grid flowing = mild positive.
      USO rising steeply = supply crunch / geopolitical = mild negative.
      USO stable         = energy system balanced = positive.

    [ESTABLISHED] — EIA crude inventory as energy system indicator (with real data).
    [TESTABLE]    — ETF proxy for inventory dynamics (indirect signal).
    Temporal layer: MEDIUM — weekly inventory cycles.
    """

    EIA_URL = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"

    def compute(self) -> WorldSignal:
        # Try EIA API if key is available
        eia_key = os.environ.get("EIA_API_KEY")
        if eia_key:
            result = self._try_eia(eia_key)
            if result is not None:
                return result

        # Fallback: USO + energy sector dynamics
        return self._energy_proxy()

    def _try_eia(self, api_key: str) -> Optional[WorldSignal]:
        try:
            params = {
                "api_key":              api_key,
                "frequency":            "weekly",
                "data[0]":              "value",
                "sort[0][column]":      "period",
                "sort[0][direction]":   "desc",
                "offset":               0,
                "length":               8,
                "facets[product][]":    "EPC0",   # crude oil
                "facets[process][]":    "SAX",    # ending stocks
            }
            resp = requests.get(self.EIA_URL, params=params, timeout=10)
            if resp.status_code != 200:
                return None

            data     = resp.json().get("response", {}).get("data", [])
            readings = [float(d["value"]) for d in data if d.get("value") is not None]
            if len(readings) < 4:
                return None

            # Weekly inventory delta: negative = drawdown = tighter supply
            weekly_delta = readings[0] - readings[1]    # latest vs previous week
            four_wk_avg  = sum(readings[1:5]) / 4       # prior 4-week average
            delta_vs_avg = (readings[0] - four_wk_avg) / (four_wk_avg + 1e-10)

            # Moderate build = adequate supply = positive. Large draw = stress.
            # Signal peaks at 0.5 for stable/moderate build, negative for sharp draw.
            value = float(np.clip(-delta_vs_avg * 20 * 0.4 + 0.1, -1, 1))

            return WorldSignal(
                domain="KPRE_ENERGY", source="eia_crude_inventory",
                region="US", value=value, confidence=0.80,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                raw={"weekly_delta_mbbl": weekly_delta,
                     "delta_vs_4wk_avg": delta_vs_avg,
                     "latest_stock_mbbl": readings[0]},
                notes=f"EIA crude stocks: delta={weekly_delta:+.0f}k bbl "
                      f"vs 4wk avg {delta_vs_avg:+.2%} [ESTABLISHED]"
            )
        except Exception as e:
            logger.debug(f"EIA API call failed: {e}")
            return None

    def _energy_proxy(self) -> WorldSignal:
        """USO/XLE proxy for energy grid health when EIA key unavailable."""
        try:
            import yfinance as yf

            uso = yf.Ticker("USO").history(period="3mo")["Close"]
            xle = yf.Ticker("XLE").history(period="3mo")["Close"]
            xlu = yf.Ticker("XLU").history(period="3mo")["Close"]   # utilities

            if uso.empty or len(uso) < 20:
                raise ValueError("USO data unavailable")

            uso_trend = (uso.iloc[-1] - uso.iloc[-20]) / (uso.iloc[-20] + 1e-10)

            # XLE vs XLU: energy sector outperforming utilities = demand-led = positive
            if not xle.empty and not xlu.empty and len(xle) >= 20 and len(xlu) >= 20:
                xle_ret = (xle.iloc[-1] - xle.iloc[-20]) / (xle.iloc[-20] + 1e-10)
                xlu_ret = (xlu.iloc[-1] - xlu.iloc[-20]) / (xlu.iloc[-20] + 1e-10)
                demand_signal = xle_ret - xlu_ret
            else:
                demand_signal = 0.0

            # Moderate USO rise = demand active (positive); steep rise = supply crunch (less positive)
            # Normalise: ±10% USO move → ±0.5 signal contribution
            uso_sig = float(np.clip(uso_trend * 5, -0.5, 0.5))
            # Penalise extreme oil price moves as supply stress signal
            if abs(uso_trend) > 0.10:
                uso_sig = uso_sig * 0.6   # dampen: extreme moves = stress

            value = float(np.clip(uso_sig * 0.6 + demand_signal * 5 * 0.4, -1, 1))

            return WorldSignal(
                domain="KPRE_ENERGY", source="energy_proxy_uso",
                region="US", value=value, confidence=0.45,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                raw={"uso_trend_20d": uso_trend, "xle_vs_xlu": demand_signal},
                notes=f"Energy proxy (no EIA key): USO trend={uso_trend:.3f}, "
                      f"XLE/XLU={demand_signal:.3f} [TESTABLE]. "
                      f"Set EIA_API_KEY for direct inventory data."
            )
        except Exception as e:
            logger.warning(f"EnergyGrid proxy failed: {e}")
            return WorldSignal(
                domain="KPRE_ENERGY", source="energy_proxy",
                region="US", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
            )


# ─── 4. Agricultural Flow Signal ─────────────────────────────────────────────

class AgriculturalFlowSignal:
    """
    Agricultural flow proxy via MOO/GLD ratio.

    MOO (VanEck Agribusiness ETF) tracks agri input/output companies.
    GLD is the gold safe-haven benchmark.

    MOO outperforming GLD = agricultural sector active, food production
    investment flowing = positive food security / generative field signal.

    Primary: MOO/GLD ratio 20-day trend
    Fallback: MOO absolute momentum + corn futures (ZC=F)

    USDA weekly export sales (apps.fas.usda.gov/gats) is the ideal primary
    source but API availability is unreliable. MOO/GLD is the working proxy.

    [TESTABLE] — MOO vs GLD ratio as agricultural flow proxy.
                 ETF captures agribusiness, not raw commodity volumes directly.
    Temporal layer: MEDIUM — agricultural cycles span weeks to months.
    """

    def compute(self) -> WorldSignal:
        try:
            import yfinance as yf

            moo = yf.Ticker("MOO").history(period="4mo")["Close"]
            gld = yf.Ticker("GLD").history(period="4mo")["Close"]

            if moo.empty or gld.empty or len(moo) < 20:
                raise ValueError("MOO/GLD data unavailable")

            # Ratio: MOO/GLD — rising = agricultural outperforming safe-haven
            ratio       = moo / (gld + 1e-10)
            ratio_trend = (ratio.iloc[-1] - ratio.iloc[-20]) / (ratio.iloc[-20] + 1e-10)

            # Also use MOO absolute trend as corroborating signal
            moo_trend = (moo.iloc[-1] - moo.iloc[-20]) / (moo.iloc[-20] + 1e-10)

            # Blend ratio trend (primary) + MOO absolute (corroboration)
            value = float(np.clip(ratio_trend * 10 * 0.60 + moo_trend * 5 * 0.40, -1, 1))

            # Blend in corn futures if available
            try:
                corn = yf.Ticker("ZC=F").history(period="3mo")["Close"]
                if not corn.empty and len(corn) >= 20:
                    corn_trend = (corn.iloc[-1] - corn.iloc[-20]) / (corn.iloc[-20] + 1e-10)
                    # Moderate corn rise = demand active (positive);
                    # very high corn = food price stress (negative)
                    corn_sig   = float(np.clip(corn_trend * 5, -0.3, 0.3))
                    value      = float(np.clip(value * 0.75 + corn_sig * 0.25, -1, 1))
            except Exception:
                pass

            return WorldSignal(
                domain="KPRE_FLOW", source="agricultural_flow",
                region="GLOBAL", value=value, confidence=0.50,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                raw={"moo_gld_ratio_trend": ratio_trend, "moo_trend": moo_trend},
                notes=f"Agri proxy MOO/GLD: ratio_trend={ratio_trend:.3f}, "
                      f"moo_trend={moo_trend:.3f} [TESTABLE]. "
                      f"USDA GATS weekly sales would improve this signal."
            )
        except Exception as e:
            logger.warning(f"AgriculturalFlow signal failed: {e}")
            return WorldSignal(
                domain="KPRE_FLOW", source="agricultural_flow",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
            )


# ─── 5. Port Congestion Signal ────────────────────────────────────────────────

class PortCongestionSignal:
    """
    Port activity and shipping throughput proxy.

    Measures container shipping demand via shipping ETF/stock momentum
    relative to historical average. Rising above historical baseline =
    high throughput = trade moving through ports = positive generative signal.

    NOTE: This proxy measures shipping demand, not port congestion directly.
    High shipping ETF = demand for shipping is high, which implies ports are
    busy. Congestion per se (bottleneck) would be a negative signal, but this
    proxy cannot distinguish busy-healthy from busy-congested.
    [SPECULATIVE] for congestion interpretation; [TESTABLE] for throughput proxy.

    Primary:  SEA (Global X Shipping & Ports ETF)
    Fallback: ZIM (Zim Integrated Shipping) or SBLK (Star Bulk Carriers)

    Metric: Price vs 60-day rolling mean (z-score).
    Temporal layer: SURFACE — container rates shift weekly.
    """

    def compute(self) -> WorldSignal:
        import yfinance as yf

        for ticker in ("SEA", "ZIM", "SBLK"):
            try:
                prices = yf.Ticker(ticker).history(period="4mo")["Close"]
                if prices.empty or len(prices) < 25:
                    continue

                mean_60 = prices.mean()
                std_60  = prices.std() + 1e-10
                z       = (prices.iloc[-1] - mean_60) / std_60

                momentum = (prices.iloc[-1] - prices.iloc[-20]) / (prices.iloc[-20] + 1e-10)

                # Above historical average = shipping demand high = ports busy = positive
                # ±1.5σ → ±1.0
                value = float(np.clip(z / 1.5 * 0.60 + momentum * 5 * 0.40, -1, 1))

                return WorldSignal(
                    domain="KPRE_FLOW", source=f"port_proxy_{ticker.lower()}",
                    region="GLOBAL", value=value, confidence=0.40,
                    evidence_level="SPECULATIVE",
                    timestamp=datetime.utcnow(),
                    temporal_layer="SURFACE",
                    raw={"ticker": ticker, "z_vs_60d_mean": float(z),
                         "momentum_20d": momentum,
                         "price": float(prices.iloc[-1]),
                         "mean_60d": float(mean_60)},
                    notes=f"Port proxy via {ticker}: z={z:.2f}, momentum={momentum:.3f} "
                          f"[SPECULATIVE — measures shipping demand, not congestion directly]"
                )
            except Exception as e:
                logger.debug(f"PortCongestion {ticker} failed: {e}")

        logger.warning("PortCongestionSignal: all tickers failed")
        return WorldSignal(
            domain="KPRE_FLOW", source="port_proxy",
            region="GLOBAL", value=0.0, confidence=0.0,
            evidence_level="SPECULATIVE", timestamp=datetime.utcnow(),
            temporal_layer="SURFACE",
        )


# ─── KPRE Layer — Physical Flow Composite ────────────────────────────────────

class KPRELayer:
    """
    KPRE Physical Flow Layer — instrument-agnostic background generative field.

    Aggregates all five physical flow sub-signals into a single composite
    WorldSignal with domain="KPRE". Confidence is scaled by the fraction of
    sub-signals successfully retrieved (max 0.70 — TESTABLE ceiling).

    This layer is added to ALL instruments as a background signal.
    Domain weight in KEPEProfile: 0.35 (see syntropy_engine.DOMAIN_WEIGHTS).

    Usage:
        sig = KPRELayer().compute()        # full pipeline with real APIs
        sig = KPRELayer._aggregate(sigs)   # testable aggregation from WorldSignals
    """

    _SUB_SIGNAL_CLASSES = [
        BalticDrySignal,
        FreightSignal,
        EnergyGridSignal,
        AgriculturalFlowSignal,
        PortCongestionSignal,
    ]

    def compute(self) -> WorldSignal:
        """Compute KPRE composite by running all sub-signal fetchers."""
        sub_signals: List[WorldSignal] = []
        for cls in self._SUB_SIGNAL_CLASSES:
            try:
                sig = cls().compute()
                if sig.confidence > 0:
                    sub_signals.append(sig)
            except Exception as e:
                logger.warning(f"KPRE sub-signal {cls.__name__} failed: {e}")

        return self._aggregate(sub_signals)

    @staticmethod
    def _aggregate(sub_signals: List[WorldSignal]) -> WorldSignal:
        """
        Confidence-weighted aggregation of KPRE sub-signals.
        Testable independently of network calls.
        """
        if not sub_signals:
            return WorldSignal(
                domain="KPRE", source="physical_flow_composite",
                region="GLOBAL", value=0.0, confidence=0.0,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                notes="No KPRE sub-signals available"
            )

        total_w = sum(s.confidence for s in sub_signals)
        if total_w < 1e-10:
            composite_value = 0.0
            composite_conf  = 0.0
        else:
            composite_value = sum(s.value * s.confidence for s in sub_signals) / total_w
            # Scale confidence by completeness: all 5 present → avg sub-signal confidence
            # Cap at 0.70 — composite is always TESTABLE ceiling
            completeness    = len(sub_signals) / 5.0
            avg_conf        = total_w / len(sub_signals)
            composite_conf  = min(0.70, avg_conf * completeness)

        sources    = [s.source for s in sub_signals]
        # Evidence: downgrade to lowest evidence level present
        ev_order   = {"ESTABLISHED": 0, "TESTABLE": 1, "SPECULATIVE": 2}
        worst_ev   = max(sub_signals, key=lambda s: ev_order.get(s.evidence_level, 1))
        ev_level   = worst_ev.evidence_level if sub_signals else "TESTABLE"
        # Composite is always at most TESTABLE (proxy-of-proxies)
        ev_final   = "TESTABLE" if ev_level == "ESTABLISHED" else ev_level

        return WorldSignal(
            domain="KPRE", source="physical_flow_composite",
            region="GLOBAL",
            value=float(np.clip(composite_value, -1, 1)),
            confidence=float(np.clip(composite_conf, 0, 0.70)),
            evidence_level=ev_final,
            timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
            raw={
                "n_signals":    len(sub_signals),
                "n_attempted":  5,
                "completeness": len(sub_signals) / 5.0,
                "sources":      sources,
                "sub_values":   {s.source: round(s.value, 3) for s in sub_signals},
            },
            notes=(
                f"KPRE composite: {len(sub_signals)}/5 signals "
                f"(value={composite_value:.3f}, conf={composite_conf:.2f}) "
                f"[{ev_final}] — physical flow background generative field"
            )
        )
