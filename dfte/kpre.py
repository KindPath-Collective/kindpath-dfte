"""
KPRE — KindPath Relational Engine — Rec 10
==========================================
Pre-price signal layer that scores Physical, Capital, and Language field states
for a symbol before the DFTE engine processes market data.

Three signal streams:
  P  — Physical field: supply chain stress, energy, logistics, weather events
  C  — Capital field: insider activity, institutional flow, credit spreads, fund flows
  L  — Language field: earnings call sentiment, news NLP, social velocity, mgmt tone

KPRE Score = (P + C + L) / 3  → fed as a modifier into DFTE nu calculation

Integration:
  from dfte.kpre import KPREEngine
  kpre = KPREEngine()
  score = kpre.score("AAPL")   # -1.0 to +1.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class KPRESignal:
    symbol: str
    physical: float = 0.0       # -1 to +1
    capital: float = 0.0        # -1 to +1
    language: float = 0.0       # -1 to +1
    score: float = 0.0          # composite
    confidence: float = 0.0
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "kpre_score": round(self.score, 4),
            "physical": round(self.physical, 4),
            "capital": round(self.capital, 4),
            "language": round(self.language, 4),
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


class KPREEngine:
    """
    KindPath Relational Engine.
    Scores the pre-price relational field for a symbol.

    Each sub-engine is pluggable. Add real data sources as feeds mature.
    Current implementation: lightweight heuristics + yfinance fast data.
    """

    def score(self, symbol: str) -> KPRESignal:
        sig = KPRESignal(symbol=symbol)
        evidence = {}

        p_score, p_conf, p_ev = self._physical(symbol)
        c_score, c_conf, c_ev = self._capital(symbol)
        l_score, l_conf, l_ev = self._language(symbol)

        sig.physical = p_score
        sig.capital = c_score
        sig.language = l_score
        sig.score = (p_score + c_score + l_score) / 3.0
        sig.confidence = (p_conf + c_conf + l_conf) / 3.0
        evidence.update(p_ev)
        evidence.update(c_ev)
        evidence.update(l_ev)
        sig.evidence = evidence

        logger.info(
            "KPRE %s: score=%.3f (P=%.2f C=%.2f L=%.2f conf=%.2f)",
            symbol, sig.score, sig.physical, sig.capital, sig.language, sig.confidence
        )
        return sig

    # ------------------------------------------------------------------
    # Physical field — supply chain, commodity exposure, sector stress
    # ------------------------------------------------------------------

    def _physical(self, symbol: str) -> tuple[float, float, dict]:
        """
        Physical field score.
        Heuristic: sector mapping → energy price sensitivity, supply chain proxies.
        Upgrade path: integrate shipping indices (BDI), energy futures, weather APIs.
        Evidence level: SPECULATIVE until real feeds connected.
        """
        evidence = {"physical_source": "[SPECULATIVE] sector heuristic"}
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            # Use 52-week range position as a proxy for momentum field
            low_52 = getattr(info, "year_low", None)
            high_52 = getattr(info, "year_high", None)
            price = getattr(info, "last_price", None)

            if low_52 and high_52 and price and high_52 > low_52:
                position = (price - low_52) / (high_52 - low_52)  # 0–1
                score = (position - 0.5) * 2  # -1 to +1
                evidence["physical_source"] = "[TESTABLE] 52w range position"
                evidence["price_position_52w"] = round(position, 3)
                return score, 0.4, evidence
        except Exception as e:
            logger.debug("Physical field yf fetch failed for %s: %s", symbol, e)

        return 0.0, 0.1, evidence

    # ------------------------------------------------------------------
    # Capital field — institutional flow, insider activity
    # ------------------------------------------------------------------

    def _capital(self, symbol: str) -> tuple[float, float, dict]:
        """
        Capital field score.
        Uses yfinance institutional ownership delta as proxy.
        Upgrade path: SEC Form 4 (insider buys), 13F changes, dark pool proxies.
        """
        evidence = {"capital_source": "[SPECULATIVE] volume proxy"}
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info

            # Volume ratio: current vs average → institutional activity proxy
            avg_vol = getattr(info, "three_month_average_volume", None)
            curr_vol = getattr(info, "last_volume", None)

            if avg_vol and curr_vol and avg_vol > 0:
                ratio = curr_vol / avg_vol
                # >1.5 = unusual volume (+ if price up, - if price down)
                price_change = getattr(info, "regular_market_change_percent", 0) or 0
                direction = 1.0 if price_change >= 0 else -1.0
                score = direction * min(abs(ratio - 1.0), 1.0)
                evidence["capital_source"] = "[TESTABLE] volume anomaly"
                evidence["volume_ratio"] = round(ratio, 3)
                evidence["price_change_pct"] = round(price_change, 3)
                return score, 0.5, evidence
        except Exception as e:
            logger.debug("Capital field yf fetch failed for %s: %s", symbol, e)

        return 0.0, 0.1, evidence

    # ------------------------------------------------------------------
    # Language field — news sentiment, earnings tone
    # ------------------------------------------------------------------

    def _language(self, symbol: str) -> tuple[float, float, dict]:
        """
        Language field score.
        Uses yfinance news + simple keyword sentiment.
        Upgrade path: full NLP (FinBERT), earnings call transcript analysis,
                      social velocity (StockTwits, Reddit).
        """
        evidence = {"language_source": "[SPECULATIVE] keyword sentiment"}
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news or []

            if not news:
                return 0.0, 0.1, evidence

            positive_words = {
                "beat", "surge", "growth", "record", "strong", "upgrade",
                "outperform", "bullish", "expansion", "profit", "gain", "rally",
            }
            negative_words = {
                "miss", "decline", "loss", "downgrade", "bearish", "cut",
                "concern", "risk", "fall", "weak", "contraction", "layoff",
            }

            pos = neg = 0
            for article in news[:10]:
                title = (article.get("title") or "").lower()
                words = set(re.findall(r"\w+", title))
                pos += len(words & positive_words)
                neg += len(words & negative_words)

            total = pos + neg
            if total == 0:
                return 0.0, 0.15, evidence

            score = (pos - neg) / total  # -1 to +1
            confidence = min(total / 10.0, 0.6)  # max 0.6 for keyword method
            evidence["language_source"] = "[TESTABLE] news headline sentiment"
            evidence["news_pos"] = pos
            evidence["news_neg"] = neg
            evidence["news_articles"] = len(news[:10])
            return score, confidence, evidence

        except Exception as e:
            logger.debug("Language field failed for %s: %s", symbol, e)

        return 0.0, 0.1, evidence
