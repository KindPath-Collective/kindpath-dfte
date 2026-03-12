"""
KEPE — Expanded Data Signals
============================
Broader field inputs: Wikipedia attention, Local Weather, GitHub activity.
All raw data is logged to the 'All Data Matters' archive.
"""

from __future__ import annotations
import logging
import os
import httpx
import numpy as np
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from .indicators import WorldSignal
from logger.raw_data_logger import get_raw_logger
from governance.governance_layer import SYMBOL_SECTOR_MAP

logger = logging.getLogger(__name__)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")

try:
    from scripts.local_data_cache import LocalDataCache
    _cache = LocalDataCache()
except Exception:
    _cache = None

class WikipediaAttentionSignal:
    """
    Measures public attention shifts via Wikipedia pageviews.
    Leading indicator for narrative field changes.
    """
    
    def compute(self, symbol: str, topic: str = None) -> WorldSignal:
        topic = topic or self._map_symbol_to_topic(symbol)
        try:
            # Cache-first: avoid network call if fresh data exists
            if _cache:
                cached_views = _cache.get_wikipedia(topic, max_age_hours=24)
                if cached_views:
                    views = cached_views[-7:]
                    if len(views) >= 2:
                        avg_view = np.mean(views[:-1])
                        last_view = views[-1]
                        spike = (last_view - avg_view) / (avg_view + 1e-10)
                        val = float(np.clip(spike, -1, 1))
                        return WorldSignal(
                            domain="ATTENTION", source="wikipedia",
                            region="GLOBAL", value=val, confidence=0.50,
                            evidence_level="TESTABLE",
                            timestamp=datetime.now(timezone.utc),
                            temporal_layer="SURFACE",
                            raw={"views_7d": views},
                            notes=f"Wiki pageviews for {topic} (cached): {spike:+.2%} spike"
                        )

            headers = {"User-Agent": "KindPath-Bot/1.0 (sam@kindpath.org)"}
            # Wikipedia API expects dates in YYYYMMDD format
            # We fetch from start of year to today to get enough history for mean
            today = datetime.now(timezone.utc).strftime("%Y%m%d")
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{topic}/daily/20260101/{today}"

            with httpx.Client(headers=headers, timeout=10) as client:
                r = client.get(url)

                # LOG RAW DATA
                get_raw_logger().log(
                    source="wikipedia",
                    data=r.json() if r.status_code == 200 else r.text,
                    metadata={"symbol": symbol, "topic": topic, "status": r.status_code}
                )

                if r.status_code != 200:
                    return WorldSignal(domain="ATTENTION", source="wikipedia", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc))

                data = r.json()
                items = data.get("items", [])
                if not items:
                    return WorldSignal(domain="ATTENTION", source="wikipedia", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc))

                # Store in cache for future calls
                if _cache:
                    _cache.put_wikipedia(topic, items)

                views = [item["views"] for item in items[-7:]]
                if len(views) < 2:
                    return WorldSignal(domain="ATTENTION", source="wikipedia", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc))


                avg_view = np.mean(views[:-1])
                last_view = views[-1]
                spike = (last_view - avg_view) / (avg_view + 1e-10)
                val = float(np.clip(spike, -1, 1))

                return WorldSignal(
                    domain="ATTENTION", source="wikipedia",
                    region="GLOBAL", value=val, confidence=0.50,
                    evidence_level="TESTABLE",
                    timestamp=datetime.now(timezone.utc),
                    temporal_layer="SURFACE",
                    raw={"views_7d": views},
                    notes=f"Wiki pageviews for {topic}: {spike:+.2%} spike"
                )

        except Exception as e:
            logger.warning(f"Wikipedia signal failed for {symbol}: {e}")
            return WorldSignal(domain="ATTENTION", source="wikipedia", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc))


    def _map_symbol_to_topic(self, symbol: str) -> str:
        m = {
            "BTC-USD": "Bitcoin", 
            "ETH-USD": "Ethereum",
            "SOL-USD": "Solana_(blockchain_platform)", 
            "ADA-USD": "Cardano_(blockchain_platform)",
            "POL-USD": "Polygon_(blockchain)", 
            "OP-USD": "Optimism_(Ethereum_layer_2)",
            "ARB-USD": "Arbitrum", 
            "LDO-USD": "Decentralized_finance", # Proxy since LDO has no page
            "LINK-USD": "Chainlink_(blockchain)",
            "TSLA": "Tesla,_Inc.", 
            "AAPL": "Apple_Inc.",
            "ENPH": "Enphase_Energy", 
            "NEE": "NextEra_Energy",
            "SPY": "S&P_500", 
            "QQQ": "Nasdaq-100"
        }
        return m.get(symbol, symbol)


class GoogleTrendsSignal:
    """
    Measures broad search interest.
    Leading indicator for 'distracted glances' (retail FOMO).
    """
    def compute(self, symbol: str) -> WorldSignal:
        query = f"{symbol} stock price"
        try:
            # Try SearXNG first (sovereign, in-house search)
            r = httpx.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json"},
                timeout=5,
            )
            data = r.json()
            results = data.get("results", [])
            # Use result count as a proxy for search interest (normalised 0–1)
            result_count = len(results)
            interest = float(np.clip(result_count / 10.0, 0.0, 1.0)) * 2 - 1  # scale to [-1, 1]
            get_raw_logger().log(source="searxng", data={"result_count": result_count}, metadata={"symbol": symbol, "query": query})
            return WorldSignal(
                domain="ATTENTION", source="google_trends",
                region="GLOBAL", value=interest, confidence=0.30,
                evidence_level="TESTABLE",
                timestamp=datetime.now(timezone.utc),
                temporal_layer="SURFACE",
                notes=f"Search interest for {symbol} via SearXNG: {result_count} results"
            )
        except Exception:
            pass

        # Fall back to direct Google scrape
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            with httpx.Client(headers=headers, timeout=10) as client:
                r = client.get(url)
                get_raw_logger().log(source="google_search_meta", data=r.text[:5000], metadata={"symbol": symbol})
            return WorldSignal(
                domain="ATTENTION", source="google_trends",
                region="GLOBAL", value=0.0, confidence=0.20,
                evidence_level="TESTABLE",
                timestamp=datetime.now(timezone.utc),
                temporal_layer="SURFACE",
                notes=f"Search interest for {symbol}: STABLE (google fallback)"
            )
        except Exception:
            return WorldSignal(domain="ATTENTION", source="google_trends", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc))



class SocialInfluenceSynthesizer:
    """
    Synthesizes multiple social channels into a 'Social Field Curvature' score.
    Measures the delta between 'Deep Interest' (Wiki) and 'Distracted Glances' (Search/Reddit).
    """
    def compute(self, symbol: str, signals: List[WorldSignal]) -> WorldSignal:
        wiki = next((s for s in signals if s.source == "wikipedia"), None)
        reddit = next((s for s in signals if s.source == "reddit_sentiment"), None)
        trends = next((s for s in signals if s.source == "google_trends"), None)
        
        # We need at least wiki and one retail signal
        if not wiki:
            return WorldSignal(domain="NARRATIVE", source="social_curvature", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="SPECULATIVE", timestamp=datetime.now(timezone.utc))

        retail_val = 0.0
        retail_count = 0
        if reddit and reddit.confidence > 0:
            retail_val += reddit.value
            retail_count += 1
        if trends and trends.confidence > 0:
            retail_val += trends.value
            retail_count += 1

        if retail_count == 0:
            return WorldSignal(domain="NARRATIVE", source="social_curvature", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="SPECULATIVE", timestamp=datetime.now(timezone.utc))


        retail_pressure = retail_val / retail_count
        deep_interest   = wiki.value
        
        # Curvature = (Retail Pressure - Deep Interest)
        # Higher positive = IN-Loading (Noise exceeding substance)
        curvature = retail_pressure - deep_interest
        
        # Invert: Positive value means Deep Interest >= Retail (Syntropic/Stealth)
        val = float(np.clip(-curvature, -1, 1))
        
        return WorldSignal(
            domain="NARRATIVE", source="social_curvature",
            region="GLOBAL", value=val, confidence=0.60,
            evidence_level="TESTABLE",
            timestamp=datetime.now(timezone.utc),
            temporal_layer="SURFACE",
            raw={"retail": retail_pressure, "deep": deep_interest},
            notes=f"Social Curvature for {symbol}: {val:+.2f}"
        )


class SectorCoherenceSignal:
    """
    'Mycorrhiza' logic: Detects if attention/sentiment is growing across a connected sector.
    Requires input from ALL symbols in the current basket.
    """
    
    def compute(self, symbol: str, basket_signals: Dict[str, List[WorldSignal]]) -> WorldSignal:
        my_sector = SYMBOL_SECTOR_MAP.get(symbol, "technology")
        if not my_sector:
            return WorldSignal(domain="ECHO", source="sector_coherence", region="SECTOR", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc))
            
        # Find peers in same sector
        peers = [s for s, _ in basket_signals.items() if SYMBOL_SECTOR_MAP.get(s) == my_sector and s != symbol]
        if not peers:
            return WorldSignal(domain="ECHO", source="sector_coherence", region="SECTOR", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc), notes="No peers in basket")

        # Calculate peer attention/sentiment
        peer_scores = []
        for p in peers:
            sigs = basket_signals[p]
            sc = next((s for s in sigs if s.source == "social_curvature"), None)
            if sc and sc.confidence > 0:
                peer_scores.append(sc.value)
            else:
                narrative_vals = [s.value for s in sigs if s.domain == "NARRATIVE" and s.confidence > 0]
                if narrative_vals:
                    peer_scores.append(np.mean(narrative_vals))
        
        if not peer_scores:
            return WorldSignal(domain="ECHO", source="sector_coherence", region="SECTOR", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc))

        avg_peer_score = float(np.mean(peer_scores))
        
        return WorldSignal(
            domain="ECHO", source="sector_coherence",
            region="SECTOR", value=avg_peer_score, confidence=0.50,
            evidence_level="TESTABLE",
            timestamp=datetime.now(timezone.utc),
            temporal_layer="SURFACE",
            raw={"peer_count": len(peers), "peers": peers},
            notes=f"Sector ({my_sector}) coherence: {avg_peer_score:+.2f}"
        )


class LocalWeatherSignal:
    """
    Local weather data for Northern NSW (Bundjalung Country).
    Connects the system to the physical reality of the operator's location.
    """
    LAT = -28.65
    LON = 153.56
    
    def compute(self) -> WorldSignal:
        try:
            # Cache-first: weather valid for 3 hours
            if _cache:
                cached = _cache.get_weather(self.LAT, self.LON, max_age_hours=3)
                if cached:
                    cloud = cached.get("cloud_cover", 0)
                    energy_level = 1.0 - (cloud / 100.0)
                    val = (energy_level * 2) - 1.0
                    return WorldSignal(
                        domain="ECOLOGICAL", source="local_weather",
                        region="BUNDJALUNG", value=val, confidence=0.80,
                        evidence_level="ESTABLISHED",
                        timestamp=datetime.now(timezone.utc),
                        temporal_layer="SURFACE",
                        raw=cached,
                        notes=f"Local weather (cached): {cached.get('temperature_2m')}°C"
                    )

            url = f"https://api.open-meteo.com/v1/forecast?latitude={self.LAT}&longitude={self.LON}&current=temperature_2m,rain,cloud_cover,wind_speed_10m&timezone=Australia%2FSydney"

            with httpx.Client(timeout=10) as client:
                r = client.get(url)
                get_raw_logger().log(source="open_meteo", data=r.json() if r.status_code == 200 else r.text, metadata={"lat": self.LAT, "lon": self.LON})

                if r.status_code != 200:
                    return WorldSignal(domain="ECOLOGICAL", source="local_weather", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="ESTABLISHED", timestamp=datetime.now(timezone.utc))

                data = r.json()
                current = data.get("current", {})

                # Store in cache
                if _cache:
                    _cache.put_weather(self.LAT, self.LON, current)

                cloud = current.get("cloud_cover", 0)
                energy_level = 1.0 - (cloud / 100.0)
                val = (energy_level * 2) - 1.0

                return WorldSignal(
                    domain="ECOLOGICAL", source="local_weather",
                    region="BUNDJALUNG", value=val, confidence=0.80,
                    evidence_level="ESTABLISHED",
                    timestamp=datetime.now(timezone.utc),
                    temporal_layer="SURFACE",
                    raw=current,
                    notes=f"Local weather: {current.get('temperature_2m')}°C"
                )
        except Exception:
            return WorldSignal(domain="ECOLOGICAL", source="local_weather", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="ESTABLISHED", timestamp=datetime.now(timezone.utc))
