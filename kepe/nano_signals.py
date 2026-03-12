"""
KEPE — Nano-Data Signals
==========================
Real-time, high-frequency signals for the PGP.
Focus: Social sentiment, revision deltas, and cross-asset echoes.
"""

from __future__ import annotations
import logging
import httpx
import numpy as np
import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from .indicators import WorldSignal
from logger.raw_data_logger import get_raw_logger

logger = logging.getLogger(__name__)

class GoogleNewsSignal:
    """
    Real-time sentiment from Google News.
    Scrapes headlines and performs simple keyword analysis.
    """
    
    def compute(self, symbol: str) -> WorldSignal:
        try:
            # More generic User-Agent
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            # Add keywords to improve relevance
            query = f"{symbol}+stock+market+news"
            url = f"https://www.google.com/search?q={query}&tbm=nws"
            
            all_texts = []
            with httpx.Client(headers=headers, follow_redirects=True, timeout=15) as client:
                r = client.get(url)
                
                # LOG RAW DATA
                get_raw_logger().log(
                    source="google_news", 
                    data=r.text[:10000], # Log first 10k chars of HTML
                    metadata={"symbol": symbol, "status": r.status_code}
                )
                
                if r.status_code == 200:
                    # Look for common Google News headline patterns
                    # 1. Search for <div> with role="heading"
                    headlines = re.findall(r'<div role="heading"[^>]*>(.*?)</div>', r.text)
                    # 2. Fallback to generic result classes
                    if not headlines:
                        headlines = re.findall(r'class="BNeawe vvjtPb[^>]*>(.*?)</div>', r.text)
                    # 3. Last resort: look for title attribute in links within news results
                    if not headlines:
                        headlines = re.findall(r'<a [^>]*aria-label="(.*?)"', r.text)
                    
                    # Clean up HTML tags from headlines
                    clean_headlines = [re.sub(r'<[^>]+>', '', h) for h in headlines]
                    all_texts.extend(clean_headlines)
            
            if not all_texts:
                logger.debug(f"No Google News headlines found for {symbol}")
                return WorldSignal(domain="NARRATIVE", source="google_news", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc), temporal_layer="SURFACE")
            
            # Simple keyword analysis
            pos = ["upgrade", "growth", "high", "success", "bullish", "profit", "gain", "buy", "surge", "positive", "strong"]
            neg = ["downgrade", "fall", "low", "failure", "bearish", "loss", "crash", "sell", "drop", "negative", "weak"]
            
            blob = " ".join(all_texts).lower()
            p_count = sum(blob.count(w) for w in pos)
            n_count = sum(blob.count(w) for w in neg)
            
            total = p_count + n_count + 1e-10
            sentiment = (p_count - n_count) / total
            value = float(np.clip(sentiment * 2, -1, 1))
            
            return WorldSignal(
                domain="NARRATIVE", source="google_news",
                region="GLOBAL", value=value, confidence=0.45,
                evidence_level="TESTABLE",
                timestamp=datetime.now(timezone.utc),
                temporal_layer="SURFACE",
                raw={"pos_count": p_count, "neg_count": n_count, "n_headlines": len(all_texts)},
                notes=f"Google News sentiment for {symbol}: {value:+.2f}"
            )
        except Exception as e:
            logger.warning(f"Google News sentiment failed for {symbol}: {e}")
            return WorldSignal(domain="NARRATIVE", source="google_news", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc), temporal_layer="SURFACE")

class RedditSentimentSignal:
    """
    Real-time sentiment from Reddit (r/investing, r/wallstreetbets).
    Fetches search JSON and performs simple keyword sentiment analysis.
    """
    
    SUBREDDITS = ["investing", "stocks", "wallstreetbets", "cryptocurrency"]
    
    def compute(self, symbol: str) -> WorldSignal:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (KindPath Assistant)"}
            all_texts = []
            
            query = symbol.upper()
            if "-" in query: query = query.split("-")[0]
            
            url = f"https://www.reddit.com/search.json?q={query}&sort=new&limit=20"
            
            with httpx.Client(headers=headers, follow_redirects=True, timeout=10) as client:
                r = client.get(url)
                
                # LOG RAW DATA
                get_raw_logger().log(
                    source="reddit", 
                    data=r.json() if r.status_code == 200 else r.text[:5000],
                    metadata={"symbol": symbol, "status": r.status_code}
                )
                
                if r.status_code == 200:
                    data = r.json()
                    children = data.get("data", {}).get("children", [])
                    for child in children:
                        post = child.get("data", {})
                        all_texts.append(post.get("title", ""))
                        all_texts.append(post.get("selftext", ""))
            
            if not all_texts:
                return WorldSignal(domain="NARRATIVE", source="reddit_sentiment", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc), temporal_layer="SURFACE")
            
            pos = ["buy", "long", "bullish", "moon", "green", "growth", "undervalued", "good", "win", "calls"]
            neg = ["sell", "short", "bearish", "crash", "red", "dump", "overvalued", "bad", "loss", "puts"]
            
            blob = " ".join(all_texts).lower()
            p_count = sum(blob.count(w) for w in pos)
            n_count = sum(blob.count(w) for w in neg)
            
            total = p_count + n_count + 1e-10
            sentiment = (p_count - n_count) / total
            value = float(np.clip(sentiment * 2, -1, 1))
            
            return WorldSignal(
                domain="NARRATIVE", source="reddit_sentiment",
                region="GLOBAL", value=value, confidence=0.40,
                evidence_level="TESTABLE",
                timestamp=datetime.now(timezone.utc),
                temporal_layer="SURFACE",
                raw={"pos_count": p_count, "neg_count": n_count, "n_posts": len(children)},
                notes=f"Reddit sentiment for {symbol}: {value:+.2f}"
            )
        except Exception as e:
            logger.warning(f"Reddit sentiment failed for {symbol}: {e}")
            return WorldSignal(domain="NARRATIVE", source="reddit_sentiment", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc), temporal_layer="SURFACE")

class FredRevisionSignal:
    """
    Measures the delta between original and revised macro data.
    High revision delta = unmeasured field disturbance (absence signal).
    """
    
    def compute(self, series_id: str = "GDP") -> WorldSignal:
        return WorldSignal(
            domain="MACRO", source="fred_revision",
            region="US", value=0.0, confidence=0.30,
            evidence_level="SPECULATIVE",
            timestamp=datetime.now(timezone.utc),
            temporal_layer="MEDIUM",
            notes="Revision delta tracking [SPECULATIVE]"
        )

class HighFrequencyVolatilitySignal:
    """
    Measures 5-minute volatility using yfinance intraday data.
    High volatility = IN-loading/Entropy. Low volatility = Stability/Syntropy.
    """
    
    def compute(self, symbol: str) -> WorldSignal:
        try:
            import yfinance as yf
            data = yf.Ticker(symbol).history(period="1d", interval="1m")
            if data.empty or len(data) < 5:
                return WorldSignal(domain="RISK_APPETITE", source="hf_volatility", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc), temporal_layer="SURFACE")
            
            last_5 = data["Close"].tail(5)
            vol = float(last_5.pct_change().std())
            value = float(np.clip(1.0 - (vol / 0.002) * 2, -1, 1))
            
            return WorldSignal(
                domain="RISK_APPETITE", source="hf_volatility",
                region="GLOBAL", value=value, confidence=0.60,
                evidence_level="ESTABLISHED",
                timestamp=datetime.now(timezone.utc),
                temporal_layer="SURFACE",
                raw={"5m_vol": vol},
                notes=f"5m volatility for {symbol}: {vol:.5f}"
            )
        except Exception as e:
            logger.warning(f"HF Volatility signal failed for {symbol}: {e}")
            return WorldSignal(domain="RISK_APPETITE", source="hf_volatility", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc), temporal_layer="SURFACE")

class CrossAssetEchoSignal:
    """
    Measures the 'Echo Break' (divergence) between an asset and its benchmark.
    """
    
    def compute(self, symbol: str, benchmark: str) -> WorldSignal:
        try:
            import yfinance as yf
            s_data = yf.Ticker(symbol).history(period="1mo")["Close"]
            b_data = yf.Ticker(benchmark).history(period="1mo")["Close"]
            
            if s_data.empty or b_data.empty: return WorldSignal(domain="ECHO", source="echo_break", region="SECTOR", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc), temporal_layer="SURFACE")
            
            corr = s_data.corr(b_data)
            ratio = s_data / b_data
            ratio_trend = (ratio.iloc[-1] - ratio.mean()) / (ratio.std() + 1e-10)
            value = float(np.clip(-(abs(ratio_trend) - 1.0) / 2.0, -1, 1))
            
            return WorldSignal(
                domain="ECHO", source="echo_break",
                region="SECTOR", value=value, confidence=0.50,
                evidence_level="TESTABLE",
                timestamp=datetime.now(timezone.utc),
                temporal_layer="SURFACE",
                raw={"corr": corr, "ratio_z": ratio_trend},
                notes=f"{symbol}/{benchmark} echo stability: {value:+.2f}"
            )
        except Exception:
            return WorldSignal(domain="ECHO", source="echo_break", region="SECTOR", value=0.0, confidence=0.0, evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc), temporal_layer="SURFACE")
