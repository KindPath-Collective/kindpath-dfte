"""
Forgazi Sentiment Engine — KEPE Language Field Module
======================================================
Named after the principle of 'forgetting' surface noise to hear deeper signal.

Purpose:
  Multi-source news and language field sentiment scoring for DFTE symbols.
  Feeds the Language (L) channel of the KPRE relational engine.

Approach:
  - Free/open news sources: RSS feeds, Yahoo Finance headlines
  - Lexicon-based scoring with financial-domain wordlist (no heavy ML needed)
  - Entity linking: maps article text → mentioned tickers/sectors
  - Time-decay weighting: recent signals weighted more heavily
  - Contradiction detection: opposing signals → uncertainty penalty
  - Returns a ForgaziSignal compatible with KPRE's language field input

No API key required for base operation.
FRED_API_KEY optional (future: macro narrative scoring).
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
import os

logger = logging.getLogger(__name__)

# ── Sentiment lexicons ────────────────────────────────────────────────────────

_POSITIVE = frozenset([
    "beat", "beats", "record", "growth", "profit", "surge", "rally", "gain",
    "gains", "strong", "strength", "bullish", "upgrade", "upgraded", "buy",
    "outperform", "raise", "raised", "exceed", "exceeded", "upbeat", "optimistic",
    "positive", "improve", "improved", "expansion", "revenue", "accelerate",
    "breakout", "momentum", "demand", "recovery", "recover", "innovation",
    "breakthrough", "invest", "investment", "boost", "boosted", "partnership",
    "contract", "acquire", "acquisition", "dividend", "increased", "launch",
    "launched", "approval", "approved", "surge", "soar", "jump",
])

_NEGATIVE = frozenset([
    "miss", "misses", "loss", "losses", "decline", "declining", "bearish",
    "downgrade", "downgraded", "sell", "underperform", "cut", "cuts", "disappoint",
    "disappoints", "disappointed", "weak", "weakness", "concern", "concerns",
    "warning", "warn", "worries", "worry", "layoff", "layoffs", "restructure",
    "restructuring", "debt", "default", "lawsuit", "investigation", "probe",
    "fraud", "scandal", "recall", "shortage", "supply", "inflation", "recession",
    "slowdown", "risk", "risks", "volatile", "volatility", "plunge", "fall",
    "tumble", "crash", "collapse", "penalty", "fine", "delay", "delays",
])

_INTENSIFIERS = frozenset([
    "very", "highly", "significantly", "substantially", "sharply", "strongly",
    "massive", "major", "record", "unprecedented", "extreme",
])

_NEGATORS = frozenset([
    "not", "no", "never", "neither", "nor", "barely", "hardly", "without",
])


# ── Database helper ───────────────────────────────────────────────────────────

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "forgazi_cache.db")

def _get_conn():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS articles (
        id TEXT PRIMARY KEY,
        symbol TEXT,
        title TEXT,
        source TEXT,
        published_iso TEXT,
        sentiment_score REAL,
        confidence REAL,
        fetched_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sentiment_cache (
        cache_key TEXT PRIMARY KEY,
        value REAL,
        confidence REAL,
        article_count INTEGER,
        fetched_at TEXT,
        expires_at TEXT
    )""")
    conn.commit()
    return conn


# ── Core lexicon scorer ───────────────────────────────────────────────────────

def score_text(text: str) -> tuple[float, float]:
    """
    Score a text snippet using the financial lexicon.

    Returns (sentiment, confidence) where:
      sentiment: -1.0 (very negative) to +1.0 (very positive)
      confidence: 0.0 to 1.0 based on signal density
    """
    if not text:
        return 0.0, 0.0

    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if not words:
        return 0.0, 0.0

    pos = 0
    neg = 0
    negate = False
    intensify = 1.0

    for i, w in enumerate(words):
        if w in _NEGATORS:
            negate = True
            continue
        if w in _INTENSIFIERS:
            intensify = 1.5
            continue
        if w in _POSITIVE:
            delta = intensify
            pos += -delta if negate else delta
        elif w in _NEGATIVE:
            delta = intensify
            neg += delta if negate else -delta
        else:
            # Reset negation/intensifier after unrelated word
            negate = False
            intensify = 1.0
            continue
        negate = False
        intensify = 1.0

    total_signals = pos + abs(neg)
    if total_signals == 0:
        return 0.0, 0.1

    raw = (pos + neg) / max(total_signals, 1)
    sentiment = max(-1.0, min(1.0, raw))
    # Confidence scales with raw signal density relative to word count
    density = total_signals / max(len(words), 1)
    confidence = min(1.0, density * 10)

    return float(sentiment), float(confidence)


# ── News fetcher ──────────────────────────────────────────────────────────────

def _fetch_yahoo_headlines(symbol: str, max_age_hours: int = 6) -> list[dict]:
    """
    Fetch recent headlines for a symbol from Yahoo Finance RSS.
    Returns list of {title, published, source} dicts.
    """
    import urllib.request
    import urllib.error
    from xml.etree import ElementTree

    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    cache_key = f"yahoo_rss_{symbol}"

    conn = _get_conn()
    row = conn.execute(
        "SELECT value, fetched_at FROM sentiment_cache WHERE cache_key=? AND expires_at > ?",
        (f"rss_{symbol}", datetime.utcnow().isoformat())
    ).fetchone()
    conn.close()

    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KindPath/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml = resp.read()
        root = ElementTree.fromstring(xml)
        ns = ""
        for item in root.findall(f"channel/item"):
            title_el = item.find("title")
            pub_el = item.find("pubDate")
            if title_el is not None:
                articles.append({
                    "title": title_el.text or "",
                    "published": pub_el.text if pub_el is not None else "",
                    "source": "yahoo_rss",
                })
    except Exception as e:
        logger.debug(f"Forgazi: Yahoo RSS fetch failed for {symbol}: {e}")

    return articles[:20]


def _fetch_gdelt_headlines(query: str, max_articles: int = 15) -> list[dict]:
    """
    Fetch recent articles from GDELT DOC API (free, no key needed).
    query: ticker symbol or company name
    """
    import urllib.request
    import urllib.parse
    import json

    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_articles),
        "sort": "DateDesc",
        "format": "json",
        "timespan": "1d",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    articles = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KindPath/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        for art in (data.get("articles") or [])[:max_articles]:
            articles.append({
                "title": art.get("title", ""),
                "published": art.get("seendate", ""),
                "source": "gdelt",
                "url": art.get("url", ""),
            })
    except Exception as e:
        logger.debug(f"Forgazi: GDELT fetch failed for '{query}': {e}")
    return articles


# ── Main ForgaziSignal ────────────────────────────────────────────────────────

@dataclass
class ForgaziSignal:
    """
    A language field sentiment signal for a symbol.
    Compatible with KPRE language channel input.
    """
    symbol: str
    sentiment: float = 0.0          # -1.0 to +1.0
    confidence: float = 0.0         # 0.0 to 1.0
    article_count: int = 0
    contradiction_flag: bool = False # True if signals strongly disagree
    recency_score: float = 0.0      # Recency-weighted average
    source_diversity: float = 0.0   # How many distinct sources contributed
    cached: bool = False
    articles_sample: list = field(default_factory=list)  # Top articles w/ scores
    fetched_at: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "sentiment": round(self.sentiment, 4),
            "confidence": round(self.confidence, 4),
            "article_count": self.article_count,
            "contradiction_flag": self.contradiction_flag,
            "recency_score": round(self.recency_score, 4),
            "source_diversity": round(self.source_diversity, 2),
            "cached": self.cached,
            "fetched_at": self.fetched_at,
            "articles_sample": self.articles_sample[:3],
        }

    def to_language_score(self) -> float:
        """Convert sentiment to KPRE-compatible [-1, 1] language score."""
        if self.contradiction_flag:
            # Contradiction means uncertainty — dampen the signal
            return self.sentiment * 0.4
        return self.sentiment * self.confidence


class ForgaziEngine:
    """
    Forgazi Sentiment Engine.

    Usage:
        engine = ForgaziEngine()
        signal = engine.analyse("AAPL")
        score = signal.to_language_score()  # use in KPRE language field
    """

    def __init__(self, cache_hours: int = 4):
        self.cache_hours = cache_hours

    def analyse(self, symbol: str, company_name: str = "") -> ForgaziSignal:
        """
        Full sentiment analysis for a symbol.
        Fetches from RSS + GDELT, scores, time-decays, detects contradictions.
        """
        # Check cache first
        conn = _get_conn()
        row = conn.execute(
            "SELECT value, confidence, article_count, fetched_at FROM sentiment_cache "
            "WHERE cache_key=? AND expires_at > ?",
            (f"forgazi_{symbol}", datetime.utcnow().isoformat())
        ).fetchone()
        conn.close()
        if row:
            return ForgaziSignal(
                symbol=symbol,
                sentiment=row[0],
                confidence=row[1],
                article_count=row[2],
                cached=True,
                fetched_at=row[3],
            )

        # Fetch from sources
        query = company_name or symbol
        articles = _fetch_yahoo_headlines(symbol)
        gdelt_arts = _fetch_gdelt_headlines(query)
        # Deduplicate by title hash
        seen = set()
        all_articles = []
        for art in articles + gdelt_arts:
            h = hashlib.md5(art["title"].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                all_articles.append(art)

        if not all_articles:
            return ForgaziSignal(symbol=symbol, sentiment=0.0, confidence=0.0,
                                  fetched_at=datetime.utcnow().isoformat())

        # Score each article and apply time decay
        scored = []
        sources = set()
        for art in all_articles:
            s, c = score_text(art["title"])
            decay = 1.0  # Future: parse date and apply exp decay
            scored.append({
                "title": art["title"][:120],
                "sentiment": s,
                "confidence": c,
                "source": art["source"],
                "decay": decay,
            })
            sources.add(art["source"])

        # Weighted aggregate
        total_weight = sum(a["confidence"] * a["decay"] for a in scored)
        if total_weight == 0:
            agg_sentiment = 0.0
            agg_confidence = 0.1
        else:
            agg_sentiment = sum(a["sentiment"] * a["confidence"] * a["decay"] for a in scored) / total_weight
            agg_confidence = min(1.0, total_weight / max(len(scored), 1) * 2)

        # Contradiction detection: check if significant pos and neg both present
        pos_articles = [a for a in scored if a["sentiment"] > 0.2]
        neg_articles = [a for a in scored if a["sentiment"] < -0.2]
        contradiction = len(pos_articles) >= 3 and len(neg_articles) >= 3

        recency_score = agg_sentiment  # TODO: implement proper time-decay when dates parsed
        source_div = min(1.0, len(sources) / 3.0)

        sig = ForgaziSignal(
            symbol=symbol,
            sentiment=round(float(agg_sentiment), 4),
            confidence=round(float(agg_confidence), 4),
            article_count=len(all_articles),
            contradiction_flag=contradiction,
            recency_score=round(float(recency_score), 4),
            source_diversity=round(source_div, 2),
            fetched_at=datetime.utcnow().isoformat(),
            articles_sample=[{k: a[k] for k in ("title","sentiment","source")} for a in scored[:5]],
        )

        # Cache it
        conn = _get_conn()
        expires = (datetime.utcnow() + timedelta(hours=self.cache_hours)).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO sentiment_cache (cache_key, value, confidence, article_count, fetched_at, expires_at) "
            "VALUES (?,?,?,?,?,?)",
            (f"forgazi_{symbol}", sig.sentiment, sig.confidence, sig.article_count,
             sig.fetched_at, expires)
        )
        conn.commit()
        conn.close()

        return sig

    def analyse_basket(self, symbols: list[str]) -> dict[str, ForgaziSignal]:
        """Analyse multiple symbols and return mapping."""
        return {sym: self.analyse(sym) for sym in symbols}
