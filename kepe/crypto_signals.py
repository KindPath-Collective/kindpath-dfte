"""
KEPE — Crypto Signals (On-Chain & Market)
=========================================
Specific signals for digital assets.
Source: CoinGecko (Free Tier).
"All data matters."
"""

from __future__ import annotations
import logging
import httpx
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from .indicators import WorldSignal
from logger.raw_data_logger import get_raw_logger

logger = logging.getLogger(__name__)

class CoinGeckoSignal:
    """
    Fetches crypto market data and developer/community stats.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def compute(self, symbol: str) -> WorldSignal:
        coin_id = self._map_symbol_to_id(symbol)
        if not coin_id:
            return WorldSignal(domain="MARKET_PHYSICS", source="coingecko", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="ESTABLISHED", timestamp=datetime.now(timezone.utc))
            
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=true&developer_data=true&sparkline=false"
            
            with httpx.Client(timeout=10) as client:
                r = client.get(url)
                get_raw_logger().log(source="coingecko", data=r.json() if r.status_code == 200 else r.text, metadata={"symbol": symbol, "coin_id": coin_id})
                
                if r.status_code != 200:
                    return WorldSignal(domain="MARKET_PHYSICS", source="coingecko", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="ESTABLISHED", timestamp=datetime.now(timezone.utc))
                
                data = r.json()
                market_data = data.get("market_data", {})
                comm_data = data.get("community_data", {})
                dev_data = data.get("developer_data", {})
                
                reddit_active = comm_data.get("reddit_accounts_active_48h", 0)
                commits = dev_data.get("commit_count_4_weeks", 0)
                
                activity_score = min(reddit_active / 5000, 1.0)
                dev_score = min(commits / 50, 1.0)
                
                network_health = (activity_score * 0.6) + (dev_score * 0.4)
                val = (network_health * 2) - 1.0
                
                return WorldSignal(
                    domain="NETWORK_HEALTH", source="coingecko",
                    region="GLOBAL", value=val, confidence=0.70,
                    evidence_level="ESTABLISHED",
                    timestamp=datetime.now(timezone.utc),
                    temporal_layer="MEDIUM",
                    raw={"commits_4w": commits, "reddit_active": reddit_active},
                    notes=f"Crypto Network Health: {val:.2f}"
                )

        except Exception as e:
            logger.warning(f"CoinGecko signal failed for {symbol}: {e}")
            return WorldSignal(domain="MARKET_PHYSICS", source="coingecko", region="GLOBAL", value=0.0, confidence=0.0, evidence_level="ESTABLISHED", timestamp=datetime.now(timezone.utc))

    def _map_symbol_to_id(self, symbol: str) -> Optional[str]:
        m = {
            "BTC-USD": "bitcoin", "ETH-USD": "ethereum",
            "SOL-USD": "solana", "ADA-USD": "cardano",
            "POL-USD": "polygon-ecosystem", "OP-USD": "optimism",
            "ARB-USD": "arbitrum", "LDO-USD": "lido-dao",
            "LINK-USD": "chainlink"
        }
        return m.get(symbol)
