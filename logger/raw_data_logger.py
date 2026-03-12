"""
Raw Data Logger — The 'All Data Matters' Archive.
===================================================
Ingests and persists raw JSON/text from every API call.
Nothing is discarded. 'Immediately irrelevant' data is stored for future context.

Storage:
  - Local: raw_field_data/YYYY-MM-DD/source_timestamp.json
  - Cloud: Synced to gs://kindpath-raw-data (if configured)

"We are looking to predict market trends through widespread data analysis, 
keeping all immediately irrelevant data for further analysis within new contexts."
"""

from __future__ import annotations
import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class RawDataLogger:
    def __init__(self, base_dir: str = "raw_field_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log(self, source: str, data: Any, metadata: Optional[dict] = None) -> str:
        """
        Log raw data.
        source: 'reddit', 'fred', 'google_news', 'weather', etc.
        data: The raw response object (dict, list, str).
        metadata: Context (symbol, query, latency, etc).
        
        Returns the file path written.
        """
        try:
            now = datetime.now(timezone.utc)
            date_str = now.strftime("%Y-%m-%d")
            ts_str = now.strftime("%H-%M-%S-%f")
            
            # Create daily directory
            day_dir = self.base_dir / date_str
            day_dir.mkdir(exist_ok=True)
            
            # Generate filename
            # content_hash to avoid duplicates? Maybe unnecessary if we timestamp precisely.
            # actually, let's keep it simple: source_symbol_timestamp.json
            
            symbol = (metadata or {}).get("symbol", "GLOBAL")
            filename = f"{source}_{symbol}_{ts_str}.json"
            file_path = day_dir / filename
            
            payload = {
                "timestamp": now.isoformat(),
                "source": source,
                "metadata": metadata or {},
                "raw_data": data
            }
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
                
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to log raw data for {source}: {e}")
            return ""

# Singleton instance
_logger_instance = None

def get_raw_logger() -> RawDataLogger:
    global _logger_instance
    if _logger_instance is None:
        # Check environment for path override
        path = os.environ.get("RAW_DATA_PATH", "raw_field_data")
        _logger_instance = RawDataLogger(path)
    return _logger_instance
