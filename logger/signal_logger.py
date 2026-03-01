"""
Signal Logger — Phase 7b
=========================
Logs every live DFTE signal run to a SQLite database so we can
run honest backtests against real ν, WFS, and STS data in 6+ months.

KINDFIELD principle: measure before claiming.
Price-proxy approximations cannot validate what live signal data can.

Schema:
  signals   — one row per symbol per run
  outcomes  — forward returns linked after they are realised

The real backtesting begins now. Six months of logged data will give
us the first honest ν, STS, and LSII validation against live signals.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

_HERE   = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "signal_history.db")

# Schema version — bump if schema changes (triggers migration warning)
SCHEMA_VERSION = 1


# ─── Schema ───────────────────────────────────────────────────────────────────

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_version   INTEGER NOT NULL DEFAULT 1,
    timestamp        TEXT    NOT NULL,   -- ISO-8601 UTC
    symbol           TEXT    NOT NULL,
    -- ν / BMR market field
    nu               REAL,
    mfs              REAL,
    mfs_label        TEXT,
    field_state      TEXT,
    lsii             REAL,
    lsii_flag        TEXT,
    curvature        REAL,
    -- KEPE world field
    wfs              REAL,
    wfs_label        TEXT,
    spi              REAL,
    opc              REAL,
    ei               REAL,
    interference_load REAL,
    sts_state        TEXT,
    wfs_history      TEXT,   -- JSON array of recent WFS floats
    -- KPRE domain scores (JSON object)
    domain_scores    TEXT,   -- full domain_scores dict from KEPEProfile
    kpre_score       REAL,   -- KPRE domain score
    kpre_capital_score REAL, -- KPRE_CAPITAL domain score
    language_score   REAL,   -- LANGUAGE domain score
    -- SAS
    sas_score        REAL,
    wolf_score       REAL,
    opacity_score    REAL,
    revenue_coherence REAL,
    capex_direction  REAL,
    ssi_gap          REAL,
    -- DFTE decision
    action           TEXT,
    tier             TEXT,
    conviction       REAL,
    position_size    REAL,
    rationale        TEXT,
    all_gates_passed INTEGER,  -- boolean 0/1
    -- CMAM
    cmam_mode        TEXT,
    sar              REAL,
    trade_type       TEXT,
    -- metadata
    run_mode         TEXT DEFAULT 'paper'
)
"""

_CREATE_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id   INTEGER NOT NULL REFERENCES signals(id),
    symbol      TEXT    NOT NULL,
    signal_ts   TEXT    NOT NULL,   -- copy of signals.timestamp for convenience
    forward_5d  REAL,
    forward_10d REAL,
    forward_20d REAL,
    forward_60d REAL,
    filled_at   TEXT,               -- ISO-8601 when this row was updated
    UNIQUE(signal_id)
)
"""

_CREATE_META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


# ─── Connection context manager ───────────────────────────────────────────────

@contextmanager
def _connection(db_path: str = DB_PATH):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ─── SignalLogger ─────────────────────────────────────────────────────────────

class SignalLogger:
    """
    Logs DFTE signal readings to SQLite for future live backtest validation.

    Usage:
        sl = SignalLogger()
        sl.log_signal(symbol, dfte_signal, kepe_profile, sas_profile, cmam_profile, tc)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with _connection(self.db_path) as con:
            con.execute(_CREATE_SIGNALS)
            con.execute(_CREATE_OUTCOMES)
            con.execute(_CREATE_META)
            # Store schema version
            con.execute(
                "INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),)
            )
            con.execute(
                "INSERT OR IGNORE INTO meta VALUES ('created_at', ?)",
                (datetime.now(timezone.utc).isoformat(),)
            )
        logger.debug(f"SignalLogger: schema ready at {self.db_path}")

    def log_signal(
        self,
        symbol: str,
        dfte_signal,              # DFTESignal
        kepe_profile,             # KEPEProfile
        sas_profile=None,         # SASProfile | None
        cmam_profile=None,        # CMAMProfile | None
        trade_classification=None, # TradeClassification | None
        run_mode: str = "paper",
    ) -> int:
        """
        Log one signal reading. Returns the new row id.
        Safe — never raises; logs warnings on failure.
        """
        try:
            return self._insert_signal(
                symbol, dfte_signal, kepe_profile,
                sas_profile, cmam_profile, trade_classification,
                run_mode,
            )
        except Exception as e:
            logger.warning(f"SignalLogger.log_signal failed for {symbol}: {e}")
            return -1

    def _insert_signal(
        self,
        symbol: str,
        sig,
        kepe,
        sas,
        cmam,
        tc,
        run_mode: str,
    ) -> int:
        # ── KEPE fields ──────────────────────────────────────────────────────
        domain = getattr(kepe, "domain_scores", {}) or {}
        wfs_history = getattr(kepe, "wfs_history", []) or []

        kpre_score         = domain.get("KPRE", None)
        kpre_capital_score = domain.get("KPRE_CAPITAL", None)
        language_score     = domain.get("LANGUAGE", None)

        # ── SAS fields ───────────────────────────────────────────────────────
        sas_score       = getattr(sas, "sas_score",       None) if sas else None
        wolf_score      = getattr(sas, "wolf_score",      None) if sas else None
        opacity_score   = getattr(sas, "opacity_score",   None) if sas else None
        rev_coherence   = getattr(sas, "revenue_coherence", None) if sas else None
        capex_dir       = getattr(sas, "capex_direction", None) if sas else None
        ssi_gap         = getattr(sas, "ssi_gap",         None) if sas else None

        # ── CMAM fields ──────────────────────────────────────────────────────
        cmam_mode = getattr(cmam, "mode", None) if cmam else None
        sar       = getattr(cmam, "sar",  None) if cmam else None
        trade_type = getattr(tc, "trade_type", None) if tc else None

        row = (
            SCHEMA_VERSION,
            datetime.now(timezone.utc).isoformat(),
            symbol,
            # BMR
            getattr(sig, "nu",          None),
            getattr(sig, "mfs",         None),
            getattr(sig, "mfs_label",   None),
            getattr(sig, "field_state", None),
            getattr(sig, "lsii",        None),
            getattr(sig, "lsii_flag",   None),
            getattr(sig, "curvature_k", None),
            # KEPE
            getattr(kepe, "wfs",               None),
            getattr(kepe, "wfs_label",         None),
            getattr(kepe, "spi",               None),
            getattr(kepe, "opc",               None),
            getattr(kepe, "entropy_indicator", None),
            getattr(kepe, "interference_load", None),
            getattr(kepe, "sts",               None),
            json.dumps(wfs_history),
            json.dumps(domain),
            kpre_score,
            kpre_capital_score,
            language_score,
            # SAS
            sas_score, wolf_score, opacity_score,
            rev_coherence, capex_dir, ssi_gap,
            # DFTE
            getattr(sig, "action",           None),
            getattr(sig, "tier",             None),
            getattr(sig, "conviction",       None),
            getattr(sig, "position_size_pct", None),
            getattr(sig, "rationale",        None),
            1 if getattr(sig, "all_gates_passed", False) else 0,
            # CMAM
            cmam_mode, sar, trade_type,
            run_mode,
        )

        with _connection(self.db_path) as con:
            cur = con.execute("""
                INSERT INTO signals (
                    schema_version, timestamp, symbol,
                    nu, mfs, mfs_label, field_state, lsii, lsii_flag, curvature,
                    wfs, wfs_label, spi, opc, ei, interference_load,
                    sts_state, wfs_history, domain_scores,
                    kpre_score, kpre_capital_score, language_score,
                    sas_score, wolf_score, opacity_score,
                    revenue_coherence, capex_direction, ssi_gap,
                    action, tier, conviction, position_size, rationale,
                    all_gates_passed,
                    cmam_mode, sar, trade_type,
                    run_mode
                ) VALUES (
                    ?,?,?,
                    ?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,
                    ?,?,?,
                    ?,?,?,
                    ?,?,?,?,?,?,
                    ?,?,?,?,?,?,
                    ?,?,?,
                    ?
                )
            """, row)
            signal_id = cur.lastrowid

        logger.debug(f"Logged signal id={signal_id} {symbol} {getattr(sig, 'action', '?')}")
        return signal_id

    # ─── Query helpers ────────────────────────────────────────────────────────

    def get_history(
        self,
        symbol: Optional[str] = None,
        days: int = 180,
    ) -> List[Dict]:
        """
        Return signal rows as list of dicts, newest first.
        Pass symbol=None for all symbols.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with _connection(self.db_path) as con:
            if symbol:
                rows = con.execute(
                    "SELECT * FROM signals WHERE symbol=? AND timestamp>=? ORDER BY timestamp DESC",
                    (symbol, cutoff)
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM signals WHERE timestamp>=? ORDER BY timestamp DESC",
                    (cutoff,)
                ).fetchall()
        return [dict(r) for r in rows]

    def link_outcome(
        self,
        signal_id: int,
        forward_5d:  Optional[float] = None,
        forward_10d: Optional[float] = None,
        forward_20d: Optional[float] = None,
        forward_60d: Optional[float] = None,
    ) -> None:
        """Insert or update the outcome row for a given signal_id."""
        with _connection(self.db_path) as con:
            # Fetch symbol + timestamp from signals table
            row = con.execute(
                "SELECT symbol, timestamp FROM signals WHERE id=?", (signal_id,)
            ).fetchone()
            if row is None:
                logger.warning(f"link_outcome: signal_id={signal_id} not found")
                return

            con.execute("""
                INSERT INTO outcomes (signal_id, symbol, signal_ts,
                    forward_5d, forward_10d, forward_20d, forward_60d, filled_at)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(signal_id) DO UPDATE SET
                    forward_5d  = excluded.forward_5d,
                    forward_10d = excluded.forward_10d,
                    forward_20d = excluded.forward_20d,
                    forward_60d = excluded.forward_60d,
                    filled_at   = excluded.filled_at
            """, (
                signal_id,
                row["symbol"],
                row["timestamp"],
                forward_5d, forward_10d, forward_20d, forward_60d,
                datetime.now(timezone.utc).isoformat(),
            ))
        logger.debug(f"Outcome linked for signal_id={signal_id}")

    def get_validation_ready_signals(
        self,
        min_age_days: int = 60,
    ) -> List[Dict]:
        """
        Return signals that are old enough to have 60-day outcomes available
        and don't yet have a complete outcome row.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
        with _connection(self.db_path) as con:
            rows = con.execute("""
                SELECT s.*
                FROM signals s
                LEFT JOIN outcomes o ON o.signal_id = s.id
                WHERE s.timestamp <= ?
                  AND (o.id IS NULL OR o.forward_60d IS NULL)
                ORDER BY s.timestamp ASC
            """, (cutoff,)).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> Dict:
        """Return a quick summary of the database state."""
        with _connection(self.db_path) as con:
            n_signals  = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            n_outcomes = con.execute(
                "SELECT COUNT(*) FROM outcomes WHERE forward_60d IS NOT NULL"
            ).fetchone()[0]
            n_pending  = con.execute("""
                SELECT COUNT(*) FROM signals s
                LEFT JOIN outcomes o ON o.signal_id = s.id
                WHERE o.id IS NULL
            """).fetchone()[0]
            earliest   = con.execute(
                "SELECT MIN(timestamp) FROM signals"
            ).fetchone()[0]
            latest     = con.execute(
                "SELECT MAX(timestamp) FROM signals"
            ).fetchone()[0]
            symbols    = [
                r[0] for r in con.execute(
                    "SELECT DISTINCT symbol FROM signals ORDER BY symbol"
                ).fetchall()
            ]
        return {
            "total_signals":          n_signals,
            "outcomes_complete":      n_outcomes,
            "outcomes_pending":       n_pending,
            "earliest_signal":        earliest,
            "latest_signal":          latest,
            "symbols":                symbols,
            "validation_ready_count": len(self.get_validation_ready_signals()),
        }

    def to_dataframe(
        self,
        symbol: Optional[str] = None,
        days: int = 180,
        include_outcomes: bool = True,
    ):
        """
        Return signals (and optionally outcomes) as a pandas DataFrame.
        Requires pandas — fails gracefully if not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise RuntimeError("pip install pandas — required for to_dataframe()")

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        if include_outcomes:
            query = """
                SELECT s.*, o.forward_5d, o.forward_10d, o.forward_20d, o.forward_60d
                FROM signals s
                LEFT JOIN outcomes o ON o.signal_id = s.id
                WHERE s.timestamp >= ?
            """
            params: tuple = (cutoff,)
        else:
            query = "SELECT * FROM signals WHERE timestamp >= ?"
            params = (cutoff,)

        if symbol:
            query += " AND s.symbol = ?" if include_outcomes else " AND symbol = ?"
            params += (symbol,)

        query += " ORDER BY timestamp DESC"

        with sqlite3.connect(self.db_path) as con:
            df = pd.read_sql_query(query, con, params=params)

        # Parse JSON columns
        for col in ("wfs_history", "domain_scores"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )

        return df
