"""
residue_logger.py — Forgazi Residue Accumulator for DFTE

Every prediction cycle generates two outputs: the prediction (what the model acts on)
and the residue (what the model can't account for). The standard practice is to throw
away the residue and aggregate it into error statistics. We don't.

We keep every individual residue item. The corpus, not the summary.

When the corpus is large enough, diverse enough, and spans sufficient ecological time,
the residue stops looking like random error and starts showing the shape of the market's
actual structure — the part the model was wrong about. That shape is the next model.

This is the BEC preparation layer. See kindpath-canon/BEC_DETECTION_GUIDE.md.
"""

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "residue_corpus.db"

# ── Residue types ──────────────────────────────────────────────────────────
# These map to the residue type taxonomy in BEC_DETECTION_GUIDE.md.
# Add new types rather than modifying existing ones — the historical
# categorisation must remain stable for BEC temporal span to be valid.

RESIDUE_TYPE_PREDICTION_MISS_TRENDING = "prediction_miss_trending"
RESIDUE_TYPE_PREDICTION_MISS_RANGING = "prediction_miss_ranging"
RESIDUE_TYPE_PREDICTION_MISS_VOLATILE = "prediction_miss_volatile"
RESIDUE_TYPE_FORGAZI_SIGNAL_SUPPRESSED = "forgazi_signal_suppressed"
RESIDUE_TYPE_TEMPORAL_DELTA_MISS = "temporal_delta_miss"      # Right direction, wrong timing
RESIDUE_TYPE_FRED_FRAME_ARTIFACT = "fred_frame_artifact"      # Signal was extraction pattern
RESIDUE_TYPE_FIELD_TIME_DELTA = "field_time_delta"            # UTC vs field-time divergence


@dataclass
class ResidueItem:
    """
    One item of prediction residue. Immutable once created.

    Not an error log. The residue is the signal that the current model
    cannot accommodate. Accumulating it honestly is the preparatory work
    for the next model.
    """
    id: str
    item_type: str                        # One of RESIDUE_TYPE_* constants
    created_utc: str                      # Forgazi reference — kept, never discarded

    # The prediction and reality
    predicted_value: float
    actual_value: float
    residual_magnitude: float             # abs(actual - predicted)
    residual_direction: str               # 'overshoot' | 'undershoot'

    # Signal context at prediction time
    input_signals: dict                   # The signals fed to the model
    fred_series_used: list[str]           # Which FRED series were active
    extraction_tags: list[str]            # Tagged extraction patterns present in inputs

    # Field-time context (see TEMPORAL_SOVEREIGNTY.md)
    field_time_context: dict              # TemporalContext-compatible dict

    # What the model couldn't see — free-text observation
    observer_note: str = ""

    # BEC tracking
    bec_cluster_id: Optional[str] = None      # Set when this item joins a condensate
    bec_threshold_crossed: bool = False       # True if this item contributed to BEC crossing

    # Provenance
    model_version: str = "unknown"
    source_repo: str = "kindpath-dfte"


@dataclass
class BECAssessment:
    """
    Current state of the residue corpus toward the BEC threshold.

    See BEC_DETECTION_GUIDE.md for full threshold conditions.
    """
    threshold_crossed: bool
    confidence: float                         # 0-1

    # Four conditions
    volume_met: bool
    diversity_met: bool
    temporal_span_met: bool
    cross_source_met: bool

    # Current state
    volume_current: int
    volume_threshold: int
    diversity_current: int                    # Distinct residue types with >5% representation
    diversity_threshold: int
    temporal_span_days: float
    temporal_span_threshold_days: float
    cross_source_repos: list[str]

    # If threshold crossed
    condensate_summary: Optional[str] = None
    condensate_cluster_id: Optional[str] = None


class ResidueLogger:
    """
    Accumulates prediction residue for the DFTE engine.

    Usage:
        logger = ResidueLogger()

        # After each prediction cycle resolves
        item = logger.log_prediction_miss(
            predicted=0.0023,
            actual=-0.0041,
            item_type=RESIDUE_TYPE_PREDICTION_MISS_TRENDING,
            input_signals={'drift': 0.0023, 'momentum': 0.71},
            fred_series_used=['UNRATE', 'FEDFUNDS'],
            extraction_tags=['post-fomc-window'],
            field_time_context=get_current_field_time(),
            observer_note="Model missed the reversal — volatility spike preceded by FOMC minutes"
        )

        # Check BEC status periodically
        assessment = logger.check_bec_threshold()
        if assessment.threshold_crossed:
            handle_bec_event(assessment)
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS residue_items (
                    id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    predicted_value REAL NOT NULL,
                    actual_value REAL NOT NULL,
                    residual_magnitude REAL NOT NULL,
                    residual_direction TEXT NOT NULL,
                    input_signals TEXT NOT NULL,
                    fred_series_used TEXT NOT NULL,
                    extraction_tags TEXT NOT NULL,
                    field_time_context TEXT NOT NULL,
                    observer_note TEXT DEFAULT '',
                    bec_cluster_id TEXT,
                    bec_threshold_crossed INTEGER DEFAULT 0,
                    model_version TEXT DEFAULT 'unknown',
                    source_repo TEXT DEFAULT 'kindpath-dfte'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_residue_type
                ON residue_items(item_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_utc
                ON residue_items(created_utc)
            """)
            conn.commit()

    def log_prediction_miss(
        self,
        predicted: float,
        actual: float,
        item_type: str,
        input_signals: dict,
        fred_series_used: list[str],
        extraction_tags: list[str],
        field_time_context: dict,
        observer_note: str = "",
        model_version: str = "unknown",
    ) -> ResidueItem:
        """
        Log a prediction residue item.

        Call this when a prediction cycle has resolved and the actual outcome
        is known. The item is stored permanently — it cannot be edited or removed.
        """
        magnitude = abs(actual - predicted)
        direction = "overshoot" if predicted > actual else "undershoot"

        item = ResidueItem(
            id=str(uuid.uuid4()),
            item_type=item_type,
            created_utc=datetime.now(timezone.utc).isoformat(),
            predicted_value=predicted,
            actual_value=actual,
            residual_magnitude=magnitude,
            residual_direction=direction,
            input_signals=input_signals,
            fred_series_used=fred_series_used,
            extraction_tags=extraction_tags,
            field_time_context=field_time_context,
            observer_note=observer_note,
            model_version=model_version,
        )
        self._store(item)
        return item

    def log_forgazi_signal(
        self,
        signal_source: str,
        signal_content: dict,
        suppression_reason: str,
        extraction_pattern_tag: str,
        field_time_context: dict,
    ) -> ResidueItem:
        """
        Log a signal that was present but suppressed by the current forgazi frame.

        A signal is 'suppressed' when the current model's extraction assumptions
        cause it to be discarded rather than acted on. These are the most important
        residue items — they represent the systematic blind spots of the frame.
        """
        return self.log_prediction_miss(
            predicted=0.0,
            actual=0.0,
            item_type=RESIDUE_TYPE_FORGAZI_SIGNAL_SUPPRESSED,
            input_signals={"signal_source": signal_source, "content": signal_content},
            fred_series_used=[signal_source] if signal_source.startswith("FRED:") else [],
            extraction_tags=[extraction_pattern_tag],
            field_time_context=field_time_context,
            observer_note=suppression_reason,
        )

    def log_field_time_context(
        self,
        utc_context: dict,
        field_time_context: dict,
        alignment_delta: float,
        note: str = "",
    ) -> ResidueItem:
        """
        Log when field-time and UTC-time produce a notable divergence in
        market state interpretation.

        alignment_delta: how far the field-time reading diverges from what
        UTC-based session analysis would predict (0.0 = identical, 1.0 = completely
        different reading of the same moment).
        """
        return self.log_prediction_miss(
            predicted=0.0,
            actual=alignment_delta,
            item_type=RESIDUE_TYPE_FIELD_TIME_DELTA,
            input_signals={"utc_context": utc_context, "field_time_context": field_time_context},
            fred_series_used=[],
            extraction_tags=["utc-vs-field-time-divergence"],
            field_time_context=field_time_context,
            observer_note=note,
        )

    def get_corpus(
        self,
        item_type: Optional[str] = None,
        since_utc: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve residue items with optional filtering.

        Returns raw dicts for flexibility — callers can deserialise as needed.
        """
        query = "SELECT * FROM residue_items WHERE 1=1"
        params = []
        if item_type:
            query += " AND item_type = ?"
            params.append(item_type)
        if since_utc:
            query += " AND created_utc >= ?"
            params.append(since_utc)
        query += " ORDER BY created_utc DESC"
        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        items = []
        for row in rows:
            d = dict(row)
            # Deserialise JSON fields
            for jf in ("input_signals", "fred_series_used", "extraction_tags", "field_time_context"):
                if isinstance(d[jf], str):
                    d[jf] = json.loads(d[jf])
            items.append(d)
        return items

    def check_bec_threshold(self) -> BECAssessment:
        """
        Assess whether the residue corpus has crossed the BEC threshold.

        References BEC_DETECTION_GUIDE.md: four conditions must all be met.
        This is intentionally conservative — false positives waste calibration cycles.
        """
        with sqlite3.connect(self.db_path) as conn:
            volume = conn.execute("SELECT COUNT(*) FROM residue_items").fetchone()[0]

            type_counts = dict(
                conn.execute(
                    "SELECT item_type, COUNT(*) FROM residue_items GROUP BY item_type"
                ).fetchall()
            )

            if volume > 0:
                span_row = conn.execute(
                    "SELECT MIN(created_utc), MAX(created_utc) FROM residue_items"
                ).fetchone()
                from dateutil.parser import parse as parse_dt
                earliest = parse_dt(span_row[0])
                latest = parse_dt(span_row[1])
                temporal_span_days = (latest - earliest).days
            else:
                temporal_span_days = 0.0

        # Condition 1: Volume (domain-appropriate threshold)
        # DFTE is a complex domain: ~500 items needed
        volume_threshold = 500
        volume_met = volume >= volume_threshold

        # Condition 2: Diversity (at least 5 types with >5% representation)
        diversity_threshold = 5
        significant_types = sum(
            1 for count in type_counts.values()
            if volume > 0 and (count / volume) >= 0.05
        )
        diversity_met = significant_types >= diversity_threshold

        # Condition 3: Temporal span (at least 4 ecological seasons = ~120 days)
        temporal_threshold_days = 120
        temporal_met = temporal_span_days >= temporal_threshold_days

        # Condition 4: Cross-source (this repo is one source; the KCE aggregator handles multi-repo)
        # For single-repo assessment, check if any items carry cross-source markers
        cross_source_repos = ["kindpath-dfte"]
        cross_source_met = False  # Full assessment requires the KCE aggregator

        all_met = volume_met and diversity_met and temporal_met

        return BECAssessment(
            threshold_crossed=all_met and cross_source_met,
            confidence=0.7 if all_met else 0.0,
            volume_met=volume_met,
            diversity_met=diversity_met,
            temporal_span_met=temporal_met,
            cross_source_met=cross_source_met,
            volume_current=volume,
            volume_threshold=volume_threshold,
            diversity_current=significant_types,
            diversity_threshold=diversity_threshold,
            temporal_span_days=temporal_span_days,
            temporal_span_threshold_days=temporal_threshold_days,
            cross_source_repos=cross_source_repos,
        )

    def _store(self, item: ResidueItem):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO residue_items (
                    id, item_type, created_utc,
                    predicted_value, actual_value, residual_magnitude, residual_direction,
                    input_signals, fred_series_used, extraction_tags, field_time_context,
                    observer_note, bec_cluster_id, bec_threshold_crossed,
                    model_version, source_repo
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id, item.item_type, item.created_utc,
                    item.predicted_value, item.actual_value,
                    item.residual_magnitude, item.residual_direction,
                    json.dumps(item.input_signals),
                    json.dumps(item.fred_series_used),
                    json.dumps(item.extraction_tags),
                    json.dumps(item.field_time_context),
                    item.observer_note,
                    item.bec_cluster_id,
                    int(item.bec_threshold_crossed),
                    item.model_version,
                    item.source_repo,
                ),
            )
            conn.commit()
