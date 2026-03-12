"""
residue_backtest.py — Residue-Accumulating Backtest Runner

Standard backtesting aggregates prediction errors into summary statistics.
This backtester doesn't. It keeps every individual prediction miss as a
residue item and accumulates them into the corpus.

The corpus that grows during backtesting is not a diagnostic tool.
It is the first layer of the BEC accumulation process — the ground truth
for where the model is systematically wrong, in what circadian phases,
under what FRED signal conditions, and with what residue magnitude.

See kindpath-canon/BEC_DETECTION_GUIDE.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from logger.residue_logger import (
    ResidueLogger,
    ResidueItem,
    RESIDUE_TYPE_PREDICTION_MISS_TRENDING,
    RESIDUE_TYPE_PREDICTION_MISS_RANGING,
    RESIDUE_TYPE_PREDICTION_MISS_VOLATILE,
    RESIDUE_TYPE_TEMPORAL_DELTA_MISS,
)
from dfte.field_time_bridge import build_temporal_context


@dataclass
class BacktestCycle:
    """
    One resolved prediction cycle in the backtest.

    All fields are available at resolution time — the predicted value
    was locked before the cycle ran; the actual is now known.
    """
    cycle_id: str
    utc_timestamp: str

    predicted_direction: float          # +1 long, -1 short, 0 flat
    predicted_magnitude: float          # Expected pip move or return %
    actual_direction: float             # What actually happened
    actual_magnitude: float             # How much it moved

    market_regime: str                  # 'trending' | 'ranging' | 'volatile'
    fred_series_active: list[str]       # Which FRED series were in the signal set
    input_signal_snapshot: dict         # Full signal state at prediction time

    # UTC-based session label (forgazi reference, kept for archive)
    session_label: str                  # 'London' | 'NY' | 'Asia' | 'overlap'

    # Field-time context
    latitude_band: int = -35
    hemisphere: str = "south"
    utc_offset_hours: float = 10.0


@dataclass
class BacktestResidueReport:
    """
    Summary of residue accumulation from a backtest run.

    The individual items are in the SQLite corpus — this report gives
    the structural overview for immediate consumption.
    """
    total_cycles: int
    total_residue_items: int
    residue_rate: float                    # items / cycles

    # Distribution by regime type
    trending_residue_count: int
    ranging_residue_count: int
    volatile_residue_count: int

    # Distribution by circadian quarter
    residue_by_quarter: dict[int, int]

    # Largest residues
    top_residues_by_magnitude: list[dict]

    # BEC assessment
    bec_assessment: Optional[dict] = None


class ResidueBacktester:
    """
    Runs a historical backtest while accumulating prediction miss residue.

    Integrates with the standard DFTE backtest engine (backtest_engine.py)
    by accepting its cycle outputs and routing residue items to the
    ResidueLogger corpus.

    Usage:
        backtester = ResidueBacktester()

        for cycle in load_historical_cycles("2023-01-01", "2024-01-01"):
            residue_item = backtester.process_cycle(cycle)
            # None if the prediction was correct (no residue)
            # ResidueItem if a miss occurred

        report = backtester.get_report()
        print(report.residue_rate)
        print(report.bec_assessment)
    """

    def __init__(self, residue_db_path: Optional[Path] = None):
        self.residue_logger = ResidueLogger(residue_db_path) if residue_db_path else ResidueLogger()
        self._cycle_count = 0
        self._residue_count = 0

    def process_cycle(self, cycle: BacktestCycle) -> Optional[ResidueItem]:
        """
        Process one resolved backtest cycle.

        Returns a ResidueItem if the prediction missed, None if it was correct.

        A 'miss' is defined by two independent criteria:
        1. Direction wrong (caught_direction = actual_direction != np.sign(predicted))
        2. Magnitude error > 1.5x predicted (structural miss, not just sizing)
        """
        self._cycle_count += 1

        # Parse timestamp
        try:
            from dateutil.parser import parse as parse_dt
            utc_dt = parse_dt(cycle.utc_timestamp).replace(tzinfo=timezone.utc)
        except Exception:
            utc_dt = datetime.now(timezone.utc)

        # Build field-time context
        field_time = build_temporal_context(
            utc_dt=utc_dt,
            latitude_band=cycle.latitude_band,
            hemisphere=cycle.hemisphere,  # type: ignore[arg-type]
            utc_offset_hours=cycle.utc_offset_hours,
        )

        # Check if this is a residue miss
        direction_correct = (
            (cycle.predicted_direction > 0 and cycle.actual_direction > 0)
            or (cycle.predicted_direction < 0 and cycle.actual_direction < 0)
            or (cycle.predicted_direction == 0 and abs(cycle.actual_direction) < 0.001)
        )

        magnitude_ratio = (
            abs(cycle.actual_magnitude / max(abs(cycle.predicted_magnitude), 1e-8))
        )
        magnitude_miss = magnitude_ratio > 1.5 or magnitude_ratio < 0.5

        if direction_correct and not magnitude_miss:
            # Correct prediction — no residue
            return None

        # Classify miss type
        if not direction_correct:
            if cycle.market_regime == "trending":
                item_type = RESIDUE_TYPE_PREDICTION_MISS_TRENDING
            elif cycle.market_regime == "volatile":
                item_type = RESIDUE_TYPE_PREDICTION_MISS_VOLATILE
            else:
                item_type = RESIDUE_TYPE_PREDICTION_MISS_RANGING
        else:
            # Magnitude miss — right direction, wrong size
            item_type = RESIDUE_TYPE_TEMPORAL_DELTA_MISS

        # Compute residual
        predicted_net = cycle.predicted_direction * cycle.predicted_magnitude
        actual_net = cycle.actual_direction * cycle.actual_magnitude

        note_parts = []
        if not direction_correct:
            note_parts.append(
                f"Direction wrong: predicted {cycle.predicted_direction:+.0f}, "
                f"actual {cycle.actual_direction:+.0f}"
            )
        if magnitude_miss:
            note_parts.append(
                f"Magnitude ratio: {magnitude_ratio:.2f} "
                f"(predicted {cycle.predicted_magnitude:.4f}, "
                f"actual {cycle.actual_magnitude:.4f})"
            )
        note_parts.append(f"Regime: {cycle.market_regime}")
        note_parts.append(f"FRED series: {', '.join(cycle.fred_series_active)}")
        note_parts.append(f"Session [forgazi]: {cycle.session_label}")
        note_parts.append(
            f"Field-time: Q{field_time.circadian_quarter} {field_time.circadian_state}"
        )

        item = self.residue_logger.log_prediction_miss(
            predicted=predicted_net,
            actual=actual_net,
            item_type=item_type,
            input_signals=cycle.input_signal_snapshot,
            fred_series_used=cycle.fred_series_active,
            extraction_tags=[cycle.session_label, cycle.market_regime, "backtest"],
            field_time_context=field_time.to_dict(),
            observer_note=" | ".join(note_parts),
        )

        self._residue_count += 1
        return item

    def process_cycles_batch(
        self, cycles: Iterator[BacktestCycle]
    ) -> list[Optional[ResidueItem]]:
        """Process an iterator of cycles, returning residue items where they occurred."""
        return [self.process_cycle(c) for c in cycles]

    def get_report(self) -> BacktestResidueReport:
        """
        Build a residue report from the current corpus state.
        """
        corpus = self.residue_logger.get_corpus()

        quarterly: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        type_counts: dict[str, int] = {
            RESIDUE_TYPE_PREDICTION_MISS_TRENDING: 0,
            RESIDUE_TYPE_PREDICTION_MISS_RANGING: 0,
            RESIDUE_TYPE_PREDICTION_MISS_VOLATILE: 0,
        }

        for item in corpus:
            ftc = item.get("field_time_context", {})
            if isinstance(ftc, str):
                import json
                ftc = json.loads(ftc)
            q = ftc.get("circadian_quarter", 0)
            if q in quarterly:
                quarterly[q] += 1

            t = item.get("item_type", "")
            if t in type_counts:
                type_counts[t] += 1

        # Top residues by magnitude
        sorted_corpus = sorted(corpus, key=lambda x: x.get("residual_magnitude", 0), reverse=True)
        top_items = [
            {
                "id": i.get("id", ""),
                "type": i.get("item_type", ""),
                "magnitude": i.get("residual_magnitude", 0),
                "created_utc": i.get("created_utc", ""),
                "note": i.get("observer_note", "")[:120],
            }
            for i in sorted_corpus[:10]
        ]

        bec = None
        try:
            bec_obj = self.residue_logger.check_bec_threshold()
            bec = {
                "threshold_crossed": bec_obj.threshold_crossed,
                "volume_met": bec_obj.volume_met,
                "diversity_met": bec_obj.diversity_met,
                "temporal_span_met": bec_obj.temporal_span_met,
                "volume_current": bec_obj.volume_current,
                "volume_threshold": bec_obj.volume_threshold,
            }
        except Exception:
            pass

        total = len(corpus)
        return BacktestResidueReport(
            total_cycles=self._cycle_count,
            total_residue_items=total,
            residue_rate=total / max(self._cycle_count, 1),
            trending_residue_count=type_counts[RESIDUE_TYPE_PREDICTION_MISS_TRENDING],
            ranging_residue_count=type_counts[RESIDUE_TYPE_PREDICTION_MISS_RANGING],
            volatile_residue_count=type_counts[RESIDUE_TYPE_PREDICTION_MISS_VOLATILE],
            residue_by_quarter=quarterly,
            top_residues_by_magnitude=top_items,
            bec_assessment=bec,
        )
