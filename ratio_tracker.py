"""
Ratio Tracker
=============
From theory session: the cross-context ratio matrix is a field snapshot.
6 contexts = 15 ratio pairs. Ratios over time = velocity vectors of the
underlying field.

Six contexts and their source scores
-------------------------------------
C1: physical_flow    KPRE physical score
C2: capital          KPRE capital score
C3: language         KPRE language score
C4: market           BMR/MFS score
C5: authenticity     SAS score
C6: echo             cross-asset correlation stability (computed here)

The ratio ratio(A,B) = score_A / score_B captures the relative loading
between two dimensions of the field. When ratios drift from their
rolling baseline, a field disturbance is detected (echo break).
When the disturbance exceeds known constraint bounds, the anomalous
variable is flagged as a "missing string" — an unmeasured influence
making the ratio anomalous.

Evidence posture: [TESTABLE]
Ratio trajectories as predictive signals require outcome validation.
Echo break detection requires calibration of threshold values.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Context definitions ──────────────────────────────────────────────────────

CONTEXTS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

CONTEXT_LABELS = {
    "C1": "physical_flow",
    "C2": "capital",
    "C3": "language",
    "C4": "macro_field",
    "C5": "psychosomatic",
    "C6": "echo_stability",
    "C7": "market_bmr",
    "C8": "authenticity_sas",
}

# All 28 ratio pairs (C(8,2))
RATIO_PAIRS: List[str] = [
    f"{a}_{b}"
    for i, a in enumerate(CONTEXTS)
    for b in CONTEXTS[i+1:]
]


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RatioSnapshot:
    """All 15 ratios from one signal cycle."""
    timestamp: str
    context_scores: Dict[str, float]
    ratios:         Dict[str, float]  # {pair_name: ratio_value}
    echo_stability_score: float       # mean(|ratio - baseline|) across all pairs
    anomalies: List[str]              # pairs flagged as echo breaks


@dataclass
class MissingString:
    """A ratio anomaly that exceeds all known constraint bounds."""
    pair_name:       str
    actual_ratio:    float
    predicted_ratio: float
    gap_magnitude:   float


# ─── SQLite helpers ───────────────────────────────────────────────────────────

@contextmanager
def _conn(db_path: str):
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


# ─── RatioTracker ─────────────────────────────────────────────────────────────

class RatioTracker:
    """
    Computes, logs, and analyses cross-context ratio dynamics.

    Usage:
        tracker = RatioTracker(db_path)
        snapshot = tracker.compute_and_log(signal_id, context_scores)
        # snapshot.echo_stability_score → float
        # snapshot.anomalies            → List[pair_name]
    """

    # Rolling window for baseline computation
    BASELINE_PERIODS = 20
    # Echo break threshold: |current - baseline| > this → anomaly
    ECHO_BREAK_THRESHOLD = 0.30
    # Missing string: |current - baseline| > this → unknown variable
    MISSING_STRING_THRESHOLD = 0.60

    def __init__(self, db_path: str):
        self.db_path = db_path

    # ─── Core ratio computation ────────────────────────────────────────────

    def compute_ratios(self, context_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Compute all 15 ratios from 6 context scores.
        ratio(A,B) = score_A / score_B — handles div/0 gracefully.
        Scores are expected in range [-1, 1]; small positive floor applied
        before division to avoid sign-flip artefacts.
        """
        ratios: Dict[str, float] = {}
        for pair in RATIO_PAIRS:
            a, b = pair.split("_")
            sa = context_scores.get(a, 0.0)
            sb = context_scores.get(b, 0.0)
            # Shift to [0.01, 2.01] range for stable ratios
            sa_pos = sa + 1.01
            sb_pos = max(sb + 1.01, 0.01)
            ratio = sa_pos / sb_pos
            ratios[pair] = round(ratio, 6)
        return ratios

    def log_ratios(self, signal_id: Optional[int], ratios: Dict[str, float], region: str = "GLOBAL") -> None:
        """Write ratio snapshot to ratio_history table with regionality."""
        ts = datetime.now(timezone.utc).isoformat()
        rows = [
            (signal_id, ts, pair, value, region)
            for pair, value in ratios.items()
        ]
        try:
            with _conn(self.db_path) as con:
                con.executemany(
                    "INSERT INTO ratio_history (signal_id, timestamp, pair_name, ratio_value, region)"
                    " VALUES (?,?,?,?,?)",
                    rows,
                )
        except Exception as e:
            logger.warning(f"RatioTracker.log_ratios failed: {e}")

    def log_missing_string(self, ms: MissingString) -> None:
        """Write a missing string detection to missing_strings table."""
        try:
            with _conn(self.db_path) as con:
                con.execute(
                    "INSERT INTO missing_strings"
                    " (timestamp, pair_name, actual_ratio, predicted_ratio, gap_magnitude)"
                    " VALUES (?,?,?,?,?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        ms.pair_name,
                        ms.actual_ratio,
                        ms.predicted_ratio,
                        ms.gap_magnitude,
                    ),
                )
        except Exception as e:
            logger.warning(f"RatioTracker.log_missing_string failed: {e}")

    # ─── History queries ───────────────────────────────────────────────────

    def _get_history(
        self,
        pair_name: str,
        n_periods: int,
    ) -> List[Tuple[str, float]]:
        """Return last n_periods (timestamp, ratio_value) for a pair, newest first."""
        try:
            with _conn(self.db_path) as con:
                rows = con.execute(
                    "SELECT timestamp, ratio_value FROM ratio_history"
                    " WHERE pair_name=? ORDER BY timestamp DESC LIMIT ?",
                    (pair_name, n_periods),
                ).fetchall()
            return [(r["timestamp"], r["ratio_value"]) for r in rows]
        except Exception as e:
            logger.debug(f"RatioTracker history query failed: {e}")
            return []

    def _rolling_baseline(
        self,
        pair_name: str,
        n_periods: Optional[int] = None,
    ) -> Optional[float]:
        """Mean ratio over the last n_periods observations. None if insufficient history."""
        n = n_periods or self.BASELINE_PERIODS
        history = self._get_history(pair_name, n)
        if len(history) < 3:
            return None
        values = [v for _, v in history]
        return sum(values) / len(values)

    # ─── Analysis methods ──────────────────────────────────────────────────

    def compute_ratio_trajectory(
        self,
        pair_name: str,
        n_periods: int = 10,
    ) -> float:
        """
        Rate of change of ratio over last n_periods.
        Returns velocity: positive = ratio growing, negative = declining.
        Zero if insufficient history.
        """
        history = self._get_history(pair_name, n_periods)
        if len(history) < 2:
            return 0.0
        # Endpoint-to-endpoint delta (same approach as STS)
        oldest_val = history[-1][1]
        newest_val = history[0][1]
        return round(newest_val - oldest_val, 6)

    def detect_echo_break(
        self,
        pair_name: str,
        current_ratio: float,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Compare current ratio against rolling baseline.
        Returns True if |current - baseline| > threshold.
        Echo break = field disturbance signal.
        """
        t = threshold or self.ECHO_BREAK_THRESHOLD
        baseline = self._rolling_baseline(pair_name)
        if baseline is None:
            return False  # Insufficient history — not enough data to judge
        return abs(current_ratio - baseline) > t

    def compute_echo_stability_score(
        self,
        ratios: Dict[str, float],
    ) -> float:
        """
        mean(|ratio_t - ratio_baseline|) across all 15 pairs.
        Lower = more stable field.
        Returns 0.0 if insufficient history for all pairs.
        """
        deviations = []
        for pair, current in ratios.items():
            baseline = self._rolling_baseline(pair)
            if baseline is not None:
                deviations.append(abs(current - baseline))

        if not deviations:
            return 0.0
        return round(sum(deviations) / len(deviations), 6)

    def detect_missing_string(
        self,
        pair_name: str,
        current_ratio: float,
        threshold: Optional[float] = None,
    ) -> Optional[MissingString]:
        """
        Flag when ratio behaves outside all known constraint bounds.
        Missing string = the unknown variable making the ratio anomalous.
        Returns MissingString if flagged, None otherwise.
        """
        t = threshold or self.MISSING_STRING_THRESHOLD
        baseline = self._rolling_baseline(pair_name)
        if baseline is None:
            return None

        gap = abs(current_ratio - baseline)
        if gap > t:
            ms = MissingString(
                pair_name       = pair_name,
                actual_ratio    = current_ratio,
                predicted_ratio = baseline,
                gap_magnitude   = round(gap, 6),
            )
            self.log_missing_string(ms)
            logger.info(
                f"Missing string detected: {pair_name} "
                f"actual={current_ratio:.4f} predicted={baseline:.4f} "
                f"gap={gap:.4f}"
            )
            return ms
        return None

    # ─── Top-level convenience method ─────────────────────────────────────

    def compute_and_log(
        self,
        signal_id: Optional[int],
        context_scores: Dict[str, float],
        region: str = "GLOBAL"
    ) -> RatioSnapshot:
        """
        Full pipeline: compute 28 ratios, log per region, analyse for anomalies.
        Returns RatioSnapshot with echo_stability_score and anomaly list.
        Safe — never raises.
        """
        try:
            ratios = self.compute_ratios(context_scores)
            self.log_ratios(signal_id, ratios, region)

            echo_stability = self.compute_echo_stability_score(ratios)

            anomalies: List[str] = []
            for pair, ratio in ratios.items():
                if self.detect_echo_break(pair, ratio):
                    anomalies.append(pair)
                    self.detect_missing_string(pair, ratio)

            return RatioSnapshot(
                timestamp            = datetime.now(timezone.utc).isoformat(),
                context_scores       = context_scores,
                ratios               = ratios,
                echo_stability_score = echo_stability,
                anomalies            = anomalies,
            )
        except Exception as e:
            logger.warning(f"RatioTracker.compute_and_log failed: {e}")
            return RatioSnapshot(
                timestamp            = datetime.now(timezone.utc).isoformat(),
                context_scores       = context_scores,
                ratios               = {},
                echo_stability_score = 0.0,
                anomalies            = [],
            )

    def get_top_anomalies(self, n: int = 3) -> List[dict]:
        """
        Return the n most recently anomalous ratio pairs from missing_strings.
        Useful for dashboard display.
        """
        try:
            with _conn(self.db_path) as con:
                rows = con.execute(
                    "SELECT pair_name, actual_ratio, predicted_ratio, gap_magnitude, timestamp"
                    " FROM missing_strings ORDER BY timestamp DESC LIMIT ?",
                    (n,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
