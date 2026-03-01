"""
CMAM — Capital Maturity Allocation Model
==========================================
Governs ST/LT capital routing as a function of fund size F.

Governing equation:
  SAR(F) = 0                               when F < X1  (ST mode)
  SAR(F) = (F − X1) / (X2 − X1) * SAR_max when X1 ≤ F < X2  (transition)
  SAR(F) = SAR_max                         when F ≥ X2  (mature mode)

  ST_allocation = 1 − SAR(F)
  LT_allocation = SAR(F)

Default parameters (operator-configurable):
  X1            = 10,000   pure ST capital-gains mode below this
  X2            = 100,000  full SAR_max syntropy allocation above this
  SAR_max       = 0.40     never more than 40% in LT positions
  short_book_cap = 0.15    max 15% of fund in short positions simultaneously

Capital routing rules:
  ST trades : any non-extractive asset, NANO/MID tier, MFS-driven
  LT trades : syntropic category ONLY, STS must be LOADING or STABLE
  Short profits : mandatory route to lt_budget — cannot grow short book
  Extractive    : hard blocked always, regardless of fund size

Gates:
  mirror_check()         — flags if trade requires behaving like what we oppose
  cooling_off_required() — flags entropy→syntropy conversion reasoning for 24h hold
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("dfte.cmam")

# ─── Default parameters ───────────────────────────────────────────────────────

_DEFAULT_X1          = 10_000.0
_DEFAULT_X2          = 100_000.0
_DEFAULT_SAR_MAX     = 0.40
_DEFAULT_SHORT_CAP   = 0.15

# ─── Output dataclasses ───────────────────────────────────────────────────────

@dataclass
class CMAMProfile:
    fund_value:      float
    sar:             float   # current syntropy allocation ratio 0 → SAR_max
    st_allocation:   float
    lt_allocation:   float
    mode:            str     # ST_MODE | TRANSITION | MATURE
    st_budget:       float   # dollar amount available for ST trades
    lt_budget:       float   # dollar amount available for LT trades
    short_remaining: float   # remaining short book capacity ($)
    short_used:      float   # short book already deployed ($)
    x1:              float
    x2:              float
    sar_max:         float


@dataclass
class TradeClassification:
    trade_type:    str    # ST | LT | BLOCKED
    budget_source: str    # st_budget | lt_budget | none
    max_size_pct:  float
    routing_note:  str    # where profits route


@dataclass
class MirrorCheckResult:
    passed: bool
    flag:   Optional[str]


# ─── Gate patterns ────────────────────────────────────────────────────────────

# Mirror gate: any of these patterns in trade rationale triggers a 24h hold
_MIRROR_PATTERNS: dict[str, str] = {
    "information asymmetry": (
        "Information asymmetry exploitation — behaving like extractive predator"
    ),
    "opacity": (
        "Opacity requirement — trade requires concealment"
    ),
    "extraction timing": (
        "Extraction timing — profit timed to others' loss"
    ),
}

# Cooling-off gate: entropy→syntropy conversion reasoning
_COOLING_OFF_PATTERNS: list[str] = [
    "entropy to syntropy",
    "entropy→syntropy",
    "entropic to syntropic",
    "conversion reasoning",
    "converting entropy",
]


# ─── CMAMEngine ───────────────────────────────────────────────────────────────

class CMAMEngine:
    """
    Capital Maturity Allocation Model.

    Usage:
        engine = CMAMEngine()
        profile = engine.profile(fund_value=50_000, short_used=2_000)
        tc = engine.classify_trade(dfte_signal, kepe_profile)
        mc = CMAMEngine.mirror_check(signal.rationale)
        if CMAMEngine.cooling_off_required(signal.rationale):
            defer_24h(signal)
    """

    def __init__(
        self,
        x1:             float = _DEFAULT_X1,
        x2:             float = _DEFAULT_X2,
        sar_max:        float = _DEFAULT_SAR_MAX,
        short_book_cap: float = _DEFAULT_SHORT_CAP,
    ):
        if x1 >= x2:
            raise ValueError(f"x1 ({x1}) must be less than x2 ({x2})")
        if not 0.0 < sar_max <= 1.0:
            raise ValueError(f"sar_max must be in (0, 1], got {sar_max}")
        if not 0.0 < short_book_cap <= 1.0:
            raise ValueError(f"short_book_cap must be in (0, 1], got {short_book_cap}")

        self.x1             = x1
        self.x2             = x2
        self.sar_max        = sar_max
        self.short_book_cap = short_book_cap
        self._current_profile: Optional[CMAMProfile] = None

    # ── Core SAR equation — [TESTABLE] ───────────────────────────────────────

    def compute_sar(self, fund_value: float) -> float:
        """
        Pure SAR piecewise-linear equation.
        Returns value in [0, SAR_max]. [TESTABLE]
        """
        if fund_value < self.x1:
            return 0.0
        if fund_value >= self.x2:
            return self.sar_max
        return (fund_value - self.x1) / (self.x2 - self.x1) * self.sar_max

    # ── Profile snapshot ──────────────────────────────────────────────────────

    def profile(self, fund_value: float, short_used: float = 0.0) -> CMAMProfile:
        """
        Compute a CMAMProfile for the current fund state.
        Caches result for subsequent classify_trade calls. [TESTABLE]
        """
        sar = self.compute_sar(fund_value)

        if fund_value < self.x1:
            mode = "ST_MODE"
        elif fund_value >= self.x2:
            mode = "MATURE"
        else:
            mode = "TRANSITION"

        short_capacity  = fund_value * self.short_book_cap
        short_remaining = max(0.0, short_capacity - short_used)

        p = CMAMProfile(
            fund_value      = fund_value,
            sar             = sar,
            st_allocation   = 1.0 - sar,
            lt_allocation   = sar,
            mode            = mode,
            st_budget       = fund_value * (1.0 - sar),
            lt_budget       = fund_value * sar,
            short_remaining = short_remaining,
            short_used      = short_used,
            x1              = self.x1,
            x2              = self.x2,
            sar_max         = self.sar_max,
        )
        self._current_profile = p
        return p

    # ── Trade classification — [TESTABLE] ────────────────────────────────────

    def classify_trade(
        self,
        dfte_signal: Any,
        kepe_profile: Any,
    ) -> TradeClassification:
        """
        Route a trade to ST or LT budget, or BLOCK it.

        Rules (in priority order):
          1. Extractive asset          → BLOCKED (hard, always)
          2. profile() not yet called  → BLOCKED (safety)
          3. Short sell + book full    → BLOCKED
          4. Short sell + book has room→ ST (profits → lt_budget mandatory)
          5. Syntropic + STS LOADING|STABLE + lt_budget > 0 → LT
          6. Syntropic + STS DETERIORATING → downgrade to ST
          7. All others                → ST

        [TESTABLE]
        """
        # 1. Extractive hard block — regardless of fund size
        if getattr(kepe_profile, "is_extractive_asset", False):
            return TradeClassification(
                trade_type    = "BLOCKED",
                budget_source = "none",
                max_size_pct  = 0.0,
                routing_note  = (
                    "Extractive asset — hard blocked regardless of fund size"
                ),
            )

        # 2. Safety: profile must be initialised
        p = self._current_profile
        if p is None:
            logger.error("CMAM classify_trade called before profile() — blocking trade")
            return TradeClassification(
                trade_type    = "BLOCKED",
                budget_source = "none",
                max_size_pct  = 0.0,
                routing_note  = "CMAM profile not initialised — call profile() first",
            )

        action       = getattr(dfte_signal, "action", "HOLD")
        req_size_pct = getattr(dfte_signal, "position_size_pct", 1.0)
        is_syntropic = getattr(kepe_profile, "is_syntropic_asset", False)
        sts          = getattr(kepe_profile, "sts", "STABLE")

        # 3. Short position — check book cap
        if action == "SELL":
            if p.short_remaining <= 0:
                return TradeClassification(
                    trade_type    = "BLOCKED",
                    budget_source = "none",
                    max_size_pct  = 0.0,
                    routing_note  = (
                        f"Short book cap reached "
                        f"({self.short_book_cap * 100:.0f}% of fund fully deployed). "
                        "Short profits → lt_budget."
                    ),
                )
            max_pct = min(
                req_size_pct,
                (p.short_remaining / p.fund_value) * 100.0,
            )
            return TradeClassification(
                trade_type    = "ST",
                budget_source = "st_budget",
                max_size_pct  = max_pct,
                routing_note  = (
                    "Short position. Profits mandatory → lt_budget. "
                    "Cannot grow short book."
                ),
            )

        # 4. LT: syntropic + STS is LOADING or STABLE + lt budget available
        if is_syntropic and sts in ("LOADING", "STABLE") and p.lt_budget > 0:
            max_pct = min(
                req_size_pct,
                (p.lt_budget / p.fund_value) * 100.0,
            )
            return TradeClassification(
                trade_type    = "LT",
                budget_source = "lt_budget",
                max_size_pct  = max_pct,
                routing_note  = (
                    "LT syntropic position. STS=LOADING|STABLE. "
                    "Profits reinvested into LT syntropy basket."
                ),
            )

        # 5. Syntropic but STS=DETERIORATING → downgrade to ST with note
        if is_syntropic and sts == "DETERIORATING":
            logger.info(
                f"Syntropic asset downgraded ST→LT: STS=DETERIORATING "
                f"(action={action})"
            )

        # 6. ST fallthrough (non-extractive, non-LT-eligible)
        max_pct = min(
            req_size_pct,
            (p.st_budget / p.fund_value) * 100.0 if p.fund_value > 0 else 0.0,
        )
        return TradeClassification(
            trade_type    = "ST",
            budget_source = "st_budget",
            max_size_pct  = max_pct,
            routing_note  = "ST capital-gains trade. MFS-driven. Profits → ST pool.",
        )

    # ── Mirror gate — [TESTABLE] ──────────────────────────────────────────────

    @staticmethod
    def mirror_check(trade_rationale: str) -> MirrorCheckResult:
        """
        Check if trade reasoning requires behaving like what we oppose.
        Any pattern match → 24h hold should be logged by caller. [TESTABLE]
        """
        text = trade_rationale.lower()
        for pattern, flag in _MIRROR_PATTERNS.items():
            if pattern in text:
                logger.warning(f"Mirror gate triggered: {flag}")
                return MirrorCheckResult(passed=False, flag=flag)
        return MirrorCheckResult(passed=True, flag=None)

    # ── Cooling-off gate — [TESTABLE] ─────────────────────────────────────────

    @staticmethod
    def cooling_off_required(rationale: str) -> bool:
        """
        Returns True if the rationale uses entropy→syntropy conversion logic.
        Caller should log full rationale with timestamp and defer 24h. [TESTABLE]
        """
        text = rationale.lower()
        return any(p in text for p in _COOLING_OFF_PATTERNS)
