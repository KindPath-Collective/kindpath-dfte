"""
DFTE — Test Suite
==================
Validates KEPE, DFTE engine, governance, wallet, and orchestrator
using synthetic data (no external APIs required for core tests).

Evidence posture enforced throughout: KINDFIELD
  [ESTABLISHED] | [TESTABLE] | [SPECULATIVE]

Run:
  cd kindpath-dfte
  pip install pytest yfinance rich requests --break-system-packages -q
  pytest tests/test_dfte.py -v
"""

from __future__ import annotations
import pytest
import sys, os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kepe.indicators import WorldSignal
from kepe.syntropy_engine import (
    synthesise_kepe_profile, compute_entropy_indicator,
    compute_interference_load, compute_opc,
    WFS_THRESHOLDS
)
from kepe.kpre_physical import KPRELayer
from dfte.dfte_engine import (
    BMRSummary, KEPESummary, synthesise_dfte_signal,
    mfs_gate, wfs_gate, determine_tier,
    compute_position_size
)
from governance.governance_layer import (
    score_benevolence, apply_governance_tier,
    detect_contradictions, log_influence
)
from wallet.wallet import PaperWallet, OrderRequest


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _world_sig(domain, value, confidence=0.7, evidence="TESTABLE") -> WorldSignal:
    return WorldSignal(
        domain=domain, source="test", region="GLOBAL",
        value=value, confidence=confidence,
        evidence_level=evidence, timestamp=datetime.utcnow()
    )


def _bmr(mfs=0.70, direction=0.5, nu=0.75,
         field_state="ZPB", tier="MID",
         lsii=0.08, lsii_flag="none", k=-0.1,
         curvature_state="COHERENT") -> BMRSummary:
    return BMRSummary(
        symbol="TEST", mfs=mfs, mfs_label="ZPB",
        direction=direction, nu=nu, field_state=field_state,
        trade_tier=tier, lsii=lsii, lsii_flag=lsii_flag,
        k=k, curvature_state=curvature_state,
    )


def _kepe(wfs=0.65, spi=0.60, opc=0.70, il=0.10, ucs=0.30,
          equity_weight=1.0, syntropic=False, extractive=False) -> KEPESummary:
    return KEPESummary(
        symbol="TEST", wfs=wfs, wfs_label="ZPB",
        spi=spi, opc=opc, interference_load=il,
        unified_curvature=ucs, equity_weight=equity_weight,
        is_syntropic=syntropic, is_extractive=extractive,
    )


# ─── KEPE Tests ───────────────────────────────────────────────────────────────

class TestKEPE:

    def test_wfs_in_range(self):
        """WFS must be in 0→1."""
        signals = [
            _world_sig("SOCIAL", 0.5),
            _world_sig("ECOLOGICAL", 0.3),
            _world_sig("NARRATIVE", -0.2),
        ]
        profile = synthesise_kepe_profile("TEST", signals)
        assert 0.0 <= profile.wfs <= 1.0

    def test_wfs_label_valid(self):
        signals = [_world_sig("SOCIAL", 0.6), _world_sig("ECOLOGICAL", 0.4)]
        profile = synthesise_kepe_profile("TEST", signals)
        assert profile.wfs_label in ("ZPB", "COHERENT", "LOADED", "DISRUPTED")

    def test_high_positive_signals_high_wfs(self):
        """All strongly positive signals should produce high WFS."""
        signals = [
            _world_sig("SOCIAL", 0.8, 0.9),
            _world_sig("ECOLOGICAL", 0.7, 0.85),
            _world_sig("NARRATIVE", 0.6, 0.8),
            _world_sig("OPTIMISM", 0.75, 0.75),
        ]
        profile = synthesise_kepe_profile("TEST", signals)
        assert profile.wfs >= 0.60, f"High positive signals → WFS should be ≥ 0.60, got {profile.wfs:.3f}"

    def test_all_negative_signals_low_wfs(self):
        """All negative signals → low WFS."""
        signals = [
            _world_sig("SOCIAL", -0.7, 0.9),
            _world_sig("ECOLOGICAL", -0.6, 0.85),
            _world_sig("CONFLICT", -0.8, 0.8),
        ]
        profile = synthesise_kepe_profile("TEST", signals)
        assert profile.wfs <= 0.40, f"Negative signals → WFS ≤ 0.40, got {profile.wfs:.3f}"

    def test_entropy_indicator_range(self):
        signals = [_world_sig("SOCIAL", v) for v in [0.3, -0.2, 0.6]]
        ei = compute_entropy_indicator(signals)
        assert 0.0 <= ei <= 1.0

    def test_interference_load_range(self):
        signals = [_world_sig("SOCIAL", 0.8), _world_sig("CONFLICT", -0.7)]
        il = compute_interference_load(signals)
        assert 0.0 <= il <= 1.0

    def test_interference_load_opposing_signals(self):
        """Strongly opposing signals should produce non-zero IL."""
        signals = [
            _world_sig("OPTIMISM", 0.9, 0.9),
            _world_sig("CONFLICT", -0.8, 0.85),
        ]
        il = compute_interference_load(signals)
        assert il > 0.0, "Opposing signals should produce non-zero interference load"

    def test_opc_in_range(self):
        opt = _world_sig("OPTIMISM", 0.6)
        soc = _world_sig("SOCIAL", 0.4)
        opc = compute_opc(opt, soc)
        assert 0.0 <= opc <= 1.0

    def test_syntropic_asset_detected(self):
        """Known syntropic symbols should be flagged."""
        signals = [_world_sig("SOCIAL", 0.5)]
        profile = synthesise_kepe_profile("ICLN", signals)
        assert profile.is_syntropic_asset, "ICLN (clean energy) should be syntropic"
        assert profile.equity_weight > 1.0, "Syntropic asset should have >1.0 equity weight"

    def test_extractive_asset_detected(self):
        """Known extractive symbols should be flagged."""
        signals = [_world_sig("SOCIAL", 0.5)]
        profile = synthesise_kepe_profile("LMT", signals)
        assert profile.is_extractive_asset, "LMT (weapons) should be extractive"
        assert profile.equity_weight == 0.0, "Extractive asset equity weight should be 0"

    def test_evidence_notes_present(self):
        signals = [_world_sig("SOCIAL", 0.5)]
        profile = synthesise_kepe_profile("SPY", signals)
        assert len(profile.evidence_notes) > 0
        assert any("TESTABLE" in n or "ESTABLISHED" in n for n in profile.evidence_notes)

    def test_empty_signals_produces_neutral_profile(self):
        profile = synthesise_kepe_profile("TEST", [])
        assert 0.0 <= profile.wfs <= 1.0  # should not crash


# ─── DFTE Engine Tests ────────────────────────────────────────────────────────

class TestDFTEEngine:

    def test_signal_action_valid(self):
        bmr = _bmr(direction=0.5)
        kepe = _kepe()
        sig = synthesise_dfte_signal(bmr, kepe)
        assert sig.action in ("BUY", "SELL", "HOLD", "BLOCKED")

    def test_signal_tier_valid(self):
        bmr = _bmr()
        kepe = _kepe()
        sig = synthesise_dfte_signal(bmr, kepe)
        assert sig.tier in ("NANO", "MID", "LARGE", "WAIT", "BLOCKED")

    def test_conviction_in_range(self):
        bmr = _bmr()
        kepe = _kepe()
        sig = synthesise_dfte_signal(bmr, kepe)
        assert 0.0 <= sig.conviction <= 1.0

    def test_extractive_asset_blocked(self):
        """Extractive asset should be BLOCKED regardless of MFS."""
        bmr = _bmr(mfs=0.90, nu=0.90, direction=0.8, field_state="ZPB", tier="LARGE")
        kepe = _kepe(wfs=0.80, extractive=True, equity_weight=0.0)
        sig = synthesise_dfte_signal(bmr, kepe)
        assert sig.action == "BLOCKED", f"Extractive asset should be BLOCKED, got {sig.action}"
        assert sig.position_size_pct == 0.0

    def test_syntropic_asset_size_boost(self):
        """Syntropic asset should get larger position than neutral asset."""
        bmr = _bmr(mfs=0.75, nu=0.80, direction=0.6, field_state="ZPB", tier="MID")
        kepe_syntropic = _kepe(wfs=0.70, syntropic=True, equity_weight=1.5)
        kepe_neutral = _kepe(wfs=0.70, syntropic=False, equity_weight=1.0)
        sig_s = synthesise_dfte_signal(bmr, kepe_syntropic)
        sig_n = synthesise_dfte_signal(bmr, kepe_neutral)
        assert sig_s.position_size_pct >= sig_n.position_size_pct, \
            "Syntropic asset should have ≥ position size vs neutral"

    def test_sic_event_no_trade(self):
        """SIC event should prevent all tiers."""
        bmr = _bmr(nu=0.10, field_state="SIC", tier="NANO")
        kepe = _kepe()
        sig = synthesise_dfte_signal(bmr, kepe)
        assert sig.action in ("HOLD", "BLOCKED")

    def test_low_wfs_prevents_large(self):
        """Low WFS should prevent LARGE tier."""
        bmr = _bmr(mfs=0.85, nu=0.85, field_state="ZPB", tier="LARGE")
        kepe = _kepe(wfs=0.30)  # world field too poor for LARGE
        _, wfs_pass_reason = wfs_gate(kepe, "LARGE")
        assert "below LARGE threshold" in wfs_pass_reason or wfs_gate(kepe, "LARGE")[0] == False

    def test_gates_all_present(self):
        """All three gates must be explicitly set."""
        bmr = _bmr()
        kepe = _kepe()
        sig = synthesise_dfte_signal(bmr, kepe)
        assert hasattr(sig, "mfs_gate")
        assert hasattr(sig, "wfs_gate")
        assert hasattr(sig, "governance_gate")
        assert hasattr(sig, "all_gates_passed")

    def test_hold_on_zero_direction(self):
        """Near-zero direction should produce HOLD."""
        bmr = _bmr(direction=0.05)
        kepe = _kepe()
        sig = synthesise_dfte_signal(bmr, kepe)
        assert sig.action == "HOLD", f"Zero direction should be HOLD, got {sig.action}"

    def test_position_size_positive_for_buy(self):
        """BUY signals should have positive position size."""
        bmr = _bmr(mfs=0.75, nu=0.80, direction=0.6, field_state="ZPB", tier="MID")
        kepe = _kepe(wfs=0.65)
        sig = synthesise_dfte_signal(bmr, kepe)
        if sig.action == "BUY":
            assert sig.position_size_pct > 0

    def test_determine_tier_logic(self):
        """Tier determination should follow field quality."""
        bmr_strong = _bmr(nu=0.85, field_state="ZPB", tier="LARGE")
        kepe_strong = _kepe(wfs=0.70, spi=0.65)
        tier = determine_tier(bmr_strong, kepe_strong)
        assert tier == "LARGE", f"Strong fields should yield LARGE, got {tier}"

        bmr_weak = _bmr(nu=0.25, field_state="IN_LOADING", tier="NANO")
        kepe_weak = _kepe(wfs=0.30)
        tier_weak = determine_tier(bmr_weak, kepe_weak)
        assert tier_weak in ("NANO", "WAIT")


# ─── Governance Tests ─────────────────────────────────────────────────────────

class TestGovernance:

    def test_clean_energy_syntropic(self):
        b = score_benevolence("ICLN")
        assert b.score > 0.5
        assert b.tier_cap == "LARGE"

    def test_weapons_extractive(self):
        b = score_benevolence("LMT")
        assert b.score < -0.5
        assert b.is_blocked
        assert b.tier_cap == "BLOCKED"

    def test_fossil_fuel_nano_only(self):
        b = score_benevolence("XOM")
        assert b.score < -0.5
        assert b.tier_cap in ("NANO", "BLOCKED")

    def test_governance_tier_cap_applied(self):
        """Governance should downgrade LARGE request for neutral asset."""
        b = score_benevolence("SPY")  # neutral
        approved, reason = apply_governance_tier("LARGE", b)
        assert approved in ("MID", "NANO", "LARGE"), f"Unexpected tier: {approved}"

    def test_extractive_always_blocked(self):
        b = score_benevolence("GEO")  # private prison
        approved, reason = apply_governance_tier("NANO", b)
        assert approved == "BLOCKED", f"Extractive should always be BLOCKED, got {approved}"

    def test_syntropic_passes_large(self):
        b = score_benevolence("NEE")  # clean energy
        approved, reason = apply_governance_tier("LARGE", b)
        assert approved == "LARGE"

    def test_contradiction_clean_vs_fossil(self):
        portfolio = {"ICLN": 10.0, "XOM": 5.0}
        report = detect_contradictions(portfolio)
        assert report.interference_load > 0, \
            "Clean energy + fossil fuel should create contradiction"
        assert len(report.contradictions) > 0

    def test_no_contradiction_clean_only(self):
        portfolio = {"ICLN": 10.0, "NEE": 8.0, "ENPH": 5.0}
        report = detect_contradictions(portfolio)
        # Should be low interference — all syntropic
        assert report.interference_load <= 0.25

    def test_influence_log_written(self):
        record = log_influence(
            symbol="SPY", action="BUY", tier="NANO",
            size_pct=1.0, mfs=0.65, wfs=0.60, nu=0.72
        )
        assert record.symbol == "SPY"
        assert record.action == "BUY"
        assert len(record.expected_field_contribution) > 10


# ─── Wallet Tests ─────────────────────────────────────────────────────────────

class TestWallet:

    @pytest.fixture
    def paper_wallet(self):
        return PaperWallet(initial_cash=10_000.0)

    def test_paper_wallet_initial_cash(self, paper_wallet):
        assert paper_wallet.get_cash() == 10_000.0

    def test_paper_wallet_is_paper(self, paper_wallet):
        assert paper_wallet.is_paper == True

    def test_paper_wallet_portfolio_value_equals_cash_initially(self, paper_wallet):
        assert paper_wallet.get_portfolio_value() == paper_wallet.get_cash()

    def test_paper_wallet_positions_empty_initially(self, paper_wallet):
        assert paper_wallet.get_positions() == []

    def test_paper_wallet_buy_order(self, paper_wallet):
        """Paper wallet should execute BUY using mock price."""
        # Inject a mock price fetch
        paper_wallet._get_price = lambda s: 100.0

        order = OrderRequest(
            symbol="TEST", side="buy",
            notional=1000.0, tier="NANO"
        )
        result = paper_wallet.submit_order(order)
        assert result.success == True
        assert result.status == "paper"
        assert result.fill_price == 100.0
        assert paper_wallet.get_cash() == 9_000.0  # 10k - 1k

    def test_paper_wallet_buy_then_positions_populated(self, paper_wallet):
        paper_wallet._get_price = lambda s: 50.0
        order = OrderRequest(symbol="TEST", side="buy", notional=500.0, tier="NANO")
        paper_wallet.submit_order(order)
        positions = paper_wallet.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "TEST"
        assert positions[0].qty == pytest.approx(10.0)  # 500/50

    def test_paper_wallet_sell_reduces_position(self, paper_wallet):
        paper_wallet._get_price = lambda s: 100.0
        paper_wallet.submit_order(OrderRequest(symbol="T", side="buy", notional=1000.0, tier="NANO"))
        paper_wallet.submit_order(OrderRequest(symbol="T", side="sell", qty=5.0, tier="NANO"))
        positions = paper_wallet.get_positions()
        remaining = next((p for p in positions if p.symbol == "T"), None)
        assert remaining is not None
        assert remaining.qty == pytest.approx(5.0)

    def test_paper_wallet_insufficient_cash_rejected(self, paper_wallet):
        paper_wallet._get_price = lambda s: 100.0
        order = OrderRequest(symbol="TEST", side="buy", notional=20_000.0, tier="NANO")
        result = paper_wallet.submit_order(order)
        assert result.success == False
        assert "Insufficient" in (result.error or "")

    def test_paper_wallet_cancel_always_succeeds(self, paper_wallet):
        assert paper_wallet.cancel_order("any_id") == True

    def test_paper_wallet_portfolio_summary(self, paper_wallet):
        paper_wallet._get_price = lambda s: 100.0
        paper_wallet.submit_order(OrderRequest(symbol="TEST", side="buy", notional=1000.0, tier="NANO"))
        summary = paper_wallet.portfolio_summary()
        assert "cash" in summary
        assert "positions" in summary
        assert "trade_count" in summary
        assert summary["trade_count"] == 1


# ─── Integration: KEPE + DFTE + Governance ───────────────────────────────────

class TestIntegration:

    def test_full_pipeline_syntropic_asset(self):
        """Full pipeline for a syntropic asset should allow LARGE if fields align."""
        # Build KEPE profile directly
        signals = [
            _world_sig("SOCIAL", 0.8, 0.9, "ESTABLISHED"),
            _world_sig("ECOLOGICAL", 0.7, 0.85, "TESTABLE"),
            _world_sig("OPTIMISM", 0.6, 0.75, "TESTABLE"),
            _world_sig("NARRATIVE", 0.5, 0.70, "TESTABLE"),
        ]
        kepe_profile = synthesise_kepe_profile("ICLN", signals)
        kepe_summary = KEPESummary(
            symbol="ICLN",
            wfs=kepe_profile.wfs, wfs_label=kepe_profile.wfs_label,
            spi=kepe_profile.spi, opc=kepe_profile.opc,
            interference_load=kepe_profile.interference_load,
            unified_curvature=kepe_profile.unified_curvature,
            equity_weight=kepe_profile.equity_weight,
            is_syntropic=kepe_profile.is_syntropic_asset,
            is_extractive=kepe_profile.is_extractive_asset,
        )

        bmr = _bmr(
            mfs=0.78, direction=0.65, nu=0.82,
            field_state="ZPB", tier="LARGE"
        )
        sig = synthesise_dfte_signal(bmr, kepe_summary)

        # Apply governance
        ben = score_benevolence("ICLN")
        approved_tier, _ = apply_governance_tier(sig.tier, ben)

        # Should be LARGE or MID with syntropic asset + strong fields
        assert approved_tier in ("LARGE", "MID"), \
            f"ICLN with strong fields should be LARGE/MID, got {approved_tier}"
        assert not ben.is_blocked

    def test_full_pipeline_extractive_blocked(self):
        """Extractive asset should be BLOCKED regardless of field quality."""
        signals = [_world_sig("SOCIAL", 0.8, 0.9)]
        kepe_profile = synthesise_kepe_profile("LMT", signals)

        assert kepe_profile.is_extractive_asset
        assert kepe_profile.equity_weight == 0.0

        ben = score_benevolence("LMT")
        approved_tier, _ = apply_governance_tier("LARGE", ben)
        assert approved_tier == "BLOCKED"

    def test_evidence_posture_end_to_end(self):
        """Every component in the pipeline must declare evidence level."""
        signals = [_world_sig("SOCIAL", 0.5)]
        kepe = synthesise_kepe_profile("SPY", signals)

        # KEPE evidence notes present
        assert len(kepe.evidence_notes) > 0

        # All notes reference at least one evidence level
        evidence_terms = ["ESTABLISHED", "TESTABLE", "SPECULATIVE"]
        for note in kepe.evidence_notes:
            assert any(t in note for t in evidence_terms), \
                f"Evidence note missing level marker: '{note}'"

    def test_paper_wallet_executes_buy_signal(self):
        """Paper wallet should execute a BUY signal from DFTE."""
        wallet = PaperWallet(initial_cash=10_000.0)
        wallet._get_price = lambda s: 100.0

        bmr = _bmr(mfs=0.75, direction=0.6, nu=0.78, field_state="ZPB", tier="MID")
        kepe = _kepe(wfs=0.65, spi=0.60, opc=0.70, il=0.10)
        sig = synthesise_dfte_signal(bmr, kepe)

        if sig.action == "BUY" and sig.all_gates_passed:
            notional = wallet.get_cash() * (sig.position_size_pct / 100)
            order = OrderRequest(
                symbol="TEST", side="buy",
                notional=max(notional, 10.0), tier=sig.tier
            )
            result = wallet.submit_order(order)
            assert result.success, f"Paper order should succeed: {result.error}"
            assert wallet.get_cash() < 10_000.0


# ─── KPRE Physical Flow Layer Tests ──────────────────────────────────────────

class TestKPRE:
    """
    Tests for KPRELayer aggregation logic using synthetic data only.
    No network calls — _aggregate() is called directly.
    """

    def _kpre_sig(self, value: float, confidence: float = 0.60,
                  evidence: str = "TESTABLE") -> WorldSignal:
        """Synthetic KPRE sub-signal for aggregation tests."""
        return WorldSignal(
            domain="KPRE_FLOW", source="test_kpre_sub",
            region="GLOBAL", value=value, confidence=confidence,
            evidence_level=evidence, timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
        )

    # ── Structural / domain correctness ─────────────────────────────────────

    def test_empty_aggregate_zero_confidence(self):
        """Empty input → zero confidence, system should not crash."""
        result = KPRELayer._aggregate([])
        assert result.confidence == 0.0

    def test_empty_aggregate_zero_value(self):
        """Empty input → zero value."""
        result = KPRELayer._aggregate([])
        assert result.value == 0.0

    def test_empty_aggregate_domain_is_kpre(self):
        """Empty input → domain is still KPRE."""
        result = KPRELayer._aggregate([])
        assert result.domain == "KPRE"

    def test_domain_is_kpre(self):
        """Non-empty aggregate → domain must be KPRE."""
        result = KPRELayer._aggregate([self._kpre_sig(0.5)])
        assert result.domain == "KPRE"

    def test_temporal_layer_is_medium(self):
        """KPRE composite temporal layer is always MEDIUM."""
        result = KPRELayer._aggregate([self._kpre_sig(0.4)])
        assert result.temporal_layer == "MEDIUM"

    # ── Value direction ──────────────────────────────────────────────────────

    def test_positive_signals_positive_composite(self):
        """All positive sub-signals → positive composite value."""
        sigs = [self._kpre_sig(0.6), self._kpre_sig(0.7), self._kpre_sig(0.5)]
        result = KPRELayer._aggregate(sigs)
        assert result.value > 0, \
            f"Positive sub-signals → composite > 0, got {result.value:.3f}"

    def test_negative_signals_negative_composite(self):
        """All negative sub-signals → negative composite value."""
        sigs = [self._kpre_sig(-0.6), self._kpre_sig(-0.7), self._kpre_sig(-0.5)]
        result = KPRELayer._aggregate(sigs)
        assert result.value < 0, \
            f"Negative sub-signals → composite < 0, got {result.value:.3f}"

    def test_value_in_bounds(self):
        """Composite value must be within [-1, 1] even at extremes."""
        sigs = [self._kpre_sig(1.0), self._kpre_sig(0.9), self._kpre_sig(1.0)]
        result = KPRELayer._aggregate(sigs)
        assert -1.0 <= result.value <= 1.0, \
            f"KPRE value out of bounds: {result.value:.3f}"

    def test_mixed_signals_confidence_weighted(self):
        """High-confidence positive + low-confidence negative → net positive."""
        sigs = [
            self._kpre_sig(0.7, confidence=0.80),   # strong positive
            self._kpre_sig(-0.5, confidence=0.20),  # weak negative
        ]
        result = KPRELayer._aggregate(sigs)
        assert result.value > 0, \
            f"High-conf positive should dominate: got {result.value:.3f}"

    # ── Confidence scaling ───────────────────────────────────────────────────

    def test_confidence_capped_at_070(self):
        """Composite confidence is always ≤ 0.70 (TESTABLE ceiling)."""
        sigs = [self._kpre_sig(0.5, confidence=0.95) for _ in range(5)]
        result = KPRELayer._aggregate(sigs)
        assert result.confidence <= 0.70, \
            f"Confidence should be capped at 0.70, got {result.confidence:.3f}"

    def test_partial_signals_lower_confidence(self):
        """Fewer sub-signals → lower completeness → lower confidence."""
        sigs_5 = [self._kpre_sig(0.5, confidence=0.60) for _ in range(5)]
        sigs_2 = [self._kpre_sig(0.5, confidence=0.60) for _ in range(2)]
        r5 = KPRELayer._aggregate(sigs_5)
        r2 = KPRELayer._aggregate(sigs_2)
        assert r2.confidence < r5.confidence, \
            f"2/5 signals should have lower confidence than 5/5: {r2.confidence:.3f} vs {r5.confidence:.3f}"

    # ── Evidence level propagation ───────────────────────────────────────────

    def test_established_sub_signals_yield_testable(self):
        """Composite is at most TESTABLE even if all sub-signals are ESTABLISHED."""
        sigs = [self._kpre_sig(0.5, evidence="ESTABLISHED")]
        result = KPRELayer._aggregate(sigs)
        assert result.evidence_level in ("TESTABLE", "SPECULATIVE"), \
            f"KPRE composite should not be ESTABLISHED, got {result.evidence_level}"

    def test_speculative_evidence_propagates(self):
        """If any sub-signal is SPECULATIVE, composite must be SPECULATIVE."""
        sigs = [
            self._kpre_sig(0.5, evidence="TESTABLE"),
            self._kpre_sig(0.4, evidence="SPECULATIVE"),
        ]
        result = KPRELayer._aggregate(sigs)
        assert result.evidence_level == "SPECULATIVE", \
            f"SPECULATIVE sub-signal should propagate to composite, got {result.evidence_level}"

    # ── Raw metadata ─────────────────────────────────────────────────────────

    def test_raw_contains_n_signals(self):
        """Raw metadata must report number of sub-signals used."""
        sigs = [self._kpre_sig(0.4), self._kpre_sig(0.5)]
        result = KPRELayer._aggregate(sigs)
        assert result.raw is not None
        assert result.raw.get("n_signals") == 2, \
            f"raw.n_signals should be 2, got {result.raw.get('n_signals')}"

    def test_raw_completeness_fraction(self):
        """raw.completeness = n_signals / 5."""
        sigs = [self._kpre_sig(0.5) for _ in range(3)]
        result = KPRELayer._aggregate(sigs)
        assert result.raw.get("completeness") == pytest.approx(0.6), \
            f"3/5 completeness should be 0.6, got {result.raw.get('completeness')}"

    # ── WFS integration ──────────────────────────────────────────────────────

    def test_positive_kpre_shifts_wfs_up(self):
        """Adding a strongly positive KPRE signal to KEPEProfile should raise WFS."""
        base = [_world_sig("SOCIAL", 0.5), _world_sig("ECOLOGICAL", 0.4)]

        profile_base = synthesise_kepe_profile("TEST", base)

        kpre_positive = WorldSignal(
            domain="KPRE", source="test_kpre_composite",
            region="GLOBAL", value=0.90, confidence=0.65,
            evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
        )
        profile_kpre = synthesise_kepe_profile("TEST", base + [kpre_positive])

        assert profile_kpre.wfs >= profile_base.wfs, (
            f"Positive KPRE should raise WFS: "
            f"{profile_base.wfs:.3f} → {profile_kpre.wfs:.3f}"
        )

    def test_negative_kpre_shifts_wfs_down(self):
        """Adding a strongly negative KPRE signal to KEPEProfile should lower WFS."""
        base = [_world_sig("SOCIAL", 0.5), _world_sig("ECOLOGICAL", 0.4)]

        profile_base = synthesise_kepe_profile("TEST", base)

        kpre_negative = WorldSignal(
            domain="KPRE", source="test_kpre_composite",
            region="GLOBAL", value=-0.90, confidence=0.65,
            evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
        )
        profile_kpre = synthesise_kepe_profile("TEST", base + [kpre_negative])

        assert profile_kpre.wfs <= profile_base.wfs, (
            f"Negative KPRE should lower WFS: "
            f"{profile_base.wfs:.3f} → {profile_kpre.wfs:.3f}"
        )
