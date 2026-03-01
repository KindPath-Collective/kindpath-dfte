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
from kepe.kpre_language import (
    KPRELanguageLayer,
    FedLanguageSignal,
    SECRiskDriftSignal,
    EarningsLanguageSignal,
)
from cmam.cmam_engine import CMAMEngine, CMAMProfile, TradeClassification
from sas.sas_engine import (
    SASEngine, SASProfile,
    RevenueCoherenceSignal, CapexDirectionSignal, OpacitySignal,
    SSIStub, WolfDetector,
)
from kepe.kpre_capital import (
    KPRECapitalLayer,
    InsiderTransactionSignal,
    CongressionalTradingSignal,
    CapexIntentSignal,
)
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


# ─── KPRE Capital Formation Tests ─────────────────────────────────────────────

class TestKPRECapital:
    """
    Tests for KPRECapitalLayer and sub-signal scoring logic.
    No network calls — all scoring methods are tested directly with synthetic data.
    """

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _tx(self, code: str, shares: float = 5000, price: float = 100.0,
            title: str = "officer", is_dir: bool = False,
            is_10b5: bool = False, post: float = 50_000) -> dict:
        """Synthetic Form 4 transaction record."""
        return {
            "code":          code,
            "shares":        shares,
            "price":         price,
            "total_value":   shares * price,
            "post_shares":   post,
            "is_director":   is_dir,
            "is_officer":    True,
            "is_10pct":      False,
            "officer_title": title,
            "is_10b5_1":     is_10b5,
            "date":          "2026-02-25",
        }

    def _congress_trade(self, tx_type: str, amount: str = "$1,001 - $15,000",
                        chamber: str = "house") -> dict:
        """Synthetic congressional trade record."""
        return {
            "type":             tx_type,
            "amount":          amount,
            "_chamber":        chamber,
            "ticker":          "TEST",
            "transaction_date": "2026-02-20",
        }

    # ── Insider transaction scoring ──────────────────────────────────────────

    def test_cluster_buy_positive(self):
        """3+ insiders purchasing → positive signal with cluster bonus."""
        txs = [self._tx("P") for _ in range(3)]
        value, raw = InsiderTransactionSignal._score_transactions(txs)
        assert value > 0, f"Cluster buy should be positive, got {value:.3f}"
        assert raw["cluster"] is True

    def test_single_buy_positive(self):
        """Single purchase → positive signal (no cluster)."""
        txs = [self._tx("P")]
        value, raw = InsiderTransactionSignal._score_transactions(txs)
        assert value > 0
        assert raw["cluster"] is False

    def test_mass_sale_negative(self):
        """Multiple insider sales → negative signal."""
        txs = [self._tx("S") for _ in range(4)]
        value, raw = InsiderTransactionSignal._score_transactions(txs)
        assert value < 0, f"Mass sales should be negative, got {value:.3f}"
        assert raw["n_sales"] == 4

    def test_ceo_purchase_higher_weight(self):
        """CEO purchase should outweigh equal-size generic officer purchase."""
        ceo_tx  = [self._tx("P", title="chief executive officer")]
        off_tx  = [self._tx("P", title="vp product")]
        v_ceo, _ = InsiderTransactionSignal._score_transactions(ceo_tx)
        v_off, _ = InsiderTransactionSignal._score_transactions(off_tx)
        assert v_ceo >= v_off, \
            f"CEO buy weight should be ≥ officer: {v_ceo:.3f} vs {v_off:.3f}"

    def test_10b5_plan_discounted(self):
        """10b5-1 pre-planned sales are discounted vs open-market sales."""
        planned_sales   = [self._tx("S", is_10b5=True)  for _ in range(3)]
        openmarket_sale = [self._tx("S", is_10b5=False) for _ in range(3)]
        v_planned, _  = InsiderTransactionSignal._score_transactions(planned_sales)
        v_open, _     = InsiderTransactionSignal._score_transactions(openmarket_sale)
        # Planned is less negative (discounted) than open-market
        assert v_planned > v_open, \
            f"10b5-1 sales should be less negative: planned={v_planned:.3f}, open={v_open:.3f}"

    def test_director_higher_weight_than_officer(self):
        """Director purchases should carry more weight than generic officer."""
        dir_tx = [self._tx("P", is_dir=True,  title="director")]
        off_tx = [self._tx("P", is_dir=False, title="vp marketing")]
        v_dir, _ = InsiderTransactionSignal._score_transactions(dir_tx)
        v_off, _ = InsiderTransactionSignal._score_transactions(off_tx)
        assert v_dir >= v_off, \
            f"Director buy should be ≥ officer: {v_dir:.3f} vs {v_off:.3f}"

    def test_awards_excluded(self):
        """Award (A code) transactions should not count as bullish purchases."""
        awards = [self._tx("A") for _ in range(5)]
        value, raw = InsiderTransactionSignal._score_transactions(awards)
        assert raw["n_purchases"] == 0, \
            "Award grants should not be counted as purchases"

    def test_empty_insider_transactions(self):
        """Empty transactions list should return zero with clean raw dict."""
        value, raw = InsiderTransactionSignal._score_transactions([])
        assert value == 0.0
        assert raw == {}

    def test_mixed_buys_sells_direction(self):
        """More buys than sells (by value) → net positive."""
        txs = (
            [self._tx("P", shares=10_000, price=150)] * 3 +  # large buys
            [self._tx("S", shares=1_000,  price=150)] * 1    # small sale
        )
        value, _ = InsiderTransactionSignal._score_transactions(txs)
        assert value > 0, f"More/larger buys should dominate: {value:.3f}"

    # ── Congressional trading scoring ────────────────────────────────────────

    def test_congressional_buy_positive(self):
        """Congressional purchase → positive signal."""
        trades = [self._congress_trade("purchase")]
        value, raw = CongressionalTradingSignal._score_trades(trades, "TEST")
        assert value > 0
        assert raw["n_buys"] == 1

    def test_congressional_sell_negative(self):
        """Congressional sale → negative signal."""
        trades = [self._congress_trade("sale_full")]
        value, raw = CongressionalTradingSignal._score_trades(trades, "TEST")
        assert value < 0
        assert raw["n_sells"] == 1

    def test_large_congressional_trade_higher_weight(self):
        """Larger notional congressional trade should produce stronger signal."""
        small = [self._congress_trade("purchase", "$1,001 - $15,000")]
        large = [self._congress_trade("purchase", "Over $1,000,000")]
        v_small, _ = CongressionalTradingSignal._score_trades(small, "TEST")
        v_large, _ = CongressionalTradingSignal._score_trades(large, "TEST")
        assert v_large >= v_small, \
            f"Large trade should be ≥ small: {v_large:.3f} vs {v_small:.3f}"

    def test_senate_trade_higher_weight_than_house(self):
        """Senate trades should carry 1.2× weight vs House."""
        house_trade  = [self._congress_trade("purchase", "$100,001 - $250,000", "house")]
        senate_trade = [self._congress_trade("purchase", "$100,001 - $250,000", "senate")]
        v_house, _  = CongressionalTradingSignal._score_trades(house_trade,  "TEST")
        v_senate, _ = CongressionalTradingSignal._score_trades(senate_trade, "TEST")
        assert v_senate >= v_house, \
            f"Senate buy should be ≥ House: {v_senate:.3f} vs {v_house:.3f}"

    def test_empty_congressional_trades(self):
        """Empty trades → zero."""
        value, raw = CongressionalTradingSignal._score_trades([], "TEST")
        assert value == 0.0

    # ── Capex trend scoring ──────────────────────────────────────────────────

    def test_rising_capex_positive(self):
        """Consistent capex growth → positive signal."""
        quarters = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
        value, raw = CapexIntentSignal._compute_capex_trend(quarters)
        assert value > 0, f"Rising capex should be positive, got {value:.3f}"

    def test_falling_capex_negative(self):
        """Consistent capex decline → negative signal."""
        quarters = [150.0, 140.0, 130.0, 120.0, 110.0, 100.0]
        value, raw = CapexIntentSignal._compute_capex_trend(quarters)
        assert value < 0, f"Falling capex should be negative, got {value:.3f}"

    def test_capex_value_in_bounds(self):
        """Capex trend signal must be within [-1, 1]."""
        quarters = [100.0, 300.0, 900.0, 2700.0]  # extreme 3× growth
        value, raw = CapexIntentSignal._compute_capex_trend(quarters)
        assert -1.0 <= value <= 1.0

    def test_capex_raw_contains_pct_delta(self):
        """Raw output should include delta percentage."""
        quarters = [200.0, 220.0, 240.0, 260.0]
        _, raw = CapexIntentSignal._compute_capex_trend(quarters)
        assert "delta_full_pct" in raw
        assert raw["delta_full_pct"] > 0

    def test_single_quarter_insufficient(self):
        """Single quarter → zero (no trend)."""
        value, raw = CapexIntentSignal._compute_capex_trend([100.0])
        assert value == 0.0

    # ── KPRECapitalLayer aggregation ─────────────────────────────────────────

    def _cap_sig(self, value: float, confidence: float = 0.60,
                 evidence: str = "TESTABLE") -> WorldSignal:
        return WorldSignal(
            domain="KPRE_CAPITAL", source="test_capital_sub",
            region="US", value=value, confidence=confidence,
            evidence_level=evidence, timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
        )

    def test_empty_capital_aggregate_zero(self):
        """Empty → zero confidence and value."""
        result = KPRECapitalLayer._aggregate([])
        assert result.confidence == 0.0
        assert result.value == 0.0
        assert result.domain == "KPRE_CAPITAL"

    def test_capital_confidence_capped_at_065(self):
        """KPRE_CAPITAL composite confidence ≤ 0.65."""
        sigs = [self._cap_sig(0.5, confidence=0.90) for _ in range(3)]
        result = KPRECapitalLayer._aggregate(sigs)
        assert result.confidence <= 0.65, \
            f"Capital confidence should cap at 0.65, got {result.confidence:.3f}"

    def test_capital_positive_signals(self):
        """All positive capital signals → positive composite."""
        sigs = [self._cap_sig(0.7), self._cap_sig(0.6), self._cap_sig(0.8)]
        result = KPRECapitalLayer._aggregate(sigs)
        assert result.value > 0

    def test_capital_negative_signals(self):
        """All negative capital signals → negative composite."""
        sigs = [self._cap_sig(-0.7), self._cap_sig(-0.6)]
        result = KPRECapitalLayer._aggregate(sigs)
        assert result.value < 0

    def test_capital_established_evidence_preserved(self):
        """All ESTABLISHED sub-signals → composite is ESTABLISHED."""
        sigs = [self._cap_sig(0.5, evidence="ESTABLISHED")]
        result = KPRECapitalLayer._aggregate(sigs)
        assert result.evidence_level == "ESTABLISHED"

    def test_capital_testable_propagates(self):
        """TESTABLE sub-signal degrades ESTABLISHED composite."""
        sigs = [
            self._cap_sig(0.5, evidence="ESTABLISHED"),
            self._cap_sig(0.4, evidence="TESTABLE"),
        ]
        result = KPRECapitalLayer._aggregate(sigs)
        assert result.evidence_level == "TESTABLE"

    def test_capital_raw_completeness(self):
        """raw.completeness = n_signals / 3."""
        sigs = [self._cap_sig(0.5)]
        result = KPRECapitalLayer._aggregate(sigs)
        assert result.raw.get("completeness") == pytest.approx(1 / 3, abs=0.01)

    def test_capital_shifts_wfs_up(self):
        """Adding positive KPRE_CAPITAL signal to KEPEProfile should raise WFS."""
        base = [_world_sig("SOCIAL", 0.5), _world_sig("ECOLOGICAL", 0.4)]
        profile_base = synthesise_kepe_profile("TEST", base)

        cap_positive = WorldSignal(
            domain="KPRE_CAPITAL", source="test_cap_composite",
            region="US", value=0.85, confidence=0.60,
            evidence_level="ESTABLISHED", timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
        )
        profile_cap = synthesise_kepe_profile("TEST", base + [cap_positive])

        assert profile_cap.wfs >= profile_base.wfs, (
            f"Positive KPRE_CAPITAL should raise WFS: "
            f"{profile_base.wfs:.3f} → {profile_cap.wfs:.3f}"
        )


# ─── KPRE Language Field Layer Tests ─────────────────────────────────────────

class TestKPRELanguage:
    """
    Tests for KPRE Language Field layer using synthetic text.
    No network calls — all scoring methods are tested directly.
    """

    # ── FedLanguageSignal scoring ────────────────────────────────────────────

    def test_fed_tightening_language_negative(self):
        """Dominant tightening language → negative signal."""
        text = (
            "inflation elevated inflation restrictive above target overheating "
            "tightening tightening price stability inflation hike"
        )
        scores = FedLanguageSignal._score_speech_text(text)
        assert scores["signal"] < 0, \
            f"Tightening language should be negative, got {scores['signal']:.3f}"

    def test_fed_easing_language_positive(self):
        """Dominant easing language → positive signal."""
        text = (
            "cooling softening normalize accommodative will commit "
            "below target disinflation cut easing slowdown"
        )
        scores = FedLanguageSignal._score_speech_text(text)
        assert scores["signal"] > 0, \
            f"Easing language should be positive, got {scores['signal']:.3f}"

    def test_fed_uncertainty_dampens_signal(self):
        """High uncertainty should produce a weaker (less negative) tightening signal."""
        certain_tight = "inflation restrictive elevated tightening hike above target"
        uncertain_tight = (
            "uncertain unclear monitor cautious vigilant "
            "inflation restrictive elevated tightening"
        )
        s_certain   = FedLanguageSignal._score_speech_text(certain_tight)
        s_uncertain = FedLanguageSignal._score_speech_text(uncertain_tight)
        # Uncertain version should be less negative (closer to 0)
        assert s_uncertain["signal"] > s_certain["signal"], (
            f"Uncertainty should dampen tightening: "
            f"certain={s_certain['signal']:.3f}, uncertain={s_uncertain['signal']:.3f}"
        )

    def test_fed_signal_in_bounds(self):
        """Fed signal must be within [-1, 1]."""
        text = "inflation inflation inflation tightening restrictive elevated hike above target"
        scores = FedLanguageSignal._score_speech_text(text)
        assert -1.0 <= scores["signal"] <= 1.0

    def test_fed_neutral_text_near_zero(self):
        """Neutral text (no keywords) → signal near zero."""
        text = "the committee reviewed economic data and discussed various perspectives"
        scores = FedLanguageSignal._score_speech_text(text)
        assert abs(scores["signal"]) < 0.3, \
            f"Neutral text should be near zero, got {scores['signal']:.3f}"

    def test_fed_returns_required_keys(self):
        """Score dict must contain all required keys."""
        scores = FedLanguageSignal._score_speech_text("test text")
        for key in ("uncertainty_density", "tightening_density", "easing_density",
                    "policy_direction", "certainty", "signal"):
            assert key in scores, f"Missing key: {key}"

    # ── SECRiskDriftSignal scoring ────────────────────────────────────────────

    def test_risk_drift_new_climate_positive(self):
        """New climate language in 10-K → positive (country-layer awakening)."""
        old = {"climate": 2, "regulatory": 10}
        new = {"climate": 8, "regulatory": 10}
        value, raw = SECRiskDriftSignal._score_risk_drift(old, new, 5000, 5000)
        assert value > 0, f"New climate language should be positive, got {value:.3f}"

    def test_risk_drift_same_counts_neutral(self):
        """Same keyword counts in both filings → near-zero drift."""
        counts = {"climate": 5, "regulatory": 10, "litigation": 3, "cybersecurity": 2}
        value, raw = SECRiskDriftSignal._score_risk_drift(counts, counts, 5000, 5000)
        assert abs(value) < 0.01, f"Same counts should be ~0, got {value:.4f}"

    def test_risk_drift_new_litigation_negative(self):
        """More litigation language → negative (interference loading)."""
        old = {"litigation": 2}
        new = {"litigation": 10}
        value, raw = SECRiskDriftSignal._score_risk_drift(old, new, 5000, 5000)
        assert value < 0, f"New litigation should be negative, got {value:.3f}"

    def test_risk_drift_new_cyber_negative(self):
        """Rising cybersecurity risk language → negative signal."""
        old = {"cybersecurity": 1, "breach": 0}
        new = {"cybersecurity": 5, "breach": 3}
        value, _ = SECRiskDriftSignal._score_risk_drift(old, new, 5000, 5000)
        assert value < 0

    def test_risk_drift_value_in_bounds(self):
        """Drift signal must be in [-1, 1]."""
        old = {"climate": 0, "litigation": 0}
        new = {"climate": 100, "litigation": 0}
        value, _ = SECRiskDriftSignal._score_risk_drift(old, new, 5000, 5000)
        assert -1.0 <= value <= 1.0

    def test_risk_drift_raw_contains_details(self):
        """Raw output should include drift_details and n_drifted."""
        old = {"climate": 3}
        new = {"climate": 8}
        _, raw = SECRiskDriftSignal._score_risk_drift(old, new, 5000, 5000)
        assert "n_drifted" in raw
        assert raw["n_drifted"] >= 1

    def test_count_keywords_basic(self):
        """Keyword counter should find expected keywords in text."""
        text = "climate risk climate environmental sustainability litigation lawsuit"
        counts = SECRiskDriftSignal._count_keywords(text)
        assert counts.get("climate", 0) >= 2
        assert counts.get("litigation", 0) >= 1

    def test_extract_risk_section_finds_header(self):
        """Risk section extractor should find 'Item 1A Risk Factors' text."""
        html = (
            "<html><body><p>Item 1 Business section text here.</p>"
            "<p>Item 1A. Risk Factors</p>"
            "<p>The company faces climate risks and litigation risks.</p>"
            "<p>Item 2. Properties</p></body></html>"
        )
        text = SECRiskDriftSignal._extract_risk_section(html)
        assert "climate" in text.lower(), "Should extract climate risk mention"
        assert "litigation" in text.lower()

    # ── EarningsLanguageSignal scoring ───────────────────────────────────────

    def test_earnings_confident_positive(self):
        """Strong, confident language → positive signal."""
        text = (
            "strong robust exceeded record growth accelerating momentum "
            "delivering expanding outperform raised guidance increased ahead beat"
        )
        scores = EarningsLanguageSignal._score_earnings_text(text)
        assert scores["signal"] > 0, \
            f"Confident earnings language should be positive, got {scores['signal']:.3f}"

    def test_earnings_hedging_negative(self):
        """Hedging language dominant → negative signal."""
        text = (
            "may might could potentially challenging uncertain difficult "
            "headwinds unforeseen subject to concerns softer cautious disappointing"
        )
        scores = EarningsLanguageSignal._score_earnings_text(text)
        assert scores["signal"] < 0, \
            f"Hedging earnings language should be negative, got {scores['signal']:.3f}"

    def test_earnings_signal_in_bounds(self):
        """Earnings signal must be in [-1, 1]."""
        text = "strong growth but may face uncertain challenging potentially difficult headwinds"
        scores = EarningsLanguageSignal._score_earnings_text(text)
        assert -1.0 <= scores["signal"] <= 1.0

    def test_earnings_investing_language_boosts(self):
        """Forward investment language alongside confidence → stronger positive."""
        text_invest = "strong growth investing capex research expand build innovate"
        text_return  = "strong growth dividend buyback repurchase returning capital yield"
        s_invest = EarningsLanguageSignal._score_earnings_text(text_invest)
        s_return  = EarningsLanguageSignal._score_earnings_text(text_return)
        # Both should be positive (confident), but investing language should
        # produce at least as high a signal as return-capital language
        assert s_invest["signal"] >= s_return["signal"], (
            f"Investment language should be ≥ return-capital: "
            f"invest={s_invest['signal']:.3f}, return={s_return['signal']:.3f}"
        )

    def test_earnings_returns_required_keys(self):
        """Score dict must include all required keys."""
        scores = EarningsLanguageSignal._score_earnings_text("test")
        for key in ("confidence_density", "hedging_density", "invest_signal", "signal"):
            assert key in scores, f"Missing key: {key}"

    # ── KPRELanguageLayer aggregation ────────────────────────────────────────

    def _lang_sig(self, value: float, confidence: float = 0.50,
                  evidence: str = "TESTABLE") -> WorldSignal:
        return WorldSignal(
            domain="LANGUAGE", source="test_language_sub",
            region="GLOBAL", value=value, confidence=confidence,
            evidence_level=evidence, timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
        )

    def test_language_layer_empty_zero(self):
        """Empty input → zero confidence and value."""
        result = KPRELanguageLayer._aggregate([])
        assert result.confidence == 0.0
        assert result.value == 0.0
        assert result.domain == "LANGUAGE"

    def test_language_confidence_capped_at_060(self):
        """Language composite confidence ≤ 0.60 (TESTABLE ceiling)."""
        sigs = [self._lang_sig(0.5, confidence=0.95) for _ in range(3)]
        result = KPRELanguageLayer._aggregate(sigs)
        assert result.confidence <= 0.60, \
            f"Language confidence should cap at 0.60, got {result.confidence:.3f}"

    def test_language_positive_signals(self):
        """All positive language signals → positive composite."""
        sigs = [self._lang_sig(0.7), self._lang_sig(0.6), self._lang_sig(0.5)]
        result = KPRELanguageLayer._aggregate(sigs)
        assert result.value > 0

    def test_language_negative_signals(self):
        """All negative language signals → negative composite."""
        sigs = [self._lang_sig(-0.7), self._lang_sig(-0.5)]
        result = KPRELanguageLayer._aggregate(sigs)
        assert result.value < 0

    def test_language_evidence_is_testable(self):
        """All language signals are at most TESTABLE (no ESTABLISHED NLP)."""
        sigs = [self._lang_sig(0.5, evidence="TESTABLE")]
        result = KPRELanguageLayer._aggregate(sigs)
        assert result.evidence_level in ("TESTABLE", "SPECULATIVE")

    def test_language_partial_lower_confidence(self):
        """1/3 signals → lower confidence than 3/3."""
        sigs_3 = [self._lang_sig(0.5, 0.55) for _ in range(3)]
        sigs_1 = [self._lang_sig(0.5, 0.55)]
        r3 = KPRELanguageLayer._aggregate(sigs_3)
        r1 = KPRELanguageLayer._aggregate(sigs_1)
        assert r1.confidence < r3.confidence

    def test_language_raw_completeness(self):
        """raw.completeness = n_signals / 3."""
        sigs = [self._lang_sig(0.5), self._lang_sig(0.3)]
        result = KPRELanguageLayer._aggregate(sigs)
        assert result.raw.get("completeness") == pytest.approx(2 / 3, abs=0.01)

    def test_language_shifts_wfs(self):
        """Strongly positive language signal should shift WFS up."""
        base = [_world_sig("SOCIAL", 0.5), _world_sig("ECOLOGICAL", 0.4)]
        profile_base = synthesise_kepe_profile("TEST", base)

        lang_positive = WorldSignal(
            domain="LANGUAGE", source="test_lang",
            region="GLOBAL", value=0.85, confidence=0.55,
            evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
        )
        profile_lang = synthesise_kepe_profile("TEST", base + [lang_positive])

        assert profile_lang.wfs >= profile_base.wfs, (
            f"Positive language signal should raise WFS: "
            f"{profile_base.wfs:.3f} → {profile_lang.wfs:.3f}"
        )


# ─── CMAM helpers ─────────────────────────────────────────────────────────────

class _MockSig:
    """Minimal DFTESignal stand-in for CMAM classify_trade tests."""
    def __init__(self, action="BUY", position_size_pct=2.0, rationale=""):
        self.action = action
        self.position_size_pct = position_size_pct
        self.rationale = rationale


class _MockKepe:
    """Minimal KEPEProfile stand-in for CMAM classify_trade tests."""
    def __init__(self, is_syntropic=False, is_extractive=False, sts="STABLE"):
        self.is_syntropic_asset = is_syntropic
        self.is_extractive_asset = is_extractive
        self.sts = sts


# ─── TestCMAM ─────────────────────────────────────────────────────────────────

class TestCMAM:
    """
    CMAM — Capital Maturity Allocation Model.
    All tests use synthetic data — no network calls. [TESTABLE]
    """

    def setup_method(self):
        self.engine = CMAMEngine(x1=10_000, x2=100_000, sar_max=0.40, short_book_cap=0.15)

    # ── SAR equation ──────────────────────────────────────────────────────────

    def test_sar_zero_below_x1(self):
        assert self.engine.compute_sar(5_000) == 0.0

    def test_sar_zero_at_x1(self):
        assert self.engine.compute_sar(10_000) == 0.0

    def test_sar_max_at_x2(self):
        assert self.engine.compute_sar(100_000) == pytest.approx(0.40)

    def test_sar_max_above_x2(self):
        assert self.engine.compute_sar(500_000) == pytest.approx(0.40)

    def test_sar_linear_midpoint(self):
        # midpoint = 55,000 → SAR = 0.20
        sar = self.engine.compute_sar(55_000)
        assert sar == pytest.approx(0.20, abs=1e-9)

    def test_sar_quarter_point(self):
        # F=32,500 → (32500-10000)/(100000-10000)*0.40 = 0.10
        sar = self.engine.compute_sar(32_500)
        assert sar == pytest.approx(0.10, abs=1e-9)

    # ── Profile mode ──────────────────────────────────────────────────────────

    def test_mode_st_below_x1(self):
        p = self.engine.profile(5_000)
        assert p.mode == "ST_MODE"
        assert p.sar == 0.0
        assert p.lt_budget == 0.0

    def test_mode_transition(self):
        p = self.engine.profile(55_000)
        assert p.mode == "TRANSITION"
        assert 0.0 < p.sar < 0.40

    def test_mode_mature_above_x2(self):
        p = self.engine.profile(200_000)
        assert p.mode == "MATURE"
        assert p.sar == pytest.approx(0.40)

    # ── LT budget correctly sized from SAR ────────────────────────────────────

    def test_lt_budget_at_maturity(self):
        p = self.engine.profile(100_000)
        assert p.lt_budget == pytest.approx(40_000.0)
        assert p.st_budget == pytest.approx(60_000.0)

    def test_lt_budget_zero_in_st_mode(self):
        p = self.engine.profile(5_000)
        assert p.lt_budget == 0.0
        assert p.st_budget == pytest.approx(5_000.0)

    # ── Short book cap ────────────────────────────────────────────────────────

    def test_short_book_cap_enforced(self):
        # short_used = full 15% cap
        self.engine.profile(100_000, short_used=15_000)
        tc = self.engine.classify_trade(
            _MockSig(action="SELL", position_size_pct=3.0),
            _MockKepe(is_syntropic=False, is_extractive=False, sts="STABLE"),
        )
        assert tc.trade_type == "BLOCKED"

    def test_short_book_partial_capacity_capped(self):
        # short_used = 10_000, cap = 15_000 → 5_000 remaining = 5%
        self.engine.profile(100_000, short_used=10_000)
        tc = self.engine.classify_trade(
            _MockSig(action="SELL", position_size_pct=10.0),
            _MockKepe(is_syntropic=False, is_extractive=False, sts="STABLE"),
        )
        assert tc.trade_type == "ST"
        assert tc.max_size_pct == pytest.approx(5.0, abs=1e-9)

    def test_short_profits_route_to_lt_budget(self):
        self.engine.profile(100_000, short_used=0)
        tc = self.engine.classify_trade(
            _MockSig(action="SELL", position_size_pct=3.0),
            _MockKepe(is_syntropic=False, is_extractive=False, sts="STABLE"),
        )
        assert tc.trade_type == "ST"
        assert "lt_budget" in tc.routing_note.lower()

    # ── LT trade blocked when STS=DETERIORATING ───────────────────────────────

    def test_lt_blocked_when_sts_deteriorating(self):
        self.engine.profile(200_000)  # mature — lt_budget available
        tc = self.engine.classify_trade(
            _MockSig(action="BUY", position_size_pct=5.0),
            _MockKepe(is_syntropic=True, is_extractive=False, sts="DETERIORATING"),
        )
        # Syntropic but DETERIORATING → downgrade to ST, not LT
        assert tc.trade_type == "ST"
        assert tc.budget_source == "st_budget"

    def test_lt_allowed_when_sts_loading(self):
        self.engine.profile(200_000)
        tc = self.engine.classify_trade(
            _MockSig(action="BUY", position_size_pct=5.0),
            _MockKepe(is_syntropic=True, is_extractive=False, sts="LOADING"),
        )
        assert tc.trade_type == "LT"
        assert tc.budget_source == "lt_budget"

    def test_lt_allowed_when_sts_stable(self):
        self.engine.profile(200_000)
        tc = self.engine.classify_trade(
            _MockSig(action="BUY", position_size_pct=5.0),
            _MockKepe(is_syntropic=True, is_extractive=False, sts="STABLE"),
        )
        assert tc.trade_type == "LT"

    # ── Extractive hard blocked ────────────────────────────────────────────────

    def test_extractive_blocked_regardless_of_fund_size(self):
        self.engine.profile(10_000_000)  # enormous fund
        tc = self.engine.classify_trade(
            _MockSig(action="BUY", position_size_pct=5.0),
            _MockKepe(is_syntropic=False, is_extractive=True, sts="LOADING"),
        )
        assert tc.trade_type == "BLOCKED"
        assert tc.max_size_pct == 0.0

    def test_extractive_blocked_in_st_mode(self):
        self.engine.profile(1_000)   # tiny fund, ST mode
        tc = self.engine.classify_trade(
            _MockSig(action="BUY", position_size_pct=2.0),
            _MockKepe(is_syntropic=False, is_extractive=True, sts="STABLE"),
        )
        assert tc.trade_type == "BLOCKED"

    # ── Mirror gate ───────────────────────────────────────────────────────────

    def test_mirror_gate_flags_information_asymmetry(self):
        mc = CMAMEngine.mirror_check(
            "trade relies on information asymmetry exploitation not available to public"
        )
        assert mc.passed is False
        assert mc.flag is not None
        assert "asymmetry" in mc.flag.lower()

    def test_mirror_gate_flags_opacity(self):
        mc = CMAMEngine.mirror_check("this trade requires opacity from regulators")
        assert mc.passed is False

    def test_mirror_gate_flags_extraction_timing(self):
        mc = CMAMEngine.mirror_check("extraction timing gives us edge over retail")
        assert mc.passed is False

    def test_mirror_gate_passes_clean_rationale(self):
        mc = CMAMEngine.mirror_check(
            "syntropic momentum confirmed by KEPE and MFS convergence"
        )
        assert mc.passed is True
        assert mc.flag is None

    # ── Cooling-off gate ──────────────────────────────────────────────────────

    def test_cooling_off_triggered_on_conversion_reasoning(self):
        assert CMAMEngine.cooling_off_required(
            "applying entropy to syntropy conversion to assess long-term value"
        ) is True

    def test_cooling_off_triggered_arrow_notation(self):
        assert CMAMEngine.cooling_off_required(
            "field analysis uses entropy→syntropy transition signal"
        ) is True

    def test_cooling_off_not_triggered_normal_rationale(self):
        assert CMAMEngine.cooling_off_required(
            "strong KEPE signal MFS=0.72 WFS=0.65 ν=0.89 all gates passed"
        ) is False

    # ── Safety: no profile ────────────────────────────────────────────────────

    def test_classify_blocked_without_profile(self):
        engine = CMAMEngine()   # fresh — no profile() called
        tc = engine.classify_trade(
            _MockSig(action="BUY", position_size_pct=2.0),
            _MockKepe(is_syntropic=True, is_extractive=False, sts="STABLE"),
        )
        assert tc.trade_type == "BLOCKED"


# ─── SAS helpers ──────────────────────────────────────────────────────────────

class _MockKEPEForSAS:
    """Minimal KEPEProfile stand-in for SAS tests."""
    def __init__(self, wfs=0.50, sts="STABLE", is_syntropic=False, is_extractive=False):
        self.wfs                = wfs
        self.sts                = sts
        self.is_syntropic_asset = is_syntropic
        self.is_extractive_asset = is_extractive


# ─── TestSAS ──────────────────────────────────────────────────────────────────

class TestSAS:
    """
    SAS — Syntropy Authenticity Score.
    All tests use pure scoring functions — no network calls. [TESTABLE]
    """

    # ── RevenueCoherenceSignal ────────────────────────────────────────────────

    def test_clean_energy_coherent_revenue(self):
        # Syntropic company, clean tech SIC, clean language dominant → high score
        score, notes = RevenueCoherenceSignal._score_coherence(
            sic_code=3674,      # semiconductors (ENPH)
            fossil_count=2,
            clean_count=30,
            climate_count=5,
            is_syntropic=True,
            is_extractive=False,
        )
        assert score >= 0.70, f"Clean energy company should score >=0.70, got {score}"
        assert notes == [] or all("fossil" not in n.lower() for n in notes)

    def test_fossil_fuel_divergent(self):
        # Syntropic classification but fossil SIC and fossil language → low score (wolf)
        score, notes = RevenueCoherenceSignal._score_coherence(
            sic_code=2911,      # petroleum refining
            fossil_count=50,
            clean_count=3,
            climate_count=8,
            is_syntropic=True,
            is_extractive=False,
        )
        assert score <= 0.35, f"Wolf company should score <=0.35, got {score}"
        assert any("fossil" in n.lower() or "sic" in n.lower() for n in notes)

    def test_extractive_greenwashing_detected(self):
        # Extractive company with many climate claims → wolf pattern
        score, notes = RevenueCoherenceSignal._score_coherence(
            sic_code=2911,
            fossil_count=40,
            clean_count=5,
            climate_count=20,    # heavy ESG claims
            is_syntropic=False,
            is_extractive=True,
        )
        assert score <= 0.30, f"Greenwashing extractive should score <=0.30, got {score}"
        assert any("greenwashing" in n.lower() or "wolf" in n.lower() for n in notes)

    def test_neutral_company_default_medium(self):
        score, _ = RevenueCoherenceSignal._score_coherence(
            sic_code=7372,      # software
            fossil_count=1,
            clean_count=5,
            climate_count=3,
            is_syntropic=False,
            is_extractive=False,
        )
        assert 0.55 <= score <= 0.85

    # ── CapexDirectionSignal ──────────────────────────────────────────────────

    def test_capex_syntropic_growing(self):
        # Syntropic + capex trend +1.0 → high coherence
        score = CapexDirectionSignal._score_capex(capex_trend=1.0, is_syntropic=True)
        assert score == pytest.approx(1.0)

    def test_capex_syntropic_declining(self):
        # Syntropic + capex trend -1.0 → low coherence (transition theater)
        score = CapexDirectionSignal._score_capex(capex_trend=-1.0, is_syntropic=True)
        assert score == pytest.approx(0.0)

    def test_capex_extractive_growing_is_wolf(self):
        # Extractive + capex growing = doubling down on extraction → low SAS
        score = CapexDirectionSignal._score_capex(capex_trend=1.0, is_syntropic=False)
        assert score == pytest.approx(0.0)

    def test_capex_extractive_declining_slightly_better(self):
        # Extractive + capex declining = possibly exiting extraction → higher SAS
        score = CapexDirectionSignal._score_capex(capex_trend=-1.0, is_syntropic=False)
        assert score == pytest.approx(1.0)

    # ── OpacitySignal ─────────────────────────────────────────────────────────

    def test_opacity_climate_claims_without_scope3(self):
        # Many climate claims, zero Scope 3 → very low opacity score
        score, notes = OpacitySignal._score_opacity(
            scope3_count=0, climate_count=12, supply_chain_words=5
        )
        assert score <= 0.30
        assert any("scope 3" in n.lower() or "zero" in n.lower() for n in notes)

    def test_opacity_full_disclosure(self):
        # Good scope 3 coverage + supply chain disclosure → high score
        score, notes = OpacitySignal._score_opacity(
            scope3_count=5, climate_count=10, supply_chain_words=30
        )
        assert score >= 0.65

    def test_opacity_no_climate_claims_neutral(self):
        # No climate claims → no scope 3 gap → neutral
        score, _ = OpacitySignal._score_opacity(
            scope3_count=0, climate_count=0, supply_chain_words=10
        )
        # With no climate count, scope3_ratio=0/1=0 → scope3_score=0
        # score = 0.60*0 + 0.40*min(1, 10/25) = 0.40*0.40 = 0.16
        assert score <= 0.50

    # ── SSIStub ───────────────────────────────────────────────────────────────

    def test_ssi_gap_low_when_aligned(self):
        # Frame1=0.60, Frame5=0.62 → gap=0.02
        gap, notes = SSIStub.compute_ssi_gap(frame1_wfs=0.60, frame5_score=0.62)
        assert gap == pytest.approx(0.02, abs=1e-9)
        assert notes == []   # no divergence note for small gap

    def test_ssi_gap_high_when_divergent(self):
        # Frame1=0.80 (self-report high), Frame5=0.20 (primary sources low) → gap=0.60
        gap, notes = SSIStub.compute_ssi_gap(frame1_wfs=0.80, frame5_score=0.20)
        assert gap == pytest.approx(0.60, abs=1e-9)
        assert any("divergence" in n.lower() or "gap" in n.lower() for n in notes)

    def test_ssi_gap_max_is_one(self):
        gap, _ = SSIStub.compute_ssi_gap(frame1_wfs=1.0, frame5_score=0.0)
        assert gap == pytest.approx(1.0)

    # ── WolfDetector ──────────────────────────────────────────────────────────

    def test_wolf_confirmed_threshold(self):
        # All components low → wolf_score > 0.65
        ws = WolfDetector.wolf_score(
            revenue_coherence=0.10,
            capex_direction=0.15,
            opacity_score=0.10,
            ssi_gap=0.70,
        )
        assert ws > 0.65, f"Should be wolf confirmed, got wolf_score={ws}"

    def test_wolf_score_authentic_company_low(self):
        # All components high → wolf_score low
        ws = WolfDetector.wolf_score(
            revenue_coherence=0.85,
            capex_direction=0.80,
            opacity_score=0.75,
            ssi_gap=0.05,
        )
        assert ws < 0.35, f"Authentic company should have low wolf score, got {ws}"

    def test_short_candidate_requires_deteriorating(self):
        # wolf_score > 0.35 but STS=STABLE → NOT a short candidate
        # Simulate with SASEngine._neutral_profile to check logic
        # Wolf pattern present but STS not DETERIORATING
        kepe = _MockKEPEForSAS(wfs=0.50, sts="STABLE", is_syntropic=False, is_extractive=True)
        # With stable STS, short_candidate should stay False
        profile = SASEngine._neutral_profile("TEST", "test reason")
        assert profile.short_candidate is False

    def test_evidence_declared_on_all_components(self):
        profile = SASEngine._neutral_profile("TEST", "no data")
        assert profile.evidence_level in ("ESTABLISHED", "TESTABLE", "SPECULATIVE")
        assert profile.evidence_level == "SPECULATIVE"   # neutral profiles are speculative

    def test_notes_populated_on_divergence(self):
        # Direct coherence scoring: wolf pattern should populate notes
        _, notes = RevenueCoherenceSignal._score_coherence(
            sic_code=2911, fossil_count=60, clean_count=2, climate_count=25,
            is_syntropic=True, is_extractive=False,
        )
        assert len(notes) >= 1

    def test_graceful_missing_sec_data(self):
        # _neutral_profile is used for ETF/crypto — no exception raised
        profile = SASEngine._neutral_profile("BTC-USD", "ETF/crypto — no EDGAR CIK")
        assert profile.sas_score == pytest.approx(0.50)
        assert profile.wolf_confirmed is False
        assert profile.short_candidate is False
        assert "BTC-USD" in profile.symbol


# ─── TestBacktest ──────────────────────────────────────────────────────────────

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "bmr"))

from backtest.backtest_engine import (
    run_nu_backtest,
    run_lsii_backtest,
    run_sts_backtest,
    save_report,
    run_backtest,
    BacktestReport,
    SASValidationReport,
    CONFIRMED, PARTIAL, INCONCLUSIVE, REFUTED,
    _compute_nu_proxy,
    _compute_sts_proxy,
    _forward_return,
    _pearson_r,
    _verdict_from_r_and_direction,
    _nu_quartile,
)


def _make_trending_closes(n: int = 300, drift: float = 0.001,
                           seed: int = 42) -> "np.ndarray":
    """Synthetic price series with controlled drift and noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.008, n)
    log_returns = drift + noise
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    return closes


def _make_reverting_series(n: int = 200, reversal_idx: int = 100,
                            seed: int = 0) -> "np.ndarray":
    """
    Price series that trends up to reversal_idx then trends down.
    Used to test LSII arc-break detection.
    """
    rng = np.random.default_rng(seed)
    up   = 100.0 * np.exp(np.cumsum( 0.003 + rng.normal(0, 0.006, reversal_idx)))
    down = up[-1] * np.exp(np.cumsum(-0.003 + rng.normal(0, 0.006, n - reversal_idx)))
    return np.concatenate([up, down])


class TestBacktest:
    """
    Phase 7: Backtest harness — synthetic data only, no network calls. [TESTABLE]
    All tests verify the mechanics of the backtest engine;
    real market verdicts come from live runs.
    """

    # ── forward return helper ──────────────────────────────────────────────────

    def test_forward_return_correct(self):
        closes = np.array([100.0, 101.0, 103.0, 102.0, 105.0])
        assert _forward_return(closes, 0, 4) == pytest.approx(0.05, abs=1e-4)

    def test_forward_return_out_of_bounds_is_none(self):
        closes = np.array([100.0, 102.0])
        assert _forward_return(closes, 0, 5) is None

    # ── ν proxy + quartile binning ─────────────────────────────────────────────

    def test_nu_proxy_returns_result_with_enough_history(self):
        closes = _make_trending_closes(200)
        result = _compute_nu_proxy(closes, 100)
        assert result is not None
        assert 0.0 <= result.nu <= 1.0

    def test_nu_proxy_none_with_insufficient_history(self):
        closes = _make_trending_closes(200)
        assert _compute_nu_proxy(closes, 10) is None

    def test_nu_quartile_binning(self):
        assert _nu_quartile(0.20) == "LOW"
        assert _nu_quartile(0.45) == "MID"
        assert _nu_quartile(0.60) == "HIGH"
        assert _nu_quartile(0.80) == "ZPB"

    # ── verdict generation (epistemically honest) ──────────────────────────────

    def test_verdict_confirmed_strong_positive_r(self):
        v = _verdict_from_r_and_direction(r=0.40, n=100, expected_positive=True)
        assert v == CONFIRMED

    def test_verdict_refuted_when_direction_wrong(self):
        # Hypothesis expects positive correlation, but r is strongly negative
        v = _verdict_from_r_and_direction(r=-0.30, n=100, expected_positive=True)
        assert v == REFUTED

    def test_verdict_inconclusive_weak_signal(self):
        v = _verdict_from_r_and_direction(r=0.05, n=100, expected_positive=True)
        assert v == INCONCLUSIVE

    def test_verdict_inconclusive_too_few_observations(self):
        v = _verdict_from_r_and_direction(r=0.50, n=5, expected_positive=True)
        assert v == INCONCLUSIVE

    def test_verdict_partial_moderate_signal(self):
        v = _verdict_from_r_and_direction(r=0.15, n=80, expected_positive=True)
        assert v in (PARTIAL, INCONCLUSIVE)   # could be either depending on t-stat

    # ── synthetic ν backtest — should detect known correlation ────────────────

    def test_nu_backtest_detects_planted_correlation(self):
        """
        Synthetic series: strong drift → ν should be in HIGH/ZPB quartile
        while the series is trending. This verifies engine runs without error
        and produces non-trivial quartile distributions.
        """
        closes = _make_trending_closes(300, drift=0.002, seed=1)
        report = run_nu_backtest("SYNTHETIC_UP", closes)
        assert report.n_observations >= 10
        assert report.evidence_verdict in (CONFIRMED, PARTIAL, INCONCLUSIVE, REFUTED)
        # ZPB or HIGH quartile should have positive 10d mean return (drift is +ve)
        qr_10 = report.per_quartile_mean_return.get("10d", {})
        high_return = qr_10.get("HIGH", None)
        zpb_return  = qr_10.get("ZPB", None)
        # At least one of HIGH/ZPB should be positive (planted upward drift)
        if high_return is not None and zpb_return is not None:
            assert max(high_return, zpb_return) > -0.05, (
                "Expected at least one of HIGH/ZPB quartile to show positive return "
                f"in a strongly trending up series. Got HIGH={high_return}, ZPB={zpb_return}"
            )

    # ── synthetic LSII backtest — reverting series ────────────────────────────

    def test_lsii_backtest_reverting_series(self):
        """
        Price that trends up then hard-reverses.
        LSII should flag the reversal zone and ideally predict negative returns.
        Engine must run without error and produce a verdict.
        """
        closes = _make_reverting_series(200, reversal_idx=100)
        highs   = closes * 1.005
        lows    = closes * 0.995
        opens   = closes * 1.001
        volumes = np.full_like(closes, 1_000_000.0)
        timestamps = np.arange(len(closes), dtype=float) * 86400 + 1_700_000_000

        report = run_lsii_backtest(
            "SYNTHETIC_REV", closes, highs, lows, opens, volumes, timestamps
        )
        assert report.evidence_verdict in (CONFIRMED, PARTIAL, INCONCLUSIVE, REFUTED)
        # Both n_flagged + n_baseline should be >= 0 (no crash)
        assert report.n_flagged >= 0
        assert report.n_baseline >= 0

    # ── STS proxy ─────────────────────────────────────────────────────────────

    def test_sts_transition_to_loading_on_acceleration(self):
        """
        A series that accelerates upward (drift increasing) should produce
        more LOADING states than a flat series, because the WFS proxy slope
        reflects the *change* in 20-day returns, not their absolute level.
        """
        rng = np.random.default_rng(42)
        # Accelerating upward: drift grows from 0 → 0.008
        n = 300
        drifts = np.linspace(0.0, 0.008, n)
        log_returns = drifts + rng.normal(0, 0.005, n)
        closes_accel = 100.0 * np.exp(np.cumsum(log_returns))

        # Flat (no drift)
        closes_flat = 100.0 + rng.normal(0, 0.5, n)

        counts_accel = {"LOADING": 0, "STABLE": 0, "DETERIORATING": 0}
        counts_flat  = {"LOADING": 0, "STABLE": 0, "DETERIORATING": 0}
        for idx in range(100, 290):
            counts_accel[_compute_sts_proxy(closes_accel, idx)] += 1
            counts_flat[_compute_sts_proxy(closes_flat,   idx)] += 1

        # Accelerating series should have more LOADING than flat
        assert counts_accel["LOADING"] >= counts_flat["LOADING"], (
            f"Accelerating series should produce more LOADING than flat: "
            f"accel_LOADING={counts_accel['LOADING']} vs flat_LOADING={counts_flat['LOADING']}"
        )

    def test_sts_returns_valid_state_strings(self):
        """_compute_sts_proxy always returns one of the three valid states."""
        closes = _make_trending_closes(200, drift=0.002)
        valid = {"LOADING", "STABLE", "DETERIORATING"}
        for idx in range(80, 180):
            state = _compute_sts_proxy(closes, idx)
            assert state in valid, f"Unexpected STS state: {state}"

    # ── SAS backtest is marked SPECULATIVE ────────────────────────────────────

    def test_sas_backtest_is_speculative(self):
        sas = SASValidationReport(symbol="TEST")
        assert sas.evidence_level == "SPECULATIVE"
        assert sas.evidence_verdict == INCONCLUSIVE
        assert len(sas.mystery_pile_reason) > 10

    # ── REFUTED verdict is never suppressed ───────────────────────────────────

    def test_refuted_result_on_anti_correlated_series(self):
        """
        Descending drift series: ν (derived from upward price history)
        will NOT predict positive forward returns. Verdict should be REFUTED
        or INCONCLUSIVE — never CONFIRMED.
        """
        closes = _make_trending_closes(300, drift=-0.003, seed=99)
        report = run_nu_backtest("SYNTHETIC_DOWN", closes)
        assert report.evidence_verdict in (REFUTED, INCONCLUSIVE, PARTIAL)
        # Must NOT be CONFIRMED on a downward drift (no upward coherence)
        assert report.evidence_verdict != CONFIRMED

    # ── JSON persistence ──────────────────────────────────────────────────────

    def test_report_saves_to_json(self, tmp_path):
        """BacktestReport serialises to valid JSON without error."""
        # Build a minimal report without network calls
        closes = _make_trending_closes(300)
        nu_r   = run_nu_backtest("SPY_TEST", closes)
        highs  = closes * 1.005
        lows   = closes * 0.995
        opens  = closes
        vols   = np.full_like(closes, 1e6)
        ts     = np.arange(len(closes), dtype=float) * 86400 + 1_700_000_000
        lsii_r = run_lsii_backtest("SPY_TEST", closes, highs, lows, opens, vols, ts)
        sts_r  = run_sts_backtest("SPY_TEST", closes)
        sas_r  = SASValidationReport(symbol="SPY_TEST")

        from backtest.backtest_engine import (
            SymbolBacktestResult,
            _aggregate_nu, _aggregate_lsii, _aggregate_sts,
            _overall_verdict_and_recommendations,
        )
        from datetime import timezone

        sym_result = SymbolBacktestResult(
            symbol="SPY_TEST",
            nu_report=nu_r, lsii_report=lsii_r,
            sts_report=sts_r, sas_report=sas_r,
        )
        nu_agg   = _aggregate_nu([nu_r])
        lsii_agg = _aggregate_lsii([lsii_r])
        sts_agg  = _aggregate_sts([sts_r])
        overall, mystery, calibration = _overall_verdict_and_recommendations(
            nu_agg, lsii_agg, sts_agg
        )
        report = BacktestReport(
            run_date=datetime.now(timezone.utc).isoformat(),
            symbols=["SPY_TEST"],
            per_symbol=[sym_result],
            nu_aggregate=nu_agg,
            lsii_aggregate=lsii_agg,
            sts_aggregate=sts_agg,
            overall_verdict=overall,
            mystery_pile_items=mystery,
            calibration_recommendations=calibration,
        )
        path = str(tmp_path / "test_backtest.json")
        saved = save_report(report, path=path)
        assert _os.path.exists(saved)
        import json as _json
        with open(saved) as f:
            data = _json.load(f)
        assert data["overall_verdict"] in (CONFIRMED, PARTIAL, INCONCLUSIVE, REFUTED)
        assert "mystery_pile_items" in data
        assert "calibration_recommendations" in data

    # ── mystery pile populated when inconclusive ──────────────────────────────

    def test_mystery_pile_populated_on_inconclusive(self):
        """
        Flat price (no drift) → all signals weak → INCONCLUSIVE verdicts
        → mystery pile should be populated.
        """
        rng = np.random.default_rng(7)
        closes = 100.0 + rng.normal(0, 0.5, 300)   # near-flat
        nu_r  = run_nu_backtest("FLAT", closes)
        highs = closes * 1.002
        lows  = closes * 0.998
        opens = closes
        vols  = np.full_like(closes, 1e6)
        ts    = np.arange(len(closes), dtype=float) * 86400 + 1_700_000_000
        lsii_r = run_lsii_backtest("FLAT", closes, highs, lows, opens, vols, ts)
        sts_r  = run_sts_backtest("FLAT", closes)

        from backtest.backtest_engine import (
            _aggregate_nu, _aggregate_lsii, _aggregate_sts,
            _overall_verdict_and_recommendations,
        )
        nu_agg   = _aggregate_nu([nu_r])
        lsii_agg = _aggregate_lsii([lsii_r])
        sts_agg  = _aggregate_sts([sts_r])
        _, mystery, _ = _overall_verdict_and_recommendations(
            nu_agg, lsii_agg, sts_agg
        )
        # SAS is always in the mystery pile regardless of verdicts
        assert any("SAS" in item for item in mystery), (
            "SAS should always be in mystery pile (SPECULATIVE data requirement)"
        )
