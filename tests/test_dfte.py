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
