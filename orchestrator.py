"""
DFTE — Main Orchestrator
==========================
Ties together:
  KEPE     → World Field Score (WFS)
  BMR      → Market Field Score (MFS) via bmr_server API
  DFTE     → Unified signal + trade decision
  Governance → Benevolence + contradiction gate
  Wallet   → Execution

Usage:
  # Start BMR server first (port 8001):
  cd kindpath-bmr && python bmr_server.py

  # Then run orchestrator:
  cd kindpath-dfte
  python orchestrator.py --symbols SPY QQQ GLD BTC-USD --mode paper
  python orchestrator.py --symbols ICLN NEE --mode paper  # syntropic basket
  python orchestrator.py --watch  # continuous loop with dashboard

Architecture:
  orchestrator.py
    ├── fetch_bmr_signal(symbol)      → BMRSummary via HTTP
    ├── fetch_kepe_signal(symbol)     → KEPEProfile via indicator stack
    ├── synthesise_dfte_signal()      → DFTESignal
    ├── governance_layer.py           → tier cap + contradiction check
    ├── wallet.py                     → execute or simulate
    └── dashboard(signals)            → rich terminal output
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import argparse
import requests
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict

# Path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kepe.indicators import (
    EcologicalSignal, WorldBankSignal, GDELTSignal,
    OptimismSignal, ConflictPressureSignal
)
from kepe.syntropy_engine import synthesise_kepe_profile, KEPEProfile
from dfte.dfte_engine import (
    BMRSummary, KEPESummary, synthesise_dfte_signal, DFTESignal
)
from governance.governance_layer import (
    score_benevolence, apply_governance_tier,
    detect_contradictions, log_influence,
    get_influence_summary
)
from wallet.wallet import get_wallet, OrderRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("dfte.orchestrator")

BMR_SERVER = os.environ.get("BMR_SERVER", "http://localhost:8001")


# ─── Data fetchers ────────────────────────────────────────────────────────────

def fetch_bmr_signal(symbol: str, timeframe: str = "1d") -> Optional[BMRSummary]:
    """Fetch Market Field Score from BMR server."""
    try:
        resp = requests.post(
            f"{BMR_SERVER}/analyse",
            json={
                "symbol": symbol,
                "timeframe": timeframe,
                "periods": 200,
                "include_lsii": True,
                "include_curvature": True,
                "multi_timeframe": False,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"BMR server returned {resp.status_code} for {symbol}")
            return None

        data = resp.json()
        lsii = data.get("lsii") or {}
        curv = data.get("curvature") or {}

        return BMRSummary(
            symbol=symbol,
            mfs=float(data.get("mfs", 0.5)),
            mfs_label=data.get("mfs_label", "DRIFT"),
            direction=float(data.get("direction", 0.0)),
            nu=float(data.get("nu", {}).get("score", 0.5)),
            field_state=data.get("nu", {}).get("field_state", "DRIFT"),
            trade_tier=data.get("trade_tier", "NANO"),
            lsii=lsii.get("score"),
            lsii_flag=lsii.get("flag"),
            k=curv.get("k"),
            curvature_state=curv.get("state"),
        )
    except requests.exceptions.ConnectionError:
        logger.warning(
            f"Cannot reach BMR server at {BMR_SERVER}. "
            "Start it with: cd kindpath-bmr && python bmr_server.py"
        )
        return None
    except Exception as e:
        logger.error(f"BMR fetch error for {symbol}: {e}")
        return None


def fetch_kepe_signal(
    symbol: str,
    market_curvature_k: Optional[float] = None
) -> KEPEProfile:
    """Fetch and synthesise World Field Score."""
    signals = []

    # Collect all world field indicators
    indicators = [
        EcologicalSignal(),
        WorldBankSignal(),
        GDELTSignal(),
        OptimismSignal(),
        ConflictPressureSignal(),
    ]

    for ind in indicators:
        try:
            sig = ind.compute()
            if sig.confidence > 0:
                signals.append(sig)
        except Exception as e:
            logger.warning(f"Indicator {ind.__class__.__name__} failed: {e}")

    return synthesise_kepe_profile(
        symbol=symbol,
        signals=signals,
        market_curvature_k=market_curvature_k,
    )


def kepe_to_summary(kepe: KEPEProfile) -> KEPESummary:
    return KEPESummary(
        symbol=kepe.symbol,
        wfs=kepe.wfs,
        wfs_label=kepe.wfs_label,
        spi=kepe.spi,
        opc=kepe.opc,
        interference_load=kepe.interference_load,
        unified_curvature=kepe.unified_curvature,
        equity_weight=kepe.equity_weight,
        is_syntropic=kepe.is_syntropic_asset,
        is_extractive=kepe.is_extractive_asset,
    )


# ─── Full analysis pipeline ───────────────────────────────────────────────────

def analyse_symbol(
    symbol: str,
    timeframe: str = "1d",
    base_risk_pct: float = 1.0,
) -> Optional[DFTESignal]:
    """Run full KEPE + BMR + DFTE pipeline for one symbol."""
    logger.info(f"Analysing {symbol}...")

    # 1. BMR — market field
    bmr = fetch_bmr_signal(symbol, timeframe)
    if bmr is None:
        logger.warning(f"No BMR signal for {symbol} — BMR server may be offline")
        return None

    # 2. KEPE — world field
    kepe = fetch_kepe_signal(symbol, market_curvature_k=bmr.k)
    kepe_summary = kepe_to_summary(kepe)

    # 3. DFTE — unified signal
    dfte_signal = synthesise_dfte_signal(bmr, kepe_summary, base_risk_pct)

    # 4. Governance — apply benevolence tier cap
    ben = score_benevolence(symbol)
    approved_tier, gov_reason = apply_governance_tier(dfte_signal.tier, ben)

    if approved_tier != dfte_signal.tier:
        logger.info(f"Governance override: {dfte_signal.tier} → {approved_tier} ({gov_reason})")
        dfte_signal.tier = approved_tier
        if approved_tier == "BLOCKED":
            dfte_signal.action = "BLOCKED"
            dfte_signal.position_size_pct = 0.0
            dfte_signal.governance_gate = False
            dfte_signal.all_gates_passed = False
        dfte_signal.rationale += f" | GOV: {gov_reason}"

    return dfte_signal


def run_basket(
    symbols: List[str],
    timeframe: str = "1d",
    base_risk_pct: float = 1.0,
    execute: bool = False,
    wallet_mode: str = "paper",
) -> Dict[str, DFTESignal]:
    """
    Analyse and optionally execute a basket of symbols.
    Returns dict of symbol → DFTESignal.
    """
    signals: Dict[str, DFTESignal] = {}
    portfolio: Dict[str, float] = {}

    for symbol in symbols:
        sig = analyse_symbol(symbol, timeframe, base_risk_pct)
        if sig:
            signals[symbol] = sig
            portfolio[symbol] = sig.position_size_pct if sig.action == "BUY" else 0.0

    # Portfolio-level contradiction check
    if len(portfolio) > 1:
        contradiction_report = detect_contradictions(portfolio)
        if contradiction_report.interference_load > 0.5:
            logger.warning(
                f"Portfolio interference load: {contradiction_report.interference_load:.2f}\n"
                + "\n".join(f"  • {c}" for c in contradiction_report.contradictions)
            )

    # Execute if requested
    if execute:
        wallet = get_wallet(wallet_mode)
        cash = wallet.get_cash()
        logger.info(f"Wallet: ${cash:,.2f} available ({wallet_mode})")

        for symbol, sig in signals.items():
            if sig.action in ("BUY", "SELL") and sig.all_gates_passed and sig.position_size_pct > 0:
                notional = cash * (sig.position_size_pct / 100)
                order = OrderRequest(
                    symbol=symbol,
                    side=sig.action.lower(),
                    notional=notional,
                    tier=sig.tier,
                    rationale=sig.rationale,
                )
                result = wallet.submit_order(order)

                # Log influence
                log_influence(
                    symbol=symbol,
                    action=sig.action,
                    tier=sig.tier,
                    size_pct=sig.position_size_pct,
                    mfs=sig.mfs,
                    wfs=sig.wfs,
                    nu=sig.nu,
                )

                if result.success:
                    logger.info(
                        f"✓ {sig.action} {symbol} ${notional:,.2f} "
                        f"[{sig.tier}] filled @ {result.fill_price}"
                    )
                else:
                    logger.error(f"✗ Order failed for {symbol}: {result.error}")

    return signals


# ─── Terminal dashboard ───────────────────────────────────────────────────────

def print_dashboard(signals: Dict[str, DFTESignal]):
    """Rich terminal dashboard of DFTE signals."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        console.print(Panel.fit(
            "[bold cyan]DFTE — Dual Field Trading Engine[/bold cyan]\n"
            "[dim]M = [(Participant × Institutional × Sovereign) · ν]² | "
            "KEPE × BMR → Trade Decision[/dim]",
            border_style="cyan"
        ))

        # Main signals table
        table = Table(show_header=True, header_style="bold blue", box=None, padding=(0,1))
        table.add_column("Symbol",    width=10)
        table.add_column("Action",    width=8)
        table.add_column("Tier",      width=6)
        table.add_column("MFS",       width=6)
        table.add_column("WFS",       width=6)
        table.add_column("ν",         width=6)
        table.add_column("Conv.",     width=6)
        table.add_column("Size%",     width=6)
        table.add_column("Gates",     width=12)
        table.add_column("State",     width=12)

        for symbol, sig in sorted(signals.items()):
            action_colour = {
                "BUY":     "green",
                "SELL":    "red",
                "HOLD":    "yellow",
                "BLOCKED": "red",
            }.get(sig.action, "white")

            tier_colour = {
                "LARGE": "green",
                "MID":   "yellow",
                "NANO":  "cyan",
                "WAIT":  "dim",
                "BLOCKED": "red",
            }.get(sig.tier, "white")

            gates = (
                f"{'✓' if sig.mfs_gate else '✗'}"
                f"{'✓' if sig.wfs_gate else '✗'}"
                f"{'✓' if sig.governance_gate else '✗'}"
            )

            table.add_row(
                f"[bold]{symbol}[/bold]",
                f"[{action_colour}]{sig.action}[/{action_colour}]",
                f"[{tier_colour}]{sig.tier}[/{tier_colour}]",
                f"{sig.mfs:.2f}",
                f"{sig.wfs:.2f}",
                f"{sig.nu:.3f}",
                f"{sig.conviction:.2f}",
                f"{sig.position_size_pct:.2f}",
                gates,
                sig.field_state if hasattr(sig, 'field_state') else "-",
            )

        console.print(table)

        # Warnings
        all_warnings = []
        for symbol, sig in signals.items():
            for w in sig.warnings:
                all_warnings.append(f"[yellow]{symbol}[/yellow]: {w}")

        if all_warnings:
            console.print(Panel(
                "\n".join(all_warnings),
                title="[yellow]Warnings[/yellow]",
                border_style="yellow"
            ))

        # Recent influence log
        influence = get_influence_summary(5)
        if influence:
            console.print("\n[dim]Recent field contributions:[/dim]")
            for rec in influence[-3:]:
                console.print(
                    f"  [dim]{rec['timestamp'][:16]}[/dim] "
                    f"{rec['action']} {rec['symbol']} [{rec['tier']}] "
                    f"MFS={rec['mfs']:.2f} WFS={rec['wfs']:.2f} ν={rec['nu']:.3f}"
                )

        console.print(f"\n[dim]Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}[/dim]")

    except ImportError:
        # Fallback: plain text
        print("\n=== DFTE Signal Report ===")
        print(f"{'Symbol':<10} {'Action':<8} {'Tier':<6} {'MFS':>6} {'WFS':>6} {'ν':>6} {'Conv':>6} {'Size%':>6}")
        print("-" * 60)
        for symbol, sig in sorted(signals.items()):
            print(
                f"{symbol:<10} {sig.action:<8} {sig.tier:<6} "
                f"{sig.mfs:>6.2f} {sig.wfs:>6.2f} {sig.nu:>6.3f} "
                f"{sig.conviction:>6.2f} {sig.position_size_pct:>6.2f}"
            )
        print(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DFTE — Dual Field Trading Engine"
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=["SPY", "QQQ", "GLD", "BTC-USD"],
        help="Symbols to analyse"
    )
    parser.add_argument(
        "--timeframe", default="1d",
        help="Timeframe (1d, 1h, 4h, 1w)"
    )
    parser.add_argument(
        "--mode", default="paper",
        choices=["paper", "alpaca"],
        help="Wallet mode"
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute trades (paper mode by default)"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Continuous loop (refresh every 5 minutes)"
    )
    parser.add_argument(
        "--risk", type=float, default=1.0,
        help="Base risk per trade as %% of portfolio"
    )
    parser.add_argument(
        "--bmr-server", default="http://localhost:8001",
        help="BMR server URL"
    )

    args = parser.parse_args()

    global BMR_SERVER
    BMR_SERVER = args.bmr_server

    logger.info(f"DFTE starting — symbols: {args.symbols}")
    logger.info(f"BMR server: {BMR_SERVER} | Mode: {args.mode} | Execute: {args.execute}")

    if args.watch:
        while True:
            signals = run_basket(
                symbols=args.symbols,
                timeframe=args.timeframe,
                base_risk_pct=args.risk,
                execute=args.execute,
                wallet_mode=args.mode,
            )
            print_dashboard(signals)
            logger.info("Sleeping 5 minutes...")
            time.sleep(300)
    else:
        signals = run_basket(
            symbols=args.symbols,
            timeframe=args.timeframe,
            base_risk_pct=args.risk,
            execute=args.execute,
            wallet_mode=args.mode,
        )
        print_dashboard(signals)


if __name__ == "__main__":
    main()
