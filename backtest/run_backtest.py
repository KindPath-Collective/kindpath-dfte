#!/usr/bin/env python3
"""
DFTE Backtest CLI
=================
Usage:
  python3 run_backtest.py --quick          (SPY only — fast validation)
  python3 run_backtest.py --symbols ENPH NEE XOM SPY QQQ
  python3 run_backtest.py --full           (all 8 symbols, full report)

Reports EVERYTHING: confirmed and refuted equally.
KINDFIELD principle: we do not prove, we record.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BMR  = os.path.join(_ROOT, "bmr")
for _p in [_ROOT, _BMR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backtest_engine import (
    run_backtest, save_report, BacktestReport,
    NuValidationReport, LSIIValidationReport, STSValidationReport,
    CONFIRMED, PARTIAL, INCONCLUSIVE, REFUTED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("backtest.cli")

ALL_SYMBOLS = ["ENPH", "NEE", "XOM", "AAPL", "QQQ", "SPY", "GLD", "BTC-USD"]
QUICK_SYMBOLS = ["SPY"]


def _verdict_colour(verdict: str) -> str:
    return {
        CONFIRMED:    "[bold green]CONFIRMED[/bold green]",
        PARTIAL:      "[bold yellow]PARTIAL[/bold yellow]",
        INCONCLUSIVE: "[dim]INCONCLUSIVE[/dim]",
        REFUTED:      "[bold red]REFUTED[/bold red]",
    }.get(verdict, verdict)


def print_report(report: BacktestReport) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        console = Console()
    except ImportError:
        _print_plain(report)
        return

    console.print()
    console.print(Panel(
        "[bold white]DFTE Phase 7 — Backtest Results[/bold white]\n"
        "[dim]KINDFIELD principle: we do not prove, we record.[/dim]\n"
        "[dim]Refuted results are recorded with equal weight to confirmed.[/dim]",
        border_style="white"
    ))

    # Per-symbol table
    sym_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    sym_table.add_column("Symbol", width=9)
    sym_table.add_column("ν Verdict",    width=14)
    sym_table.add_column("LSII Verdict", width=14)
    sym_table.add_column("STS Verdict",  width=14)
    sym_table.add_column("ν 10d r", width=8)
    sym_table.add_column("LSII 10d Δ",  width=10)
    sym_table.add_column("STS 20d Δ",   width=10)
    sym_table.add_column("n(ν)", width=6)

    for res in report.per_symbol:
        nu   = res.nu_report
        lsii = res.lsii_report
        sts  = res.sts_report
        nu_r   = nu.correlation_nu_vs_return.get("10d", 0)
        lsii_d = lsii.flag_predictive_value.get("10d", 0)
        sts_d  = sts.loading_vs_deteriorating.get("20d", 0)

        def _col(v: float) -> str:
            if v > 0.005:   return f"[green]{v:+.4f}[/green]"
            elif v < -0.005: return f"[red]{v:+.4f}[/red]"
            return f"[dim]{v:+.4f}[/dim]"

        sym_table.add_row(
            f"[bold]{res.symbol}[/bold]",
            _verdict_colour(nu.evidence_verdict),
            _verdict_colour(lsii.evidence_verdict),
            _verdict_colour(sts.evidence_verdict),
            _col(nu_r),
            _col(lsii_d),
            _col(sts_d),
            str(nu.n_observations),
        )

    console.print(sym_table)

    # Aggregate panel
    agg_lines = [
        f"[bold]ν aggregate:[/bold]   {_verdict_colour(report.nu_aggregate.evidence_verdict)}  "
        f"10d r={report.nu_aggregate.correlation_nu_vs_return.get('10d', 0):.3f}  "
        f"20d r={report.nu_aggregate.correlation_nu_vs_return.get('20d', 0):.3f}",
        f"[bold]LSII aggregate:[/bold] {_verdict_colour(report.lsii_aggregate.evidence_verdict)}  "
        f"10d spread={report.lsii_aggregate.flag_predictive_value.get('10d', 0):+.4f}  "
        f"n_flagged={report.lsii_aggregate.n_flagged}",
        f"[bold]STS aggregate:[/bold]  {_verdict_colour(report.sts_aggregate.evidence_verdict)}  "
        f"20d spread={report.sts_aggregate.loading_vs_deteriorating.get('20d', 0):+.4f}  "
        f"60d spread={report.sts_aggregate.loading_vs_deteriorating.get('60d', 0):+.4f}",
        "",
        f"[bold]OVERALL:[/bold] {_verdict_colour(report.overall_verdict)}",
    ]
    console.print(Panel(
        "\n".join(agg_lines),
        title="[bold cyan]Aggregate Verdicts[/bold cyan]",
        border_style="cyan",
    ))

    # ν detail panel
    nu_lines = [report.nu_aggregate.notes, ""]
    for res in report.per_symbol:
        nu = res.nu_report
        nu_lines.append(
            f"  [bold]{nu.symbol:<8}[/bold] "
            f"n={nu.n_observations}  "
            f"5d r={nu.correlation_nu_vs_return.get('5d', 0):.3f}  "
            f"10d r={nu.correlation_nu_vs_return.get('10d', 0):.3f}  "
            f"20d r={nu.correlation_nu_vs_return.get('20d', 0):.3f}"
        )
        # Quartile returns at 10d
        qr = nu.per_quartile_mean_return.get("10d", {})
        if qr:
            nu_lines.append(
                f"           Quartile 10d returns: "
                f"LOW={qr.get('LOW', 0):+.3f}  "
                f"MID={qr.get('MID', 0):+.3f}  "
                f"HIGH={qr.get('HIGH', 0):+.3f}  "
                f"ZPB={qr.get('ZPB', 0):+.3f}"
            )
    console.print(Panel(
        "\n".join(nu_lines),
        title="[bold magenta]Backtest 1 — ν Predictive Validity[/bold magenta]",
        border_style="magenta",
    ))

    # LSII detail panel
    lsii_lines = [report.lsii_aggregate.notes, ""]
    for res in report.per_symbol:
        lsii = res.lsii_report
        col = (
            "[green]" if lsii.flag_predictive_value.get("10d", 0) < 0 else "[red]"
        )
        lsii_lines.append(
            f"  [bold]{lsii.symbol:<8}[/bold] "
            f"flagged={lsii.n_flagged}  "
            f"baseline={lsii.n_baseline}  "
            f"10d flag={col}{lsii.flagged_mean_return.get('10d', 0):+.4f}[/]  "
            f"baseline={lsii.baseline_mean_return.get('10d', 0):+.4f}  "
            f"spread={col}{lsii.flag_predictive_value.get('10d', 0):+.4f}[/]  "
            f"opt_thresh={lsii.optimal_threshold}"
        )
    console.print(Panel(
        "\n".join(lsii_lines),
        title="[bold yellow]Backtest 2 — LSII Arc Break Prediction[/bold yellow]",
        border_style="yellow",
    ))

    # STS detail panel
    sts_lines = [report.sts_aggregate.notes, ""]
    for res in report.per_symbol:
        sts = res.sts_report
        spr_20 = sts.loading_vs_deteriorating.get("20d", 0)
        spr_60 = sts.loading_vs_deteriorating.get("60d", 0)
        col20 = "[green]" if spr_20 > 0 else "[red]"
        col60 = "[green]" if spr_60 > 0 else "[red]"
        sts_lines.append(
            f"  [bold]{sts.symbol:<8}[/bold] "
            f"LOAD={sts.loading_mean_return.get('20d', 0):+.4f}/"
            f"{sts.loading_mean_return.get('60d', 0):+.4f}  "
            f"DETM={sts.deteriorating_mean_return.get('20d', 0):+.4f}/"
            f"{sts.deteriorating_mean_return.get('60d', 0):+.4f}  "
            f"spread(20d)={col20}{spr_20:+.4f}[/]  "
            f"spread(60d)={col60}{spr_60:+.4f}[/]"
        )
    console.print(Panel(
        "\n".join(sts_lines),
        title="[bold blue]Backtest 3 — STS Trajectory Prediction[/bold blue]",
        border_style="blue",
    ))

    # Mystery pile
    if report.mystery_pile_items:
        mp_lines = []
        for i, item in enumerate(report.mystery_pile_items, 1):
            mp_lines.append(f"  {i}. {item}")
        console.print(Panel(
            "\n".join(mp_lines),
            title="[bold red]Mystery Pile — Inconclusive / Refuted[/bold red]",
            border_style="red",
        ))

    # Calibration recommendations
    if report.calibration_recommendations:
        cal_lines = []
        for i, rec in enumerate(report.calibration_recommendations, 1):
            cal_lines.append(f"  {i}. {rec}")
        console.print(Panel(
            "\n".join(cal_lines),
            title="[bold green]Calibration Recommendations[/bold green]",
            border_style="green",
        ))

    console.print(f"\n[dim]ν proxy note: {report.nu_proxy_note}[/dim]")
    console.print(f"[dim]Run date: {report.run_date}[/dim]")


def _print_plain(report: BacktestReport) -> None:
    """Fallback plain-text output if rich is not available."""
    print(f"\n=== DFTE Phase 7 Backtest — {report.run_date} ===\n")
    print(f"Symbols: {', '.join(report.symbols)}")
    print(f"Overall verdict: {report.overall_verdict}\n")
    for res in report.per_symbol:
        print(f"  {res.symbol}: ν={res.nu_report.evidence_verdict} "
              f"LSII={res.lsii_report.evidence_verdict} "
              f"STS={res.sts_report.evidence_verdict}")
    print(f"\nν aggregate:    {report.nu_aggregate.evidence_verdict}")
    print(f"LSII aggregate: {report.lsii_aggregate.evidence_verdict}")
    print(f"STS aggregate:  {report.sts_aggregate.evidence_verdict}")
    print("\nMystery pile:")
    for item in report.mystery_pile_items:
        print(f"  - {item}")
    print("\nCalibration recommendations:")
    for rec in report.calibration_recommendations:
        print(f"  - {rec}")


def _run_live_backtest(symbols: list) -> None:
    """
    --live mode: validate against signals logged to signal_history.db.
    Reads real ν, WFS, STS values logged by the orchestrator and
    computes accuracy vs actual forward returns (via outcomes table).
    Requires outcome_updater.py to have been run first.
    """
    try:
        from logger.signal_logger import SignalLogger
    except ImportError:
        print("ERROR: logger module not found. Run from kindpath-dfte root.")
        return

    sl = SignalLogger()
    stats = sl.get_stats()

    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
    except ImportError:
        console = None

    n_total    = stats["total_signals"]
    n_complete = stats["outcomes_complete"]
    n_ready    = stats["validation_ready_count"]

    header = (
        f"[bold]Live signal database:[/bold] {sl.db_path}\n"
        f"[bold]Total signals logged:[/bold]  {n_total}\n"
        f"[bold]Outcomes complete:[/bold]     {n_complete}\n"
        f"[bold]Validation-ready:[/bold]      {n_ready} (≥60 days old)\n"
        f"[bold]Symbols in DB:[/bold]         {', '.join(stats['symbols']) or 'none'}\n"
        f"[bold]Date range:[/bold]            "
        f"{stats.get('earliest_signal', 'none')[:10] if stats.get('earliest_signal') else 'none'}"
        f" → {stats.get('latest_signal', 'none')[:10] if stats.get('latest_signal') else 'none'}"
    )

    if console:
        console.print(Panel(
            header,
            title="[bold cyan]Live Backtest — Signal Logger Status[/bold cyan]",
            border_style="cyan",
        ))
    else:
        print(header)

    if n_complete < 10:
        msg = (
            f"\nInsufficient outcome data for live backtest validation.\n"
            f"  Outcomes complete: {n_complete} (need ≥10)\n"
            f"  Run `python3 logger/outcome_updater.py` to populate outcomes\n"
            f"  for signals older than 60 days.\n\n"
            f"  KINDFIELD: the real backtesting begins after 6 months of logging.\n"
            f"  Price-proxy approximations cannot validate what live signal data can."
        )
        if console:
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            print(msg)
        return

    # Enough data — compute live accuracy
    try:
        df = sl.to_dataframe(days=730, include_outcomes=True)
    except RuntimeError:
        print("pandas required for live backtest: pip install pandas")
        return

    df_complete = df[df["forward_60d"].notna()].copy()

    if console:
        console.print(
            f"\n[bold]Analysing {len(df_complete)} completed signals "
            f"across {df_complete['symbol'].nunique()} symbols...[/bold]"
        )

    # ν live validation: compare ν quartile to forward_20d return
    import numpy as np
    quartile_bins = {"LOW": (0, 0.35), "MID": (0.35, 0.55),
                     "HIGH": (0.55, 0.75), "ZPB": (0.75, 1.01)}

    lines = []
    for sym in sorted(df_complete["symbol"].unique()):
        if symbols and sym not in symbols:
            continue
        sub = df_complete[df_complete["symbol"] == sym]
        nu_vals = sub["nu"].dropna().values
        ret_20  = sub["forward_20d"].dropna().values
        n_sym   = min(len(nu_vals), len(ret_20))
        if n_sym < 5:
            continue
        r = float(np.corrcoef(nu_vals[:n_sym], ret_20[:n_sym])[0, 1])
        lines.append(
            f"  [bold]{sym:<8}[/bold] n={n_sym}  "
            f"ν↔20d_return r=[{'green' if r > 0.1 else 'red' if r < -0.1 else 'dim'}]"
            f"{r:+.3f}[/]  "
            f"mean_ν={nu_vals.mean():.2f}  "
            f"mean_20d={ret_20.mean():+.4f}"
        )

    if lines and console:
        console.print(Panel(
            "\n".join(lines),
            title="[bold magenta]Live ν Validation (real signal data)[/bold magenta]",
            border_style="magenta",
        ))

    print(
        f"\nLive backtest complete. Run `python3 logger/outcome_updater.py` "
        f"regularly to add outcome data as signals age."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DFTE Phase 7 Backtesting Harness")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick",   action="store_true",
                       help="SPY only — fast price-proxy validation")
    group.add_argument("--full",    action="store_true",
                       help="All 8 symbols, full price-proxy report")
    group.add_argument("--symbols", nargs="+",
                       help="Custom symbol list")
    group.add_argument("--live",    action="store_true",
                       help="Validate against real logged signals (requires 60+ day history)")
    parser.add_argument("--period", default="2y",
                        help="yfinance period string for price-proxy mode (default: 2y)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save JSON report")
    args = parser.parse_args()

    if args.live:
        live_symbols = args.symbols or []
        _run_live_backtest(live_symbols)
        return

    if args.quick:
        symbols = QUICK_SYMBOLS
    elif args.full:
        symbols = ALL_SYMBOLS
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = ["SPY", "QQQ", "ENPH", "NEE", "XOM"]

    logger.info(f"Starting backtest: symbols={symbols}, period={args.period}")
    report = run_backtest(symbols, period=args.period)
    print_report(report)

    if not args.no_save:
        path = save_report(report)
        print(f"\nReport saved: {path}")


if __name__ == "__main__":
    main()
