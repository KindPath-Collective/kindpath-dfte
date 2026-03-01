#!/usr/bin/env python3
"""
Outcome Updater — Phase 7b
===========================
Fetches actual forward returns for all signals older than 60 days
and populates the outcomes table.

Run weekly (cron) or manually:
  python3 outcome_updater.py
  python3 outcome_updater.py --dry-run   (shows what would be updated)
  python3 outcome_updater.py --min-age 20  (lower minimum age for testing)

The mystery pile items have a path to resolution:
  ν validation:   6 months + outcome_updater.py
  STS validation: 6 months
  LSII recalib:   instrument-specific thresholds from live data

KINDFIELD: measure before claiming. This file is how we measure.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

import yfinance as yf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from logger.signal_logger import SignalLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("outcome_updater")

FORWARD_WINDOWS = [5, 10, 20, 60]   # trading days
_RATE_LIMIT_SLEEP = 0.5              # seconds between yfinance requests


def _fetch_price_on_date(
    symbol: str,
    signal_ts: str,
    windows: List[int],
) -> Dict[str, Optional[float]]:
    """
    Fetch actual forward returns for a signal recorded at signal_ts.
    Returns {forward_5d: float|None, forward_10d: float|None, ...}

    Uses yfinance: downloads ~3 months of daily data around the signal date
    and computes actual close-to-close returns.
    """
    results: Dict[str, Optional[float]] = {f"forward_{w}d": None for w in windows}

    try:
        sig_dt = datetime.fromisoformat(signal_ts)
        if sig_dt.tzinfo is None:
            sig_dt = sig_dt.replace(tzinfo=timezone.utc)

        # Fetch 3× the longest window to ensure we have enough trading days
        max_window   = max(windows)
        fetch_days   = max_window * 3
        start_date   = sig_dt.date()
        end_date     = (sig_dt + timedelta(days=fetch_days)).date()

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval="1d",
            auto_adjust=True,
        )

        if df.empty:
            logger.warning(f"No price data for {symbol} from {start_date}")
            return results

        closes = df["Close"].values.astype(float)
        if len(closes) < 2:
            return results

        # Signal price: first trading day on or after signal timestamp
        entry_price = float(closes[0])
        if entry_price <= 0:
            return results

        for w in windows:
            if w < len(closes):
                exit_price = float(closes[w])
                results[f"forward_{w}d"] = (exit_price - entry_price) / entry_price
            # else: not enough future data yet — stays None

    except Exception as e:
        logger.warning(f"Failed to fetch outcome for {symbol} @ {signal_ts}: {e}")

    return results


def run_outcome_update(
    sl: SignalLogger,
    min_age_days: int = 60,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Find signals ready for outcome linkage and populate the outcomes table.

    Returns (n_processed, n_updated, n_failed).
    """
    pending = sl.get_validation_ready_signals(min_age_days=min_age_days)
    if limit:
        pending = pending[:limit]

    if not pending:
        logger.info(f"No signals older than {min_age_days} days need outcome data.")
        return 0, 0, 0

    logger.info(
        f"Found {len(pending)} signal(s) needing outcome data "
        f"(age ≥ {min_age_days} days). dry_run={dry_run}"
    )

    n_processed = 0
    n_updated   = 0
    n_failed    = 0

    # Group by symbol to minimise yfinance calls
    by_symbol: Dict[str, List[Dict]] = {}
    for row in pending:
        sym = row["symbol"]
        by_symbol.setdefault(sym, []).append(row)

    for symbol, rows in by_symbol.items():
        logger.info(f"Processing {len(rows)} signal(s) for {symbol}...")

        for row in rows:
            n_processed += 1
            signal_id = row["id"]
            signal_ts = row["timestamp"]

            outcome = _fetch_price_on_date(symbol, signal_ts, FORWARD_WINDOWS)

            # Only update if we have at least the 5d return
            if outcome.get("forward_5d") is None:
                logger.warning(
                    f"  {symbol} id={signal_id} @ {signal_ts[:10]}: "
                    f"no 5d price data available yet"
                )
                n_failed += 1
                continue

            if dry_run:
                logger.info(
                    f"  [DRY-RUN] {symbol} id={signal_id}: "
                    f"5d={outcome.get('forward_5d', 'N/A'):.4f}  "
                    f"10d={outcome.get('forward_10d') or 'N/A'}  "
                    f"20d={outcome.get('forward_20d') or 'N/A'}  "
                    f"60d={outcome.get('forward_60d') or 'N/A'}"
                )
            else:
                sl.link_outcome(
                    signal_id=signal_id,
                    forward_5d=outcome.get("forward_5d"),
                    forward_10d=outcome.get("forward_10d"),
                    forward_20d=outcome.get("forward_20d"),
                    forward_60d=outcome.get("forward_60d"),
                )
                logger.info(
                    f"  ✓ {symbol} id={signal_id} @ {signal_ts[:10]}: "
                    f"5d={outcome['forward_5d']:+.4f}  "
                    f"10d={outcome.get('forward_10d') or '--':>8}  "
                    f"20d={outcome.get('forward_20d') or '--':>8}  "
                    f"60d={outcome.get('forward_60d') or '--':>8}"
                )
                n_updated += 1

            time.sleep(_RATE_LIMIT_SLEEP)  # yfinance rate limit

    return n_processed, n_updated, n_failed


def print_db_stats(sl: SignalLogger) -> None:
    stats = sl.get_stats()
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        lines = [
            f"[bold]Total signals logged:[/bold]  {stats['total_signals']}",
            f"[bold]Outcomes complete:[/bold]     {stats['outcomes_complete']}",
            f"[bold]Outcomes pending:[/bold]      {stats['outcomes_pending']}",
            f"[bold]Validation-ready:[/bold]      {stats['validation_ready_count']} "
            f"(≥60 days old, no outcome yet)",
            f"[bold]Earliest signal:[/bold]       {stats.get('earliest_signal') or 'none'}",
            f"[bold]Latest signal:[/bold]         {stats.get('latest_signal') or 'none'}",
            f"[bold]Symbols:[/bold]               {', '.join(stats['symbols']) or 'none'}",
        ]
        console.print(Panel(
            "\n".join(lines),
            title="[bold cyan]Signal Logger — Database Status[/bold cyan]",
            border_style="cyan",
        ))
    except ImportError:
        for k, v in stats.items():
            print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DFTE Outcome Updater — populate forward returns for logged signals"
    )
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would be updated without writing")
    parser.add_argument("--min-age",  type=int, default=60,
                        help="Minimum signal age in days (default: 60)")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Process at most N signals (for testing)")
    parser.add_argument("--stats",    action="store_true",
                        help="Show database stats only, no updating")
    parser.add_argument("--db",       default=None,
                        help="Path to signal_history.db (default: logger/signal_history.db)")
    args = parser.parse_args()

    sl = SignalLogger(db_path=args.db) if args.db else SignalLogger()
    print_db_stats(sl)

    if args.stats:
        return

    n_proc, n_upd, n_fail = run_outcome_update(
        sl,
        min_age_days=args.min_age,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    print()
    logger.info(
        f"Done. processed={n_proc}  updated={n_upd}  failed/pending={n_fail}"
    )
    if n_proc > 0 and not args.dry_run:
        print_db_stats(sl)


if __name__ == "__main__":
    main()
