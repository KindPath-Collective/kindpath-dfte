"""
KPRE Capital Formation Layer
=============================
Pre-price capital formation intent signals from sovereign and Me-layer actors:

1. InsiderTransactionSignal  — SEC Form 4 insider buying/selling
2. CongressionalTradingSignal — House/Senate stock disclosure filings
3. CapexIntentSignal          — Forward capital expenditure (SEC XBRL)

This layer is SYMBOL-SPECIFIC — each signal is pulled per instrument.
Unlike the physical flow layer (instrument-agnostic background), capital
formation signals directly measure the conviction of informed participants
in a specific security.

Evidence posture:
  [ESTABLISHED] — Insider buying with personal capital is among the most
                  reliable pre-price signals in academic literature.
                  Academic basis: Seyhun (1986), Lakonishok & Lee (2001).
                  Capex is a direct measure of forward capital commitment.
  [TESTABLE]    — Congressional trading directional validity well-documented;
                  however, use of committee information advantage is
                  ethically complex. Flag: this signal detects sovereign-layer
                  pre-price positioning. It does not endorse the behaviour
                  it measures. Calibration against forward returns: [TESTABLE].
  [SPECULATIVE] — Cluster buy timing as ZPB precursor signal (KindPath-specific
                  framing) requires out-of-sample validation.

SEC EDGAR note:
  All EDGAR API calls require a User-Agent header with contact info.
  Rate limit: ≤ 10 requests/second. We add 0.15s delay between calls.
  Ref: https://www.sec.gov/developer

Congressional trading note:
  House Stock Watcher: https://housestockwatcher.com/api
  Senate Stock Watcher: https://senatestockwatcher.com/api
  Both are free public APIs with no auth.
"""

from __future__ import annotations
import time
import logging
import requests
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any, Tuple

from kepe.indicators import WorldSignal

logger = logging.getLogger(__name__)

# SEC EDGAR requires contact info in User-Agent
_EDGAR_UA = {"User-Agent": "KindPath Research admin@kindpath.org", "Accept": "application/json"}
_EDGAR_DELAY = 0.15  # seconds between EDGAR requests — SEC asks for ≤10 req/s

# Module-level CIK cache — avoids repeated company_tickers.json lookups
_CIK_CACHE: Dict[str, str] = {}


# ─── SEC EDGAR helpers ────────────────────────────────────────────────────────

def _edgar_get(url: str, **kwargs) -> Optional[requests.Response]:
    """GET with EDGAR User-Agent and rate-limiting delay."""
    time.sleep(_EDGAR_DELAY)
    try:
        return requests.get(url, headers=_EDGAR_UA, timeout=12, **kwargs)
    except Exception as e:
        logger.debug(f"EDGAR GET {url}: {e}")
        return None


def _get_cik(ticker: str) -> Optional[str]:
    """
    Resolve ticker symbol → zero-padded 10-digit CIK.
    Caches result for the session.
    """
    key = ticker.upper()
    if key in _CIK_CACHE:
        return _CIK_CACHE[key]

    resp = _edgar_get("https://www.sec.gov/files/company_tickers.json")
    if resp is None or resp.status_code != 200:
        return None

    data = resp.json()
    for entry in data.values():
        if entry.get("ticker", "").upper() == key:
            cik = str(entry["cik_str"]).zfill(10)
            _CIK_CACHE[key] = cik
            return cik

    logger.debug(f"CIK not found for {ticker}")
    return None


def _fetch_submissions(cik: str) -> Optional[dict]:
    """Fetch company submissions JSON from EDGAR — includes recent filing list."""
    resp = _edgar_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if resp is None or resp.status_code != 200:
        return None
    return resp.json()


def _fetch_form4_xml(cik: str, accession_raw: str, primary_doc: str) -> Optional[str]:
    """
    Fetch Form 4 XML filing text.
    accession_raw: '0001234567-24-000123' format from submissions JSON.
    primary_doc: filename like '0001234567-24-000123.xml'
    """
    acc_nodash = accession_raw.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary_doc}"
    resp = _edgar_get(url)
    if resp is None or resp.status_code != 200:
        return None
    return resp.text


def _parse_form4_xml(xml_text: str) -> List[Dict[str, Any]]:
    """
    Parse Form 4 XML → list of transaction dicts.

    Each dict:
      code       : str   — 'P' (purchase), 'S' (sale), 'A' (award/grant), 'D' (dispose), etc.
      shares     : float — number of shares in transaction
      price      : float — price per share (0.0 if not disclosed)
      total_value: float — shares × price
      post_shares: float — shares owned after transaction
      is_director: bool
      is_officer : bool
      is_10pct   : bool
      officer_title: str
      is_10b5_1  : bool  — footnote references Rule 10b5-1 (pre-planned)
      date       : str   — transaction date YYYY-MM-DD
    """
    transactions = []
    try:
        root = ET.fromstring(xml_text)
        ns = ""  # Form 4 XML often has no namespace

        def find_val(node, path: str, default=None):
            el = node.find(path)
            if el is None:
                return default
            text = el.text
            if text is None:
                # Try child <value>
                v = el.find("value")
                return v.text if v is not None else default
            return text

        # Reporting owner relationship
        owner_rel = root.find(".//reportingOwnerRelationship")
        is_director = find_val(owner_rel, "isDirector", "0") == "1" if owner_rel is not None else False
        is_officer  = find_val(owner_rel, "isOfficer",  "0") == "1" if owner_rel is not None else False
        is_10pct    = find_val(owner_rel, "isTenPercentOwner", "0") == "1" if owner_rel is not None else False
        officer_title = (find_val(owner_rel, "officerTitle", "") or "").strip()

        # Footnotes — check for 10b5-1
        footnotes_text = " ".join(
            (fn.text or "") for fn in root.findall(".//footnote")
        ).upper()
        is_10b5_1 = "10B5-1" in footnotes_text or "10B5" in footnotes_text

        # Non-derivative transactions (common stock purchases/sales)
        for tx in root.findall(".//nonDerivativeTransaction"):
            code  = find_val(tx, ".//transactionCode") or ""
            try:
                shares = float(find_val(tx, ".//transactionShares/value", 0) or 0)
            except (TypeError, ValueError):
                shares = 0.0
            try:
                price = float(find_val(tx, ".//transactionPricePerShare/value", 0) or 0)
            except (TypeError, ValueError):
                price = 0.0
            try:
                post_shares = float(find_val(tx, ".//sharesOwnedFollowingTransaction/value", 0) or 0)
            except (TypeError, ValueError):
                post_shares = 0.0

            tx_date = find_val(tx, ".//transactionDate/value") or find_val(tx, "transactionDate") or ""

            transactions.append({
                "code":          code.strip().upper(),
                "shares":        shares,
                "price":         price,
                "total_value":   shares * price,
                "post_shares":   post_shares,
                "is_director":   is_director,
                "is_officer":    is_officer,
                "is_10pct":      is_10pct,
                "officer_title": officer_title,
                "is_10b5_1":     is_10b5_1,
                "date":          tx_date,
            })
    except Exception as e:
        logger.debug(f"Form 4 XML parse failed: {e}")

    return transactions


# ─── 1. Insider Transaction Signal ───────────────────────────────────────────

class InsiderTransactionSignal:
    """
    SEC Form 4 insider transaction signal.

    Reads Form 4 filings from SEC EDGAR for the past 30 days.
    Scores based on:
      - Transaction direction (P=buy, S=sell)
      - Insider role (CEO/CFO/Director weigh more than generic officer)
      - Cluster detection (multiple insiders buying same window = strong)
      - 10b5-1 plan discounting (pre-planned scheduled sales = neutral)
      - Transaction size relative to post-transaction holdings

    [ESTABLISHED] — insider cluster buying with personal capital is a
                    well-validated pre-price signal (Seyhun 1986, Lakonishok 2001).
    [SPECULATIVE] — Cluster buy as ZPB precursor framing (KindPath-specific).
    Temporal layer: MEDIUM — insider conviction plays out over weeks.
    """

    _SENIOR_TITLES = {"ceo", "chief executive", "cfo", "chief financial", "president",
                      "chairman", "coo", "chief operating"}

    def compute(self, symbol: str) -> WorldSignal:
        """Fetch Form 4s and compute insider signal for symbol."""
        cik = _get_cik(symbol)
        if cik is None:
            return self._no_data(symbol, "CIK not found")

        transactions = self._fetch_transactions(cik, days=30)
        if not transactions:
            return self._no_data(symbol, "No Form 4 filings in last 30 days")

        value, raw = self._score_transactions(transactions)

        return WorldSignal(
            domain="KPRE_CAPITAL",
            source=f"sec_form4_{symbol.lower()}",
            region="US",
            value=float(np.clip(value, -1, 1)),
            confidence=0.65,
            evidence_level="ESTABLISHED",
            timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
            raw=raw,
            notes=(
                f"Form 4 insider signal [{symbol}]: "
                f"{raw.get('n_purchases', 0)} buys, {raw.get('n_sales', 0)} sales "
                f"({raw.get('n_filings', 0)} filings, 30d) [ESTABLISHED]"
            )
        )

    def _fetch_transactions(self, cik: str, days: int = 30) -> List[Dict[str, Any]]:
        """Fetch and parse Form 4 filings filed in last `days` days."""
        subs = _fetch_submissions(cik)
        if subs is None:
            return []

        filings = subs.get("filings", {}).get("recent", {})
        forms    = filings.get("form", [])
        dates    = filings.get("filingDate", [])
        accnos   = filings.get("accessionNumber", [])
        pdocs    = filings.get("primaryDocument", [])

        cutoff = (date.today() - timedelta(days=days)).isoformat()
        transactions: List[Dict[str, Any]] = []
        filings_fetched = 0

        for form, filing_date, accno, pdoc in zip(forms, dates, accnos, pdocs):
            if form not in ("4", "4/A"):
                continue
            if filing_date < cutoff:
                break  # filings are reverse-chronological, stop early

            xml_text = _fetch_form4_xml(cik, accno, pdoc)
            if xml_text is None:
                continue

            txs = _parse_form4_xml(xml_text)
            transactions.extend(txs)
            filings_fetched += 1

            if filings_fetched >= 15:  # cap to avoid excessive API calls
                break

        return transactions

    @staticmethod
    def _score_transactions(transactions: List[Dict[str, Any]]) -> Tuple[float, dict]:
        """
        Score a list of parsed Form 4 transactions.
        Returns (signal_value, raw_metadata).
        Testable independently of network.

        Scoring logic:
          Each purchase (P) earns positive weight; each sale (S) earns negative weight.
          Senior role (CEO/CFO/President/Chairman) = 2× weight.
          Director = 1.5× weight.
          10b5-1 pre-planned = 0.3× weight (discount: routine, not conviction).
          Large buy (>0.5% post-holding) = 1.5× bonus.
          Cluster: 3+ distinct purchase transactions = cluster bonus (1.3×).
        """
        if not transactions:
            return 0.0, {}

        senior_titles = InsiderTransactionSignal._SENIOR_TITLES

        buy_score    = 0.0
        sell_score   = 0.0
        n_purchases  = 0
        n_sales      = 0
        cluster_sigs = []

        for tx in transactions:
            code     = tx.get("code", "")
            shares   = tx.get("shares", 0.0)
            value    = tx.get("total_value", 0.0)
            post     = tx.get("post_shares", 1.0) or 1.0
            is_10b5  = tx.get("is_10b5_1", False)
            title    = (tx.get("officer_title") or "").lower()
            is_dir   = tx.get("is_director", False)
            is_off   = tx.get("is_officer", False)
            is_10pct = tx.get("is_10pct", False)

            # Skip awards/grants — not discretionary personal purchases
            if code in ("A", "M", "C", "X"):
                continue

            # Role multiplier
            role_mult = 1.0
            if any(s in title for s in senior_titles) or is_10pct:
                role_mult = 2.0
            elif is_dir:
                role_mult = 1.5
            elif is_off:
                role_mult = 1.2

            # 10b5-1 discount
            plan_mult = 0.3 if is_10b5 else 1.0

            # Size signal: transaction as % of post-holding
            size_mult = 1.0
            if post > 0 and shares > 0:
                pct_holding = shares / post
                if pct_holding > 0.005:   # > 0.5% of holdings
                    size_mult = 1.5
                elif pct_holding > 0.001:
                    size_mult = 1.2

            # Base weight — normalise by $100k notional as unit
            notional_units = min(value / 100_000.0, 10.0) if value > 0 else 1.0
            weight = max(notional_units, 0.5) * role_mult * plan_mult * size_mult

            if code == "P":
                buy_score  += weight
                n_purchases += 1
                cluster_sigs.append(1)
            elif code == "S":
                sell_score  += weight
                n_sales += 1
                cluster_sigs.append(-1)
            elif code == "D":
                sell_score  += weight * 0.5   # dispose back to company — lighter negative
                n_sales += 1

        total = buy_score + sell_score
        if total < 1e-10:
            return 0.0, {
                "n_filings": len(transactions),
                "n_purchases": n_purchases, "n_sales": n_sales,
                "buy_score": 0.0, "sell_score": 0.0,
            }

        # Cluster bonus: 3+ purchases = 1.3× on buy score
        if n_purchases >= 3:
            buy_score *= 1.3

        # Net signal: range [-1, +1]
        raw_signal = (buy_score - sell_score) / (buy_score + sell_score + 1e-10)

        raw = {
            "n_filings":   len(transactions),
            "n_purchases": n_purchases,
            "n_sales":     n_sales,
            "buy_score":   round(buy_score, 2),
            "sell_score":  round(sell_score, 2),
            "cluster":     n_purchases >= 3,
        }

        return float(raw_signal), raw

    def _no_data(self, symbol: str, reason: str) -> WorldSignal:
        return WorldSignal(
            domain="KPRE_CAPITAL", source=f"sec_form4_{symbol.lower()}",
            region="US", value=0.0, confidence=0.0,
            evidence_level="ESTABLISHED", timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
            notes=f"InsiderTransaction [{symbol}]: no data — {reason}"
        )


# ─── 2. Congressional Trading Signal ─────────────────────────────────────────

class CongressionalTradingSignal:
    """
    Congressional stock disclosure signal.

    Sources:
      House: https://housestockwatcher.com/api (all transactions JSON)
      Senate: https://senatestockwatcher.com/api (all transactions JSON)

    Parses disclosures for the target symbol over last 30 days.
    Congressional purchases with personal capital = sovereign-layer
    pre-price positioning signal. Most powerful when trade is against
    obvious committee interest (i.e. the senator is not on Energy committee
    and buys clean energy — no regulatory advantage is the motivator).

    Evidence: [TESTABLE] — directional validity documented in the academic
    literature (Ziobrowski et al. 2004, 2011). Exploitation of information
    advantage is the ethical concern. This signal measures the behaviour;
    it does not endorse it.

    Temporal layer: MEDIUM — congressional filings lag the trade by up to 45 days.
    """

    _HOUSE_API  = "https://housestockwatcher.com/api"
    _SENATE_API = "https://senatestockwatcher.com/api"

    # Amount band midpoints (USD)
    _AMOUNT_MAP = {
        "$1,001 - $15,000":       8_000,
        "$15,001 - $50,000":     32_500,
        "$50,001 - $100,000":    75_000,
        "$100,001 - $250,000":  175_000,
        "$250,001 - $500,000":  375_000,
        "$500,001 - $1,000,000": 750_000,
        "Over $1,000,000":     1_500_000,
    }

    def compute(self, symbol: str) -> WorldSignal:
        trades = self._fetch_recent_trades(symbol, days=30)
        if not trades:
            return WorldSignal(
                domain="KPRE_CAPITAL", source=f"congress_{symbol.lower()}",
                region="US", value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                notes=f"Congressional trading [{symbol}]: no disclosures in last 30 days [TESTABLE]"
            )

        value, raw = self._score_trades(trades, symbol)
        return WorldSignal(
            domain="KPRE_CAPITAL", source=f"congress_{symbol.lower()}",
            region="US",
            value=float(np.clip(value, -1, 1)),
            confidence=0.45,   # moderate — lag in disclosure reduces timeliness
            evidence_level="TESTABLE",
            timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
            raw=raw,
            notes=(
                f"Congressional trading [{symbol}]: "
                f"{raw.get('n_buys', 0)} buys / {raw.get('n_sells', 0)} sells "
                f"(30d) — detects sovereign-layer positioning, does not endorse it [TESTABLE]"
            )
        )

    def _fetch_recent_trades(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """Fetch all congress trades and filter to symbol + date window."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        trades: List[Dict[str, Any]] = []

        for api_url, chamber in [
            (self._HOUSE_API,  "house"),
            (self._SENATE_API, "senate"),
        ]:
            try:
                resp = requests.get(api_url, headers=_EDGAR_UA, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                # Handle both list and dict response shapes
                if isinstance(data, dict):
                    items = data.get("data", data.get("transactions", []))
                else:
                    items = data

                for item in items:
                    ticker = (item.get("ticker") or item.get("asset_description") or "").upper()
                    # Accept if ticker matches or if asset_description contains symbol
                    if symbol.upper() not in ticker and symbol.upper() not in (item.get("asset_description") or "").upper():
                        continue
                    tx_date = (
                        item.get("transaction_date") or
                        item.get("disclosure_date") or ""
                    )
                    if tx_date < cutoff:
                        continue
                    item["_chamber"] = chamber
                    trades.append(item)

            except Exception as e:
                logger.debug(f"Congressional {api_url} failed: {e}")

        return trades

    @staticmethod
    def _score_trades(trades: List[Dict[str, Any]], symbol: str) -> Tuple[float, dict]:
        """
        Score congressional trades.
        Testable independently of network.

        Scoring:
          Purchase = positive, Sale/Exchange = negative.
          Notional size weight from amount band.
          Senate trade = 1.2× (longer term, fewer trades = stronger signal).
          No role/committee data available from free APIs → uniform weight.
        """
        if not trades:
            return 0.0, {}

        buy_score  = 0.0
        sell_score = 0.0
        n_buys     = 0
        n_sells    = 0

        for trade in trades:
            tx_type = (
                trade.get("type") or trade.get("transaction") or ""
            ).lower()
            amount_str = trade.get("amount", "$1,001 - $15,000")
            notional = CongressionalTradingSignal._AMOUNT_MAP.get(
                amount_str, 8_000
            )
            weight = (notional / 100_000.0)   # normalise to $100k unit
            weight = max(weight, 0.1)
            # Senate trades carry slightly more weight (rarer, longer horizon)
            if trade.get("_chamber") == "senate":
                weight *= 1.2

            if "purchase" in tx_type or "buy" in tx_type:
                buy_score += weight
                n_buys    += 1
            elif "sale" in tx_type or "sell" in tx_type or "exchange" in tx_type:
                sell_score += weight
                n_sells    += 1

        total = buy_score + sell_score
        if total < 1e-10:
            return 0.0, {"n_buys": n_buys, "n_sells": n_sells}

        signal = (buy_score - sell_score) / total
        raw = {
            "n_buys": n_buys, "n_sells": n_sells,
            "buy_score": round(buy_score, 2),
            "sell_score": round(sell_score, 2),
            "chambers":  list({t.get("_chamber", "?") for t in trades}),
        }
        return float(signal), raw


# ─── 3. CapEx Intent Signal ───────────────────────────────────────────────────

class CapexIntentSignal:
    """
    Capital expenditure trend from SEC XBRL company facts.

    Extracts quarterly capex (PaymentsToAcquirePropertyPlantAndEquipment)
    from the SEC XBRL API and computes a trend signal.

    Rising capex = forward capital commitment = management conviction in growth.
    For syntropic assets: rising capex in clean energy / sustainability = strong
    positive signal (real-world build-out of generative capacity).
    For neutral assets: capex direction as business health proxy.

    [ESTABLISHED] — capex is a direct measure of forward capital commitment,
                    not a proxy. Source: SEC 10-K/10-Q mandatory reporting.
    Temporal layer: STRUCTURAL — capex decisions reflect 1-3 year planning horizons.
    """

    _XBRL_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    # Primary + fallback XBRL concept names for capital expenditure
    _CAPEX_CONCEPTS = [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "CapitalExpendituresIncurredButNotYetPaid",
        "PaymentsForCapitalImprovements",
    ]

    def compute(self, symbol: str) -> WorldSignal:
        """Fetch XBRL capex data and compute trend signal."""
        cik = _get_cik(symbol)
        if cik is None:
            return self._no_data(symbol, "CIK not found")

        quarters, concept_used = self._fetch_capex_quarters(cik)
        if not quarters or len(quarters) < 2:
            return self._no_data(symbol, "Insufficient capex data")

        value, raw = self._compute_capex_trend(quarters)
        raw["concept"] = concept_used

        return WorldSignal(
            domain="KPRE_CAPITAL",
            source=f"sec_capex_{symbol.lower()}",
            region="US",
            value=float(np.clip(value, -1, 1)),
            confidence=0.75,
            evidence_level="ESTABLISHED",
            timestamp=datetime.utcnow(),
            temporal_layer="STRUCTURAL",
            raw=raw,
            notes=(
                f"CapEx trend [{symbol}]: {len(quarters)}q of data "
                f"via {concept_used} (XBRL) [ESTABLISHED]"
            )
        )

    def _fetch_capex_quarters(self, cik: str) -> Tuple[List[float], str]:
        """
        Fetch discrete quarterly capex values, return (list of values, concept_name_used).

        XBRL 10-Q capex entries are often YTD-cumulative within the fiscal year.
        We prefer entries where the period is ≈ 90 days (discrete quarterly),
        falling back to 10-K annual values if quarterly are unavailable.
        """
        from datetime import date as _date

        resp = _edgar_get(self._XBRL_URL.format(cik=cik))
        if resp is None or resp.status_code != 200:
            return [], ""

        facts = resp.json()
        usgaap = facts.get("facts", {}).get("us-gaap", {})

        for concept in self._CAPEX_CONCEPTS:
            concept_data = usgaap.get(concept)
            if not concept_data:
                continue
            usd_data = concept_data.get("units", {}).get("USD", [])
            if not usd_data:
                continue

            # Prefer annual 10-K entries — always discrete full-year values.
            # 10-Q capex is typically YTD-cumulative from fiscal year start
            # (Q2=6mo, Q3=9mo), making direct comparison unreliable without
            # subtraction logic. Annual data is clean for trend computation.
            quarterly = [
                e for e in usd_data
                if e.get("form") == "10-K" and e.get("val") is not None
                and e.get("fp") == "FY"
            ]

            # Supplement with Q1 10-Q (fp='Q1') if more recent than latest 10-K.
            # Q1 is always a discrete 3-month period, so annualize (×4) before
            # adding to annual series to keep the scale comparable.
            if quarterly:
                latest_annual_end = max(e.get("end", "") for e in quarterly)
                q1_new = [
                    e for e in usd_data
                    if e.get("form") == "10-Q"
                    and e.get("fp") == "Q1"
                    and e.get("val") is not None
                    and e.get("end", "") > latest_annual_end
                ]
                # Deduplicate Q1 entries by end date
                seen_q1: Dict[str, dict] = {}
                for e in q1_new:
                    end = e.get("end", "")
                    if end not in seen_q1 or e.get("filed", "") > seen_q1[end].get("filed", ""):
                        seen_q1[end] = e
                # Annualise Q1 values (×4) to make comparable to annual 10-K
                annualised_q1 = [
                    {**e, "val": e["val"] * 4, "_annualised": True}
                    for e in seen_q1.values()
                ]
                quarterly = quarterly + annualised_q1

            # Deduplicate by end date — each fiscal year appears in multiple
            # 10-K filings as comparative periods. Keep latest-filed per end date.
            seen: Dict[str, dict] = {}
            for e in quarterly:
                end = e.get("end", "")
                if end not in seen or e.get("filed", "") > seen[end].get("filed", ""):
                    seen[end] = e
            quarterly = sorted(seen.values(), key=lambda e: e.get("end", ""))

            recent  = quarterly[-8:]
            values  = [float(e["val"]) for e in recent]

            if len(values) >= 2:
                return values, concept

        return [], ""

    @staticmethod
    def _compute_capex_trend(quarters: List[float]) -> Tuple[float, dict]:
        """
        Score capex trend from quarterly values.
        Testable independently of network.

        Approach:
          - Short-term (last 2 quarters): recent acceleration
          - Medium-term (all quarters): endpoint delta as % of baseline
          - Consistent growth > 5% = strong positive
          - Consistent decline > 5% = negative
          - Flat = weak positive (stable commitment)
        """
        if len(quarters) < 2:
            return 0.0, {}

        baseline  = quarters[0] if quarters[0] > 0 else 1.0
        latest    = quarters[-1]
        mid       = quarters[len(quarters) // 2] if len(quarters) > 2 else quarters[0]

        # Full-period delta as % of baseline
        delta_full = (latest - baseline) / (abs(baseline) + 1e-10)

        # Recent acceleration: last 2 quarters vs prior 2
        if len(quarters) >= 4:
            recent_avg = (quarters[-1] + quarters[-2]) / 2
            prior_avg  = (quarters[-3] + quarters[-4]) / 2
            accel      = (recent_avg - prior_avg) / (abs(prior_avg) + 1e-10)
        else:
            accel = (quarters[-1] - quarters[-2]) / (abs(quarters[-2]) + 1e-10)

        # Combine: full-period trend (60%) + recent acceleration (40%)
        # ±20% capex growth → ±0.5 signal contribution
        raw_signal = delta_full * 2.5 * 0.60 + accel * 2.5 * 0.40
        value = float(np.clip(raw_signal, -1, 1))

        raw = {
            "n_quarters":    len(quarters),
            "baseline_capex": round(baseline, 0),
            "latest_capex":  round(latest, 0),
            "delta_full_pct": round(delta_full * 100, 1),
            "recent_accel_pct": round(accel * 100, 1),
        }

        return value, raw

    def _no_data(self, symbol: str, reason: str) -> WorldSignal:
        return WorldSignal(
            domain="KPRE_CAPITAL", source=f"sec_capex_{symbol.lower()}",
            region="US", value=0.0, confidence=0.0,
            evidence_level="ESTABLISHED", timestamp=datetime.utcnow(),
            temporal_layer="STRUCTURAL",
            notes=f"CapEx [{symbol}]: no data — {reason}"
        )


# ─── KPRE Capital Formation Layer ─────────────────────────────────────────────

class KPRECapitalLayer:
    """
    KPRE Capital Formation Layer — symbol-specific pre-price capital intent.

    Aggregates InsiderTransactionSignal, CongressionalTradingSignal,
    and CapexIntentSignal into a single WorldSignal per instrument.

    Unlike KPRELayer (physical flow — background, instrument-agnostic),
    this layer is called with a symbol argument and returns a signal
    specific to that instrument.

    Domain weight in KEPE: "KPRE_CAPITAL" → 0.20
    (see syntropy_engine.DOMAIN_WEIGHTS)

    Usage:
        sig = KPRECapitalLayer().compute("AAPL")     # full pipeline with real APIs
        sig = KPRECapitalLayer._aggregate([...])      # testable aggregation
    """

    def compute(self, symbol: str) -> WorldSignal:
        """Run all capital formation sub-signals for symbol."""
        sub_signals: List[WorldSignal] = []

        for cls in (InsiderTransactionSignal, CongressionalTradingSignal, CapexIntentSignal):
            try:
                sig = cls().compute(symbol)
                if sig.confidence > 0:
                    sub_signals.append(sig)
            except Exception as e:
                logger.warning(f"KPRECapital sub-signal {cls.__name__} failed for {symbol}: {e}")

        result = self._aggregate(sub_signals)
        logger.debug(
            f"KPRECapital [{symbol}]: {result.value:+.3f} "
            f"(conf={result.confidence:.2f}, {len(sub_signals)}/3 sub-signals)"
        )
        return result

    @staticmethod
    def _aggregate(sub_signals: List[WorldSignal]) -> WorldSignal:
        """
        Confidence-weighted aggregation of KPRE_CAPITAL sub-signals.
        Testable independently of network.

        Confidence ceiling: 0.65 (capital formation signals have higher
        individual evidence quality than physical flow, but are less
        comprehensive as a composite).
        """
        if not sub_signals:
            return WorldSignal(
                domain="KPRE_CAPITAL", source="capital_formation_composite",
                region="US", value=0.0, confidence=0.0,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                temporal_layer="MEDIUM",
                notes="No KPRE_CAPITAL sub-signals available"
            )

        total_w = sum(s.confidence for s in sub_signals)
        if total_w < 1e-10:
            composite_value = 0.0
            composite_conf  = 0.0
        else:
            composite_value = sum(s.value * s.confidence for s in sub_signals) / total_w
            completeness    = len(sub_signals) / 3.0
            avg_conf        = total_w / len(sub_signals)
            composite_conf  = min(0.65, avg_conf * completeness)

        # Evidence level: worst of sub-signals (SPECULATIVE propagates)
        ev_order = {"ESTABLISHED": 0, "TESTABLE": 1, "SPECULATIVE": 2}
        worst_ev = max(sub_signals, key=lambda s: ev_order.get(s.evidence_level, 1))
        ev_level = worst_ev.evidence_level

        sources = [s.source for s in sub_signals]

        return WorldSignal(
            domain="KPRE_CAPITAL", source="capital_formation_composite",
            region="US",
            value=float(np.clip(composite_value, -1, 1)),
            confidence=float(np.clip(composite_conf, 0, 0.65)),
            evidence_level=ev_level,
            timestamp=datetime.utcnow(),
            temporal_layer="MEDIUM",
            raw={
                "n_signals":    len(sub_signals),
                "n_attempted":  3,
                "completeness": round(len(sub_signals) / 3.0, 2),
                "sources":      sources,
                "sub_values":   {s.source: round(s.value, 3) for s in sub_signals},
            },
            notes=(
                f"KPRE_CAPITAL composite: {len(sub_signals)}/3 signals "
                f"(value={composite_value:.3f}, conf={composite_conf:.2f}) "
                f"[{ev_level}] — capital formation intent"
            )
        )
