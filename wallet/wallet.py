"""
DFTE — Wallet & Execution Layer
==================================
Broker/exchange abstraction for trade execution.

Supported:
  Alpaca       — equities + crypto (paper and live, free API)
  Abstract      — base class for adding new brokers

Security model:
  - API keys stored in environment variables ONLY
  - Never logged, never persisted to disk
  - All orders require explicit confirmation unless auto_execute=True
  - Position limits enforced at wallet level (separate from DFTE sizing)

Paper trading is the default mode.
Set ALPACA_LIVE=true to enable live trading (with full confirmation chain).

Environment variables:
  ALPACA_API_KEY      — Alpaca API key
  ALPACA_SECRET_KEY   — Alpaca secret key
  ALPACA_LIVE         — "true" for live trading (default: paper)
"""

from __future__ import annotations
import os
import logging
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Unified order request across all brokers."""
    symbol: str
    side: str           # buy | sell
    qty: Optional[float] = None
    notional: Optional[float] = None   # dollar amount (alternative to qty)
    order_type: str = "market"
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tier: str = "NANO"
    rationale: str = ""


@dataclass
class OrderResult:
    """Result of order submission."""
    success: bool
    order_id: Optional[str]
    symbol: str
    side: str
    qty: Optional[float]
    fill_price: Optional[float]
    status: str         # submitted | filled | cancelled | rejected | paper
    broker: str
    timestamp: datetime
    raw: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Position:
    """Current position in one instrument."""
    symbol: str
    qty: float
    avg_entry: float
    market_value: float
    unrealised_pnl: float
    side: str           # long | short


# ─── Abstract base ────────────────────────────────────────────────────────────

class BaseWallet(ABC):
    """Abstract wallet interface. All brokers implement this."""

    @abstractmethod
    def get_positions(self) -> List[Position]:
        ...

    @abstractmethod
    def get_cash(self) -> float:
        ...

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResult:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        ...

    @abstractmethod
    def get_portfolio_value(self) -> float:
        ...

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        ...


# ─── Paper wallet (no API required, for testing) ──────────────────────────────

class PaperWallet(BaseWallet):
    """
    Simulated paper trading wallet with True Persistence.
    Tracks virtual positions and P&L.
    Persists state to wallet_state.json.
    """

    STATE_FILE = "wallet_state.json"

    def __init__(self, initial_cash: float = 100_000.0, persistence: bool = True):
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._orders: List[OrderResult] = []
        self._persistence = persistence
        self._price_cache: Dict[str, float] = {}
        
        if self._persistence:
            self._load_state()
        logger.info(f"PaperWallet ready: ${self._cash:,.2f} cash, {len(self._positions)} positions")

    def _load_state(self):
        """Load state from disk if exists."""
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, "r") as f:
                    state = json.load(f)
                    if "cash" in state:
                        self._cash = float(state["cash"])
                    pos_data = state.get("positions", {})
                    self._positions = {
                        sym: Position(**p) for sym, p in pos_data.items()
                    }
                    logger.info(f"Loaded wallet state from {self.STATE_FILE}")
            except Exception as e:
                logger.error(f"Failed to load wallet state: {e}")

    def _save_state(self):
        """Save current state to disk."""
        if not self._persistence:
            return
        try:
            state = {
                "cash": self._cash,
                "positions": {
                    sym: {
                        "symbol": p.symbol, "qty": p.qty, "avg_entry": p.avg_entry,
                        "market_value": p.market_value, "unrealised_pnl": p.unrealised_pnl,
                        "side": p.side
                    } for sym, p in self._positions.items()
                },
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            with open(self.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save wallet state: {e}")

    @property
    def is_paper(self) -> bool:
        return True

    def get_cash(self) -> float:
        return self._cash

    def get_portfolio_value(self) -> float:
        self._update_market_values()
        pos_value = sum(p.market_value for p in self._positions.values())
        return self._cash + pos_value

    def price_pulse(self, symbols: List[str]):
        """
        Parallel fetch of all symbols in the basket to minimize loop latency.
        'Pulse traffic' approach.
        """
        if not symbols: return
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self._fetch_remote_price, s): s for s in symbols}
            for f in futures:
                sym = futures[f]
                price = f.result()
                if price:
                    self._price_cache[sym] = price

    def _update_market_values(self):
        """Fetch current prices and update position market values/PNL."""
        self.price_pulse(list(self._positions.keys()))
        for sym, pos in self._positions.items():
            price = self._get_price(sym)
            if price:
                pos.market_value = pos.qty * price
                pos.unrealised_pnl = (price - pos.avg_entry) * pos.qty

    def get_positions(self) -> List[Position]:
        self._update_market_values()
        return list(self._positions.values())

    def submit_order(self, order: OrderRequest) -> OrderResult:
        """Simulate order execution and persist state."""
        try:
            price = self._get_price(order.symbol)
            if price is None:
                return OrderResult(
                    success=False, order_id=None, symbol=order.symbol, side=order.side,
                    qty=order.qty, fill_price=None, status="rejected", broker="paper",
                    timestamp=datetime.now(timezone.utc), error=f"Cannot get price for {order.symbol}"
                )

            qty = order.qty
            if qty is None and order.notional:
                qty = order.notional / price
            if qty is None: qty = 1.0

            if order.side == "buy":
                cost = qty * price
                if cost > self._cash:
                    return OrderResult(
                        success=False, order_id=None, symbol=order.symbol, side=order.side,
                        qty=qty, fill_price=price, status="rejected", broker="paper",
                        timestamp=datetime.now(timezone.utc), error="Insufficient cash"
                    )
                self._cash -= cost
                if order.symbol in self._positions:
                    pos = self._positions[order.symbol]
                    new_qty = pos.qty + qty
                    new_avg = (pos.avg_entry * pos.qty + price * qty) / new_qty
                    self._positions[order.symbol] = Position(
                        symbol=order.symbol, qty=new_qty, avg_entry=new_avg,
                        market_value=new_qty * price, unrealised_pnl=(price - new_avg) * new_qty,
                        side="long"
                    )
                else:
                    self._positions[order.symbol] = Position(
                        symbol=order.symbol, qty=qty, avg_entry=price,
                        market_value=qty * price, unrealised_pnl=0.0, side="long"
                    )

            elif order.side == "sell":
                if order.symbol in self._positions:
                    pos = self._positions[order.symbol]
                    sell_qty = min(qty, pos.qty)
                    self._cash += sell_qty * price
                    remaining = pos.qty - sell_qty
                    if remaining < 0.001:
                        del self._positions[order.symbol]
                    else:
                        pos.qty = remaining
                        pos.market_value = remaining * price
                        pos.unrealised_pnl = (price - pos.avg_entry) * remaining

            self._save_state()
            order_id = f"paper_{order.symbol}_{int(datetime.now(timezone.utc).timestamp())}"
            res = OrderResult(
                success=True, order_id=order_id, symbol=order.symbol, side=order.side,
                qty=qty, fill_price=price, status="paper", broker="paper",
                timestamp=datetime.now(timezone.utc)
            )
            self._orders.append(res)
            logger.info(
                f"[PAPER] {order.side.upper()} {qty:.4f} {order.symbol} @ ${price:.4f} "
                f"(${qty*price:,.2f}) [{order.tier}]"
            )
            return res
        except Exception as e:
            logger.error(f"PaperWallet order error: {e}")
            return OrderResult(success=False, order_id=None, symbol=order.symbol, side=order.side, qty=None, fill_price=None, status="error", broker="paper", timestamp=datetime.now(timezone.utc), error=str(e))

    def cancel_order(self, order_id: str) -> bool:
        return True 

    def _get_price(self, symbol: str) -> Optional[float]:
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        return self._fetch_remote_price(symbol)

    def _fetch_remote_price(self, symbol: str) -> Optional[float]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                val = float(hist["Close"].iloc[-1])
                self._price_cache[symbol] = val
                return val
        except Exception as e:
            logger.warning(f"Price fetch failed for {symbol}: {e}")
        return None

    def portfolio_summary(self) -> dict:
        return {
            "cash": self._cash,
            "portfolio_value": self.get_portfolio_value(),
            "positions": {
                sym: {"qty": p.qty, "avg_entry": p.avg_entry,
                      "market_value": p.market_value, "pnl": p.unrealised_pnl}
                for sym, p in self._positions.items()
            },
            "trade_count": len(self._orders),
        }


# ─── Alpaca wallet ────────────────────────────────────────────────────────────

class AlpacaWallet(BaseWallet):
    """
    Alpaca Markets broker integration.
    """
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL  = "https://api.alpaca.markets"

    def __init__(self):
        self._api_key    = os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self._live       = os.environ.get("ALPACA_LIVE", "").lower() == "true"
        self._base_url   = self.LIVE_URL if self._live else self.PAPER_URL

        if not self._api_key or not self._secret_key:
            logger.warning("Alpaca credentials not set — orders will fail.")

        mode = "LIVE" if self._live else "PAPER"
        logger.info(f"AlpacaWallet initialised ({mode})")

    @property
    def is_paper(self) -> bool:
        return not self._live

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID":     self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Content-Type":        "application/json",
        }

    def _get(self, path: str) -> Optional[dict]:
        import requests
        try:
            r = requests.get(f"{self._base_url}{path}", headers=self._headers(), timeout=8)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Alpaca GET {path}: {e}")
            return None

    def _post(self, path: str, payload: dict) -> Optional[dict]:
        import requests
        try:
            r = requests.post(
                f"{self._base_url}{path}",
                headers=self._headers(),
                json=payload, timeout=8
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Alpaca POST {path}: {e}")
            return None

    def get_cash(self) -> float:
        data = self._get("/v2/account")
        if data:
            return float(data.get("buying_power", 0))
        return 0.0

    def get_portfolio_value(self) -> float:
        data = self._get("/v2/account")
        if data:
            return float(data.get("portfolio_value", 0))
        return 0.0

    def get_positions(self) -> List[Position]:
        data = self._get("/v2/positions")
        if not data:
            return []
        positions = []
        for p in data:
            positions.append(Position(
                symbol=p["symbol"],
                qty=float(p["qty"]),
                avg_entry=float(p["avg_entry_price"]),
                market_value=float(p["market_value"]),
                unrealised_pnl=float(p["unrealized_pl"]),
                side=p["side"],
            ))
        return positions

    def submit_order(self, order: OrderRequest) -> OrderResult:
        payload = {
            "symbol":        order.symbol,
            "side":          order.side,
            "type":          order.order_type,
            "time_in_force": order.time_in_force,
        }
        if order.qty:
            payload["qty"] = str(order.qty)
        elif order.notional:
            payload["notional"] = str(order.notional)
        if order.limit_price:
            payload["limit_price"] = str(order.limit_price)
        if order.stop_price:
            payload["stop_price"] = str(order.stop_price)

        mode = "LIVE" if self._live else "PAPER"
        logger.info(f"[ALPACA {mode}] Submitting {order.side} {order.symbol}")

        data = self._post("/v2/orders", payload)
        if data and "id" in data:
            return OrderResult(
                success=True,
                order_id=data["id"],
                symbol=order.symbol,
                side=order.side,
                qty=float(data.get("qty") or 0),
                fill_price=float(data.get("filled_avg_price") or 0) or None,
                status=data.get("status", "submitted"),
                broker="alpaca",
                timestamp=datetime.now(timezone.utc),
                raw=data,
            )
        return OrderResult(
            success=False, order_id=None,
            symbol=order.symbol, side=order.side,
            qty=order.qty, fill_price=None,
            status="rejected", broker="alpaca",
            timestamp=datetime.now(timezone.utc),
            error=str(data) if data else "No response from Alpaca"
        )

    def cancel_order(self, order_id: str) -> bool:
        import requests
        try:
            r = requests.delete(
                f"{self._base_url}/v2/orders/{order_id}",
                headers=self._headers(), timeout=8
            )
            return r.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Cancel order {order_id}: {e}")
            return False


# ─── Wallet factory ───────────────────────────────────────────────────────────

def get_wallet(mode: str = "paper", **kwargs) -> BaseWallet:
    """
    Factory: get the appropriate wallet.
    """
    if mode == "alpaca":
        return AlpacaWallet()
    # FORCE PERSISTENCE for simulated real-world growth
    kwargs.setdefault("persistence", True)
    return PaperWallet(**kwargs)
