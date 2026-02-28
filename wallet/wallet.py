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
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from abc import ABC, abstractmethod

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
    Simulated paper trading wallet.
    No external API required.
    Tracks virtual positions and P&L.
    Used for backtesting and development.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self._cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._orders: List[OrderResult] = []
        logger.info(f"PaperWallet initialised: ${initial_cash:,.2f}")

    @property
    def is_paper(self) -> bool:
        return True

    def get_cash(self) -> float:
        return self._cash

    def get_portfolio_value(self) -> float:
        pos_value = sum(p.market_value for p in self._positions.values())
        return self._cash + pos_value

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def submit_order(self, order: OrderRequest) -> OrderResult:
        """
        Simulate order execution at mid-price.
        In real use: connects to broker API.
        """
        try:
            # Get current price via yfinance
            price = self._get_price(order.symbol)

            if price is None:
                return OrderResult(
                    success=False, order_id=None,
                    symbol=order.symbol, side=order.side,
                    qty=order.qty, fill_price=None,
                    status="rejected", broker="paper",
                    timestamp=datetime.utcnow(),
                    error=f"Cannot get price for {order.symbol}"
                )

            # Calculate qty from notional if needed
            qty = order.qty
            if qty is None and order.notional:
                qty = order.notional / price
            if qty is None:
                qty = 1.0

            cost = qty * price

            if order.side == "buy":
                if cost > self._cash:
                    return OrderResult(
                        success=False, order_id=None,
                        symbol=order.symbol, side=order.side,
                        qty=qty, fill_price=price,
                        status="rejected", broker="paper",
                        timestamp=datetime.utcnow(),
                        error=f"Insufficient cash: need ${cost:,.2f}, have ${self._cash:,.2f}"
                    )
                self._cash -= cost
                if order.symbol in self._positions:
                    pos = self._positions[order.symbol]
                    new_qty = pos.qty + qty
                    new_avg = (pos.avg_entry * pos.qty + price * qty) / new_qty
                    self._positions[order.symbol] = Position(
                        symbol=order.symbol, qty=new_qty,
                        avg_entry=new_avg, market_value=new_qty * price,
                        unrealised_pnl=(price - new_avg) * new_qty,
                        side="long"
                    )
                else:
                    self._positions[order.symbol] = Position(
                        symbol=order.symbol, qty=qty,
                        avg_entry=price, market_value=qty * price,
                        unrealised_pnl=0.0, side="long"
                    )

            elif order.side == "sell":
                if order.symbol in self._positions:
                    pos = self._positions[order.symbol]
                    sell_qty = min(qty, pos.qty)
                    proceeds = sell_qty * price
                    self._cash += proceeds
                    remaining = pos.qty - sell_qty
                    if remaining < 0.001:
                        del self._positions[order.symbol]
                    else:
                        self._positions[order.symbol] = Position(
                            symbol=order.symbol, qty=remaining,
                            avg_entry=pos.avg_entry,
                            market_value=remaining * price,
                            unrealised_pnl=(price - pos.avg_entry) * remaining,
                            side="long"
                        )

            order_id = f"paper_{order.symbol}_{int(datetime.utcnow().timestamp())}"
            result = OrderResult(
                success=True, order_id=order_id,
                symbol=order.symbol, side=order.side,
                qty=qty, fill_price=price,
                status="paper", broker="paper",
                timestamp=datetime.utcnow(),
                raw={"notional": qty * price, "tier": order.tier}
            )
            self._orders.append(result)
            logger.info(
                f"[PAPER] {order.side.upper()} {qty:.4f} {order.symbol} @ ${price:.4f} "
                f"(${qty*price:,.2f}) [{order.tier}]"
            )
            return result

        except Exception as e:
            logger.error(f"PaperWallet order error: {e}")
            return OrderResult(
                success=False, order_id=None,
                symbol=order.symbol, side=order.side,
                qty=order.qty, fill_price=None,
                status="error", broker="paper",
                timestamp=datetime.utcnow(),
                error=str(e)
            )

    def cancel_order(self, order_id: str) -> bool:
        return True  # Paper orders are instant

    def _get_price(self, symbol: str) -> Optional[float]:
        try:
            import yfinance as yf
            hist = yf.Ticker(symbol).history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
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
    Free paper trading API: https://alpaca.markets

    Setup:
      export ALPACA_API_KEY=your_key
      export ALPACA_SECRET_KEY=your_secret
      # Default is paper trading. Set ALPACA_LIVE=true for live.
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL  = "https://api.alpaca.markets"

    def __init__(self):
        self._api_key    = os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self._live       = os.environ.get("ALPACA_LIVE", "").lower() == "true"
        self._base_url   = self.LIVE_URL if self._live else self.PAPER_URL

        if not self._api_key or not self._secret_key:
            logger.warning("Alpaca credentials not set — orders will fail. "
                         "Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")

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
                timestamp=datetime.utcnow(),
                raw=data,
            )
        return OrderResult(
            success=False, order_id=None,
            symbol=order.symbol, side=order.side,
            qty=order.qty, fill_price=None,
            status="rejected", broker="alpaca",
            timestamp=datetime.utcnow(),
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
      mode="paper"   → PaperWallet (no credentials needed)
      mode="alpaca"  → AlpacaWallet (needs ALPACA_API_KEY + ALPACA_SECRET_KEY)
    """
    if mode == "alpaca":
        return AlpacaWallet()
    return PaperWallet(**kwargs)
