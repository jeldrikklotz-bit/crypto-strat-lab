"""
Broker abstraction for live trading in the trading bot.

This module defines :class:`BinanceBroker`, a simple wrapper around the
Binance REST API for placing market orders. It supports both paper
trading (simulated) and real orders, with API key and secret taken
from environment variables. Equity and positions are tracked in USD.
"""

from __future__ import annotations

import os
import time
import hmac
import hashlib
import urllib.parse
from typing import Tuple, Dict, Any

import requests
from datetime import datetime, timezone

from .data import BINANCE_REST, BINANCE_TESTNET, stream_latest_close

__all__ = ["BinanceBroker"]


class BinanceBroker:
    """A minimal broker for paper and live trading on Binance.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. ``'BTCUSDT'``.
    fee : float, optional
        Trading fee rate (per trade). Default is 0.001 (0.1%).
    paper : bool, optional
        If True, simulate trades without hitting the API. Defaults to True.
    testnet : bool, optional
        If False and paper is False, use the production API. Defaults to True.
    start_cash : float, optional
        Initial cash balance in quote currency. Defaults to 5000.0.
    """

    def __init__(self, symbol: str, fee: float = 0.001, paper: bool = True, testnet: bool = True, start_cash: float = 5000.0) -> None:
        self.symbol = symbol
        self.fee = fee
        self.paper = paper
        self.base_url = BINANCE_TESTNET if testnet else BINANCE_REST
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.cash = float(start_cash)
        self.position = 0.0
        self.trades: list[Tuple[str, float, float, datetime]] = []

    def _signed_request(self, method: str, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key or not self.api_secret:
            raise RuntimeError("API keys missing")
        params["timestamp"] = int(time.time() * 1000)
        qs = urllib.parse.urlencode(params)
        sig = hmac.new(self.api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
        headers = {"X‑MBX‑APIKEY": self.api_key}
        url = f"{self.base_url}{path}?{qs}&signature={sig}"
        resp = requests.request(method, url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def market_buy(self, qty: float, price_hint: float | None = None) -> Dict[str, Any]:
        """Execute a market buy order.

        For paper trading, this updates internal balances without hitting the API. If
        live trading with API keys, it will place an order on Binance.
        """
        if self.paper or not self.api_key:
            price = price_hint if price_hint is not None else self._last_price()
            cost = qty * price * (1 + self.fee)
            if cost > self.cash:
                qty = self.cash / (price * (1 + self.fee))
            self.cash -= qty * price * (1 + self.fee)
            self.position += qty
            self.trades.append(("BUY", price, qty, datetime.now(timezone.utc)))
            return {"paper": True, "price": price, "qty": qty}
        else:
            return self._signed_request(
                "POST",
                "/api/v3/order",
                {"symbol": self.symbol, "side": "BUY", "type": "MARKET", "quantity": qty},
            )

    def market_sell(self, qty: float, price_hint: float | None = None) -> Dict[str, Any]:
        """Execute a market sell order."""
        if self.paper or not self.api_key:
            price = price_hint if price_hint is not None else self._last_price()
            proceeds = qty * price * (1 - self.fee)
            self.cash += proceeds
            self.position -= qty
            self.trades.append(("SELL", price, qty, datetime.now(timezone.utc)))
            return {"paper": True, "price": price, "qty": qty}
        else:
            return self._signed_request(
                "POST",
                "/api/v3/order",
                {"symbol": self.symbol, "side": "SELL", "type": "MARKET", "quantity": qty},
            )

    def equity(self, price: float) -> float:
        """Return the current total equity (cash + value of held position)."""
        return self.cash + self.position * price

    def _last_price(self) -> float:
        _, px = stream_latest_close(self.symbol, base_url=self.base_url)
        return px