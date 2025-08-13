# trading_bot/data.py
"""
Data access functions for the trading bot.

This module provides helpers for fetching historical and real-time
cryptocurrency price data from the Binance REST API. All exceptions are
wrapped with ASCII-safe messages so Windows consoles that default to
ASCII won't choke on Unicode when printing tracebacks.
"""

from __future__ import annotations

import time
import urllib.parse
import requests
import pandas as pd

BINANCE_REST = "https://api.binance.com"
BINANCE_TESTNET = "https://testnet.binance.vision"


def _ascii_safe(obj) -> str:
    """
    Best-effort: convert any object to a str that contains only ASCII
    (non-ASCII characters are replaced). This prevents Windows ASCII
    consoles from throwing `UnicodeEncodeError` when printing tracebacks.
    """
    try:
        return str(obj).encode("ascii", "replace").decode("ascii")
    except Exception:
        # Fallback to repr if str() itself explodes
        try:
            return repr(obj).encode("ascii", "replace").decode("ascii")
        except Exception:
            return "unknown error"


def fetch_klines(
    symbol: str,
    interval: str = "1m",
    limit: int = 1000,
    end_time: int | None = None,
    base_url: str = BINANCE_REST,
    timeout: float = 12,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance and return a DataFrame indexed by tz-aware
    timestamps (Europe/Berlin). Any raised RuntimeError uses ASCII-safe text.
    """
    symbol = symbol.upper()
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    if end_time is not None:
        params["endTime"] = int(end_time)

    url = f"{base_url}/api/v3/klines?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "python-requests/klines", "Accept-Charset": "utf-8"}

    last_err = None
    for _ in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            # make sure requests decodes body as UTF-8 if needed
            r.encoding = r.encoding or "utf-8"
            r.raise_for_status()
            data = r.json()

            # Binance error object
            if isinstance(data, dict) and "code" in data:
                raise RuntimeError(
                    _ascii_safe(f"Binance error {data.get('code')}: {data.get('msg')}")
                )
            if not isinstance(data, list):
                raise RuntimeError(_ascii_safe(f"Unexpected payload type: {type(data)}"))

            rows = []
            for k in data:
                if not isinstance(k, list) or len(k) < 6:
                    continue
                try:
                    rows.append(
                        (int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
                    )
                except Exception as e:
                    # Skip malformed row; keep going
                    last_err = e
                    continue

            if not rows:
                raise RuntimeError(_ascii_safe("No valid candles parsed"))

            ot, o, h, l, c, v = map(list, zip(*rows))
            ts = pd.to_datetime(ot, unit="ms", utc=True).tz_convert("Europe/Berlin")
            df = pd.DataFrame(
                {"open": o, "high": h, "low": l, "close": c, "volume": v},
                index=ts,
            ).dropna(subset=["close"])

            if df.empty:
                raise RuntimeError(_ascii_safe("No valid candles (all NaN) after parsing."))
            return df

        except Exception as e:
            last_err = e
            time.sleep(1.0)

    raise RuntimeError(f"fetch_klines failed: {_ascii_safe(last_err)}")


def stream_latest_close(symbol: str, base_url: str = BINANCE_REST):
    """
    Get the latest close price for `symbol` from Binance as (timestamp, price).
    Errors raised here are also ASCII-safe.
    """
    try:
        url = f"{base_url}/api/v3/klines?symbol={symbol}&interval=1m&limit=1"
        headers = {"User-Agent": "python-requests/klines", "Accept-Charset": "utf-8"}
        r = requests.get(url, headers=headers, timeout=5)
        r.encoding = r.encoding or "utf-8"
        r.raise_for_status()
        k = r.json()[0]
        ts = pd.to_datetime(int(k[0]) // 1000, unit="s", utc=True).tz_convert("Europe/Berlin")
        px = float(k[4])
        return ts, px
    except Exception as e:
        raise RuntimeError(f"stream_latest_close failed: {_ascii_safe(e)}")
