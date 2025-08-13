"""
Concrete trading strategies for the trading bot.

This module defines several strategy classes derived from
:class:`trading_bot.base_strategy.StrategyBase`. Each strategy implements
different technical rules to generate BUY/SELL signals. The available
strategies are:

* :class:`MACDRSI` – uses MACD crossovers and RSI thresholds with a
  trailing stop.
* :class:`SMACross` – uses simple moving average crossovers combined
  with an RSI filter and a trailing stop.
* :class:`DonchianBreakout` – uses Donchian channel breakouts and ATR
  based trailing stops.

Each strategy also provides a :meth:`grid` static method that returns
a list of parameter dictionaries for grid search optimisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from .base_strategy import StrategyBase
from .indicators import rsi, macd, sma, atr

__all__ = [
    "MACDRSI",
    "SMACross",
    "DonchianBreakout",
    "BollingerReversion",
]


class MACDRSI(StrategyBase):
    """MACD and RSI based trading strategy with trailing stop."""

    def __init__(
        self,
        sid: int,
        *,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        rsi_buy: float = 55,
        rsi_sell: float = 70,
        trail_pct: float = 0.02,
        fee: float = 0.001,
    ) -> None:
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.trail_pct = trail_pct
        super().__init__(sid, f"{sid}: MACD+RSI", fee=fee)

    def _after_reset(self) -> None:
        self.high_since_entry: float | None = None

    def _decide(self, ts: pd.Timestamp, price: float, close: pd.Series, df_full: pd.DataFrame | None = None) -> str | None:
        m, s, _ = macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        if len(m) < 3:
            return None
        md_now = m.iloc[-1] - s.iloc[-1]
        md_prev = m.iloc[-2] - s.iloc[-2]
        rsi_now = rsi(close, self.rsi_period).iloc[-1]
        if self.position > 0:
            self.high_since_entry = max(self.high_since_entry or price, price)
            trail_stop = self.high_since_entry * (1 - self.trail_pct)
            if (md_now < 0 and md_prev > 0) or rsi_now >= self.rsi_sell or price <= trail_stop:
                self.high_since_entry = None
                return "SELL"
        else:
            if (md_now > 0 and md_prev < 0) and rsi_now >= self.rsi_buy:
                self.high_since_entry = price
                return "BUY"
        return None

    @staticmethod
    def grid(fee: float = 0.001) -> list[dict[str, float | int]]:
        """Return a grid of parameter combinations for optimisation."""
        return [
            {
                "macd_fast": mf,
                "macd_slow": ms,
                "macd_signal": sg,
                "rsi_period": rp,
                "rsi_buy": rb,
                "rsi_sell": rs,
                "trail_pct": tp,
                "fee": fee,
            }
            for mf in [8, 12, 16]
            for ms in [21, 26, 32]
            if mf < ms
            for sg in [7, 9, 12]
            for rp in [10, 14, 21]
            for rb in [52, 55, 58]
            for rs in [65, 70, 75]
            if rb < rs
            for tp in [0.01, 0.02, 0.03]
        ]


class SMACross(StrategyBase):
    """Simple Moving Average crossover strategy with RSI filter and trailing stop."""

    def __init__(
        self,
        sid: int,
        *,
        fast: int = 10,
        slow: int = 30,
        rsi_period: int = 14,
        rsi_filter: float = 50,
        trail_pct: float = 0.02,
        fee: float = 0.001,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_filter = rsi_filter
        self.trail_pct = trail_pct
        super().__init__(sid, f"{sid}: SMA Cross", fee=fee)

    def _after_reset(self) -> None:
        self.high_since_entry: float | None = None

    def _decide(self, ts: pd.Timestamp, price: float, close: pd.Series, df_full: pd.DataFrame | None = None) -> str | None:
        if len(close) < max(self.fast, self.slow) + 2:
            return None
        s_fast = sma(close, self.fast)
        s_slow = sma(close, self.slow)
        c_now = s_fast.iloc[-1] - s_slow.iloc[-1]
        c_prev = s_fast.iloc[-2] - s_slow.iloc[-2]
        r = rsi(close, self.rsi_period).iloc[-1]
        if self.position > 0:
            self.high_since_entry = max(self.high_since_entry or price, price)
            if (
                price <= self.high_since_entry * (1 - self.trail_pct)
                or (c_now < 0 and c_prev > 0)
                or r > 70
            ):
                self.high_since_entry = None
                return "SELL"
        else:
            if c_now > 0 and c_prev < 0 and r >= self.rsi_filter:
                self.high_since_entry = price
                return "BUY"
        return None

    @staticmethod
    def grid(fee: float = 0.001) -> list[dict[str, float | int]]:
        """Return a grid of parameter combinations for optimisation."""
        return [
            {
                "fast": f,
                "slow": s,
                "rsi_period": rp,
                "rsi_filter": rf,
                "trail_pct": tp,
                "fee": fee,
            }
            for f in [7, 10, 14]
            for s in [25, 30, 40]
            if f < s
            for rp in [10, 14, 21]
            for rf in [48, 50, 55]
            for tp in [0.01, 0.02, 0.03]
        ]


class DonchianBreakout(StrategyBase):
    """Donchian channel breakout strategy with ATR‑based trailing stops."""

    def __init__(
        self,
        sid: int,
        *,
        ch: int = 20,
        exit_ch: int = 10,
        atr_n: int = 14,
        atr_mult: float = 2.0,
        fee: float = 0.001,
    ) -> None:
        self.ch = ch
        self.exit_ch = exit_ch
        self.atr_n = atr_n
        self.atr_mult = atr_mult
        super().__init__(sid, f"{sid}: Donchian", fee=fee)

    def _after_reset(self) -> None:
        self.trailing: float | None = None

    def _decide(self, ts: pd.Timestamp, price: float, close: pd.Series, df_full: pd.DataFrame | None = None) -> str | None:
        if df_full is None or len(df_full) < max(self.ch, self.exit_ch) + 2:
            return None
        dc_high = df_full["high"].rolling(self.ch).max()
        exit_low = df_full["low"].rolling(self.exit_ch).min()
        _atr = atr(df_full, self.atr_n)
        # Entry on channel breakout
        if self.position == 0:
            if df_full["close"].iloc[-1] > dc_high.iloc[-2]:
                self.trailing = price - self.atr_mult * _atr.iloc[-1]
                return "BUY"
        else:
            self.trailing = max(self.trailing or -np.inf, price - self.atr_mult * _atr.iloc[-1])
            if df_full["close"].iloc[-1] < exit_low.iloc[-1] or price < self.trailing:
                self.trailing = None
                return "SELL"
        return None

    @staticmethod
    def grid(fee: float = 0.001) -> list[dict[str, float | int]]:
        """Return a grid of parameter combinations for optimisation."""
        return [
            {
                "ch": ch,
                "exit_ch": ex,
                "atr_n": an,
                "atr_mult": am,
                "fee": fee,
            }
            for ch in [20, 30]
            for ex in [10, 15]
            for an in [14]
            for am in [2.0, 2.5]
        ]
# --- NEW: Bollinger Band mean-reversion strategy ---

class BollingerReversion(StrategyBase):
    """
    Counter-trend mean reversion using Bollinger Bands + RSI filter, with optional ATR stop.
    Entry: close < lower_band AND RSI <= rsi_buy
    Exit:  close >= mid_band OR RSI >= rsi_exit OR ATR-stop hit
    """
    def __init__(self, sid:int,
                 bb_period=20, bb_dev=2.0,
                 rsi_period=14, rsi_buy=35, rsi_exit=50,
                 atr_n=14, atr_mult=2.0,
                 fee=0.001):
        self.bb_period = bb_period
        self.bb_dev = bb_dev
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_exit = rsi_exit
        self.atr_n = atr_n
        self.atr_mult = atr_mult
        super().__init__(sid, f"{sid}: Bollinger MR", fee=fee)

    def _after_reset(self):
        self.stop_px = None  # ATR-based protective stop

    def _decide(self, ts, price, close, df_full=None):
        if df_full is None or len(close) < max(self.bb_period, self.rsi_period, self.atr_n) + 2:
            return None

        # Bollinger bands
        mid = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std(ddof=0)
        upper = mid + self.bb_dev * std
        lower = mid - self.bb_dev * std

        # RSI
        r = rsi(close, self.rsi_period).iloc[-1]

        # ATR for protective stop
        _atr = atr(df_full, self.atr_n).iloc[-1]

        # SELL rules (flatten)
        if self.position > 0:
            # stop updates: trailing on reversion is not typical here; use static ATR from entry area
            if self.stop_px is not None and price <= self.stop_px:
                self.stop_px = None
                return "SELL"
            # exit on reversion or RSI normalization
            if price >= mid.iloc[-1] or r >= self.rsi_exit:
                self.stop_px = None
                return "SELL"

        # BUY rules (fade downside extension)
        else:
            if price <= lower.iloc[-1] and r <= self.rsi_buy:
                # set initial stop below entry by ATR multiple
                self.stop_px = price - self.atr_mult * _atr if _atr == _atr else None
                return "BUY"

        return None

    @staticmethod
    def grid(fee=0.001):
        return [{"bb_period": bp, "bb_dev": bd,
                 "rsi_period": rp, "rsi_buy": rb, "rsi_exit": re,
                 "atr_n": an, "atr_mult": am, "fee": fee}
                for bp in [14, 20, 30]
                for bd in [1.5, 2.0, 2.5]
                for rp in [10, 14]
                for rb in [30, 35, 40]
                for re in [48, 50, 55]
                for an in [14]
                for am in [1.5, 2.0]]
