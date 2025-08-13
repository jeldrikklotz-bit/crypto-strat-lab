"""
Base strategy class for the trading bot.

This module defines :class:`StrategyBase`, an abstract class that provides
common state management for trading strategies including position tracking,
cash accounting, trade logging, and performance statistics. Concrete
strategies should inherit from this class and implement the ``_decide``
method to return trading signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["StrategyBase"]


class StrategyBase:
    """Abstract base class for trading strategies.

    Strategies maintain cash and position state, handle trade execution on
    signals from subclasses, record an equity curve, and compute basic
    performance metrics. Subclasses must implement the
    :meth:`_decide` method, which returns ``"BUY"``, ``"SELL"``, or
    ``None`` based on the current market data.
    """

    def __init__(self, sid: int, name: str, fee: float = 0.001, start_cash: float = 5000.0) -> None:
        self.sid = sid
        self.name = name
        self.fee = fee
        self.start_cash = start_cash
        self.reset()

    def reset(self) -> None:
        """Reset internal state before a backtest or live trading session."""
        self.position = 0.0
        self.cash = float(self.start_cash)
        self.trades: list[tuple[pd.Timestamp, str, float, float, int]] = []
        self.equity_series: list[tuple[pd.Timestamp, float]] = []
        self._after_reset()

    def _after_reset(self) -> None:
        """Hook for subclasses to override. Called after :meth:`reset`."""
        pass

    def step(self, ts: pd.Timestamp, price: float, df_close: pd.Series, df_full: pd.DataFrame | None = None) -> str | None:
        """Advance the strategy by one bar and act on signals.

        Subclasses should not override this method. They should instead
        implement :meth:`_decide` to generate signals. This method will
        automatically handle position sizing, account for trading fees, and
        log trades.

        Parameters
        ----------
        ts : pandas.Timestamp
            Timestamp of the current bar.
        price : float
            Closing price for the current bar.
        df_close : pandas.Series
            Series of close prices up to and including the current bar.
        df_full : pandas.DataFrame, optional
            Full OHLC data up to the current bar. Some strategies may use
            additional columns like high/low for ATR.

        Returns
        -------
        str or None
            The signal returned by the strategy: ``"BUY"``, ``"SELL"``, or
            ``None`` for no action.
        """
        sig = self._decide(ts, price, df_close, df_full)
        if sig == "BUY" and self.position == 0:
            qty = self.cash / (price * (1 + self.fee))
            if qty > 1e-9:
                # Deduct cost including fees
                self.cash -= qty * price * (1 + self.fee)
                self.position = qty
                self.trades.append((ts, "BUY", price, qty, self.sid))
        elif sig == "SELL" and self.position > 0:
            qty = self.position
            # Add proceeds minus fees
            self.cash += qty * price * (1 - self.fee)
            self.position = 0.0
            self.trades.append((ts, "SELL", price, qty, self.sid))
        # Record equity regardless of action
        self.equity_series.append((ts, self.cash + self.position * price))
        return sig

    def _decide(self, ts: pd.Timestamp, price: float, df_close: pd.Series, df_full: pd.DataFrame | None = None) -> str | None:
        """Return the trading signal for the current bar.

        Subclasses must override this method. It should return ``"BUY"`` to
        initiate a long position, ``"SELL"`` to exit the position, or
        ``None`` to hold.
        """
        raise NotImplementedError

    def metrics(self) -> dict[str, float | int | str]:
        """Compute performance metrics for the strategy.

        Returns a dictionary containing final equity, profit/loss, Sharpe
        ratio, maximum drawdown, trade count, win rate, compound annual
        growth rate (CAGR), and number of days tested.
        """
        if not self.equity_series:
            return {}
        times = [t for t, _ in self.equity_series]
        vals = np.array([v for _, v in self.equity_series], dtype=float)
        mask = ~np.isnan(vals)
        vals = vals[mask]
        if len(vals) == 0:
            return {}
        # Use the first valid timestamp and the last timestamp to compute duration
        t0 = times[np.argmax(mask)]
        t1 = times[-1]
        days = max((t1 - t0).total_seconds() / 86400.0, 1e-9)
        # Calculate returns
        ret = pd.Series(vals).pct_change(fill_method=None).fillna(0.0).to_numpy()
        mu = ret.mean()
        sigma = ret.std(ddof=1) if len(ret) > 1 else 0.0
        sharpe = (mu / sigma) * np.sqrt(1440) if sigma > 0 else 0.0
        # Maximum drawdown
        peak = vals[0]
        max_dd = 0.0
        for v in vals:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak)
        start_eq = float(vals[0])
        end_eq = float(vals[-1])
        pnl = end_eq - start_eq
        years = days / 365.0
        cagr = (end_eq / start_eq) ** (1.0 / max(years, 1e-9)) - 1.0 if start_eq > 0 else 0.0
        # Win/loss statistics
        wins = losses = 0
        for i in range(0, len(self.trades), 2):
            if i + 1 >= len(self.trades):
                break
            buy = self.trades[i]
            sell = self.trades[i + 1]
            if (sell[2] - buy[2]) * buy[3] > 0:
                wins += 1
            else:
                losses += 1
        winrate = wins / max(1, wins + losses)
        return {
            "sid": self.sid,
            "name": self.name,
            "final_equity": end_eq,
            "pnl": pnl,
            "sharpe": float(sharpe),
            "max_dd": float(max_dd),
            "trades": len(self.trades),
            "winrate": float(winrate),
            "cagr": float(cagr),
            "days": float(days),
        }