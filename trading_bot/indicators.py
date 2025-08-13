"""
Technical indicators used by trading strategies.

This module implements common technical analysis indicators required by
the built‑in strategies. The functions operate on pandas Series or
DataFrames and return pandas objects of the same length where
appropriate.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["rsi", "macd", "sma", "atr"]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a price series.

    Parameters
    ----------
    series : pandas.Series
        Series of closing prices.
    period : int, optional
        The lookback period over which to compute RSI. Default is 14.

    Returns
    -------
    pandas.Series
        RSI values in the range 0–100. The returned series aligns with the
        input index and contains NaNs for the first ``period`` elements.
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    out = 100 - (100 / (1 + rs))
    return out


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute the Moving Average Convergence Divergence (MACD) indicator.

    Parameters
    ----------
    series : pandas.Series
        Series of closing prices.
    fast : int, optional
        Period for the fast EMA. Default is 12.
    slow : int, optional
        Period for the slow EMA. Default is 26.
    signal : int, optional
        Period for the signal line EMA. Default is 9.

    Returns
    -------
    tuple of pandas.Series
        A tuple of (macd_line, signal_line, histogram) where ``macd_line`` is
        the difference between the fast and slow EMAs, ``signal_line`` is the
        EMA of the MACD line, and ``histogram`` is the difference between
        ``macd_line`` and ``signal_line``.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def sma(series: pd.Series, n: int) -> pd.Series:
    """Compute a simple moving average (SMA).

    Parameters
    ----------
    series : pandas.Series
        Series of prices.
    n : int
        Number of periods over which to compute the average.

    Returns
    -------
    pandas.Series
        SMA of the input series.
    """
    return series.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ``high``, ``low``, and ``close``.
    n : int, optional
        Lookback period for ATR. Default is 14.

    Returns
    -------
    pandas.Series
        ATR values.
    """
    h = df["high"]
    l = df["low"]
    c = df["close"]
    prev_c = c.shift(1)
    true_range = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(n).mean()