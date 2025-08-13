"""
Backtesting utilities for the trading bot.

This module implements functions to run multiple strategies over a
historical dataset and return the resulting strategy objects with their
trade logs and equity curves. Backtesting is used both for manual
evaluation and as part of the optimisation and interactive viewer tools.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .strategies import StrategyBase

__all__ = ["run_backtest_multi"]


def run_backtest_multi(
    df: pd.DataFrame, strategy_specs: List[Tuple[type[StrategyBase], Dict[str, Any]]]
) -> List[StrategyBase]:
    """Run multiple strategies over the same dataset.

    For each tuple in ``strategy_specs``, instantiate the strategy class with
    a unique ``sid`` (starting at 1) and the provided parameter dict. Then
    iterate over the historical data and call ``step`` on each strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical OHLC data. Must contain a ``close`` column at minimum.
    strategy_specs : list of tuples
        A list where each element is (StrategyClass, params_dict).

    Returns
    -------
    list of StrategyBase
        A list of strategy instances after completing the backtest.
    """
    strats: List[StrategyBase] = []
    for idx, (Cls, params) in enumerate(strategy_specs, start=1):
        s = Cls(sid=idx, **params)
        strats.append(s)
    closes: List[float] = []
    for ts, row in df.iterrows():
        px = row["close"]
        if np.isnan(px):
            continue
        closes.append(px)
        cs = pd.Series(closes, index=df.index[: len(closes)])
        for s in strats:
            s.step(ts, px, cs, df_full=df.iloc[: len(closes)])
    return strats