"""
Parameter optimisation utilities for trading strategies.

This module provides functions to evaluate strategy performance using
a custom scoring function and to search a parameter grid for the best
configuration. These functions are used by the CLI to find the
optimal parameters for each built‑in strategy.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd

from .strategies import StrategyBase  # type: ignore
from .backtest import run_backtest_multi

__all__ = ["score_metrics", "grid_search_one"]


def score_metrics(
    m: Dict[str, Any],
    min_trades: int = 6,
    max_dd_cap: float = 0.25,
) -> float:
    """Score strategy performance metrics.

    The score is a linear combination of Sharpe ratio and CAGR minus a
    penalty for maximum drawdown. Strategies that trade too infrequently,
    have negative CAGR, or exceed the drawdown cap are heavily penalised.

    Parameters
    ----------
    m : dict
        Dictionary of performance metrics returned by a strategy's
        :meth:`~trading_bot.base_strategy.StrategyBase.metrics` method.
    min_trades : int, optional
        Minimum number of trades required to be considered. Defaults to 6.
    max_dd_cap : float, optional
        Maximum allowable drawdown. If the strategy exceeds this, it is
        discarded. Defaults to 0.25 (25%).

    Returns
    -------
    float
        The computed score. Higher is better.
    """
    if m.get("trades", 0) < min_trades:
        return -1e9
    if m.get("cagr", 0.0) <= 0:
        return -1e9
    if m.get("max_dd", 1.0) > max_dd_cap:
        return -1e9
    sharpe = m.get("sharpe", 0.0)
    cagr = m.get("cagr", 0.0)
    maxdd = m.get("max_dd", 0.0)
    return 0.6 * sharpe + 0.4 * (cagr * 100) - 2.0 * (maxdd * 100)


def grid_search_one(
    df: pd.DataFrame,
    StrategyCls: type[StrategyBase],
    *,
    fee: float = 0.001,
    min_trades: int = 6,
    max_dd_cap: float = 0.25,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Perform a simple parameter grid search for a single strategy.

    For each parameter combination returned by ``StrategyCls.grid()``, this
    function runs a backtest and scores the results. The best parameters by
    the scoring function are returned along with the associated metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical OHLC data on which to evaluate the strategy.
    StrategyCls : class
        Strategy class to optimise.
    fee : float, optional
        Trading fee to pass through when constructing the strategy. Default 0.001.
    min_trades : int, optional
        Minimum trades required to be considered. Default 6.
    max_dd_cap : float, optional
        Maximum drawdown allowed. Default 0.25.

    Returns
    -------
    tuple
        A tuple of (best_params_dict, metrics_dict) for the best score.
    """
    best_params: Dict[str, Any] | None = None
    best_metrics: Dict[str, Any] | None = None
    best_score = -1e9
    for params in StrategyCls.grid(fee=fee):
        strat = StrategyCls(sid=1, **params)  # sid is irrelevant for optimisation
        closes: list[float] = []
        # Run backtest inline to avoid overhead of run_backtest_multi for each param
        for ts, row in df.iterrows():
            px = row["close"]
            if np.isnan(px):
                continue
            closes.append(px)
            cs = pd.Series(closes, index=df.index[: len(closes)])
            strat.step(ts, px, cs, df_full=df.iloc[: len(closes)])
        m = strat.metrics()
        sc = score_metrics(m, min_trades=min_trades, max_dd_cap=max_dd_cap)
        if sc > best_score:
            best_params, best_metrics, best_score = params, m, sc
    assert best_params is not None and best_metrics is not None
    return best_params, best_metrics