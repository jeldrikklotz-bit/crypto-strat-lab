# -*- coding: utf-8 -*-
"""
Trading bot package exposing core components.

This package collects all of the modules needed to fetch data, compute
indicators, define strategies, optimise parameters, backtest, plot results,
provide an interactive viewer, manage a broker for live trading, and run
live trading loops. Importing this package makes it easy to access the
components directly, e.g. ``from trading_bot import MACDRSI``.
"""

from .data import fetch_klines, stream_latest_close
from .indicators import rsi, macd, sma, atr
from .base_strategy import StrategyBase
from .strategies import MACDRSI, SMACross, DonchianBreakout, BollingerReversion
from .optimizer import score_metrics, grid_search_one
from .backtest import run_backtest_multi
from .plotting import plot_with_numbered_trades
from .viewer import interactive_backtest_viewer
from .broker import BinanceBroker
from .live import run_live

__all__ = [
    "fetch_klines",
    "stream_latest_close",
    "rsi",
    "macd",
    "sma",
    "atr",
    "StrategyBase",
    "MACDRSI",
    "SMACross",
    "DonchianBreakout",
    "score_metrics",
    "grid_search_one",
    "run_backtest_multi",
    "plot_with_numbered_trades",
    "interactive_backtest_viewer",
    "BinanceBroker",
    "run_live",
]