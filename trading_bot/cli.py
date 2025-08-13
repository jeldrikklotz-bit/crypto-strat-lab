# -*- coding: utf-8 -*-
"""
Command line interface for the trading bot library.

This script exposes a simple CLI for backtesting, optimisation, and live trading
using the modules in this package. It can be executed directly or imported and
called via the :func:`main` function.

Example usage from the command line::

    python -m trading_bot.cli --mode backtest --symbols BTCUSDT,ETHUSDT --limit 1200

"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple, Dict, Any

from .data import fetch_klines
from .strategies import MACDRSI, SMACross, DonchianBreakout, BollingerReversion
from .backtest import run_backtest_multi
from .optimizer import grid_search_one
from .plotting import plot_with_numbered_trades
from .viewer import interactive_backtest_viewer
from .live import run_live
import os
import matplotlib.pyplot as plt

if os.name == "nt":
    os.environ.setdefault("MPLBACKEND", "TkAgg")


def default_choices() -> List[Tuple[type, Dict[str, Any]]]:
    """
    Return a list of default parameter sets for each of the built‑in strategies.

    Each entry in the list is a tuple of (StrategyClass, params_dict) that can
    be passed directly to backtesting functions. These defaults provide a
    reasonable starting point for evaluating the strategies.
    """
    return [
        (
            MACDRSI,
            dict(
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                rsi_period=14,
                rsi_buy=55,
                rsi_sell=70,
                trail_pct=0.02,
                fee=0.001,
            ),
        ),
        (
            SMACross,
            dict(
                fast=10,
                slow=30,
                rsi_period=14,
                rsi_filter=50,
                trail_pct=0.02,
                fee=0.001,
            ),
        ),
        (
            DonchianBreakout,
            dict(ch=20, exit_ch=10, atr_n=14, atr_mult=2.0, fee=0.001),
        ),
        (
            BollingerReversion,
            dict(
                bb_period=20,
                bb_dev=2.0,
                rsi_period=14, 
                rsi_buy=35, 
                rsi_exit=50,
                atr_n=14, 
                atr_mult=2.0, 
                fee=0.001)),
    ]


def optimise_all(df: Any) -> List[Tuple[type, Dict[str, Any]]]:
    """
    Run a grid search for all built‑in strategies on the provided data.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical OHLC data to optimise on.

    Returns
    -------
    list
        List of (StrategyClass, best_params_dict) for each built‑in strategy.
    """
    out: List[Tuple[type, Dict[str, Any]]] = []
    for Cls in [MACDRSI, SMACross, DonchianBreakout]:
        best_p, best_m = grid_search_one(df, Cls, fee=0.001)
        print(f"Best {Cls.__name__}:", json.dumps(best_m, indent=2))
        out.append((Cls, best_p))
    return out


def run_backtest_and_plot(
    symbol: str, interval: str, limit: int, chosen: List[Tuple[type, Dict[str, Any]]]
) -> None:
    """
    Backtest the given strategies on a symbol and display a plot.

    This helper downloads historical data, runs each strategy over it, prints
    performance metrics, and shows a plot with numbered buy/sell markers.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. 'BTCUSDT'.
    interval : str
        Candle interval such as '1m', '5m', etc.
    limit : int
        Number of candles to fetch.
    chosen : list
        List of (StrategyClass, params_dict) to test.
    """
    df = fetch_klines(symbol, interval=interval, limit=limit)
    strats = run_backtest_multi(df, chosen)
    for s in strats:
        print(json.dumps(s.metrics(), indent=2))
    plot_with_numbered_trades(df, strats, title_suffix=f"{symbol} backtest")


def main(argv: List[str] = None) -> None:
    """
    Entry point for the command line interface.

    Parses arguments and dispatches to backtest, optimisation, or live trading
    functions based on the selected mode.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. If None, uses sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backtest", "optimize", "live"], default="backtest")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT", help="Comma separated list for backtests")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--paper", type=str, default="true")
    parser.add_argument("--testnet", type=str, default="true")
    parser.add_argument("--reopt_every", type=int, default=120)
    parser.add_argument("--hist_bars", type=int, default=1000)
    parser.add_argument("--live_mode", choices=["best", "all"], default="best")
    args = parser.parse_args(argv)

    paper = args.paper.lower() == "true"
    testnet = args.testnet.lower() == "true"

    if args.mode == "backtest":
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        interactive_backtest_viewer(syms, args.interval, args.limit, default_choices())
    elif args.mode == "optimize":
        df = fetch_klines(args.symbol, interval=args.interval, limit=args.limit)
        choices = optimise_all(df)
        run_backtest_and_plot(args.symbol, args.interval, args.limit, choices)
    # --- LIVE mode (single entrypoint) ---
    # --- LIVE mode: open exactly ONE window ---
    elif args.mode == "live":
        from trading_bot.strategies import MACDRSI, SMACross, DonchianBreakout
        from trading_bot.live import run_live

        # Prefer --symbols; fall back to --symbol; default BTCUSDT
        raw = (getattr(args, "symbols", None) or getattr(args, "symbol", None) or "BTCUSDT").strip()
        parsed = [s.strip().upper() for s in raw.split(",") if s.strip()]

        # Deduplicate while keeping order
        all_syms = list(dict.fromkeys(parsed))
        if not all_syms:
            all_syms = ["BTCUSDT"]

        # IMPORTANT: only use the FIRST symbol for the live window
        main_sym = all_syms[0]

        # Symbols to show in the right-side control panel (can include extras)
        panel_syms = list(dict.fromkeys(all_syms + ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "BNBUSDT", "XRPUSDT"]))

        print(f"[LIVE] opening 1 window for: {main_sym}")
        # Log the panel list so we can see what's available in the dropdown
        print(f"[LIVE] panel symbols: {', '.join(panel_syms)}")

        run_live(
            main_sym,
            strategies_to_try=[MACDRSI, SMACross, DonchianBreakout],
            paper=(args.paper.lower() == "true") if isinstance(args.paper, str) else bool(args.paper),
            testnet=(args.testnet.lower() == "true") if isinstance(args.testnet, str) else bool(args.testnet),
            reopt_every=int(args.reopt_every),
            hist_bars=int(args.hist_bars),
            loop_sec=10,
            live_mode=args.live_mode,
            interval=args.interval,
            symbols=panel_syms,
        )

        # VERY IMPORTANT: return so we don't fall through and call run_live again
        return
    else:
        parser.print_help()
        return

if __name__ == "__main__":
    main()