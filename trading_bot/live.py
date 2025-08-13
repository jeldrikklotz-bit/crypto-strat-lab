"""
Live trading loop for the trading bot.

This module provides a function to run a live trading session using one
or more strategies. It fetches real‑time price data, periodically
reoptimises strategy parameters based on recent history, and executes
trades via the :class:`trading_bot.broker.BinanceBroker`. The loop can run
in paper mode or against the Binance testnet.
"""

from __future__ import annotations

import time
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple, Type

from .data import fetch_klines, stream_latest_close
from .strategies import StrategyBase
from .optimizer import grid_search_one
from .backtest import run_backtest_multi
from .broker import BinanceBroker

__all__ = ["run_live"]


def run_live(
    symbol: str,
    strategies_to_try: List[Type[StrategyBase]],
    *,
    paper: bool = True,
    testnet: bool = True,
    reopt_every: int = 120,
    hist_bars: int = 1000,
    loop_sec: int = 10,
    live_mode: str = "best",
) -> None:
    """Run a live trading loop on the given symbol.

    Parameters
    ----------
    symbol : str
        Trading pair to trade.
    strategies_to_try : list of strategy classes
        List of strategy classes to consider for live trading. If
        ``live_mode`` is ``'best'``, only the top scoring strategy is
        traded. If ``'all'``, a majority vote across strategies is used.
    paper : bool, optional
        If True, simulate trades without sending orders. Default True.
    testnet : bool, optional
        If True and not paper, use the Binance testnet. Default True.
    reopt_every : int, optional
        Number of bars to wait between reoptimising the strategy parameters.
    hist_bars : int, optional
        Number of historical candles to use for initial optimisation and
        subsequent reoptimisations.
    loop_sec : int, optional
        Delay in seconds between polling for new price data.
    live_mode : str, optional
        Either 'best' (trade the single best strategy) or 'all' (vote across
        all strategies). Default 'best'.
    """
    print(f"Live on {symbol}, paper={paper}, mode={live_mode}")
    df = fetch_klines(symbol, interval="1m", limit=hist_bars)
    # Pick best initial strategy
    best_params = None
    best_m = None
    best_score = -1e9
    for Cls in strategies_to_try:
        params, metrics = grid_search_one(df, Cls, fee=0.001)
        sc = metrics.get("sharpe", 0)  # quick pick by Sharpe; more advanced scoring available
        if sc > best_score:
            best_params = (Cls, params)
            best_m = metrics
            best_score = sc
    assert best_params is not None and best_m is not None
    print("Chosen:", best_params[0].__name__, best_m)

    broker = BinanceBroker(symbol, fee=0.001, paper=paper, testnet=testnet)
    if live_mode == "all":
        strats = [Cls(sid=i + 1, **grid_search_one(df, Cls, fee=0.001)[0]) for i, Cls in enumerate(strategies_to_try)]
    else:
        Cls, params = best_params
        strats = [Cls(sid=1, **params)]

    closes = df["close"].copy()
    fig, ax = plt.subplots()
    bars_since_opt = 0
    while True:
        try:
            ts, px = stream_latest_close(symbol)
            closes.loc[ts] = px
            if len(closes) > hist_bars:
                closes = closes.iloc[-hist_bars:]
            bars_since_opt += 1
            if bars_since_opt >= reopt_every:
                df_train = fetch_klines(symbol, interval="1m", limit=hist_bars)
                if live_mode == "best":
                    Cls, params = best_params
                    params, _ = grid_search_one(df_train, Cls, fee=0.001)
                    strats = [Cls(sid=1, **params)]
                else:
                    strats = [Cls(sid=i + 1, **grid_search_one(df_train, Cls, fee=0.001)[0]) for i, Cls in enumerate(strategies_to_try)]
                bars_since_opt = 0
            cs = pd.Series(closes.values, index=closes.index)
            signals = []
            for s in strats:
                sig = s.step(ts, px, cs, df_full=None)
                signals.append(sig)
            # Determine action
            do = None
            if live_mode == "best":
                do = signals[0]
            else:
                buys = signals.count("BUY")
                sells = signals.count("SELL")
                if buys > len(strats) // 2:
                    do = "BUY"
                if sells > len(strats) // 2:
                    do = "SELL"
            if do == "BUY" and broker.position == 0:
                broker.market_buy(broker.cash / px, price_hint=px)
            elif do == "SELL" and broker.position > 0:
                broker.market_sell(broker.position, price_hint=px)
            # Update plot
            ax.clear()
            tail = closes.tail(200)
            ax.plot(tail.index, tail.values, linewidth=0.7, label="price")
            ax.set_title(f"{symbol}  equity {broker.equity(px):.2f}  pos {broker.position:.6f}")
            ax.legend()
            plt.pause(0.01)
            time.sleep(loop_sec)
        except KeyboardInterrupt:
            print("Stopped")
            break
        except Exception as exc:
            print("Live error:", exc)
            time.sleep(loop_sec)