"""
Interactive backtest viewer for the trading bot.

This module defines a function that launches an interactive plot with two
dropdowns or radio buttons (depending on Matplotlib version) to select
a symbol and a strategy. It supports adjusting the trading fee on the fly
and updates the backtest results accordingly. The viewer remembers the
selected strategy across symbol changes and recomputes only the current
symbol when the fee is changed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

try:
    from matplotlib.widgets import Dropdown  # type: ignore
    HAVE_DROPDOWN = True
except Exception:
    from matplotlib.widgets import RadioButtons  # type: ignore
    HAVE_DROPDOWN = False

from typing import List, Tuple, Dict, Any

from .data import fetch_klines
from .backtest import run_backtest_multi
from .strategies import StrategyBase
from .plotting import _plot_one

__all__ = ["interactive_backtest_viewer"]


def interactive_backtest_viewer(
    symbols: List[str], interval: str, limit: int, chosen_specs: List[Tuple[type, Dict[str, Any]]]
) -> None:
    """Launch an interactive viewer for multiple symbols and strategies.

    Parameters
    ----------
    symbols : list of str
        Symbols to make available in the selector (e.g. ['BTCUSDT', 'ETHUSDT']).
    interval : str
        Candle interval to fetch, such as '1m' or '5m'.
    limit : int
        Number of candles to retrieve for each symbol.
    chosen_specs : list
        List of (StrategyClass, params_dict) pairs to backtest.
    """
    # Fetch data once for each symbol
    data: Dict[str, Any] = {sym: fetch_klines(sym, interval=interval, limit=limit) for sym in symbols}
    # Helper to run backtests for a specific fee
    def with_fee(specs: List[Tuple[type, Dict[str, Any]]], fee: float) -> List[Tuple[type, Dict[str, Any]]]:
        return [(Cls, {**params, "fee": fee}) for Cls, params in specs]

    current_fee = 0.0  # default fee
    results: Dict[str, List[StrategyBase]] = {
        sym: run_backtest_multi(data[sym], with_fee(chosen_specs, current_fee)) for sym in symbols
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.08, right=0.75, top=0.9, bottom=0.1)  
    # main plot only takes ~75% width now

    # right-side control positions
    ax_sym  = plt.axes([0.78, 0.75, 0.2, 0.15])   # crypto dropdown
    ax_str  = plt.axes([0.78, 0.65, 0.2, 0.05])   # strategy dropdown
    ax_fee  = plt.axes([0.78, 0.55, 0.1, 0.04])   # fee text box


    sym_options = [s.upper() for s in symbols]
    # Helper to get strategy labels for a symbol
    def labels_for(sym: str) -> List[str]:
        return [f"{s.sid}: {s.name.split(':', 1)[1]}" for s in results[sym]]

    # Create selection widgets
    if HAVE_DROPDOWN:
        dd_sym = Dropdown(ax_sym, "Symbol", sym_options)
        dd_str = Dropdown(ax_str, "Strategy", labels_for(sym_options[0]))
        dd_str_dropdown = dd_str  # keep a reference for dropdown case
    else:
        dd_sym = RadioButtons(ax_sym, labels=sym_options, active=0)
        dd_str = RadioButtons(ax_str, labels=labels_for(sym_options[0]), active=0)
        dd_str_dropdown = None

    fee_box = TextBox(ax_fee, "Fee", initial=f"{current_fee:.4f}")

    current_sym = sym_options[0]
    strategy_keys = [Cls.__name__ for Cls, _ in chosen_specs]
    current_key = strategy_keys[0]

    def current_index() -> int:
        try:
            return strategy_keys.index(current_key)
        except ValueError:
            return 0

    def set_strategy_selector(idx: int) -> None:
        nonlocal dd_str, ax_str, dd_str_dropdown
        labels = labels_for(current_sym)
        idx = max(0, min(idx, len(labels) - 1))
        if HAVE_DROPDOWN:
            dd_str_dropdown.options = labels
            dd_str_dropdown.value = labels[idx]
            dd_str = dd_str_dropdown
        else:
            # Remove old axis and create a new RadioButtons widget
            pos = ax_str.get_position()
            fig.delaxes(ax_str)
            ax_str = fig.add_axes(pos)
            new_rb = RadioButtons(ax_str, labels=labels, active=idx)
            new_rb.on_clicked(on_change_str)
            dd_str = new_rb
            fig.canvas.draw_idle()
            plt.pause(0.01)

    def title_mapping() -> str:
        return " | ".join([f"{s.sid}:{s.name.split(':',1)[1]}" for s in results[current_sym]])

    def refresh() -> None:
        idx = current_index()
        idx = max(0, min(idx, len(results[current_sym]) - 1))
        strat = results[current_sym][idx]
        _plot_one(ax, data[current_sym], strat)
        fig.suptitle(
            f"{current_sym} — {title_mapping()}  |  fee={current_fee:.4%}", y=0.95
        )

    def on_change_sym(val: Any) -> None:
        nonlocal current_sym
        current_sym = str(val)
        set_strategy_selector(current_index())
        refresh()

    def on_change_str(val: Any) -> None:
        nonlocal current_key
        label = str(val)
        try:
            idx = int(label.split(":")[0].strip()) - 1
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(strategy_keys) - 1))
        current_key = strategy_keys[idx]
        refresh()

    def parse_fee(text: str) -> float:
        t = str(text).strip().replace(",", ".")
        if t.endswith("%"):
            t = t[:-1].strip()
            v = float(t) / 100.0
        else:
            v = float(t)
        return max(0.0, min(v, 0.05))

    def on_fee_submit(text: str) -> None:
        nonlocal current_fee, results
        try:
            new_fee = parse_fee(text)
        except Exception:
            fee_box.set_val(f"{current_fee:.4f}")
            return
        if abs(new_fee - current_fee) < 1e-9:
            return
        current_fee = new_fee
        # Recompute only the current symbol with the new fee
        results[current_sym] = run_backtest_multi(
            data[current_sym], with_fee(chosen_specs, current_fee)
        )
        set_strategy_selector(current_index())
        refresh()

    # Wire events
    if HAVE_DROPDOWN:
        dd_sym.on_change(on_change_sym)
        dd_str.on_change(on_change_str)
    else:
        dd_sym.on_clicked(on_change_sym)
        dd_str.on_clicked(on_change_str)
    fee_box.on_submit(on_fee_submit)

    # Initial draw
    set_strategy_selector(current_index())
    refresh()
    plt.show(block=True)