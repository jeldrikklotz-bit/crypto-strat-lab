# trading_bot/live.py
from __future__ import annotations

from typing import List, Type, Optional
import traceback

import pandas as pd
import matplotlib.pyplot as plt

from .data import fetch_klines
from .strategies import StrategyBase
from .optimizer import grid_search_one
from .broker import BinanceBroker
from .plotting import _plot_one

__all__ = ["run_live"]

# Keep refs so timers aren’t GC’d; map symbol -> (figure, timer)
_OPEN_FIGS: dict[str, tuple[plt.Figure, any]] = {}

# Try dropdown, fall back to radio buttons
try:
    from matplotlib.widgets import Dropdown
    _WIDGET_DROPDOWN = True
except Exception:
    from matplotlib.widgets import RadioButtons as Dropdown  # type: ignore
    _WIDGET_DROPDOWN = False

from matplotlib.widgets import Button


def _latest_completed_bar(symbol: str, interval: str) -> tuple[pd.Timestamp, pd.Series]:
    df = fetch_klines(symbol, interval=interval, limit=2)
    ts = df.index[-1]
    return ts, df.iloc[-1]


def _defaults_for(cls: Type[StrategyBase]) -> dict:
    n = cls.__name__.lower()
    if "macdrsi" in n:
        return dict(macd_fast=12, macd_slow=26, macd_signal=9,
                    rsi_period=14, rsi_buy=55, rsi_sell=70,
                    trail_pct=0.02, fee=0.001)
    if "smacross" in n or "sma" in n:
        return dict(fast=10, slow=30, rsi_period=14,
                    rsi_filter=50, trail_pct=0.02, fee=0.001)
    if "donchian" in n:
        return dict(ch=20, exit_ch=10, atr_n=14, atr_mult=2.0, fee=0.001)
    return dict(fee=0.001)


def _instantiate_for_mode(
    strategies_to_try: List[Type[StrategyBase]],
    mode: str,
    pick: Optional[Type[StrategyBase]] = None,
) -> List[StrategyBase]:
    if mode == "all":
        return [Cls(sid=i + 1, **_defaults_for(Cls)) for i, Cls in enumerate(strategies_to_try)]
    if pick is None:
        pick = strategies_to_try[0]
    return [pick(sid=1, **_defaults_for(pick))]


def _label_for_strategy_class(Cls: Type[StrategyBase]) -> str:
    name = getattr(Cls, "__name__", "Strategy")
    return name.replace("Strategy", "").replace("_", " ").strip()


def run_live(
    symbol: str,
    strategies_to_try: List[Type[StrategyBase]],
    *,
    paper: bool = True,
    testnet: bool = True,
    reopt_every: int = 120,      # bars
    hist_bars: int = 1000,
    loop_sec: int = 10,          # seconds
    live_mode: str = "best",     # "best" or "all"
    interval: str = "1m",
    symbols: Optional[List[str]] = None,
) -> None:
    key = symbol.upper()

    # If a window for this symbol already exists and is alive, just focus it
    tup = _OPEN_FIGS.get(key)
    if tup is not None:
        fig, _ = tup
        try:
            if plt.fignum_exists(fig.number):
                try:
                    mgr = fig.canvas.manager
                    # Try to bring to front across common backends
                    if hasattr(mgr, "window"):
                        mgr.window.deiconify()  # TkAgg
                        mgr.window.lift()
                    fig.canvas.manager.show()
                except Exception:
                    pass
                print(f"[LIVE] window already open for {key}; focusing it")
                return
        except Exception:
            # stale handle, drop it
            _OPEN_FIGS.pop(key, None)

    print(f"[LIVE] {key}  mode={live_mode}  interval={interval}  paper={paper}")
    print(f"[LIVE] backend={plt.get_backend()}")

    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "BNBUSDT", "XRPUSDT"]

    # --- state ---
    state = {
        "symbol": key,
        "mode": live_mode,
        "strats": [],                  # list[StrategyBase]
        "broker": None,                # BinanceBroker
        "df": None,                    # DataFrame OHLCV
        "last_ts": None,
        "bars_since_opt": 0,
        "timer": None,
        "interval": interval,
        "hist_bars": int(hist_bars),
        "reopt_every": int(reopt_every),
    }

    def _load_history(sym: str) -> pd.DataFrame:
        df = fetch_klines(sym, interval=state["interval"], limit=state["hist_bars"]).copy()
        if df.empty:
            raise RuntimeError(f"No history for {sym}")
        return df

    def _reset_runtime(sym: str, pick_cls: Optional[Type[StrategyBase]], mode: str):
        state["symbol"] = sym.upper()
        state["mode"] = mode
        print(f"[LIVE] reset → {state['symbol']} | mode={state['mode']} | pick={getattr(pick_cls, '__name__', 'auto')}")
        state["df"] = _load_history(state["symbol"])
        state["last_ts"] = state["df"].index[-1]
        state["bars_since_opt"] = 0
        state["broker"] = BinanceBroker(state["symbol"], fee=0.001, paper=paper, testnet=testnet)
        state["strats"].clear()
        state["strats"].extend(_instantiate_for_mode(strategies_to_try, state["mode"], pick_cls))

    _reset_runtime(key, None if live_mode == "all" else strategies_to_try[0], live_mode)

    # --- figure (create first, THEN register it as open) ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        fig.canvas.manager.set_window_title(f"{state['symbol']} — Live")
    except Exception:
        pass
    fig.subplots_adjust(left=0.08, right=0.74, top=0.92, bottom=0.10)

    # register the figure now that it exists
    _OPEN_FIGS[key] = (fig, None)

    # --- right-side panel ---
    ax_sym  = plt.axes([0.76, 0.78, 0.22, 0.08])
    ax_str  = plt.axes([0.76, 0.66, 0.22, 0.08])
    ax_apply = plt.axes([0.76, 0.56, 0.10, 0.05])

    sym_labels = [s.upper() for s in symbols]
    strat_labels = ["[All]"] + [_label_for_strategy_class(Cls) for Cls in strategies_to_try]

    if _WIDGET_DROPDOWN:
        dd_sym = Dropdown(ax_sym, "Symbol", sym_labels, value=state["symbol"])
        dd_str = Dropdown(ax_str, "Strategy",
                          strat_labels,
                          value=("[All]" if state["mode"] == "all"
                                 else _label_for_strategy_class(type(state["strats"][0]))))
    else:
        from matplotlib.widgets import RadioButtons
        dd_sym = RadioButtons(ax_sym, labels=sym_labels, active=sym_labels.index(state["symbol"]))
        active_idx = 0 if state["mode"] == "all" else strat_labels.index(
            _label_for_strategy_class(type(state["strats"][0]))
        )
        dd_str = RadioButtons(ax_str, labels=strat_labels, active=active_idx)
    btn_apply = Button(ax_apply, "Apply")

    # --- draw helpers ---
    def _draw_first():
        ax.clear()
        tdf = state["df"].tail(600)
        _plot_one(ax, tdf, state["strats"][0])
        ax.set_title(f"{state['symbol']}  |  Equity {state['broker'].equity(float(tdf['close'].iloc[-1])):.2f}  |  Pos {state['broker'].position:.6f}")
        fig.canvas.draw_idle()
        plt.pause(0.01)

    def _reopt_short():
        try:
            look = min(400, len(state["df"]))
            train = state["df"].iloc[-look:]
            if state["mode"] == "all":
                new_list: List[StrategyBase] = []
                for i, s in enumerate(state["strats"]):
                    Cls = type(s)
                    p, _ = grid_search_one(train, Cls, fee=0.001)
                    new_list.append(Cls(sid=i + 1, **p))
                state["strats"][:] = new_list
            else:
                Cls = type(state["strats"][0])
                p, _ = grid_search_one(train, Cls, fee=0.001)
                state["strats"][:] = [Cls(sid=1, **p)]
        except Exception:
            traceback.print_exc()

    # --- timer tick ---
    def on_tick(*_):
        try:
            ts, row = _latest_completed_bar(state["symbol"], state["interval"])
            if ts <= state["last_ts"]:
                return

            df = state["df"]
            df.loc[ts, ["open", "high", "low", "close", "volume"]] = [
                float(row["open"]), float(row["high"]), float(row["low"]),
                float(row["close"]), float(row["volume"])
            ]
            if len(df) > state["hist_bars"]:
                state["df"] = df = df.iloc[-state["hist_bars"]:]
            state["last_ts"] = ts

            cs = df["close"]
            for s in state["strats"]:
                s.step(ts, float(row["close"]), cs, df_full=df)

            signals = [s.trades[-1][1] if s.trades and s.trades[-1][0] == ts else None for s in state["strats"]]
            if state["mode"] == "best":
                action = signals[0]
            else:
                buys = sum(sig == "BUY" for sig in signals)
                sells = sum(sig == "SELL" for sig in signals)
                action = "BUY" if buys > len(state["strats"]) // 2 else "SELL" if sells > len(state["strats"]) // 2 else None

            px = float(row["close"])
            if action == "BUY" and state["broker"].position == 0:
                state["broker"].market_buy(state["broker"].cash / px, price_hint=px)
            elif action == "SELL" and state["broker"].position > 0:
                state["broker"].market_sell(state["broker"].position, price_hint=px)

            tdf = df.tail(600)
            ax.clear()
            if state["mode"] == "all":
                ax.plot(tdf.index, tdf["close"], linewidth=0.7, label="close")
                for s in state["strats"]:
                    _plot_one(ax, tdf, s)
            else:
                _plot_one(ax, tdf, state["strats"][0])

            ax.set_title(f"{state['symbol']}  |  Equity {state['broker'].equity(px):.2f}  |  Pos {state['broker'].position:.6f}")
            fig.canvas.draw_idle()

            state["bars_since_opt"] += 1
            if state["bars_since_opt"] >= state["reopt_every"]:
                state["bars_since_opt"] = 0
                _reopt_short()
        except Exception:
            traceback.print_exc()

    # --- widget callbacks ---
    pending = {
        "symbol": state["symbol"],
        "strategy": "[All]" if state["mode"] == "all" else _label_for_strategy_class(type(state["strats"][0])),
    }

    def _on_symbol_change(val):
        pending["symbol"] = str(val)

    def _on_strategy_change(val):
        pending["strategy"] = str(val)

    def _apply(_=None):
        # stop timer first
        t = state.get("timer")
        if t is not None:
            try: t.stop()
            except Exception: pass
            state["timer"] = None

        label = pending["strategy"]
        if label == "[All]":
            pick_cls = None
            mode = "all"
        else:
            pick_cls = None
            for Cls in strategies_to_try:
                if _label_for_strategy_class(Cls) == label:
                    pick_cls = Cls
                    break
            mode = "best"

        _reset_runtime(pending["symbol"], pick_cls, mode)
        try:
            fig.canvas.manager.set_window_title(f"{state['symbol']} — Live")
        except Exception:
            pass

        _draw_first()
        new_t = fig.canvas.new_timer(interval=max(200, int(loop_sec * 1000)))
        new_t.add_callback(on_tick)
        new_t.start()
        state["timer"] = new_t
        _OPEN_FIGS[state["symbol"]] = (fig, new_t)

    # wire widgets
    if _WIDGET_DROPDOWN:
        dd_sym.on_change(_on_symbol_change)
        dd_str.on_change(_on_strategy_change)
    else:
        dd_sym.on_clicked(_on_symbol_change)
        dd_str.on_clicked(_on_strategy_change)
    btn_apply.on_clicked(_apply)

    # first frame + timer
    _draw_first()
    timer = fig.canvas.new_timer(interval=max(200, int(loop_sec * 1000)))
    timer.add_callback(on_tick)
    timer.start()
    state["timer"] = timer
    _OPEN_FIGS[key] = (fig, timer)  # update with timer

    # on close: stop timer and unregister this figure
    def _on_close(_event):
        t = state.get("timer")
        if t is not None:
            try: t.stop()
            except Exception: pass
            state["timer"] = None
        _OPEN_FIGS.pop(state["symbol"], None)
        print(f"[LIVE] closed {state['symbol']} window")

    fig.canvas.mpl_connect("close_event", _on_close)

    # show blocking (your setup prefers block=True)
    plt.show(block=True)
