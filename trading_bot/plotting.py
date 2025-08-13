"""
Plotting helpers for the trading bot.

This module includes functions to display trading performance, including
annotating buy/sell trades with numbered markers and profit/loss labels.
It is used by the CLI and interactive viewer to visualise backtest results.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .base_strategy import StrategyBase
from typing import List, Dict

__all__ = ["plot_with_numbered_trades", "_plot_one"]


def _pair_trades(trades: List[tuple], fee: float) -> List[Dict[str, float]]:
    """Return a list of dictionaries for consecutive BUY->SELL pairs.

    Each dict contains the buy and sell trades, the quantity used, cost,
    proceeds, net PnL, percent PnL, and total fees paid. Only closed pairs
    are considered; an unmatched BUY at the end is ignored.
    """
    out = []
    i = 0
    n = len(trades)
    while i + 1 < n:
        b = trades[i]
        s = trades[i + 1]
        if b[1] == "BUY" and s[1] == "SELL":
            _, _, pb, qb, _ = b
            _, _, ps, qs, _ = s
            qty = min(qb, qs) if qs > 0 else qb
            cost = qty * pb * (1 + fee)
            proceeds = qty * ps * (1 - fee)
            pnl_abs = proceeds - cost
            pnl_pct = (proceeds / cost - 1.0) * 100.0
            fees_abs = qty * pb * fee + qty * ps * fee
            out.append(
                dict(
                    buy=b,
                    sell=s,
                    qty=qty,
                    cost=cost,
                    proceeds=proceeds,
                    pnl_abs=pnl_abs,
                    pnl_pct=pnl_pct,
                    fees=fees_abs,
                )
            )
            i += 2
        else:
            i += 1
    return out


def _format_money(x: float) -> str:
    """Format a number as a money string with a +/- sign and thousands separators."""
    sign = "+" if x >= 0 else "−"
    return f"{sign}${abs(x):,.2f}"


def _plot_one(ax: plt.Axes, df, strategy_obj: StrategyBase) -> None:
    """Plot a single strategy's trades and PnL on an axis.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes object to plot on. Will be cleared first.
    df : pandas.DataFrame
        Historical data (with ``close`` column) used for the backtest.
    strategy_obj : StrategyBase
        Strategy instance with recorded trades and equity.
    """
    ax.clear()
    ax.plot(df.index, df["close"], linewidth=0.7, label="close")
    
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y %H:%M'))

    # Plot markers with numbered labels
    for (t, side, price, qty, sid) in strategy_obj.trades:
        ax.scatter(t, price, s=28, c=("green" if side == "BUY" else "red"), marker=("^" if side == "BUY" else "v"), zorder=3)
        ax.text(
            t,
            price * (1.0006 if side == "BUY" else 0.9994),
            str(sid),
            fontsize=7,
            ha="center",
            va=("bottom" if side == "BUY" else "top"),
        )
    # Fee‑aware per‑pair labels
    pairs = _pair_trades(strategy_obj.trades, strategy_obj.fee)
    realized = 0.0
    total_fees = 0.0
    for p in pairs:
        tb, _, pb, _, _ = p["buy"]
        ts, _, ps, _, _ = p["sell"]
        t_mid = tb + (ts - tb) / 2
        p_mid = (pb + ps) / 2
        realized += p["pnl_abs"]
        total_fees += p["fees"]
        color = "green" if p["pnl_abs"] >= 0 else "red"
        ax.text(
            t_mid,
            p_mid,
            f"{_format_money(p['pnl_abs'])}\n{p['pnl_pct']:+.2f}%",
            fontsize=7,
            ha="center",
            va="center",
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.6, alpha=0.85),
        )
    # Unrealised PnL if open
    unrealized = 0.0
    if strategy_obj.trades and strategy_obj.trades[-1][1] == "BUY":
        last_buy = strategy_obj.trades[-1]
        _, _, pb, qb, _ = last_buy
        last_px = float(df["close"].iloc[-1])
        unrealized = qb * (last_px - pb * (1 + strategy_obj.fee))
    m = strategy_obj.metrics()
    final_eq = m.get("final_equity", 0.0)
    total_pnl_calc = realized + unrealized
    ax.text(
        0.995,
        0.02,
        f"Realized {_format_money(realized)}   |   "
        f"Unrealized {_format_money(unrealized)}   |   "
        f"Fees {_format_money(-total_fees)}\n"
        f"Total {_format_money(total_pnl_calc)}   |   "
        f"Equity ${final_eq:,.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#777", lw=0.6, alpha=0.9),
        fontsize=8,
    )
    ax.set_title(f"{strategy_obj.name}  |  {df.index[0].date()} → {df.index[-1].date()}")
    ax.legend(loc="upper left")
    ax.figure.autofmt_xdate()
    ax.figure.canvas.draw_idle()


def plot_with_numbered_trades(df, strategies: List[StrategyBase], title_suffix: str = "") -> None:
    """Plot backtest results for multiple strategies on a single figure.

    Each strategy's trades are plotted with green/red markers and numbered by
    the strategy ID. A legend shows the mapping from numbers to strategy
    names. The close price series is plotted once.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical data used for the backtest.
    strategies : list of StrategyBase
        Strategies to plot.
    title_suffix : str, optional
        Additional text to append to the figure title.
    """
    fig, ax = plt.subplots()
    ax.plot(df.index, df["close"], linewidth=0.7, label="close")
    for s in strategies:
        for (t, side, price, qty, sid) in s.trades:
            ax.scatter(t, price, s=28, c=("green" if side == "BUY" else "red"), marker=("^" if side == "BUY" else "v"), zorder=3)
            ax.text(t, price * (1.0006 if side == "BUY" else 0.9994), str(sid), fontsize=7, ha="center", va=("bottom" if side == "BUY" else "top"))
    legend_txt = " | ".join([f"{s.sid}:{s.name.split(':',1)[1]}" for s in strategies])
    ax.set_title(f"{title_suffix}\n{legend_txt}")
    fig.autofmt_xdate()
    plt.show(block=True)