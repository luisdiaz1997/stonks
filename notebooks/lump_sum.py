"""One sum in from the start, everything sold and re-bought every two weeks, for each way of picking.

The heatmaps so far put a fixed sum in at every buy day. Here the whole of it
goes in at the first one and nothing follows, as it would for someone with
savings and no income to add. Every two weeks the account is sold, winners and
losers alike, and the new recommendation is bought with all of it; inside each
two-week stretch a position is also sold once its return reaches the
threshold. It is the "everything sold every buy day" row of
take_profit_heatmap.py, with one sum instead of many.

The normal, flipped and dip picks are compared at one gamma (40 unless given):
their sweeps as rows of one heatmap, how each account got there on the same
axes, and a one-year sum started at every buy day of the first year, because
with one sum the day it goes in matters. The normal picks at γ=160, spread
over more names, close the heatmap as a reference. The picks are the cached
ones.

    python notebooks/lump_sum.py [span in years, default 2] [gamma, default 40]

Writes outputs/lump_sum_sweep.png with the same numbers as a CSV, and
outputs/lump_sum_path.png. Needs the cached prices and recommendations of
take_profit_heatmap.py. Not financial advice.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

SPAN = sys.argv[1] if len(sys.argv) > 1 else "2"
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
os.environ["STONKS_BACKTEST_SPAN"] = SPAN          # read by this module and by its worker processes
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import take_profit_backtest as tb
import take_profit_heatmap as th     # the thresholds and the row labels, so the heatmaps line up

WAYS = {"normal": (False, False), "flipped": (True, False), "buy the dip": (False, True)}   # name -> (flipped, dip)
SPREAD = 160.0                        # the normal picks at this gamma close the heatmap: the same idea, more names
SHOWN = [None, 5, 10, 20]             # the sell thresholds inside the two weeks that are run
PANELS = [(None, "No sell inside the two weeks"), (10, "Also sell at +10%")]   # the two drawn against each other
COLOUR = {"normal": "black", "buy the dip": "#1f6fb4", "flipped": "#e8600a"}
SPY = "SPY, bought once and held"
OUTPUT = Path(__file__).resolve().parent.parent / "outputs"


def name_of(pct: int | None) -> str:
    return "no sell inside the two weeks" if pct is None else f"also sell at +{pct}%"


def run(tickers, weights, buy_days, end, pct: int | None) -> dict:
    """The whole sum in at the first buy day, everything sold and re-bought at every buy day up to ``end``."""
    tb.TAKE = 0.08 if pct is None else pct / 100
    return tb._simulate(tickers, weights, buy_days, end, None if pct is None else "close", tb.YEARLY, True,
                        rebalance=True, lump=True)


def worst_fall(result: dict) -> float:
    """The deepest drop from an earlier close, over the account's daily values."""
    values = result["daily"][1]
    return float((values / np.maximum.accumulate(values) - 1).min())


def main() -> None:
    tb.ensure_prices()
    close = pd.read_parquet(tb._cache("close")).sort_index()
    stocks = [t for t in close.columns if t != "SPY"]
    dates = close.index
    buy_days = tb._buy_days(dates, "biweekly")
    end = int(dates.searchsorted(tb.END))            # the anniversary
    spy = {day: pd.Series({"SPY": 1.0}) for day in buy_days}
    put_in = f"one sum of ${tb.YEARLY:,.0f} in on {dates[buy_days[0]].date()}"
    suffix = "" if tb.SPAN == 1 else f"_span{tb.SPAN}"

    # (gamma, flipped, dip, name); the three ways at GAMMA first, the spread reference last
    settings = [(GAMMA, *change, name) for name, change in WAYS.items()]
    settings.append((SPREAD, False, False, f"normal at gamma {SPREAD:g}"))
    rows, picks = [], {}
    for gamma, flip, dip, name in settings:
        # load_weights reads these from the module; the command line carries the span and the gamma here
        tb.GAMMA, tb.FLIP, tb.DIP, tb.CADENCE = float(gamma), flip, dip, "biweekly"
        picks[name] = tb.load_weights(dates, buy_days)
        for pct in (*th.THRESHOLDS, None):
            result = run(stocks, picks[name], buy_days, end, pct)
            rows.append({"row": th.label(f"{gamma:g}", flip, dip, True), "sell at": "never" if pct is None else f"{pct}%",
                         "return": 100 * result["pnl"] / tb.YEARLY})
    tb.TAKE = 0.08

    table = pd.DataFrame(rows)
    columns = [f"{p}%" for p in th.THRESHOLDS] + ["never"]
    grid = (table.pivot_table(index="row", columns="sell at", values="return")
            .reindex(index=list(dict.fromkeys(table["row"])), columns=columns))

    OUTPUT.mkdir(exist_ok=True)
    table.to_csv(OUTPUT / f"lump_sum_sweep{suffix}.csv", index=False)
    fig, ax = plt.subplots(figsize=(26, 0.8 * len(settings) + 0.6))
    sns.heatmap(grid, ax=ax, annot=True, fmt="+.1f", cmap="RdYlGn", center=0, linewidths=0.4,
                linecolor="white", annot_kws={"size": 8},
                cbar_kws={"label": f"profit or loss on the ${tb.YEARLY:,.0f} put in once, %"})
    ax.axhline(len(WAYS), color="black", linewidth=1.6)               # before the reference row
    ax.axvline(len(th.THRESHOLDS), color="black", linewidth=1.6)      # and before the "never" column
    ax.set_xlabel("Sell a position once its return reaches (everything is sold at the next buy day anyway)", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10, rotation=0)
    ax.set_title(f"{tb.SPAN}-year result on one sum by sell threshold, {dates[buy_days[0]].date()} to {dates[end].date()}: "
                 f"K=25, ${tb.YEARLY:,.0f} in at the first buy and nothing after, everything sold and re-bought every other "
                 f"Monday, sells checked at the close", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"lump_sum_sweep{suffix}.png", dpi=110)
    print(f"saved {OUTPUT / f'lump_sum_sweep{suffix}.png'} ({grid.shape[0]} rows x {grid.shape[1]} columns), "
          f"{put_in}, valued {dates[end].date()}, gamma {GAMMA:g}")
    for row, group in table.groupby("row", sort=False):
        sweep = (group[group["sell at"] != "never"].set_index("sell at")["return"].reindex([f"{p}%" for p in th.THRESHOLDS]))
        never = group[group["sell at"] == "never"]["return"].iloc[0]
        print(f"  {row.replace(chr(10), ', '):<64}no sell inside the two weeks: {never:+.1f} | +10%: {sweep['10%']:+.1f} | "
              f"best {sweep.idxmax()} at {sweep.max():+.1f}")

    # how each account got there, on the same axes
    paths = {name: {pct: run(stocks, picks[name], buy_days, end, pct) for pct in SHOWN} for name in WAYS}
    spy_held = run(["SPY"], spy, buy_days, end, None)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, (pct, title) in zip(axes, PANELS):
        for name in WAYS:
            days, values = paths[name][pct]["daily"]
            ax.plot(days, 100 * (values / tb.YEARLY - 1), color=COLOUR[name], linewidth=2.0, label=name)
        days, values = spy_held["daily"]
        ax.plot(days, 100 * (values / tb.YEARLY - 1), color="grey", linestyle="--", linewidth=1.4, label=SPY)
        ax.axhline(0, color="lightgrey", linewidth=0.8)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", color="lightgrey", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("return on the sum put in once, %", fontsize=10)
    fig.suptitle(f"{put_in[0].upper() + put_in[1:]}, everything sold and re-bought every other Monday, K=25, gamma {GAMMA:g}, "
                 f"valued at each day's close", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"lump_sum_path{suffix}.png", dpi=110)
    print(f"\nsaved {OUTPUT / f'lump_sum_path{suffix}.png'}")

    turns = 2 * (len(buy_days) - 1)                   # every buy day but the first sells all and buys it all back
    for pct, title in PANELS:
        print(f"\n{title.lower()}:")
        print("  worst fall from an earlier close: " + " | ".join(f"{name} {worst_fall(paths[name][pct]):+.1%}" for name in WAYS)
              + f" | SPY {worst_fall(spy_held):+.1%}")
        for months in range(6, 12 * tb.SPAN + 1, 6):
            mark = int(dates.searchsorted(dates[end] - pd.DateOffset(months=12 * tb.SPAN - months)))
            print(f"  valued {dates[mark].date()}, {months} months in: " + " | ".join(
                f"{name} {paths[name][pct]['daily'][1][mark - buy_days[0]] / tb.YEARLY - 1:+.1%}" for name in WAYS)
                + f" | SPY {spy_held['daily'][1][mark - buy_days[0]] / tb.YEARLY - 1:+.1%}")
        for cost in (0.001, 0.005):
            print(f"  with {cost:.1%} a side on each of the {turns} full turns: " + " | ".join(
                f"{name} {(1 + paths[name][pct]['pnl'] / tb.YEARLY) * (1 - cost) ** turns - 1:+.1%}" for name in WAYS))

    # the day the sum goes in: a one-year sum started at every buy day of the first year
    half = len(buy_days) // 2
    print(f"\none-year sums, each started on a different buy day ({half} starts from {dates[buy_days[0]].date()}), "
          f"everything sold and re-bought every other Monday, gamma {GAMMA:g}:")
    spy_scores = np.array([run(["SPY"], spy, buy_days[first:first + half], buy_days[first + half], None)["pnl"] / tb.YEARLY
                           for first in range(half)])
    scores = {(name, pct): np.array([run(stocks, picks[name], buy_days[first:first + half], buy_days[first + half], pct)["pnl"] / tb.YEARLY
                                     for first in range(half)])
              for name in WAYS for pct in (None, 10)}
    for (name, pct), returns in [*scores.items(), ((SPY, None), spy_scores)]:
        low, high = int(returns.argmin()), int(returns.argmax())
        ahead = "" if name == SPY else f"; ahead of SPY over the same year in {int((returns > spy_scores).sum())} of {half} starts"
        print(f"  {name:<24}{name_of(pct):<28} worst {returns[low]:+.1%} (from {dates[buy_days[low]].date()}), "
              f"median {np.median(returns):+.1%}, best {returns[high]:+.1%} (from {dates[buy_days[high]].date()}), "
              f"below zero in {int((returns < 0).sum())} of {half}{ahead}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
