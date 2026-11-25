"""Every position's path from the day it was bought, under one sell threshold.

For the held strategy (positions stay until the rule sells them), each stock
position is one line: it starts at zero on the day the stock was first bought
and runs until the rule sells it or the span ends. The x axis is trading days
since that first buy, so positions bought at different times line up. Returns
are capped at the threshold, which is where a sale happens.

Four panels, one per six months of the span, each showing the positions first
bought in that period. It also reports how many positions are never sold.

    python notebooks/take_profit_paths.py [span in years, default 2] [gamma, default 40] [threshold %, default 10]

Needs the same cached prices and recommendations as take_profit_heatmap.py.
Writes outputs/take_profit_paths.png. Not financial advice.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

SPAN = sys.argv[1] if len(sys.argv) > 1 else "2"
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
THRESHOLD = (float(sys.argv[3]) if len(sys.argv) > 3 else 10.0) / 100
os.environ["STONKS_BACKTEST_SPAN"] = SPAN          # read by the backtest module
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import take_profit_backtest as tb

OUTPUT = Path(__file__).resolve().parent.parent / "outputs"
PANELS, MONTHS_EACH = 4, 6
FLOOR = -70.0                                       # the lowest return shown, in percent


def episodes_from(result: dict, dates: pd.DatetimeIndex) -> list[dict]:
    """One record per position: from the first day a stock is held to the day it is sold."""
    sales = {(dates.get_loc(date), int(col)): float(ret)
             for date, col, _, _, ret in result["sell log"]}
    live: dict[int, dict] = {}
    episodes: list[dict] = []
    for day, columns, returns in result["snapshots"]:
        present = set(columns.tolist())
        for col in [c for c in live if c not in present]:
            episode = live.pop(col)
            episode.update(end=day, sold=(day, col) in sales, sale_return=sales.get((day, col)))
            episodes.append(episode)
        for col, ret in zip(columns, returns):
            col = int(col)
            if col in live:
                live[col]["points"].append((day, float(ret)))
            else:
                live[col] = {"col": col, "start": day, "points": [(day, float(ret))]}
    for episode in live.values():                   # still held on the last day
        episode.update(end=day, sold=False, sale_return=None)
        episodes.append(episode)
    return episodes


def main() -> None:
    tb.ensure_prices()
    close = pd.read_parquet(tb._cache("close")).sort_index()
    stocks = [t for t in close.columns if t != "SPY"]
    dates = close.index
    buy_days = tb._buy_days(dates, "biweekly")
    per_buy = tb.YEARLY / len(buy_days)
    end = int(dates.searchsorted(tb.END))

    tb.GAMMA, tb.FLIP, tb.CADENCE, tb.TAKE, tb.ALGO = GAMMA, False, "biweekly", THRESHOLD, "Algorithm"
    algo = tb.load_weights(dates, buy_days)
    result = tb._simulate(stocks, algo, buy_days, end, "close", per_buy, True, record=True)
    episodes = episodes_from(result, dates)
    for episode in episodes:
        first = dates[buy_days[0]]
        episode["panel"] = ((dates[episode["start"]].year - first.year) * 12
                            + dates[episode["start"]].month - first.month) // MONTHS_EACH
        episode["length"] = episode["end"] - episode["start"]
    cap = 100 * THRESHOLD

    sold = [e for e in episodes if e["sold"]]
    kept = [e for e in episodes if not e["sold"]]
    print(f"gamma {GAMMA:g}, normal, held | sell at +{cap:g}% (close check) | {dates[buy_days[0]].date()} to "
          f"{dates[end].date()} | account {result['pnl'] / tb.YEARLY:+.1%}")
    print(f"\npositions (a stock counted again each time it is re-bought after a sale): {len(episodes)}")
    print(f"  sold by the rule: {len(sold)} ({len(sold) / len(episodes):.1%}) | never sold, still held on "
          f"{dates[end].date()}: {len(kept)} ({len(kept) / len(episodes):.1%})")
    if sold:
        days = np.array([e["length"] for e in sold])
        print(f"  days from first buy to the sale: median {np.median(days):.0f}, "
              f"half of them between {np.percentile(days, 25):.0f} and {np.percentile(days, 75):.0f}")
    if kept:
        last = np.array([e["points"][-1][1] for e in kept]) * 100
        print(f"  the never-sold positions end at a median of {np.median(last):+.1f}% "
              f"(a quarter of them below {np.percentile(last, 25):+.1f}%)")

    print("\nshare still unsold after a given number of trading days since the first buy:")
    for horizon in (21, 63, 126, 252):
        # every sold position has an exact sale day, so it counts at every horizon; a
        # never-sold position only counts once it has been followed that long
        followed = [e for e in episodes if e["sold"] or e["length"] >= horizon]
        alive = [e for e in followed if not e["sold"] or e["length"] > horizon]
        print(f"  after {horizon:>3} days: {len(alive) / len(followed):.0%} of {len(followed)}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=True)
    for panel, ax in enumerate(axes.flat):
        group = [e for e in episodes if e["panel"] == panel]
        gone = [e for e in group if e["sold"]]
        stay = [e for e in group if not e["sold"]]

        def line(episode: dict, finish: bool) -> np.ndarray:
            points = [(d - episode["start"], max(min(r * 100, cap), FLOOR)) for d, r in episode["points"]]
            if finish:
                points.append((episode["length"], cap))
            return np.array(points)

        ax.add_collection(LineCollection([line(e, False) for e in stay], colors="#e8600a", alpha=0.30, linewidths=1.2))
        ax.add_collection(LineCollection([line(e, True) for e in gone], colors="#1f6fb4", alpha=0.20, linewidths=1.0))
        longest = max((e["length"] for e in group), default=0)
        ax.set_xlim(-3, longest + 3)
        ax.set_ylim(FLOOR, cap * 1.08)
        ax.axhline(cap, color="black", linestyle="--", linewidth=1)
        ax.axhline(0, color="grey", linewidth=0.6)
        span = max((e["end"] for e in group), default=0)
        ax.set_title(f"Bought {dates[min(e['start'] for e in group)].date()} to {dates[max(e['start'] for e in group)].date()} "
                     f"(followed until {dates[span].date()})\n{len(group)} positions: {len(gone)} sold at +{cap:g}%, "
                     f"{len(stay)} never sold ({len(stay) / len(group):.0%})", fontsize=10.5)
        ax.set_xlabel("trading days since the stock was first bought", fontsize=9)
        ax.set_ylabel("return since bought, %", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="lightgrey", linewidth=0.5)
        handles = [Line2D([], [], color="#1f6fb4", alpha=0.9, label=f"sold at +{cap:g}%"),
                   Line2D([], [], color="#e8600a", label="never sold")]
        ax.legend(handles=handles, loc="lower left", fontsize=9)
    fig.suptitle(f"gamma {GAMMA:g}, normal, held until sold: each line is one stock position from the day it was bought\n"
                 f"return capped at +{cap:g}%; {len(episodes)} positions over {tb.SPAN} year{'s' if tb.SPAN > 1 else ''}, "
                 f"{len(kept) / len(episodes):.0%} never sold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUTPUT.mkdir(exist_ok=True)
    stem = "take_profit_paths" + ("" if tb.SPAN == 1 else f"_span{tb.SPAN}")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=100)
    print(f"\nsaved {OUTPUT / f'{stem}.png'}")

    # The same positions on the calendar: a line starts on the day its stock is
    # bought, so new lines keep appearing along the timeline.
    import matplotlib.dates as mdates

    axis = mdates.date2num(dates[: end + 1].to_pydatetime())      # day index -> position on the date axis

    def timeline(episode: dict, finish: bool) -> np.ndarray:
        points = [(axis[d], max(min(r * 100, cap), FLOOR)) for d, r in episode["points"]]
        if finish:
            points.append((axis[episode["end"]], cap))
        return np.array(points)

    fig, ax = plt.subplots(figsize=(20, 9))
    for day in buy_days:                                          # one faint mark per buy day
        ax.axvline(axis[day], color="grey", linewidth=0.6, alpha=0.35)
    ax.add_collection(LineCollection([timeline(e, False) for e in kept], colors="#e8600a", alpha=0.30, linewidths=1.2))
    ax.add_collection(LineCollection([timeline(e, True) for e in sold], colors="#1f6fb4", alpha=0.20, linewidths=1.0))
    ax.set_xlim(axis[buy_days[0]] - 4, axis[end] + 4)
    ax.set_ylim(FLOOR, cap * 1.08)
    ax.axhline(cap, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="lightgrey", linewidth=0.5)
    ax.set_ylabel("return since the stock was bought, %", fontsize=9)
    ax.legend(handles=[Line2D([], [], color="#1f6fb4", alpha=0.9, label=f"sold at +{cap:g}%"),
                       Line2D([], [], color="#e8600a", label="never sold"),
                       Line2D([], [], color="grey", alpha=0.5, label="a buy day (every other Monday)")],
              loc="lower left", fontsize=9)
    ax.set_title(f"gamma {GAMMA:g}, normal, held until sold, on the calendar: {dates[buy_days[0]].date()} to "
                 f"{dates[end].date()}\neach line starts on the day its stock was first bought; "
                 f"return capped at +{cap:g}%; {len(kept) / len(episodes):.0%} of {len(episodes)} positions never sold",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}_timeline.png", dpi=100)
    print(f"saved {OUTPUT / f'{stem}_timeline.png'}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
