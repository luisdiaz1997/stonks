"""What each way of picking chooses on one day, and how those stocks got there.

Three ways of picking, each at three levels of gamma, on the same day: the
latest close in the cached prices. For each, every stock picked is one line:
its cumulative return from four months before that day, thick in proportion to
its weight. Black is the picks' weighted average, and the largest names are
listed beside the chart with what they had done. Four months are shown for all
three so that they line up on the same days. This shows what each way looks
for on one day; it is not a recommendation.

- normal: the usual objective on the last three months;
- flipped: the expected return negated, so it buys what fell;
- dip: the usual objective on the three months before the last month, plus the
  flipped one on the last month.

    python notebooks/pick_trends.py [gamma [gamma [gamma]]]     (default: 20 40 160)

Writes outputs/picks_normal.png, picks_flipped.png and picks_dip.png. The
recommendations are computed on first use (about half a minute each) and cached
under data/, which git ignores, with their date in the file name.
Not financial advice.
"""

from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import take_profit_backtest as tb

GAMMAS = [float(word) for word in sys.argv[1:]] or [20.0, 40.0, 160.0]
KINDS = {"normal": dict(flip=False, dip=False), "flipped": dict(flip=True, dip=False), "dip": dict(flip=False, dip=True)}
TITLE = {"normal": "Normal", "flipped": "Flipped", "dip": "Buy the dip"}
BLUE, ORANGE, GREY = "#1f6fb4", "#e8600a", "grey"      # up over the rise, down over it, neither
SHOWN = 8                                                # the largest names listed beside each chart
OUTPUT = Path(__file__).resolve().parent.parent / "outputs"


def picks_for(wanted: list[tuple], day: int, date: pd.Timestamp) -> dict:
    """Each wanted recommendation, computed by a pool of workers unless it is cached."""
    paths = {(kind, gamma): tb._cache(f"picks_{kind}_g{gamma:g}_{date:%Y-%m-%d}") for kind, gamma, _ in wanted}
    missing = [item for item in wanted if not paths[item[:2]].exists()]
    if missing:
        with ProcessPoolExecutor(max_workers=min(8, len(missing)), initializer=tb._init) as pool:
            sent = [pool.submit(partial(tb._recommend, **setting), day) for _, _, setting in missing]
            for (kind, gamma, _), future in zip(missing, sent):
                future.result().rename("weight").to_frame().to_parquet(paths[(kind, gamma)])
    return {key: pd.read_parquet(path)["weight"] for key, path in paths.items()}


def chart(kind: str, gamma: float, w: pd.Series, legs: tuple[pd.Series, pd.Series], close, dates, span: dict, row) -> None:
    """One level of gamma: the chart on the left, the largest names on the right."""
    ax, side = row
    w = w / w.sum()
    first, last = span["first"], span["last"]
    path = (close.iloc[first:last + 1][w.index] / close.iloc[first][w.index] - 1) * 100
    rise, fall = legs                                   # for normal and flipped, "rise" is the whole three months
    if kind == "dip":
        colours = np.where((rise > 0) & (fall < 0), BLUE, np.where((rise <= 0) & (fall < 0), ORANGE, GREY))
        names = ["rose three months, fell in the last", "fell in both", "neither"]
    else:
        colours = np.where(rise > 0, BLUE, ORANGE)
        names = ["up over the three months", "down over the three months"]
    thickness = 0.5 + 3.5 * w / w.max()

    if kind != "dip":                                   # the others look back three months; this one, four
        ax.axvspan(dates[span["first"]], dates[span["quarter"]], color="lightgrey", alpha=0.4)
        ax.text(dates[span["first"]], 0.985, " before the window ", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8.5, color="dimgrey")
    for colour in (GREY, ORANGE, BLUE):                 # grey first, so it never covers the others
        for ticker in w.index[colours == colour]:
            ax.plot(path.index, path[ticker], color=colour, linewidth=thickness[ticker], alpha=0.75)
    ax.plot(path.index, (path * w).sum(axis=1), color="black", linewidth=2.5)
    ax.axvline(dates[span["split"]], color="black", linestyle="--", linewidth=0.9)
    ax.text(dates[span["split"]], 0.985, " last month starts", transform=ax.get_xaxis_transform(), va="top", fontsize=9)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_ylabel("return since four months before, %", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", color="lightgrey", linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.set_title(f"γ = {gamma:g}: {len(w)} names, {1 / (w ** 2).sum():.1f} equally weighted; "
                 f"the top four carry {w.head(4).sum():.0%} of the money", loc="left", fontsize=11.5)

    side.axis("off")
    lines = [f"{t:<6}{w[t]:>4.0%}  {rise[t]:>+5.0%} then {fall[t]:>+5.0%}" if kind == "dip" else
             f"{t:<6}{w[t]:>4.0%}  {rise[t]:>+6.0%}" for t in w.head(SHOWN).index]
    if len(w) > SHOWN:
        lines.append(f"... and {len(w) - SHOWN} smaller, {w.iloc[SHOWN:].sum():.0%} of the money")
    side.text(0, 0.96, "largest weights" + ("   three months, then the last" if kind == "dip" else "   three months"),
              fontsize=9.5, style="italic", va="top")
    side.text(0, 0.88, "\n".join(lines), fontsize=11.5, va="top", family="monospace", linespacing=1.5)
    side.legend(handles=[Line2D([], [], color=colour, linewidth=2, label=f"{name} ({(colours == colour).sum()})")
                         for colour, name in zip((BLUE, ORANGE, GREY), names) if (colours == colour).any()] +
                [Line2D([], [], color="black", linewidth=2.5, label="the picks, weighted")],
                loc="lower left", fontsize=10.5, frameon=False)


def main() -> None:
    tb.ensure_prices()
    close = pd.read_parquet(tb._cache("close")).sort_index()
    dates = close.index
    day, last = len(dates), len(dates) - 1                     # every close there is
    on_or_after = lambda when: int(dates.searchsorted(when))             # noqa: E731
    span = {"last": last,
            "first": on_or_after(dates[last] - pd.DateOffset(months=4)),          # the chart starts here
            "split": on_or_after(dates[last] - pd.DateOffset(months=1)) - 1,      # the last close before the final month
            "quarter": on_or_after(dates[last] - pd.DateOffset(months=3)),        # the optimizer's window starts here
            "rise": on_or_after(dates[on_or_after(dates[last] - pd.DateOffset(months=1)) - 1]
                               - pd.DateOffset(months=3))}                         # ... and the dip's rise window here
    wanted = [(kind, gamma, dict(gamma=gamma, flip=change["flip"], dip=change["dip"], weight=tb.DIP_WEIGHT, last=tb.DIP_LAST))
              for kind, change in KINDS.items() for gamma in GAMMAS]
    picks = picks_for(wanted, day, dates[last])

    OUTPUT.mkdir(exist_ok=True)
    print(f"picks from closes up to {dates[last].date()}; the chart runs from {dates[span['first']].date()}")
    for kind in KINDS:
        fig, rows = plt.subplots(len(GAMMAS), 2, figsize=(11, 5.2 * len(GAMMAS)), squeeze=False,
                                 gridspec_kw={"width_ratios": [3.6, 1.6]})
        for gamma, row in zip(GAMMAS, rows):
            w = picks[(kind, gamma)]
            legs = (close.iloc[last][w.index] / close.iloc[span["quarter"]][w.index] - 1,
                    close.iloc[last][w.index] / close.iloc[span["split"]][w.index] - 1)
            if kind == "dip":
                legs = (close.iloc[span["split"]][w.index] / close.iloc[span["rise"]][w.index] - 1, legs[1])
            w = w / w.sum()
            chart(kind, gamma, w, legs, close, dates, span, row)
            print(f"  {kind:<8} gamma {gamma:>4g}: {len(w):>3} names, {1 / (w ** 2).sum():>5.1f} equally weighted, top four "
                  f"{w.head(4).sum():.0%}; weight-weighted, the picks did {100 * (w * legs[0]).sum():+.0f}%"
                  + (f" then {100 * (w * legs[1]).sum():+.0f}%" if kind == "dip" else "")
                  + f" over the {'rise and fall' if kind == 'dip' else 'three months'}")
        fig.suptitle(f"{TITLE[kind]} picks from closes up to {dates[last].date()}, K=25, the 500 most liquid stocks\n"
                     "each line is one stock picked, thick in proportion to its weight; black is their weighted average",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUTPUT / f"picks_{kind}.png", dpi=110)
        print(f"  saved {OUTPUT / f'picks_{kind}.png'}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
