"""One sum in from the start, sold and re-bought at every buy day, at four rhythms.

The same account as lump_sum.py and the same lines as lump_sum_paths.py, once
per rhythm: weekly, every other Monday, monthly and every two months, all
starting on the same day. Each panel holds the path of every sell threshold
from 1% to 50%, and of no sell inside the period at all.

A recommendation depends only on the closes before its day, so one set of
picks serves all four rhythms: its Mondays contain theirs. The every-other-
Monday picks are the backtests' usual ones; the other Mondays are computed on
first use (52 more over two years) and cached.

At the end the window is set against the two- and four-month windows, from
every other start day: one-year sums on the weekly rhythm, for each window
whose Monday picks are cached.

    python notebooks/lump_sum_cadence.py [span in years, default 2] [gamma, default 40] [K, default 25] [window in months, default 3]

Writes outputs/lump_sum_cadence.png. With a window other than three months every
Monday has to be computed on first use (104 over two years). Needs the cached
prices of take_profit_heatmap.py. Not financial advice.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_ARGUMENTS = sys.argv[1:]
SPAN = _ARGUMENTS[0] if _ARGUMENTS else "2"
GAMMA = float(_ARGUMENTS[1]) if len(_ARGUMENTS) > 1 else 40.0
FACTORS = int(_ARGUMENTS[2]) if len(_ARGUMENTS) > 2 else 25
WINDOW = int(_ARGUMENTS[3]) if len(_ARGUMENTS) > 3 else 3     # months of history the optimizer looks at
os.environ["STONKS_BACKTEST_SPAN"] = SPAN          # read by this module and by its worker processes
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import take_profit_backtest as tb

CADENCES = [("weekly", "Weekly"), ("biweekly", "Every two weeks"), ("monthly", "Monthly"), ("bimonthly", "Every two months")]
THRESHOLDS = list(range(1, 51))
SHOWN = (1, 5, 10, 20, 50)              # thresholds in the printed table
NEVER = "no sell inside the period"
SPY = "SPY, bought once and held"
OUTPUT = Path(__file__).resolve().parent.parent / "outputs"


def run(tickers, weights, days, end, pct: int | None) -> dict:
    """The whole sum in at the first buy day, everything sold and re-bought at every buy day up to ``end``."""
    tb.TAKE = 0.08 if pct is None else pct / 100
    return tb._simulate(tickers, weights, days, end, None if pct is None else "close", tb.YEARLY, True,
                        rebalance=True, lump=True)


def worst_fall(result: dict) -> float:
    """The deepest drop from an earlier close, over the account's daily values."""
    values = result["daily"][1]
    return float((values / np.maximum.accumulate(values) - 1).min())


def mondays(dates: pd.DatetimeIndex) -> dict:
    """The picks for every Monday of the span, computing the ones that are not cached."""
    suffix = "" if tb.SPAN == 1 else f"_span{tb.SPAN}"
    tb.GAMMA, tb.FLIP, tb.DIP, tb.K, tb.MONTHS = GAMMA, False, False, FACTORS, WINDOW
    weekly, usual = (tb._cache(tb._setting(cadence) + suffix) for cadence in ("weekly", "biweekly"))
    if not weekly.exists() and usual.exists():
        pd.read_parquet(usual).to_parquet(weekly)      # the weekly cache starts from the every-other-Monday one
    tb.CADENCE = "weekly"
    return tb.load_weights(dates, tb._buy_days(dates, "weekly"))


def figure(picks: dict) -> None:
    close = pd.read_parquet(tb._cache("close")).sort_index()
    stocks = [t for t in close.columns if t != "SPY"]
    dates = close.index
    end = int(dates.searchsorted(tb.END))            # the anniversary
    first = tb._buy_days(dates, "weekly")[0]
    spy_held = run(["SPY"], {day: pd.Series({"SPY": 1.0}) for day in tb._buy_days(dates, "weekly")}, tb._buy_days(dates, "weekly"), end, None)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    shade = plt.get_cmap("viridis")
    print(f"gamma {GAMMA:g}, K={FACTORS}, {WINDOW}-month window, normal picks: one sum of ${tb.YEARLY:,.0f} in on "
          f"{dates[first].date()}, valued {dates[end].date()}")
    for ax, (cadence, label) in zip(axes.flat, CADENCES):
        days = tb._buy_days(dates, cadence)
        weights = {day: picks[day] for day in days}
        results = {pct: run(stocks, weights, days, end, pct) for pct in (*THRESHOLDS, None)}
        final = {pct: 100 * result["pnl"] / tb.YEARLY for pct, result in results.items()}
        best = max(THRESHOLDS, key=final.get)
        for pct in THRESHOLDS:
            along, values = results[pct]["daily"]
            ax.plot(along, 100 * (values / tb.YEARLY - 1), color=shade((pct - 1) / (len(THRESHOLDS) - 1)), linewidth=1.0, alpha=0.85)
        along, values = results[None]["daily"]
        ax.plot(along, 100 * (values / tb.YEARLY - 1), color="black", linewidth=2.0, label=NEVER)
        along, values = spy_held["daily"]
        ax.plot(along, 100 * (values / tb.YEARLY - 1), color="grey", linestyle="--", linewidth=1.2, label=SPY)
        ax.axhline(0, color="lightgrey", linewidth=0.7)
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", color="lightgrey", linewidth=0.5)
        worst = worst_fall(results[None])
        ax.set_title(f"{label}: {len(days)} re-buys\n{NEVER} {final[None]:+.0f}% | +10% {final[10]:+.0f}% | "
                     f"+1% {final[1]:+.0f}% | worst fall {worst:.0%}", fontsize=10.5)
        with_cost = (1 + results[None]["pnl"] / tb.YEARLY) * (1 - 0.001) ** (2 * (len(days) - 1)) - 1
        print(f"  {label:<17}{len(days):>4} re-buys | {NEVER:<27}{final[None]:>+7.1f} | best +{best}% at {final[best]:+.1f} | "
              + " | ".join(f"+{p}% {final[p]:+.1f}" for p in SHOWN)
              + f" | worst fall {worst:+.1%} | no sell with 0.1% a side {with_cost:+.1%}")
    print(f"  {SPY}: {100 * spy_held['pnl'] / tb.YEARLY:+.1f}")
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("return on the sum put in once, %", fontsize=9)
    fig.suptitle(f"gamma {GAMMA:g}, K={FACTORS}, {WINDOW}-month window, normal picks: one sum of ${tb.YEARLY:,.0f} in on "
                 f"{dates[first].date()}, everything sold and "
                 f"re-bought at every buy day, valued at each day's close\none line per sell threshold inside the period "
                 f"(dark sell early, light sell late); black is no sell inside the period; dashed grey is SPY bought once and held",
                 fontsize=11.5)
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    out = OUTPUT / (f"lump_sum_cadence{'' if tb.SPAN == 1 else f'_span{tb.SPAN}'}_g{GAMMA:g}_k{FACTORS}"
                    + ("" if WINDOW == 3 else f"_m{WINDOW}") + ".png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    start_days(dates, stocks)


def start_days(dates: pd.DatetimeIndex, stocks: list[str]) -> None:
    """This window against the others whose Monday picks are cached, from every other start day:
    one-year sums on the weekly rhythm, so that the page's ranking does not rest on one start."""
    tb.CADENCE = "weekly"
    suffix = "" if tb.SPAN == 1 else f"_span{tb.SPAN}"
    mondays = tb._buy_days(dates, "weekly")
    picks = {}
    for window in sorted({2, 3, 4, WINDOW}):
        tb.MONTHS = window
        if tb._cache(tb._setting("weekly") + suffix).exists():
            picks[window] = tb.load_weights(dates, mondays)
    tb.MONTHS = WINDOW
    if len(picks) < 2:
        print("only one window's Monday picks are cached; nothing to compare start days across")
        return
    year = len(mondays) // 2
    starts = [(mondays[s:s + year], mondays[s + year]) for s in range(year)]
    market = np.array([100 * run(["SPY"], {d: pd.Series({"SPY": 1.0}) for d in days}, days, stop, None)["pnl"] / tb.YEARLY
                       for days, stop in starts])
    for pct in (None, 10):
        scores = {window: np.array([100 * run(stocks, {d: pick[d] for d in days}, days, stop, pct)["pnl"] / tb.YEARLY
                                    for days, stop in starts]) for window, pick in picks.items()}
        top = np.max(np.vstack(list(scores.values())), axis=0)
        print(f"\nother start days, weekly rhythm, {'no sell inside the period' if pct is None else 'sell at +10% inside'}: "
              f"one-year sums from {year} start days, {dates[starts[0][0][0]].date()} to {dates[starts[-1][0][0]].date()}")
        for window, returns in scores.items():
            low, high = int(returns.argmin()), int(returns.argmax())
            print(f"  {window} months: median {np.median(returns):+.0f}%, worst {returns[low]:+.0f}% (from {dates[starts[low][0][0]].date()}), "
                  f"best {returns[high]:+.0f}% | the best of the windows in {int((returns == top).sum())} of {year} starts | "
                  f"ahead of SPY in {int((returns > market).sum())}")
        print(f"  SPY, bought once and held: median {np.median(market):+.0f}%")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    tb.ensure_prices()
    figure(mondays(pd.read_parquet(tb._cache("close")).sort_index().index))
