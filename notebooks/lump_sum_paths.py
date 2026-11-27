"""One sum in from the start, everything sold and re-bought every two weeks: every sell threshold's path.

The same account as lump_sum.py, drawn in full. For each sell threshold from
1% to 50%, and for no sell inside the two weeks at all, this is the account's
return on the sum from the day it went in to the anniversary, at every day's
close. The end of each line is that threshold's cell in the heatmap.

    python notebooks/lump_sum_paths.py [span in years, default 2] [gamma, default 160]

Writes outputs/lump_sum_paths.png. Needs the cached prices and recommendations
of take_profit_heatmap.py. Not financial advice.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

SPAN = sys.argv[1] if len(sys.argv) > 1 else "2"
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 160.0
os.environ["STONKS_BACKTEST_SPAN"] = SPAN          # read by this module and by its worker processes
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import take_profit_backtest as tb
import take_profit_heatmap as th     # the thresholds, so the lines are the heatmap's columns

LABELLED = [1, 5, 10, 50]                     # thresholds named at the end of their lines, chosen so the names do not overlap
NEVER = "no sell inside the two weeks"
SPY = "SPY, bought once and held"
OUTPUT = Path(__file__).resolve().parent.parent / "outputs"


def run(tickers, weights, buy_days, end, pct: int | None) -> dict:
    """The whole sum in at the first buy day, everything sold and re-bought at every buy day up to ``end``."""
    tb.TAKE = 0.08 if pct is None else pct / 100
    return tb._simulate(tickers, weights, buy_days, end, None if pct is None else "close", tb.YEARLY, True,
                        rebalance=True, lump=True)


def main() -> None:
    tb.ensure_prices()
    close = pd.read_parquet(tb._cache("close")).sort_index()
    stocks = [t for t in close.columns if t != "SPY"]
    dates = close.index
    buy_days = tb._buy_days(dates, "biweekly")
    end = int(dates.searchsorted(tb.END))            # the anniversary
    # load_weights reads these from the module; the command line only carries the span and the gamma here
    tb.GAMMA, tb.FLIP, tb.DIP, tb.CADENCE = GAMMA, False, False, "biweekly"
    algo = tb.load_weights(dates, buy_days)

    results = {pct: run(stocks, algo, buy_days, end, pct) for pct in (*th.THRESHOLDS, None)}
    spy_held = run(["SPY"], {day: pd.Series({"SPY": 1.0}) for day in buy_days}, buy_days, end, None)
    final = {pct: 100 * result["pnl"] / tb.YEARLY for pct, result in results.items()}
    best = max(th.THRESHOLDS, key=final.get)

    fig, ax = plt.subplots(figsize=(15, 8))
    shade = plt.get_cmap("viridis")
    for pct in th.THRESHOLDS:
        days, values = results[pct]["daily"]
        ax.plot(days, 100 * (values / tb.YEARLY - 1), color=shade((pct - 1) / (len(th.THRESHOLDS) - 1)), linewidth=1.3, alpha=0.85)
    days, values = results[None]["daily"]
    ax.plot(days, 100 * (values / tb.YEARLY - 1), color="black", linewidth=2.6, label=NEVER)
    days, values = spy_held["daily"]
    ax.plot(days, 100 * (values / tb.YEARLY - 1), color="grey", linestyle="--", linewidth=1.4, label=SPY)
    for pct in LABELLED:                             # name a few lines where they end
        ax.annotate(f"+{pct}%", (dates[end], final[pct]), xytext=(5, 0), textcoords="offset points", fontsize=9,
                    color=shade((pct - 1) / (len(th.THRESHOLDS) - 1)), va="center")
    ax.annotate(NEVER, (dates[end], final[None]), xytext=(5, 9), textcoords="offset points", fontsize=9, va="center")
    ax.axhline(0, color="lightgrey", linewidth=0.8)
    ax.set_xlim(dates[buy_days[0]], dates[end] + pd.Timedelta(days=75))
    ax.set_ylabel("return on the sum put in once, %", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="lightgrey", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"gamma {GAMMA:g}, normal picks: one sum of ${tb.YEARLY:,.0f} in on {dates[buy_days[0]].date()}, everything "
                 f"sold and re-bought every other Monday, one line per sell threshold inside the two weeks\n"
                 f"valued at each day's close; dark lines sell early inside the two weeks, light ones late", fontsize=11.5)
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    out = OUTPUT / f"lump_sum_paths{'' if tb.SPAN == 1 else f'_span{tb.SPAN}'}_g{GAMMA:g}.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")

    print(f"gamma {GAMMA:g}, normal picks, one sum of ${tb.YEARLY:,.0f} in on {dates[buy_days[0]].date()}, valued {dates[end].date()}:")
    print(f"  {NEVER}: {final[None]:+.1f} | {SPY}: {100 * spy_held['pnl'] / tb.YEARLY:+.1f} | best threshold +{best}% at {final[best]:+.1f} | "
          f"1% to 50%: {min(final[p] for p in th.THRESHOLDS):+.1f} to {max(final[p] for p in th.THRESHOLDS):+.1f}")
    for pct in (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 45, 50):
        print(f"  +{pct:>2}%: {final[pct]:+.1f}", end="")
    print()
    worst = {pct: float((result["daily"][1] / np.maximum.accumulate(result["daily"][1]) - 1).min()) for pct, result in results.items()}
    print(f"  worst fall from an earlier close: " + " | ".join(f"{name} {worst[pct]:+.1%}" for name, pct in
          (("+1%", 1), ("+10%", 10), ("+25%", 25), ("+50%", 50), (NEVER, None))))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
