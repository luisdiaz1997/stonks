"""One sum in from the start, everything sold and re-bought every two weeks, for several K.

The same account and the same lines as lump_sum_paths.py, once per K: the
number of factors, which doubles as the concentration dial (a small K gives a
book of many names, a large K a concentrated one). Each panel holds the path
of every sell threshold from 1% to 50%, and of no sell inside the two weeks.
γ is 40 unless given.

    python notebooks/lump_sum_k.py [span in years, default 2] [gamma, default 40] [k, k, ...]
    python notebooks/lump_sum_k.py pick K [span] [gamma]

The first form draws outputs/lump_sum_k.png, computing any picks it lacks with
the usual pool of workers. For several K at once that is an hour or more, so
the second form computes one K's picks with a single worker and can be started
once per K, side by side; each recommendation is cached as it finishes, so a
run that is cut short resumes where it stopped. Needs the cached prices of
take_profit_heatmap.py. Not financial advice.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_ARGUMENTS = sys.argv[1:]
_PICK = int(_ARGUMENTS[1]) if _ARGUMENTS[:1] == ["pick"] else None      # the K whose picks are to be computed
_REST = _ARGUMENTS[2:] if _PICK is not None else _ARGUMENTS
SPAN = _REST[0] if _REST else "2"
GAMMA = float(_REST[1]) if len(_REST) > 1 else 40.0
KS = [int(word) for word in _REST[2:]] or [5, 10, 15, 20, 25, 30, 35, 40]
os.environ["STONKS_BACKTEST_SPAN"] = SPAN          # read by this module and by its worker processes
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import take_profit_backtest as tb

THRESHOLDS = list(range(1, 51))
NEVER = "no sell inside the two weeks"
SPY = "SPY, bought once and held"
OUTPUT = Path(__file__).resolve().parent.parent / "outputs"


def run(tickers, weights, buy_days, end, pct: int | None) -> dict:
    """The whole sum in at the first buy day, everything sold and re-bought at every buy day up to ``end``."""
    tb.TAKE = 0.08 if pct is None else pct / 100
    return tb._simulate(tickers, weights, buy_days, end, None if pct is None else "close", tb.YEARLY, True,
                        rebalance=True, lump=True)


def pick(k: int) -> None:
    """One K's missing recommendations, one worker, each cached as it finishes."""
    tb.WORKERS, tb.K = 1, k
    tb.GAMMA, tb.FLIP, tb.DIP, tb.CADENCE = GAMMA, False, False, "biweekly"
    tb.ensure_prices()
    dates = pd.read_parquet(tb._cache("close")).sort_index().index
    print(f"K = {k}: computing the picks that are not cached yet, one worker", flush=True)
    tb.load_weights(dates, tb._buy_days(dates, "biweekly"))
    print(f"K = {k}: every buy day is cached", flush=True)


def main() -> None:
    tb.ensure_prices()
    close = pd.read_parquet(tb._cache("close")).sort_index()
    stocks = [t for t in close.columns if t != "SPY"]
    dates = close.index
    buy_days = tb._buy_days(dates, "biweekly")
    end = int(dates.searchsorted(tb.END))            # the anniversary
    spy = {day: pd.Series({"SPY": 1.0}) for day in buy_days}
    spy_held = run(["SPY"], spy, buy_days, end, None)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=True, sharey=True)
    shade = plt.get_cmap("viridis")
    for ax, k in zip(axes.flat, KS):
        # load_weights reads these from the module; the command line carries the rest
        tb.K, tb.GAMMA, tb.FLIP, tb.DIP, tb.CADENCE = k, GAMMA, False, False, "biweekly"
        algo = tb.load_weights(dates, buy_days)
        results = {pct: run(stocks, algo, buy_days, end, pct) for pct in (*THRESHOLDS, None)}
        final = {pct: 100 * result["pnl"] / tb.YEARLY for pct, result in results.items()}
        best = max(THRESHOLDS, key=final.get)
        for pct in THRESHOLDS:
            days, values = results[pct]["daily"]
            ax.plot(days, 100 * (values / tb.YEARLY - 1), color=shade((pct - 1) / (len(THRESHOLDS) - 1)), linewidth=1.0, alpha=0.85)
        days, values = results[None]["daily"]
        ax.plot(days, 100 * (values / tb.YEARLY - 1), color="black", linewidth=2.0, label=NEVER)
        days, values = spy_held["daily"]
        ax.plot(days, 100 * (values / tb.YEARLY - 1), color="grey", linestyle="--", linewidth=1.2, label=SPY)
        ax.axhline(0, color="lightgrey", linewidth=0.7)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="lightgrey", linewidth=0.5)
        names = float(np.mean([len(weights) for weights in algo.values()]))
        worst = float((results[None]["daily"][1] / np.maximum.accumulate(results[None]["daily"][1]) - 1).min())
        ax.set_title(f"K = {k}: {names:.0f} names a buy\nno sell {final[None]:+.0f}% | +10% {final[10]:+.0f}% | "
                     f"+1% {final[1]:+.0f}% | worst fall {worst:.0%}", fontsize=10)
        print(f"K {k:>2}: {names:>5.1f} names a buy | {NEVER:<28}{final[None]:>+7.1f} | best +{best}% at {final[best]:+.1f} | "
              + " | ".join(f"+{p}% {final[p]:+.1f}" for p in (1, 5, 10, 20, 50)) + f" | worst fall with no inside sell {worst:+.1%}")
    for ax in list(axes.flat)[len(KS):]:
        ax.axis("off")
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("return on the sum put in once, %", fontsize=9)
    print(f"SPY, bought once and held: {100 * spy_held['pnl'] / tb.YEARLY:+.1f}")
    fig.suptitle(f"gamma {GAMMA:g}, normal picks: one sum of ${tb.YEARLY:,.0f} in on {dates[buy_days[0]].date()}, everything sold and "
                 f"re-bought every other Monday, valued at each day's close\none line per sell threshold inside the two weeks "
                 f"(dark sell early, light sell late); black is no sell inside the two weeks; dashed grey is SPY bought once and held",
                 fontsize=11.5)
    fig.tight_layout()
    OUTPUT.mkdir(exist_ok=True)
    out = OUTPUT / f"lump_sum_k{'' if tb.SPAN == 1 else f'_span{tb.SPAN}'}_g{GAMMA:g}.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    if _PICK is not None:
        pick(_PICK)
    else:
        main()
