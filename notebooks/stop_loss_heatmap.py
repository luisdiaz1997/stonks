"""Heatmap of the result for every stop-loss from 1% to 50%, with the sell at +10% kept.

Same test as take_profit_heatmap.py (K=25, 3-month history, top 500, the
day's share of $12,000 a year over every other Monday, sold cash reused,
sells checked at the close), evaluated on the anniversary. The take-profit
stays at +10%; on top of it a position is sold once its return falls to the
level on the x axis. "none" is the rule as it runs live: +10% and nothing else.

It is repeated for each gamma, the normal and the flipped optimizer, and two
ways of managing the account:

- "held": a position stays until a rule sells it;
- "rebalanced": every buy day starts by selling the whole account, winners and
  losers alike, and the new recommendation is bought with all of it (so the
  stop only has the two weeks until the next buy day to fire).

Cells show the profit or loss as a percentage of the money put in.

    python notebooks/stop_loss_heatmap.py [span in years, default 1]

One year buys $12,000 from November 2025; two years buy $24,000 from November
2024. Both are valued on 2026-11-02. Needs the cached prices and
recommendations of take_profit_heatmap.py, which builds them on first use.

Writes outputs/stop_loss_sweep.png and the same numbers as a CSV.
Not financial advice.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

SPAN = sys.argv[1] if len(sys.argv) > 1 else "1"
os.environ["STONKS_BACKTEST_SPAN"] = SPAN          # read by this module and by its worker processes
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import take_profit_backtest as tb

TAKE = 10                                # percent: the live threshold, held fixed
STOPS = list(range(1, 51))               # percent: a position is sold once it has fallen this far
CONFIGS = [(160, False), (160, True), (40, False), (40, True)]
MODES = [(False, "held until a rule sells it"), (True, "everything sold every buy day")]
OUTPUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    tb.ensure_prices()
    tb.TAKE = TAKE / 100
    close = pd.read_parquet(tb._cache("close")).sort_index()
    stocks = [t for t in close.columns if t != "SPY"]
    dates = close.index
    buy_days = tb._buy_days(dates, "biweekly")
    per_buy = tb.YEARLY / len(buy_days)
    end = int(dates.searchsorted(tb.END))            # the anniversary

    rows = []
    for gamma, flip in CONFIGS:
        # load_weights reads these from the module; the command line only carries the span here
        tb.GAMMA, tb.FLIP, tb.CADENCE = float(gamma), flip, "biweekly"
        tb.ALGO = f"{'Flipped' if flip else ''}algorithm"
        algo = tb.load_weights(dates, buy_days)
        for rebalance, mode in MODES:
            for pct in (None, *STOPS):
                stop = None if pct is None else pct / 100
                result = tb._simulate(stocks, algo, buy_days, end, "close", per_buy, True,
                                      rebalance=rebalance, stop=stop)
                rows.append({"row": f"gamma {gamma}, {'flipped' if flip else 'normal'}\n{mode}",
                             "stop at": "none" if pct is None else f"{pct}%",
                             "return": 100 * result["pnl"] / tb.YEARLY,
                             "sells": result["sells"],
                             # a sale at a loss as deep as the stop was made by the stop (held rows
                             # only: a rebalanced account also sells losers at every buy day)
                             "stopped": (sum(ret <= -stop for *_, ret in result["sell log"])
                                         if stop and not rebalance else 0)})
    tb.TAKE = 0.08

    table = pd.DataFrame(rows)
    order = [f"gamma {gamma}, {'flipped' if flip else 'normal'}\n{mode}"
             for gamma, flip in CONFIGS for _, mode in MODES]
    columns = ["none", *[f"{p}%" for p in STOPS]]
    grid = table.pivot_table(index="row", columns="stop at", values="return").reindex(index=order, columns=columns)

    OUTPUT.mkdir(exist_ok=True)
    stem = "stop_loss_sweep" + ("" if tb.SPAN == 1 else f"_span{tb.SPAN}")
    table.to_csv(OUTPUT / f"{stem}.csv", index=False)

    fig, ax = plt.subplots(figsize=(26, 6.4))
    # no centre at zero: the cells run from about 0 to +48, so centring there would
    # paint the whole upper half one green and hide a 20-point dip inside a row
    sns.heatmap(grid, ax=ax, annot=True, fmt="+.1f", cmap="RdYlGn", linewidths=0.4,
                linecolor="white", annot_kws={"size": 8},
                cbar_kws={"label": f"profit or loss on the ${tb.YEARLY:,.0f} put in, %"})
    for boundary in (2, 4, 6):                        # a heavier line between settings
        ax.axhline(boundary, color="black", linewidth=1.6)
    ax.axvline(1, color="black", linewidth=1.6)       # and after the "none" column
    ax.set_xlabel(f"Also sell a position once its return has fallen to (the sell at +{TAKE}% stays)", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10, rotation=0)
    ax.set_title(f"{tb.SPAN}-year result by stop-loss, {dates[buy_days[0]].date()} to {dates[end].date()}: "
                 f"K=25, 3-month history, ${tb.YEARLY:,.0f} over every other Monday, sold cash reused, "
                 f"sells checked at the close", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}.png", dpi=110)

    print(f"saved {OUTPUT / f'{stem}.png'} ({grid.shape[0]} rows x {grid.shape[1]} columns), "
          f"{len(buy_days)} buys, valued {dates[end].date()}, sell at +{TAKE}% throughout\n")
    for row, group in table.groupby("row", sort=False):
        group = group.set_index("stop at")
        sweep = group["return"].drop("none")
        sells = group.loc[["none", "5%", "10%", "15%", "20%", "30%", "50%"]]
        print(f"{row.replace(chr(10), ', ')}")
        print(f"  no stop: {group.loc['none', 'return']:+.1f}% | best stop {sweep.idxmax()} at {sweep.max():+.1f}% | "
              f"5-15%: {sweep['5%':'15%'].mean():+.1f}% avg")
        print("  " + " | ".join(f"{level}: {sells.loc[level, 'return']:+.1f}% ({int(sells.loc[level, 'sells'])} sells"
                              + (f", {int(sells.loc[level, 'stopped'])} by the stop)" if not rebalance else ")")
                              for level in sells.index))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
