"""Heatmap of the result for every sell threshold from 1% to 50%.

Same test as take_profit_backtest.py (K=25, 3-month history, top 500, $461.54
over every other Monday, sold cash reused, sells checked at the close),
evaluated on the anniversary. It is repeated for each threshold, each gamma,
the normal and the flipped optimizer, and two ways of managing the account:

- "held": a position stays until the take-profit rule sells it;
- "rebalanced": every buy day starts by selling the whole account, winners and
  losers alike, and the new recommendation is bought with all of it (so
  "never" there means no take-profit inside the two-week window).

Cells show the profit or loss as a percentage of the money put in.

    python notebooks/take_profit_heatmap.py [span in years, default 1]

One year buys $12,000 from November 2025; two years buys $24,000 from November
2024. Both are valued on 2026-11-02. The two-year span needs the three-year
price files; to create them, download with stonks.download._download_raw for
period "3y" and save its Open, High and Close frames as
backtest_top500_3y_open/high/close.parquet in the cache. The recommendations
are computed on first use (about four minutes per setting for each span) and
cached after that.

Writes outputs/take_profit_sweep.png and the same numbers as a CSV.
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

THRESHOLDS = list(range(1, 51))            # percent; 0 would mean "sell at break-even or better", not a profit target
CONFIGS = [(160, False), (160, True), (40, False), (40, True)]
MODES = [(False, "held until the rule sells it"), (True, "everything sold every buy day")]
OUTPUT = Path(__file__).resolve().parent.parent / "outputs"


def main() -> None:
    tb.ensure_prices()
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
            for pct in (*THRESHOLDS, None):
                tb.TAKE = 0.08 if pct is None else pct / 100
                result = tb._simulate(stocks, algo, buy_days, end, None if pct is None else "close",
                                      per_buy, True, rebalance=rebalance)
                rows.append({"row": f"gamma {gamma}, {'flipped' if flip else 'normal'}\n{mode}",
                             "sell at": "never" if pct is None else f"{pct}%",
                             "return": 100 * result["pnl"] / tb.YEARLY})
    tb.TAKE = 0.08

    table = pd.DataFrame(rows)
    order = [f"gamma {gamma}, {'flipped' if flip else 'normal'}\n{mode}"
             for gamma, flip in CONFIGS for _, mode in MODES]
    grid = (table.pivot_table(index="row", columns="sell at", values="return")
            .reindex(index=order, columns=[f"{p}%" for p in THRESHOLDS] + ["never"]))

    OUTPUT.mkdir(exist_ok=True)
    stem = "take_profit_sweep" + ("" if tb.SPAN == 1 else f"_span{tb.SPAN}")
    table.to_csv(OUTPUT / f"{stem}.csv", index=False)

    fig, ax = plt.subplots(figsize=(26, 6.4))
    sns.heatmap(grid, ax=ax, annot=True, fmt="+.1f", cmap="RdYlGn", center=0, linewidths=0.4,
                linecolor="white", annot_kws={"size": 8},
                cbar_kws={"label": f"profit or loss on the ${tb.YEARLY:,.0f} put in, %"})
    for boundary in (2, 4, 6):                        # a heavier line between settings
        ax.axhline(boundary, color="black", linewidth=1.6)
    ax.axvline(len(THRESHOLDS), color="black", linewidth=1.6)   # and before the "never" column
    ax.set_xlabel("Sell a position once its return reaches", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=10, rotation=0)
    ax.set_title(f"{tb.SPAN}-year result by sell threshold, {dates[buy_days[0]].date()} to {dates[end].date()}: "
                 f"K=25, 3-month history, ${tb.YEARLY:,.0f} over every other Monday, sold cash reused, "
                 "sells checked at the close", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}.png", dpi=110)

    print(f"saved {OUTPUT / f'{stem}.png'} ({grid.shape[0]} rows x {grid.shape[1]} columns), "
          f"{len(buy_days)} buys, valued {dates[end].date()}")
    for (row, _), group in table.groupby(["row", "sell at"], sort=False):
        pass
    for row, group in table.groupby("row", sort=False):
        sweep = (group[group["sell at"] != "never"].set_index("sell at")["return"]
                 .reindex([f"{p}%" for p in THRESHOLDS]))
        never = group[group["sell at"] == "never"]["return"].iloc[0]
        print(f"  {row.replace(chr(10), ', '):<58}best {sweep.idxmax():>4} at {sweep.max():+.1f}% | "
              f"5-15%: {sweep['5%':'15%'].mean():+.1f}% avg | never: {never:+.1f}%")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
