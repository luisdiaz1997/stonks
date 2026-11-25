"""What taxes do to the comparison between sell thresholds.

Every purchase is its own tax lot. A lot sold more than a year after it was
bought realizes a long-term gain; anything sooner is short-term. Taxes are
charged only on realized gains, so a position that is never sold costs nothing
during the span, and losses on positions that are never sold do not reduce the
bill either. Dividends are taxed as ordinary income (with short holding most
would not qualify for the lower rate).

Three after-tax figures are shown for each strategy:
- "keep the rest": pay tax on what was sold and the dividends, keep the rest;
- "sell the losers": additionally sell the unsold positions that are below cost
  at the end, so their losses offset the gains (tax-loss harvesting);
- "sell everything": liquidate at the end and pay tax on all gains.

Not modelled: when the tax is paid (an earlier bill leaves less to invest), the
3.8% surtax's income threshold, and any state tax. Add your state's rate to
both numbers of a scenario if it applies.

    python notebooks/take_profit_taxes.py [span in years, default 2] [gamma, default 40]

Downloads the prices and dividends it needs on first use.
Not financial or tax advice.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

SPAN = sys.argv[1] if len(sys.argv) > 1 else "2"
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
os.environ["STONKS_BACKTEST_SPAN"] = SPAN          # read by the backtest module
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import take_profit_backtest as tb
from stonks.config import Settings

STRATEGIES = [
    ("never sell", None, False),
    ("sell at +5%", 5, False),
    ("sell at +8%", 8, False),
    ("sell at +10%", 10, False),
    ("sell at +15%", 15, False),
    ("sell at +20%", 20, False),
    ("sell at +50%", 50, False),
    ("sell everything every buy day", None, True),
    ("sell everything every buy day, +10%", 10, True),
]
# (label, rate on short-term gains and dividends, rate on long-term gains)
SCENARIOS = [
    ("22% / 15%", 0.22, 0.15),
    ("24% / 15%", 0.24, 0.15),
    ("32% / 15%, +3.8%", 0.358, 0.188),
    ("35% / 20%, +3.8%", 0.388, 0.238),
]
LONG_AFTER = 365                                      # calendar days: longer than this is a long-term gain


def tax(short: float, long: float, dividends: float, short_rate: float, long_rate: float) -> float:
    """Tax on net gains, losses in one category offsetting the other."""
    if short < 0:
        long, short = long + short, 0.0
    if long < 0:
        short, long = short + long, 0.0
    return short_rate * max(short, 0.0) + long_rate * max(long, 0.0) + short_rate * dividends


def ensure_price_only() -> None:
    """Build the price-only series and the dividend table unless they are cached.

    The backtest's usual prices are total-return prices, with each dividend folded
    back into the price. For taxes we need the price that actually trades, which
    drops on the ex-dividend day, and the cash paid on that day.
    """
    cache = Settings().cache_dir / "cache"
    paths = {name: cache / f"priceonly_top500_3y_{name}.parquet" for name in ("close", "open", "high", "dividends")}
    if all(path.exists() for path in paths.values()):
        return
    import time

    import yfinance as yf

    index = pd.read_parquet(tb._cache("close")).sort_index().index
    tickers = list(pd.read_parquet(tb._cache("close")).columns)
    parts = []
    for i in range(0, len(tickers), 100):
        parts.append(yf.download(tickers[i:i + 100], period="3y", interval="1d", auto_adjust=False,
                                 actions=True, progress=False, threads=True))
        time.sleep(0.3)                                   # be polite
    raw = pd.concat(parts, axis=1)
    close, adjusted = raw["Close"].reindex(index), raw["Adj Close"].reindex(index)
    paid = raw["Dividends"].reindex(index).fillna(0.0)

    # Yahoo's adjusted series is the traded price times a backward product of
    # (1 - dividend / previous close); divide that product out again.
    share = (paid / close.ffill().shift(1)).to_numpy()
    share[~(np.isfinite(share) & (paid.to_numpy() > 0))] = 0.0
    step = pd.DataFrame(1.0 - share, index=index, columns=tickers)
    after = step.iloc[::-1].cumprod().iloc[::-1] / step    # the product over later ex-dividend days
    ratio = adjusted / after / close                      # split-adjusted price over the traded price
    series = {"close": adjusted / after, "open": raw["Open"].reindex(index) * ratio,
              "high": raw["High"].reindex(index) * ratio, "dividends": paid * ratio}
    for name, frame in series.items():
        # a blank price must stay blank; a blank dividend is simply no dividend that day
        (frame if name != "dividends" else frame.fillna(0.0)).to_parquet(paths[name])
    print(f"[taxes] built the price-only series and dividend table: {len(tickers)} stocks x {len(index)} days")


def main() -> None:
    tb.ensure_prices()
    ensure_price_only()
    close = pd.read_parquet(tb._cache("close")).sort_index()
    stocks = [t for t in close.columns if t != "SPY"]
    dates = close.index
    buy_days = tb._buy_days(dates, "biweekly")
    per_buy = tb.YEARLY / len(buy_days)
    end = int(dates.searchsorted(tb.END))

    tb.GAMMA, tb.FLIP, tb.CADENCE = GAMMA, False, "biweekly"
    algo = tb.load_weights(dates, buy_days)
    spy = {day: pd.Series({"SPY": 1.0}) for day in buy_days}
    tb.DATA = "priceonly_top500_3y"                       # dividends arrive as cash, not reinvested
    dividends = pd.read_parquet(tb._cache("dividends")).fillna(0.0)

    def run(weights, tickers, pct, rebalance):
        tb.TAKE = 0.08 if pct is None else pct / 100
        result = tb._simulate(tickers, weights, buy_days, end, None if pct is None else "close",
                              per_buy, True, rebalance=rebalance, dividends=dividends[tickers].to_numpy())
        sold = {"short": 0.0, "long": 0.0}
        for sale, _, bought, gain in result["lot log"]:
            sold["long" if (dates[sale] - dates[bought]).days > LONG_AFTER else "short"] += gain
        kept = {"short": 0.0, "long": 0.0, "losses": 0.0}
        for _, bought, gain in result["open lots"]:
            kept["long" if (dates[end] - dates[bought]).days > LONG_AFTER else "short"] += gain
            kept["losses"] += min(gain, 0.0)
        return result, sold, kept

    runs = [(label, *run(algo, stocks, pct, rebalance)) for label, pct, rebalance in STRATEGIES]
    runs.append(("SPY, never sell", *run(spy, ["SPY"], None, False)))
    tb.TAKE, tb.DATA = 0.08, "backtest_top500_3y"

    money = lambda v: f"${v:,.0f}"                        # noqa: E731
    print(f"gamma {GAMMA:g}, normal | {len(buy_days)} buys of ${per_buy:,.2f} = ${tb.YEARLY:,.0f} | "
          f"{dates[buy_days[0]].date()} to {dates[end].date()} | dividends paid as cash")
    print("\nwhat the account realizes (before any tax)")
    print(f"{'strategy':<38}{'final value':>13}{'gain':>16}{'sold: short-term':>18}{'sold: long-term':>17}"
          f"{'dividends':>11}{'still held: <=1y':>17}{'>1y':>9}{'of which losses':>17}")
    for label, r, sold, kept in runs:
        print(f"{label:<38}{money(r['value']):>13}{f'{r['pnl'] / tb.YEARLY:+.1%}':>16}{money(sold['short']):>18}"
              f"{money(sold['long']):>17}{money(r['dividends']):>11}{money(kept['short']):>17}{money(kept['long']):>9}"
              f"{money(kept['losses']):>17}")

    for name, short_rate, long_rate in SCENARIOS:
        print(f"\nafter tax at {name}: final value, and the gain on ${tb.YEARLY:,.0f}")
        print(f"{'strategy':<38}{'keep the rest':>24}{'sell the losers':>24}{'sell everything':>24}")
        for label, r, sold, kept in runs:
            owed = tax(sold["short"], sold["long"], r["dividends"], short_rate, long_rate)
            harvest = tax(sold["short"] + kept["losses"], sold["long"], r["dividends"], short_rate, long_rate)
            everything = tax(sold["short"] + kept["short"], sold["long"] + kept["long"], r["dividends"],
                             short_rate, long_rate)
            cell = lambda bill: f"{money(r['value'] - bill)} ({(r['pnl'] - bill) / tb.YEARLY:+.1%})"   # noqa: E731
            print(f"{label:<38}{cell(owed):>24}{cell(harvest):>24}{cell(everything):>24}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
