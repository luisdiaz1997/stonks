"""The take-profit sell rule on the last year: fixed new money into the optimizer's picks.

The question: on every buy day, run the production optimizer (given gamma, K=25,
3-month history, top 500) and buy that day's share of $12,000 plus whatever
earlier sells brought in; a server then sells any position whose total return
reaches +8%. What would a year of that have done?

    python notebooks/take_profit_backtest.py [gamma] [monthly|biweekly] [flip] [sweep]

"monthly" puts $1,000 on the first Monday of each month (12 buys); "biweekly"
spreads the same $12,000 over every other Monday from the start (26 buys), so
the two differ only in cadence. "flip" negates the optimizer's expected return,
which turns it into a long-only portfolio of the stocks it would otherwise
avoid (the ones that fell over the window) - the "buy the dip" candidate.
"sweep" tries every sell threshold from 5% to 15%. Setting the environment
variable STONKS_BACKTEST_SPAN=2 keeps buying for two years instead of one
(from November 2024, $24,000 in all, still evaluated on the anniversary); it
needs the three-year price files, which take_profit_heatmap.py can download.

It is compared against the same buys without selling, against an equal-weight
basket of the same stocks, and against SPY, all with identical contributions.

Mirrors the live rule where it matters:
- a stock bought more than once is ONE position with a blended average cost,
  and the whole position is sold when that blended return reaches the target;
- a position bought that day is not checked until the next day;
- proceeds are available for the next buy day.
The live job checks hourly; the two sell variants bracket that: "close" only
looks at each day's close (fewer sells), "touch" sells at the target the moment
a day's high reaches it (more sells).

Not financial advice. Run from the repository root.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import stonks.portfolio as pf
from stonks.config import Settings
from stonks.robinhood import allocate_fractional_buys

GAMMA = float(sys.argv[1]) if len(sys.argv) > 1 else 160.0
CADENCE = sys.argv[2] if len(sys.argv) > 2 else "monthly"
FLAGS = sys.argv[3:]                                    # extra words: "flip", "sweep"
FLIP = "flip" in FLAGS                                  # negate the expected return
SWEEP = "sweep" in FLAGS                                # try every sell threshold from 5% to 15%
ALGO = "Flipped algorithm" if FLIP else "Algorithm"
K, MONTHS = 25, 3                      # the optimizer exactly as it is run
# years of buying; an environment variable so worker processes read it too
SPAN = int(os.environ.get("STONKS_BACKTEST_SPAN", "1"))
YEARLY = 12000.0 * SPAN                # $1,000 a month for the whole span
TAKE = 0.08                            # sell a position once its return reaches this
END = pd.Timestamp("2026-11-02")       # the anniversary on which the account is valued
START = (END - pd.DateOffset(years=SPAN)) + pd.offsets.Week(weekday=0, n=0)
DATA = "backtest_top500_2y" if SPAN == 1 else "backtest_top500_3y"           # cached price files
PERIOD = "2y" if SPAN == 1 else "3y"   # how much history those files hold

_CLOSE: pd.DataFrame = None            # set in each worker process by _init


def _init() -> None:
    """Each worker: one torch thread, its own copy of the price history, and the flip if asked."""
    global _CLOSE
    import torch

    torch.set_num_threads(1)
    _CLOSE = pd.read_parquet(_cache("close")).sort_index().drop(columns="SPY")   # stocks only
    if FLIP:
        factor_covariance = pf._factor_covariance

        def flipped(returns, K):
            mu, sigma = factor_covariance(returns, K)
            return -mu, sigma                       # the one change: expect the opposite

        pf._factor_covariance = flipped


def _cache(name: str, data: str | None = None) -> Path:
    return Settings().cache_dir / "cache" / f"{data or DATA}_{name}.parquet"


def ensure_prices() -> None:
    """Download this span's daily bars for the 500 stocks unless they are cached.

    The stocks are the package's own ranking (``get_prices(top_n=500, period="2y")``),
    built here on first use; SPY is added as the benchmark. The files land in the
    package's ``data/cache`` directory, which git ignores.
    """
    if all(_cache(field).exists() for field in ("open", "high", "close")):
        return
    import stonks.download as dl

    settings = Settings()
    ranked = settings.prices_cache_path("prices_ranked_1d_2y_close_top500")
    if not ranked.exists():
        dl.get_prices(top_n=500, period="2y")             # builds the ranking
    tickers = pd.read_parquet(ranked).columns.tolist()
    fields, failed = dl._download_raw(                    # the package's own downloader
        tickers + ["SPY"], PERIOD, "1d", settings.chunk_size
    )
    for name in ("Open", "High", "Close"):
        fields[name].to_parquet(_cache(name.lower()))
    note = f" ({len(failed)} failed to download)" if failed else ""
    print(f"[backtest] downloaded {len(tickers)} stocks x {len(fields['Close'])} days of daily bars{note}")


def _recommend(i: int) -> pd.Series:
    """The production recommendation using closes up to the day before ``i``."""
    pf.get_prices = lambda **kwargs: _CLOSE.iloc[:i].T      # the only thing swapped
    weights = pf.optimize_portfolio(top_n=500, months=MONTHS, gamma=GAMMA, K=K)
    return weights["weight"]


def load_weights(dates: pd.DatetimeIndex, buy_days: list[int]) -> dict[int, pd.Series]:
    """The optimizer's recommendation for every buy day, computed once and cached.

    A recommendation depends only on the closes before it, so a longer span
    starts from the one-year cache wherever the buy days coincide and only the
    earlier buy days are computed.
    """
    setting = f"weights_g{GAMMA:g}_k{K}_{CADENCE}{'_flip' if FLIP else ''}"
    cache = _cache(setting if SPAN == 1 else f"{setting}_span{SPAN}")
    frame = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()

    if frame.empty and SPAN > 1 and _cache(setting, "backtest_top500_2y").exists():
        one_year_dates = pd.read_parquet(_cache("close", "backtest_top500_2y")).index
        previous = pd.read_parquet(_cache(setting, "backtest_top500_2y"))
        position = dates.get_indexer([one_year_dates[int(day)] for day in previous.columns])
        frame = previous.loc[:, position >= 0].set_axis(position[position >= 0], axis=1)

    missing = [day for day in buy_days if day not in frame.columns]
    if missing:
        with ProcessPoolExecutor(max_workers=8, initializer=_init) as pool:
            computed = dict(zip(missing, pool.map(_recommend, missing)))
        frame = pd.concat([frame, pd.DataFrame(computed)], axis=1)
        frame[buy_days].to_parquet(cache)
    return {day: frame[day].dropna() for day in buy_days}


def _buy_days(dates: pd.DatetimeIndex, cadence: str) -> list[int]:
    """Trading days on which new money arrives, all before the anniversary ``END``."""
    days: list[int] = []

    def roll(day: pd.Timestamp) -> int:
        return int(dates.searchsorted(day))                  # first trading day on or after

    if cadence == "monthly":
        month = START.to_period("M")
        while (first := (month + len(days)).to_timestamp()) < END:
            day = roll(first + pd.offsets.Week(weekday=0, n=0))         # n=0: stay if a Monday
            if dates[day] >= END:                                       # never buy on the anniversary itself
                break
            days.append(day)
    elif cadence == "biweekly":
        monday = START
        while monday < END:
            days.append(roll(monday))
            monday += pd.Timedelta(days=14)
    else:
        raise ValueError("cadence must be 'monthly' or 'biweekly'")
    return days


def _simulate(
    tickers: list[str],
    weights: dict[int, pd.Series],           # buy-day index -> target weights
    buy_days: list[int],
    end: int,
    rule: str | None,                        # None = never sell, "close" or "touch"
    per_buy: float,
    reuse_cash: bool = True,
    rebalance: bool = False,                 # True = sell the whole account at every buy day, winners and losers
    record: bool = False,                    # True = also return every position's return at every day's close
    dividends: np.ndarray | None = None,     # days x tickers: cash paid per share held on each ex-dividend day
) -> dict:
    """One account over the span. With ``rebalance`` every buy day starts by
    liquidating everything at that day's close, and the new recommendation is
    bought with all of it; the take-profit rule still applies inside each
    holding window.

    ``dividends`` models dividends paid in cash rather than reinvested in the
    same stock: whoever holds a share the day before its ex-dividend day gets
    that cash (credited on the ex-dividend day), and it waits for the next buy
    day like sale proceeds. Pass it with a price-only price series, where the
    price drops by the dividend on the ex-dividend day."""
    close = pd.read_parquet(_cache("close"))[tickers]
    high = pd.read_parquet(_cache("high"))[tickers].to_numpy()
    opening = pd.read_parquet(_cache("open"))[tickers].to_numpy()
    # value positions at the last known price; trigger only on a real print
    valued = close.ffill().to_numpy()
    raw = close.to_numpy()
    dates = close.index

    shares = np.zeros(len(tickers))
    cost = np.zeros(len(tickers))
    cash = idle = deployed = 0.0
    sells: list[tuple] = []
    skipped_buys = 0
    path: list[tuple[pd.Timestamp, float, float]] = []

    def sell(indices: np.ndarray, fill: np.ndarray) -> None:
        """Sell whole positions at ``fill``; proceeds wait for the next buy day."""
        nonlocal cash, idle
        average = cost[indices] / shares[indices]
        proceeds = shares[indices] * fill
        gains = proceeds - cost[indices]
        # (date, ticker column, gain, shares, return at the sale)
        sells.extend(zip([dates[day]] * len(indices), indices, gains, shares[indices].copy(),
                         fill / average - 1.0))
        for column, price in zip(indices, fill):         # each purchase is its own tax lot
            for bought, lot_shares, lot_cost in lots.pop(int(column), []):
                lot_log.append((day, int(column), bought, lot_shares * price - lot_cost))
        if reuse_cash:
            cash += proceeds.sum()
        else:
            idle += proceeds.sum()
        shares[indices] = 0.0
        cost[indices] = 0.0

    snapshots: list[tuple[int, np.ndarray, np.ndarray]] = []     # per day: held columns and their returns
    lots: dict[int, list[tuple[int, float, float]]] = {}         # column -> (buy day, shares, cost)
    lot_log: list[tuple] = []                                    # (sale day, column, buy day, gain) per lot sold
    dividend_cash = 0.0
    dividend_payments = 0

    for day in range(buy_days[0], end + 1):
        if dividends is not None:
            # shares still held from yesterday's close are the ones entitled to today's dividend
            paid = shares * np.nan_to_num(dividends[day])        # a blank means no dividend that day
            dividend_cash += float(paid.sum())
            dividend_payments += int(((paid > 0) & (shares > 0)).sum())
            if reuse_cash:
                cash += float(paid.sum())
            else:
                idle += float(paid.sum())

        bought_today = np.zeros(len(tickers), dtype=bool)
        if day in buy_days:
            if rebalance:
                held_now = np.flatnonzero((shares > 0) & ~np.isnan(raw[day]))   # a halted stock waits
                if len(held_now):
                    sell(held_now, raw[day, held_now])
            budget = per_buy + (cash if reuse_cash else 0.0)
            deployed += budget
            cash -= budget - per_buy
            plan = allocate_fractional_buys(weights[day], budget)
            columns = close.columns.get_indexer(plan.index)
            assert (columns >= 0).all(), "a recommendation is not in the price data"
            price = raw[day, columns]
            ok = ~np.isnan(price)
            skipped_buys += int((~ok).sum())
            bought = plan["dollars"].to_numpy()[ok]
            shares[columns[ok]] += bought / price[ok]
            cost[columns[ok]] += bought
            for column, lot_shares, lot_cost in zip(columns[ok], bought / price[ok], bought):
                lots.setdefault(int(column), []).append((day, float(lot_shares), float(lot_cost)))
            bought_today[columns] = True

        held = (shares > 0) & ~bought_today
        if rule is not None and held.any():
            average = cost[held] / shares[held]
            if rule == "close":
                trigger = raw[day, held] / average - 1.0 >= TAKE
                fill = raw[day, held]
            else:
                trigger = high[day, held] / average - 1.0 >= TAKE
                # a gap above the target at the open fills at the open, otherwise at the target
                fill = np.where(opening[day, held] >= (1 + TAKE) * average, opening[day, held],
                                (1 + TAKE) * average)
            trigger &= ~np.isnan(fill)
            sold = np.flatnonzero(held)[trigger]
            if len(sold):
                sell(sold, fill[trigger])

        if record:
            live = np.flatnonzero(shares > 0)
            snapshots.append((day, live, valued[day, live] * shares[live] / cost[live] - 1.0))

        month_end = day == end or dates[day].month != dates[day + 1].month
        if month_end:
            # nansum: a stock not listed yet has no price, but zero shares of it
            value = cash + idle + np.nansum(shares * valued[day])
            path.append((dates[day], value, per_buy * sum(d <= day for d in buy_days)))

    value = cash + idle + np.nansum(shares * valued[end])
    realized = float(sum(gain for _, _, gain, _, _ in sells))
    return {
        "value": value,
        "pnl": value - per_buy * len(buy_days),
        "sells": len(sells),
        "realized": realized,
        "unrealized": float(np.nansum(shares * valued[end]) - cost.sum()),
        "still held": int((shares > 0).sum()),
        "underwater": int(((shares > 0) & (valued[end] * shares < cost)).sum()),
        "skipped buys": skipped_buys,
        "deployed": deployed,
        "dividends": dividend_cash,      # cash dividends collected (only when a dividends table is passed)
        "dividend payments": dividend_payments,
        "lot log": lot_log,              # (sale day, column, buy day, gain) per lot sold
        "open lots": [(column, bought, lot_shares * valued[end, column] - lot_cost)
                      for column, held in lots.items() for bought, lot_shares, lot_cost in held],
        "sell log": sells,                 # (date, ticker column, gain, shares, return) per sale
        "snapshots": snapshots,            # (day, held columns, returns) per day, only when record=True
        "path": path,
    }


def _annualized(value: float, horizon: float, times: list[float], per_buy: float) -> float:
    """Annual return on the money while it was invested (equal buys at ``times`` months)."""
    def npv(rate: float) -> float:
        return sum(per_buy * (1 + rate) ** (horizon - t) for t in times) - value

    low, high = -0.99, 10.0
    for _ in range(200):
        mid = (low + high) / 2
        low, high = (mid, high) if npv(mid) < 0 else (low, mid)
    return (1 + (low + high) / 2) ** 12 - 1


def _crosscheck(close, algo, spy, buy_days, end, results, per_buy) -> None:
    """Recompute the two never-sell results by hand; the simulator must agree to the cent."""
    by_name = dict(results)
    priced = close.ffill()

    def hand(weights: dict) -> float:
        total = 0.0
        for day in buy_days:
            plan = allocate_fractional_buys(weights[day], per_buy)
            bought = plan["dollars"] / priced.iloc[day][plan.index]
            total += float((bought * priced.iloc[end][plan.index]).sum())
        return total

    for name, weights in ((f"{ALGO}, never sell", algo), ("SPY, never sell", spy)):
        gap = abs(hand(weights) - by_name[name]["value"])
        assert gap < 0.01, f"simulator and hand calculation disagree on {name}: {gap:.4f}"


def _sweep(stocks, algo, buy_days, dates, per_buy) -> None:
    """The same buys under a range of sell thresholds. The recommendations are reused, so it is quick."""
    global TAKE
    for label, when in (("one year after the first buy" if SPAN == 1 else "two years after the first buy", END), ("through the latest close", dates[-1])):
        end = int(dates.searchsorted(when))
        print(f"\n=== {ALGO}: sell-threshold sweep | {label}: {dates[buy_days[0]].date()} -> {dates[end].date()} ===")
        print(f"contributions: ${YEARLY:,.0f}. Sells are checked at the close; 'touch' sells the moment a "
              f"day's high reaches the threshold instead.\n")
        print(f"{'sell at':<9}{'end value':>11}{'p/l':>10}{'pct':>8}{'touch':>11}{'sells':>7}{'banked':>9}"
              f"{'unsold':>9}{'left':>7}{'below cost':>12}")
        for pct in (None, *range(5, 16)):
            TAKE = 0.08 if pct is None else pct / 100
            r = _simulate(stocks, algo, buy_days, end, None if pct is None else "close", per_buy, True)
            touch = r["value"] if pct is None else _simulate(
                stocks, algo, buy_days, end, "touch", per_buy, True)["value"]
            print(f"{('never' if pct is None else f'+{pct}%'):<9}{r['value']:>11,.0f}{r['pnl']:>+10,.0f}"
                  f"{r['pnl'] / YEARLY:>8.1%}{touch:>11,.0f}{r['sells']:>7}{r['realized']:>+9,.0f}"
                  f"{r['unrealized']:>+9,.0f}{r['still held']:>7}{r['underwater']:>12}")
    TAKE = 0.08


def main() -> None:
    t0 = time.time()
    ensure_prices()
    close = pd.read_parquet(_cache("close")).sort_index()
    stocks = [t for t in close.columns if t != "SPY"]
    dates = close.index
    buy_days = _buy_days(dates, CADENCE)
    per_buy = YEARLY / len(buy_days)
    spy = {day: pd.Series({"SPY": 1.0}) for day in buy_days}
    first = buy_days[0]

    print(f"gamma={GAMMA:g}, K={K}, {MONTHS}-month history{' | expected return FLIPPED (dip portfolio)' if FLIP else ''}"
          f"\n{CADENCE}: {len(buy_days)} buys of ${per_buy:,.2f} = ${YEARLY:,.0f} over {SPAN} year{'s' if SPAN > 1 else ''}, "
          f"{dates[first].date()} to {dates[int(dates.searchsorted(END))].date()}")
    algo = load_weights(dates, buy_days)

    # equal weight over exactly the stocks the optimizer could use that month
    equal: dict[int, pd.Series] = {}
    for day in buy_days:
        window = close[stocks].iloc[:day].loc[dates[day] - pd.DateOffset(months=MONTHS):]
        names = window.dropna(axis=1).columns
        equal[day] = pd.Series(1.0 / len(names), index=names)

    per_buy_count = [len(algo[day]) for day in buy_days]
    print(f"optimizer picks per buy day: {min(per_buy_count)}-{max(per_buy_count)} names "
          f"(mean {np.mean(per_buy_count):.0f}); equal weight uses {len(equal[first])} names")

    if SWEEP:
        _sweep(stocks, algo, buy_days, dates, per_buy)
        print(f"\ndone in {time.time() - t0:.0f}s")
        return

    rows = [
        (f"{ALGO} + 8% sell, cash reused (day's close)", stocks, algo, "close", True),
        (f"{ALGO} + 8% sell, cash reused (intraday touch)", stocks, algo, "touch", True),
        (f"{ALGO} + 8% sell, cash left idle (intraday touch)", stocks, algo, "touch", False),
        (f"{ALGO}, never sell", stocks, algo, None, True),
        (f"{ALGO} + 8% sell, and everything sold every buy day", stocks, algo, "close", True, "rebalance"),
        (f"{ALGO}, everything sold every buy day, no take-profit", stocks, algo, None, True, "rebalance"),
        ("Equal-weight 500 + 8% sell, cash reused (intraday touch)", stocks, equal, "touch", True),
        ("Equal-weight 500, never sell", stocks, equal, None, True),
        ("SPY + 8% sell, cash reused (intraday touch)", ["SPY"], spy, "touch", True),
        ("SPY, never sell", ["SPY"], spy, None, True),
    ]

    for label, when in (("one year after the first buy" if SPAN == 1 else "two years after the first buy", END), ("through the latest close", dates[-1])):
        end = int(dates.searchsorted(when))
        months = lambda d: (dates[d] - dates[first]).days / 30.44      # noqa: E731
        horizon, times = months(end), [months(d) for d in buy_days]
        results = [(row[0], _simulate(row[1], row[2], buy_days, end, row[3], per_buy, row[4],
                                      rebalance=len(row) > 5 and row[5] == "rebalance")) for row in rows]
        _crosscheck(close, algo, spy, buy_days, end, results, per_buy)

        print(f"\n=== {label}: {dates[first].date()} -> {dates[end].date()} ===")
        print(f"contributions: ${YEARLY:,.0f} | SPY over the period: "
              f"{close['SPY'].iloc[end] / close['SPY'].iloc[first] - 1:+.1%}")
        print(f"{'strategy':<56}{'end value':>11}{'p/l':>10}{'annual':>8}{'sells':>7}{'realized':>10}{'unreal.':>10}{'held':>6}{'<cost':>6}")
        for name, r in results:
            print(f"{name:<56}{r['value']:>11,.2f}{r['pnl']:>+10,.0f}{_annualized(r['value'], horizon, times, per_buy):>8.1%}"
                  f"{r['sells']:>7}{r['realized']:>+10,.0f}{r['unrealized']:>+10,.0f}{r['still held']:>6}{r['underwater']:>6}")
        mine = results[0][1]
        print(f"your plan (first row) deployed ${mine['deployed']:,.0f}: ${YEARLY:,.0f} new money "
              f"+ ${mine['deployed'] - YEARLY:,.0f} reused from sells")

    print(f"\nmonth-end value of the whole account, {SPAN} year{'s' if SPAN > 1 else ''}")
    paths = {name: {d: (v, put) for d, v, put in r["path"]} for name, r in results[:4]}
    months = [d for d, _, _ in results[0][1]["path"]]
    print(f"{'month':<10}{'put in':>8}" + "".join(f"{n[:30]:>32}" for n in paths))
    for month in months:
        put_in = results[0][1]["path"][[d for d, _, _ in results[0][1]["path"]].index(month)][2]
        print(f"{month.strftime('%Y-%m'):<10}{put_in:>8,.0f}"
              + "".join(f"{paths[n][month][0]:>32,.0f}" for n in paths))
    if any(r["skipped buys"] for _, r in results):
        print("buys skipped for lack of a price that day:", {n: r["skipped buys"] for n, r in results if r["skipped buys"]})
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
