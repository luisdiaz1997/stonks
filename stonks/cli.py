"""Command-line interface for stonks, built on Click.

Examples:
    stonks fetch --tickers AAPL,MSFT,GOOG --period 1y
    stonks fetch --top-n 40 --period 1y --field close
    python -m stonks fetch --tickers AAPL --interval 1wk
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import click
import pandas as pd

from . import __version__
from .download import get_prices


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """stonks: simple, exploratory algorithms for stocks and investments."""


@cli.command()
@click.option(
    "--top-n",
    type=int,
    default=1500,
    show_default=True,
    help="Most-liquid N common stocks to keep (ignored if --tickers is given).",
)
@click.option(
    "--tickers",
    type=str,
    default=None,
    help="Comma-separated tickers, e.g. AAPL,MSFT. Skips ranking (fast).",
)
@click.option(
    "--period", type=str, default="1y", show_default=True, help="History window (1y, 5y, max, ...)."
)
@click.option(
    "--interval",
    type=str,
    default="1d",
    show_default=True,
    help="Bar resolution: 1d, 1wk, 1mo, or intraday 1m..90m (last 60d).",
)
@click.option(
    "--field",
    type=click.Choice(["open", "high", "low", "close", "volume"], case_sensitive=False),
    default="close",
    show_default=True,
    help="OHLCV field to return.",
)
@click.option("--force-refresh", is_flag=True, default=False, help="Bypass cache and re-download.")
def fetch(
    top_n: int,
    tickers: str | None,
    period: str,
    interval: str,
    field: str,
    force_refresh: bool,
) -> None:
    """Fetch an N×T price matrix and print a preview."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else None

    matrix = get_prices(
        top_n=top_n,
        tickers=ticker_list,
        period=period,
        interval=interval,
        field=field,
        force_refresh=force_refresh,
    )

    click.echo(
        f"\nmatrix: {matrix.shape[0]} tickers × {matrix.shape[1]} bars "
        f"(field={field}, interval={interval}, period={period})"
    )
    # Top-left 5×5 preview.
    click.echo(matrix.iloc[: min(5, len(matrix)), : min(5, matrix.shape[1])].to_string())


@cli.command()
@click.option(
    "--top-n", type=int, default=500, show_default=True, help="Most-liquid N common stocks."
)
@click.option("--months", type=int, default=3, show_default=True, help="Window length (months).")
@click.option("--interval", default="1d", show_default=True, help="Bar resolution.")
@click.option(
    "--gamma",
    type=float,
    default=20.0,
    show_default=True,
    help="Risk aversion (higher = more diversified).",
)
@click.option(
    "-k",
    "--k-factors",
    "k_factors",
    type=int,
    default=10,
    show_default=True,
    help="Number of SVD factors; also the concentration dial (larger = fewer holdings).",
)
@click.option(
    "--period", default="2y", show_default=True, help="Cache source to slice --months from."
)
@click.option("--min-weight", type=float, default=1e-3, show_default=True)
@click.option(
    "--n-restarts",
    type=int,
    default=10,
    show_default=True,
    help="Random restarts; the best final objective is kept (non-convex problem).",
)
@click.option(
    "--lr",
    type=float,
    default=0.05,
    show_default=True,
    help="Optimizer learning rate (smaller = smoother ascent).",
)
@click.option("--force-refresh", is_flag=True, default=False, help="Bypass cache.")
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="After confirmation, buy the recommendation in Robinhood.",
)
@click.option(
    "--amount",
    type=click.FloatRange(min=1.0),
    default=None,
    help="Dollars to invest with --execute (default: all available buying power).",
)
@click.option(
    "--save",
    "save_to",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write the recommendation to this file and stop without orders, for `stonks rebuy` to buy later.",
)
def portfolio(
    top_n: int,
    months: int,
    interval: str,
    gamma: float,
    k_factors: int,
    period: str,
    min_weight: float,
    n_restarts: int,
    lr: float,
    force_refresh: bool,
    execute: bool,
    amount: float | None,
    save_to: str | None,
) -> None:
    """Recommend a long-only portfolio, optionally buying it in Robinhood."""
    from .portfolio import optimize_portfolio
    from .universe import load_universe

    if amount is not None and not execute:
        raise click.UsageError("--amount requires --execute")
    if save_to is not None and (execute or amount is not None):
        raise click.UsageError(
            "--save cannot be combined with --execute or --amount: it writes the recommendation and places no orders"
        )

    w = optimize_portfolio(
        top_n=top_n,
        months=months,
        interval=interval,
        gamma=gamma,
        K=k_factors,
        period=period,
        min_weight=min_weight,
        force_refresh=force_refresh,
        n_restarts=n_restarts,
        lr=lr,
    )

    # enrich with company names when available
    try:
        names = load_universe().set_index("ticker")["name"]
        w = w.assign(name=w.index.map(names).fillna(""))
    except Exception:
        pass

    click.echo(
        f"\nRecommended long-only portfolio: {len(w)} holdings "
        f"(gamma={gamma}, K={k_factors}, {months}m window, top {top_n} stocks)"
    )
    click.echo(w.to_string())
    click.echo(f"\nsum(weight) = {w['weight'].sum():.4f}")

    if save_to is not None:
        _save_pick(
            save_to,
            w["weight"],
            {
                "top_n": top_n,
                "months": months,
                "K": k_factors,
                "gamma": gamma,
                "n_restarts": n_restarts,
                "lr": lr,
            },
        )
        click.echo(
            f"\nSaved to {save_to} without placing any orders, for: stonks rebuy --pick {save_to}"
        )
        return
    if not execute:
        return

    from .robinhood import (
        account_buying_power,
        allocate_fractional_buys,
        login,
        save_last_orders,
        submit_fractional_buys,
    )

    click.echo("\nAuthenticating with Robinhood (no orders are submitted yet)...")
    try:
        rh = login()
        buying_power, account_number = account_buying_power(rh)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    budget = buying_power if amount is None else amount
    if budget > buying_power + 1e-9:
        raise click.ClickException(
            f"requested ${budget:,.2f}, but Robinhood reports only "
            f"${buying_power:,.2f} of buying power"
        )
    try:
        plan = allocate_fractional_buys(w["weight"], budget)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"\nProposed Robinhood orders: {len(plan)} buy-only fractional market orders "
        f"totaling ${plan['dollars'].sum():,.2f}"
    )
    click.echo(plan.to_string(formatters={"dollars": "${:,.2f}".format}))
    click.echo(
        "\nExisting holdings will not be sold. Orders are GFD regular-hours orders "
        "and may queue while the market is closed."
    )
    click.echo(
        f"Submissions are throttled (2s between orders): {len(plan)} orders take "
        f"~{2 * len(plan) / 60:.0f} minutes."
    )

    if not click.confirm("Do you want to submit this portfolio to Robinhood?", default=False):
        click.echo("Canceled; no orders were submitted.")
        return

    try:
        results = submit_fractional_buys(rh, plan, account_number=account_number)
    except Exception as exc:
        raise click.ClickException(f"Robinhood order submission failed: {exc}") from exc

    click.echo("\nRobinhood order responses:")
    click.echo(results.to_string())
    save_last_orders(results)  # persist ids before anything else can fail

    if (results["order_id"] != "").any():
        import time as _time

        click.echo("\nWaiting 15s for Robinhood to acknowledge, then checking (read-only)...")
        _time.sleep(15)
        _print_live_status(rh, results, "buy")
        click.echo(
            "Queued/unconfirmed GFD orders execute at the next market open; "
            "re-check any time with: stonks orders"
        )

    _fail_on_order_errors(results)


def _save_pick(path: str, weights: pd.Series, settings: dict[str, Any]) -> None:
    """Write a recommendation for ``stonks rebuy``, with when it was made and how."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "weights": {str(ticker): float(weight) for ticker, weight in weights.items()},
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_PICK_MAX_AGE = timedelta(hours=12)  # a recommendation is for the day it was computed


def _load_pick(path: str) -> tuple[pd.Series, datetime, dict[str, Any]]:
    """The recommendation ``portfolio --save`` wrote to ``path``; it must be from the same day."""
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        saved_at = datetime.fromisoformat(str(saved["saved_at"]))
        weights = pd.Series(saved["weights"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(
            f"{path} does not hold a recommendation from `stonks portfolio --save` ({exc})"
        ) from exc
    saved_at = saved_at if saved_at.tzinfo else saved_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - saved_at
    if not -timedelta(minutes=5) <= age <= _PICK_MAX_AGE:
        raise click.ClickException(
            f"the recommendation in {path} is {age.total_seconds() / 3600:.1f} hours old; it is for the day "
            "it was computed, so save a fresh one first"
        )
    total = float(weights.sum())
    if weights.empty or not math.isclose(total, 1.0, rel_tol=1e-6) or not bool((weights > 0).all()):
        raise click.ClickException(
            f"the recommendation in {path} is not a set of positive weights summing to 1"
        )
    return weights / total, saved_at, saved.get("settings") or {}


def _print_live_status(rh: Any, results: pd.DataFrame, side: str) -> None:
    """Show the live state of every submitted order in ``results`` (read-only GETs)."""
    from .robinhood import fetch_order_states

    order_ids = results["order_id"].astype(str)
    submitted = results[order_ids != ""]
    # a book of a hundred orders is checked a second apart, to stay inside Robinhood's rate limit
    interval = 1.0 if len(submitted) > 10 else 0.5
    states = fetch_order_states(
        rh, submitted["order_id"].astype(str).tolist(), poll_interval=interval
    )
    results["live_state"] = order_ids.map(lambda oid: states.get(oid, {}).get("state", ""))
    results["filled_dollars"] = order_ids.map(lambda oid: states.get(oid, {}).get("filled_dollars"))
    size = "dollars" if side == "buy" else "quantity"
    click.echo("\nLive order status:")
    click.echo(
        results.loc[submitted.index, [size, "order_id", "live_state", "filled_dollars"]].to_string(
            formatters={
                "dollars": "${:,.2f}".format,
                "quantity": lambda v: f"{v:.6f}",
                "filled_dollars": lambda v: "—" if v is None or pd.isna(v) else f"${v:,.2f}",
            }
        )
    )


def _fail_on_order_errors(results: pd.DataFrame) -> None:
    """Exit with an error when a submission had failed or never-attempted orders."""
    failures = results[results["error"] != ""]
    skipped = results[results["state"] == "not_attempted"]
    if failures.empty:
        return
    message = f"{len(failures)} of {len(results)} orders returned an error"
    if not skipped.empty:
        message += f" and {len(skipped)} were never attempted"
    message += (
        "; review Robinhood activity and wait a few minutes before " "retrying (order rate limits)"
    )
    raise click.ClickException(message)


@cli.command("orders")
def orders_status() -> None:
    """Re-check the live status of the most recent --execute submissions."""
    from .robinhood import load_last_orders, login

    saved = {side: load_last_orders(side) for side in ("buy", "sell")}
    if all(frame is None for frame in saved.values()):
        raise click.ClickException(
            "no previous submission found (run stonks portfolio --execute "
            "or stonks sell --execute first)"
        )

    click.echo("Authenticating with Robinhood (read-only)...")
    try:
        rh = login()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    checked_any = False
    for side, frame in saved.items():
        if frame is None or (frame["order_id"].astype(str) == "").all():
            continue
        checked_any = True
        click.echo(f"\n{side.capitalize()} orders:")
        _print_live_status(rh, frame, side)
        filled = frame["filled_dollars"].fillna(0.0).sum()
        planned = f" of ${frame['dollars'].sum():,.2f}" if side == "buy" else " in proceeds"
        click.echo(f"\nfilled so far: ${filled:,.2f}{planned}")

    if not checked_any:
        raise click.ClickException("the saved submission has no order ids to check")


def _dollars(value: float) -> str:
    """$1.23, or $0.1234 for sub-dollar prices."""
    return f"${value:,.2f}" if value >= 1 else f"${value:,.4f}"


def _percent(value: float) -> str:
    return f"{value:+.2f}%"


@cli.command()
@click.option(
    "--take-profit",
    type=click.FloatRange(min=0.1),
    default=8.0,
    show_default=True,
    help="Sell at this total return in percent (price vs. average buy price).",
)
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="After confirmation, sell the triggered positions in Robinhood.",
)
@click.option(
    "--yes",
    "assume_yes",
    is_flag=True,
    default=False,
    help="With --execute, skip the confirmation prompt (for scheduled runs).",
)
@click.option(
    "--everything",
    "sell_everything",
    is_flag=True,
    default=False,
    help="Sell every position that can be sold, whatever its return; the take-profit is not used.",
)
def sell(take_profit: float, execute: bool, assume_yes: bool, sell_everything: bool) -> None:
    """Sell positions that reached a take-profit threshold, or all of them (dry run by default)."""
    from .robinhood import (
        account_buying_power,
        load_positions,
        login,
        market_status,
        record_sells,
        save_last_orders,
        select_sells,
        submit_fractional_sells,
    )

    if assume_yes and not execute:
        raise click.UsageError("--yes requires --execute")

    click.echo("Authenticating with Robinhood (no orders are submitted yet)...")
    try:
        rh = login()
        _, account_number = account_buying_power(rh)
        is_trading_day, in_session = market_status(rh)
        positions = load_positions(rh, account_number=account_number)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if not is_trading_day:
        click.echo("\nUS market is closed today; prices are the last regular-session close.")
    elif not in_session:
        click.echo(
            "\nUS market is outside regular hours (9:30 AM-4:00 PM ET); "
            "prices are the last close."
        )
    click.echo(
        "\nSelling every position that can be sold cleanly; the take-profit is not used."
        if sell_everything
        else f"\nTake-profit rule: sell at +{take_profit:g}% total return (price vs. average buy price)"
    )
    if positions.empty:
        click.echo("No open positions.")
        return

    plan = select_sells(positions, None if sell_everything else take_profit)
    action = plan["skip"].where(plan["skip"] != "", "SELL")
    display = plan.assign(action=action).drop(columns="sell")
    click.echo(
        display[
            [
                "quantity",
                "avg_buy_price",
                "price",
                "value",
                "total_return_pct",
                "day_change_pct",
                "action",
            ]
        ].to_string(
            formatters={
                "quantity": lambda v: f"{v:.6f}",
                "avg_buy_price": _dollars,
                "price": _dollars,
                "value": "${:,.2f}".format,
                "total_return_pct": _percent,
                "day_change_pct": _percent,
            }
        )
    )

    to_sell = plan[plan["sell"]]
    if to_sell.empty:
        click.echo(
            "\nNo position can be sold; there are no orders to place."
            if sell_everything
            else "\nNo position reached the threshold; nothing to sell."
        )
        return
    click.echo(
        f"\n{len(to_sell)} of {len(plan)} positions to sell, about ${to_sell['value'].sum():,.2f}."
    )

    if not execute:
        click.echo("Dry run: no orders were submitted. Add --execute to sell.")
        return
    if not in_session:
        click.echo("Market closed: no orders were submitted (sells are regular-hours orders).")
        return
    click.echo(
        "Each order sells the full position as a GFD regular-hours fractional market "
        "sell, throttled 2s apart."
    )
    if sell_everything:
        click.echo("A failed order is reported, and the run goes on to the next position.")
    if not assume_yes and not click.confirm(
        f"Sell these {len(to_sell)} positions in Robinhood?", default=False
    ):
        click.echo("Canceled; no orders were submitted.")
        return

    try:
        results = submit_fractional_sells(
            rh, to_sell, account_number=account_number, stop_on_failure=not sell_everything
        )
    except Exception as exc:
        raise click.ClickException(f"Robinhood order submission failed: {exc}") from exc

    click.echo("\nRobinhood order responses:")
    click.echo(results.to_string(formatters={"quantity": lambda v: f"{v:.6f}"}))
    save_last_orders(results, "sell")  # persist ids before anything else can fail

    if (results["order_id"] != "").any():
        import time as _time

        click.echo("\nWaiting 15s for Robinhood to acknowledge, then checking (read-only)...")
        _time.sleep(15)
        _print_live_status(rh, results, "sell")
        click.echo("Re-check any time with: stonks orders")
        try:
            proceeds = record_sells("everything" if sell_everything else "take-profit", results)
        except Exception as exc:  # noqa: BLE001 - the orders are placed; say what was lost
            raise click.ClickException(
                f"the orders were placed but could not be put on the ledger for the next re-buy: {exc}"
            ) from exc
        click.echo(f"Counted for the next re-buy: ${proceeds:,.2f} filled.")

    _fail_on_order_errors(results)


@cli.command("rebuy")
@click.option(
    "--pick",
    "pick_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="A recommendation written by: stonks portfolio --save FILE",
)
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="After confirmation, place the buy orders in Robinhood.",
)
@click.option(
    "--yes",
    "assume_yes",
    is_flag=True,
    default=False,
    help="With --execute, skip the confirmation prompt (for scheduled runs).",
)
def rebuy(pick_path: str, execute: bool, assume_yes: bool) -> None:
    """Buy a saved recommendation with what the jobs have sold since the last rebuy.

    The amount is the filled proceeds of every `stonks sell` run since the
    previous `stonks rebuy`, plus whatever that run left undeployed, capped at
    the buying power Robinhood reports. A cycle of selling and re-buying
    therefore moves the same money around and borrows nothing more. It is a
    dry run by default.
    """
    from .robinhood import (
        account_buying_power,
        allocate_fractional_buys,
        cash_to_rebuy,
        login,
        market_status,
        record_buy,
        save_last_orders,
        submit_fractional_buys,
    )

    if assume_yes and not execute:
        raise click.UsageError("--yes requires --execute")
    weights, saved_at, settings = _load_pick(pick_path)
    shown = "".join(
        f", {key}={settings[key]}" for key in ("months", "gamma", "K", "top_n") if key in settings
    )
    click.echo(
        f"Saved recommendation from {saved_at.astimezone():%Y-%m-%d %H:%M %Z}: {len(weights)} names{shown}"
    )

    available, sells = cash_to_rebuy()
    if available < 1.0:
        click.echo(
            f"\nNothing to re-buy: the ledger holds ${available:,.2f} from {sells} sell event(s) since the last re-buy."
        )
        return
    click.echo(f"\nSells since the last re-buy: ${available:,.2f} from {sells} event(s).")

    click.echo("\nAuthenticating with Robinhood (no orders are submitted yet)...")
    try:
        rh = login()
        buying_power, account_number = account_buying_power(rh)
        is_trading_day, in_session = market_status(rh)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if not is_trading_day:
        click.echo("US market is closed today.")
    elif not in_session:
        click.echo("US market is outside regular hours (9:30 AM-4:00 PM ET).")

    budget = min(available, buying_power)
    if budget < available:
        click.echo(
            f"Capped at the ${buying_power:,.2f} of buying power Robinhood reports; the rest waits for the next re-buy."
        )
    if budget < 1.0:
        click.echo("No buying power; no orders.")
        return
    try:
        plan = allocate_fractional_buys(weights, budget)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"\nProposed Robinhood orders: {len(plan)} buy-only fractional market orders totaling "
        f"${plan['dollars'].sum():,.2f}"
    )
    click.echo(plan.to_string(formatters={"dollars": "${:,.2f}".format}))
    if not execute:
        click.echo("\nDry run: no orders were submitted. Add --execute to buy.")
        return
    if not in_session:
        click.echo("\nMarket closed: no orders were submitted (buys are regular-hours orders).")
        return
    if not assume_yes and not click.confirm(
        f"Submit these {len(plan)} buy orders to Robinhood?", default=False
    ):
        click.echo("Canceled; no orders were submitted.")
        return

    try:
        results = submit_fractional_buys(rh, plan, account_number=account_number)
    except Exception as exc:
        raise click.ClickException(f"Robinhood order submission failed: {exc}") from exc

    click.echo("\nRobinhood order responses:")
    click.echo(results.to_string(formatters={"dollars": "${:,.2f}".format}))
    save_last_orders(results)  # persist ids before anything else can fail
    submitted = float(results.loc[results["order_id"].astype(str) != "", "dollars"].sum())
    try:
        record_buy(available, budget, submitted)
    except Exception as exc:  # noqa: BLE001 - the orders are placed; say what was lost
        raise click.ClickException(
            f"the orders were placed but the ledger could not be updated: {exc}"
        ) from exc
    click.echo(
        f"\nPut on the ledger: ${submitted:,.2f} sent to Robinhood, "
        f"${available - submitted:,.2f} carried to the next re-buy."
    )

    if (results["order_id"] != "").any():
        import time as _time

        click.echo("\nWaiting 15s for Robinhood to acknowledge, then checking (read-only)...")
        _time.sleep(15)
        _print_live_status(rh, results, "buy")
        click.echo(
            "Queued/unconfirmed GFD orders execute at the next market open; "
            "re-check any time with: stonks orders"
        )

    _fail_on_order_errors(results)


if __name__ == "__main__":
    cli()
