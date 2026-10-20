"""Command-line interface for stonks, built on Click.

Examples:
    stonks fetch --tickers AAPL,MSFT,GOOG --period 1y
    stonks fetch --top-n 40 --period 1y --field close
    python -m stonks fetch --tickers AAPL --interval 1wk
"""

from __future__ import annotations

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
@click.option("--period", type=str, default="1y", show_default=True, help="History window (1y, 5y, max, ...).")
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
    ticker_list = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else None
    )

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
@click.option("--top-n", type=int, default=500, show_default=True, help="Most-liquid N common stocks.")
@click.option("--months", type=int, default=3, show_default=True, help="Window length (months).")
@click.option("--interval", default="1d", show_default=True, help="Bar resolution.")
@click.option("--gamma", type=float, default=20.0, show_default=True, help="Risk aversion (higher = more diversified).")
@click.option(
    "-k", "--k-factors", "k_factors", type=int, default=10, show_default=True,
    help="Number of SVD factors; also the concentration dial (larger = fewer holdings).",
)
@click.option("--period", default="2y", show_default=True, help="Cache source to slice --months from.")
@click.option("--min-weight", type=float, default=1e-3, show_default=True)
@click.option(
    "--n-restarts", type=int, default=10, show_default=True,
    help="Random restarts; the best final objective is kept (non-convex problem).",
)
@click.option("--lr", type=float, default=0.05, show_default=True,
              help="Optimizer learning rate (smaller = smoother ascent).")
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
) -> None:
    """Recommend a long-only portfolio, optionally buying it in Robinhood."""
    from .portfolio import optimize_portfolio
    from .universe import load_universe

    if amount is not None and not execute:
        raise click.UsageError("--amount requires --execute")

    w = optimize_portfolio(
        top_n=top_n, months=months, interval=interval, gamma=gamma,
        K=k_factors, period=period, min_weight=min_weight, force_refresh=force_refresh,
        n_restarts=n_restarts, lr=lr,
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

    if not execute:
        return

    from .robinhood import (
        account_buying_power,
        allocate_fractional_buys,
        fetch_order_states,
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

    submitted = results[results["order_id"] != ""]
    if not submitted.empty:
        import time as _time

        click.echo("\nWaiting 15s for Robinhood to acknowledge, then checking (read-only)...")
        _time.sleep(15)
        states = fetch_order_states(rh, submitted["order_id"].tolist())
        results["live_state"] = results["order_id"].map(
            lambda oid: states.get(oid, {}).get("state", "")
        )
        results["filled_dollars"] = results["order_id"].map(
            lambda oid: states.get(oid, {}).get("filled_dollars")
        )
        confirm = results.loc[
            results["order_id"] != "",
            ["dollars", "order_id", "live_state", "filled_dollars"],
        ]
        click.echo("\nLive order status:")
        click.echo(
            confirm.to_string(
                formatters={
                    "dollars": "${:,.2f}".format,
                    "filled_dollars": lambda v: "—" if v is None or pd.isna(v) else f"${v:,.2f}",
                }
            )
        )
        click.echo(
            "Queued/unconfirmed GFD orders execute at the next market open; "
            "re-check any time with: stonks orders"
        )

    failures = results[results["error"] != ""]
    skipped = results[results["state"] == "not_attempted"]
    if not failures.empty:
        message = (
            f"{len(failures)} of {len(results)} orders returned an error"
        )
        if not skipped.empty:
            message += f" and {len(skipped)} were never attempted"
        message += (
            "; review Robinhood activity and wait a few minutes before "
            "retrying (order rate limits)"
        )
        raise click.ClickException(message)


@cli.command("orders")
def orders_status() -> None:
    """Re-check the live status of the most recent --execute submission."""
    from .robinhood import fetch_order_states, load_last_orders, login

    saved = load_last_orders()
    if saved is None:
        raise click.ClickException(
            "no previous submission found (run stonks portfolio --execute first)"
        )

    click.echo("Authenticating with Robinhood (read-only)...")
    try:
        rh = login()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    tracked = saved[saved["order_id"].astype(str) != ""]
    if tracked.empty:
        raise click.ClickException("the saved submission has no order ids to check")

    states = fetch_order_states(rh, tracked["order_id"].tolist())
    saved["live_state"] = saved["order_id"].map(
        lambda oid: states.get(oid, {}).get("state", "")
    )
    saved["filled_dollars"] = saved["order_id"].map(
        lambda oid: states.get(oid, {}).get("filled_dollars")
    )
    click.echo(
        saved.to_string(
            formatters={
                "dollars": "${:,.2f}".format,
                "filled_dollars": lambda v: "—" if v is None or pd.isna(v) else f"${v:,.2f}",
            }
        )
    )
    total_filled = saved["filled_dollars"].fillna(0.0).sum()
    click.echo(f"\nfilled so far: ${total_filled:,.2f} of ${saved['dollars'].sum():,.2f}")


@cli.group()
def intraday() -> None:
    """Run constrained intraday research experiments (no broker orders)."""


@intraday.command("backtest")
@click.option(
    "--ticker",
    default="SPY",
    show_default=True,
    help="Research symbol. SPY is only a liquid baseline, not a recommendation.",
)
@click.option("--period", default="30d", show_default=True, help="yfinance history window.")
@click.option(
    "--interval",
    type=click.Choice(["1m", "2m", "5m", "15m", "30m", "60m", "90m"]),
    default="5m",
    show_default=True,
)
@click.option(
    "--csv-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Use a local timestamp-indexed OHLCV CSV instead of downloading.",
)
@click.option(
    "--account-limit",
    "--cash",
    "account_limit",
    type=click.FloatRange(min=1.0),
    default=20.0,
    show_default=True,
    help="Starting simulated cash and absolute account cap.",
)
@click.option(
    "--trade-dollars",
    type=click.FloatRange(min=1.0),
    default=5.0,
    show_default=True,
    help="Maximum notional for each simulated fractional position.",
)
@click.option("--opening-minutes", type=click.IntRange(5, 120), default=30, show_default=True)
@click.option(
    "--stop-pct",
    type=click.FloatRange(min=0.01, max=25.0),
    default=1.0,
    show_default=True,
)
@click.option(
    "--take-pct",
    type=click.FloatRange(min=0.01, max=50.0),
    default=2.0,
    show_default=True,
)
@click.option("--slippage-bps", type=click.FloatRange(min=0.0), default=5.0, show_default=True)
@click.option("--force-refresh", is_flag=True, help="Bypass the local intraday-data cache.")
def intraday_backtest(
    ticker: str,
    period: str,
    interval: str,
    csv_file: str | None,
    account_limit: float,
    trade_dollars: float,
    opening_minutes: int,
    stop_pct: float,
    take_pct: float,
    slippage_bps: float,
    force_refresh: bool,
) -> None:
    """Backtest one long-only opening-range strategy with fractional shares."""
    from .intraday import (
        BacktestConfig,
        OpeningRangeBreakout,
        download_bars,
        normalize_bars,
        run_backtest,
    )

    if trade_dollars > account_limit:
        raise click.UsageError("--trade-dollars cannot exceed --account-limit")

    try:
        if csv_file:
            raw = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            bars = normalize_bars(raw)
            source = csv_file
        else:
            bars = download_bars(
                ticker,
                period=period,
                interval=interval,
                force_refresh=force_refresh,
            )
            source = f"{ticker.upper()} via yfinance"

        config = BacktestConfig(
            initial_cash=account_limit,
            account_cap=account_limit,
            trade_notional=trade_dollars,
            stop_loss_pct=stop_pct / 100,
            take_profit_pct=take_pct / 100,
            slippage_bps=slippage_bps,
        )
        result = run_backtest(
            bars,
            strategy=OpeningRangeBreakout(opening_minutes=opening_minutes),
            config=config,
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    metrics = result.metrics
    click.echo(f"\nIntraday research backtest: {source}")
    click.echo(
        f"{len(bars)} regular-session bars | opening range {opening_minutes}m | "
        f"next-bar entries | ${trade_dollars:.2f} maximum position"
    )
    click.echo("\nResults (simulation only; not evidence of future performance)")
    click.echo(f"trades:          {int(metrics['trade_count'])}")
    click.echo(f"final equity:    ${float(metrics['final_equity']):.4f}")
    click.echo(f"total P&L:       ${float(metrics['total_pnl']):+.4f}")
    click.echo(f"total return:    {float(metrics['total_return_pct']):+.3f}%")
    click.echo(f"win rate:        {float(metrics['win_rate_pct']):.1f}%")
    click.echo(f"max drawdown:    {float(metrics['max_drawdown_pct']):.3f}%")
    if not result.trades.empty:
        display = result.trades.copy()
        display["entry_time"] = display["entry_time"].map(str)
        display["exit_time"] = display["exit_time"].map(str)
        click.echo("\nTrades")
        click.echo(display.to_string(index=False))
    click.echo("\nNo Robinhood login occurred and no orders were submitted.")


if __name__ == "__main__":
    cli()
