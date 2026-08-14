"""Command-line interface for stonks, built on Click.

Examples:
    stonks fetch --tickers AAPL,MSFT,GOOG --period 1y
    stonks fetch --top-n 40 --period 1y --field close
    python -m stonks fetch --tickers AAPL --interval 1wk
"""

from __future__ import annotations

import click

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
) -> None:
    """Recommend a long-only portfolio (factor-model mean-variance)."""
    from .portfolio import optimize_portfolio
    from .universe import load_universe

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


if __name__ == "__main__":
    cli()
