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


if __name__ == "__main__":
    cli()
