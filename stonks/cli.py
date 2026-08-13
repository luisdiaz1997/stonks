"""Command-line interface: ``python -m stonks fetch ...`` -> an N×T price matrix."""

from __future__ import annotations

import argparse

from . import __version__
from .download import get_prices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stonks", description="Stock data wrapper.")
    parser.add_argument("--version", action="version", version=f"stonks {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="Fetch an N×T price matrix and print a preview.")
    f.add_argument("--top-n", type=int, default=1500, help="Most-liquid N stocks (default 1500).")
    f.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers (e.g. AAPL,MSFT). Skips ranking.",
    )
    f.add_argument("--period", default="1y", help="History window (e.g. 1y, 5y, max).")
    f.add_argument(
        "--interval", default="1d", help="Bar resolution (1d, 1wk, 1mo, or 1m..90m intraday)."
    )
    f.add_argument("--field", default="close", help="open/high/low/close/volume (default close).")
    f.add_argument("--force-refresh", action="store_true", help="Bypass cache.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "fetch":
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if args.tickers
            else None
        )
        mat = get_prices(
            top_n=args.top_n,
            tickers=tickers,
            period=args.period,
            interval=args.interval,
            field=args.field,
            force_refresh=args.force_refresh,
        )
        print(
            f"\nmatrix: {mat.shape[0]} tickers × {mat.shape[1]} bars "
            f"(field={args.field}, interval={args.interval}, period={args.period})"
        )
        # Top-left 5×5 preview.
        print(mat.iloc[: min(5, len(mat)), : min(5, mat.shape[1])])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
