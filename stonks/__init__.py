"""
stonks: Simple, exploratory algorithms for stocks and investments.

This package currently provides a thin data wrapper,
:func:`get_prices`, that returns a clean **N×T** matrix (N tickers × T time
points) of stock prices for the most-liquid US common stocks (or any ticker
list you pass in). Later layers (differencing, covariance, portfolios) will
build on top of this matrix.

No financial advice; just code to read, run, and tinker with.

Example:
    >>> from stonks import get_prices
    >>> prices = get_prices(tickers=["AAPL", "MSFT"], period="1y", interval="1d")
    >>> prices.shape  # (2 tickers, ~252 trading days)
"""

from .config import Settings
from .download import get_prices
from .portfolio import optimize_portfolio
from .returns import to_returns
from .universe import list_all_common_stocks, load_universe

__version__ = "0.1.0"

__all__ = [
    "get_prices",
    "to_returns",
    "optimize_portfolio",
    "load_universe",
    "list_all_common_stocks",
    "Settings",
    "__version__",
]
