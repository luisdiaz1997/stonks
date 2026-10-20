"""Small, deliberately constrained intraday-trading experiments.

The intraday package is separate from the multi-month portfolio optimizer.  It
contains an offline backtester and one transparent baseline strategy.  Nothing
in this package submits broker orders.
"""

from .backtest import BacktestConfig, BacktestResult, Trade, run_backtest
from .data import download_bars, normalize_bars
from .strategy import OpeningRangeBreakout

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "OpeningRangeBreakout",
    "download_bars",
    "normalize_bars",
    "run_backtest",
]
