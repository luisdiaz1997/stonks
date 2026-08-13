"""Return computation.

Convert a price matrix (N tickers × T dates) to daily returns — the stationary
increments you actually want to model. Simple (arithmetic) returns by default,
since "how much did it move" is a percent; pass ``log=True`` for log returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Daily returns from a price matrix.

    Args:
        prices: N tickers (rows) × T dates (columns), as returned by
            :func:`stonks.get_prices`. Use the adjusted close (``field="close"``)
            so the returns are total returns (splits/dividends accounted for).
        log: If True, return log returns ``log(P_t / P_{t-1})``; otherwise
            simple (arithmetic) returns ``P_t / P_{t-1} - 1`` (default).

    Returns:
        An N × (T-1) DataFrame of daily returns. The leading column (which has
        no prior day) is dropped.
    """
    prev = prices.shift(axis=1)
    ratio = prices / prev
    rets = np.log(ratio) if log else ratio - 1
    return rets.iloc[:, 1:]
