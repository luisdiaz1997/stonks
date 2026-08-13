"""Offline tests for to_returns (no network)."""

import numpy as np
import pandas as pd

from stonks import to_returns


def test_simple_returns_values() -> None:
    prices = pd.DataFrame(
        [[100.0, 102.0, 99.0, 104.0]],
        index=["AAPL"],
        columns=["d0", "d1", "d2", "d3"],
    )
    rets = to_returns(prices)

    # N x (T-1), leading column dropped.
    assert rets.shape == (1, 3)
    assert list(rets.columns) == ["d1", "d2", "d3"]

    assert np.isclose(rets.loc["AAPL", "d1"], 102.0 / 100.0 - 1.0)
    assert np.isclose(rets.loc["AAPL", "d2"], 99.0 / 102.0 - 1.0)
    assert np.isclose(rets.loc["AAPL", "d3"], 104.0 / 99.0 - 1.0)


def test_log_returns_values() -> None:
    prices = pd.DataFrame(
        [[100.0, 102.0, 99.0]],
        index=["X"],
        columns=["a", "b", "c"],
    )
    rets = to_returns(prices, log=True)
    assert rets.shape == (1, 2)
    assert np.isclose(rets.loc["X", "b"], np.log(102.0 / 100.0))
    assert np.isclose(rets.loc["X", "c"], np.log(99.0 / 102.0))


def test_leading_column_dropped_and_no_nan() -> None:
    prices = pd.DataFrame(
        [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
        index=["A", "B"],
        columns=["d0", "d1", "d2"],
    )
    rets = to_returns(prices)
    assert "d0" not in rets.columns  # the no-prior-day column is gone
    assert not rets.isna().any().any()


def test_multiple_tickers_index_preserved() -> None:
    prices = pd.DataFrame(
        [[100.0, 110.0], [50.0, 55.0]],
        index=["AAPL", "MSFT"],
        columns=["d0", "d1"],
    )
    rets = to_returns(prices)
    assert rets.index.tolist() == ["AAPL", "MSFT"]
    # both up 10%
    assert np.allclose(rets.to_numpy(), [[0.1], [0.1]])
