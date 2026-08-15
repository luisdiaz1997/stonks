"""Offline tests for Robinhood dollar-order allocation."""

import numpy as np
import pandas as pd
import pytest

from stonks.robinhood import allocate_fractional_buys, submit_fractional_buys


def test_allocate_fractional_buys_uses_exact_budget() -> None:
    weights = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2})
    plan = allocate_fractional_buys(weights, 123.45)

    assert np.isclose(plan["dollars"].sum(), 123.45)
    assert (plan["dollars"] >= 1.0).all()
    assert np.isclose(plan["target_weight"].sum(), 1.0)
    assert plan.loc["AAPL", "dollars"] > plan.loc["MSFT", "dollars"]


def test_allocate_fractional_buys_drops_smallest_weights() -> None:
    weights = pd.Series({f"T{i:02d}": 1.0 for i in range(100)})
    plan = allocate_fractional_buys(weights, 50.0)

    assert len(plan) == 50
    assert np.isclose(plan["dollars"].sum(), 50.0)
    assert (plan["dollars"] >= 1.0).all()


def test_allocate_fractional_buys_validates_budget() -> None:
    with pytest.raises(ValueError, match="at least"):
        allocate_fractional_buys(pd.Series({"AAPL": 1.0}), 0.99)


def test_submit_fractional_buys_passes_confirmed_amounts() -> None:
    class FakeRobinhood:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def order_buy_fractional_by_price(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"id": f"order-{args[0]}", "state": "queued"}

    rh = FakeRobinhood()
    plan = pd.DataFrame(
        {"target_weight": [0.6, 0.4], "dollars": [60.0, 40.0]},
        index=["AAPL", "MSFT"],
    )
    results = submit_fractional_buys(rh, plan, account_number="ABC123")

    assert len(rh.calls) == 2
    assert rh.calls[0][0] == ("AAPL", 60.0)
    assert rh.calls[0][1]["account_number"] == "ABC123"
    assert (results["state"] == "queued").all()


def test_submit_fractional_buys_stops_after_ambiguous_failure() -> None:
    class FakeRobinhood:
        def __init__(self) -> None:
            self.calls = 0

        def order_buy_fractional_by_price(self, *args, **kwargs):
            self.calls += 1
            return None

    rh = FakeRobinhood()
    plan = pd.DataFrame(
        {"target_weight": [0.6, 0.4], "dollars": [60.0, 40.0]},
        index=["AAPL", "MSFT"],
    )
    results = submit_fractional_buys(rh, plan)

    assert rh.calls == 1
    assert len(results) == 1
    assert results.loc["AAPL", "state"] == "error"
    assert "no order confirmation" in results.loc["AAPL", "error"]
