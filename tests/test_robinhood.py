"""Offline tests for Robinhood dollar-order allocation."""

import numpy as np
import pandas as pd
import pytest

from stonks.robinhood import (
    allocate_fractional_buys,
    fetch_order_states,
    load_last_orders,
    save_last_orders,
    submit_fractional_buys,
)


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


def _no_sleep(monkeypatch: pytest.MonkeyPatch, intervals: list | None = None) -> None:
    """Make submit_fractional_buys' throttle instant, recording the waits."""
    import stonks.robinhood

    def fake_sleep(seconds: float) -> None:
        if intervals is not None:
            intervals.append(seconds)

    monkeypatch.setattr(stonks.robinhood.time, "sleep", fake_sleep)


def test_submit_fractional_buys_passes_confirmed_amounts(monkeypatch) -> None:
    _no_sleep(monkeypatch)

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


def test_submit_fractional_buys_throttles_between_orders(monkeypatch) -> None:
    intervals: list[float] = []
    _no_sleep(monkeypatch, intervals)

    class FakeRobinhood:
        def order_buy_fractional_by_price(self, *args, **kwargs):
            return {"id": f"order-{args[0]}", "state": "queued"}

    plan = pd.DataFrame(
        {"target_weight": [0.5, 0.3, 0.2], "dollars": [50.0, 30.0, 20.0]},
        index=["AAPL", "MSFT", "GOOG"],
    )
    submit_fractional_buys(FakeRobinhood(), plan, order_interval=1.5, verbose=False)

    # one wait between consecutive orders, none before the first or after the last
    assert intervals == [1.5, 1.5]


def test_submit_fractional_buys_rejects_negative_interval(monkeypatch) -> None:
    _no_sleep(monkeypatch)

    plan = pd.DataFrame(
        {"target_weight": [1.0], "dollars": [50.0]}, index=["AAPL"]
    )
    with pytest.raises(ValueError, match="order_interval"):
        submit_fractional_buys(object(), plan, order_interval=-1.0)


def test_submit_fractional_buys_stops_after_ambiguous_failure(monkeypatch) -> None:
    _no_sleep(monkeypatch)

    class FakeRobinhood:
        def __init__(self) -> None:
            self.calls = 0

        def order_buy_fractional_by_price(self, *args, **kwargs):
            self.calls += 1
            return None

    rh = FakeRobinhood()
    plan = pd.DataFrame(
        {"target_weight": [0.6, 0.3, 0.1], "dollars": [60.0, 30.0, 10.0]},
        index=["AAPL", "MSFT", "GOOG"],
    )
    results = submit_fractional_buys(rh, plan, verbose=False)

    assert rh.calls == 1  # no further orders after an ambiguous failure
    assert len(results) == 3  # ...but every planned row is reported
    assert results.loc["AAPL", "state"] == "error"
    assert "no order confirmation" in results.loc["AAPL", "error"]
    assert (results.loc[["MSFT", "GOOG"], "state"] == "not_attempted").all()
    assert (results.loc[["MSFT", "GOOG"], "error"] == "").all()


def test_fetch_order_states_reports_live_state_and_fills() -> None:
    class FakeRobinhood:
        def get_stock_order_info(self, order_id):
            if order_id == "id-filled":
                return {"state": "filled", "executed_notional": {"amount": "10.50"}}
            if order_id == "id-legacy":
                return {"state": "filled", "average_price": "2.50", "quantity": "4"}
            if order_id == "id-queued":
                return {"state": "queued"}
            raise RuntimeError("boom")

    states = fetch_order_states(
        FakeRobinhood(), ["id-filled", "id-legacy", "id-queued", "id-broken"],
        poll_interval=0,
    )

    assert states["id-filled"] == {"state": "filled", "filled_dollars": 10.50}
    assert states["id-legacy"] == {"state": "filled", "filled_dollars": 10.0}
    assert states["id-queued"] == {"state": "queued", "filled_dollars": None}
    assert states["id-broken"] == {"state": "unknown", "filled_dollars": None}


def test_last_orders_roundtrip(tmp_path, monkeypatch) -> None:
    import stonks.robinhood as rb
    from stonks.config import Settings

    monkeypatch.setattr(rb, "Settings", lambda: Settings(cache_dir=tmp_path))

    results = pd.DataFrame(
        {
            "dollars": [60.0, 40.0],
            "state": ["unconfirmed", "not_attempted"],
            "order_id": ["abc-123", ""],
            "error": ["", ""],
        },
        index=pd.Index(["AAPL", "MSFT"], name="ticker"),
    )
    save_last_orders(results)

    loaded = load_last_orders()
    assert loaded is not None
    assert list(loaded.index) == ["AAPL", "MSFT"]
    assert loaded.loc["AAPL", "order_id"] == "abc-123"
    assert loaded.loc["AAPL", "dollars"] == 60.0


def test_load_last_orders_tolerates_corrupt_file(tmp_path, monkeypatch) -> None:
    import stonks.robinhood as rb
    from stonks.config import Settings

    monkeypatch.setattr(rb, "Settings", lambda: Settings(cache_dir=tmp_path))
    (tmp_path / "last_robinhood_orders.json").write_text("{not json")

    assert load_last_orders() is None
