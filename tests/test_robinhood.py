"""Offline tests for Robinhood dollar-order allocation, sells and the ledger of what they brought in."""

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from stonks.robinhood import (
    allocate_fractional_buys,
    cash_to_rebuy,
    fetch_order_states,
    ledger_path,
    load_last_orders,
    load_positions,
    login,
    market_status,
    record_buy,
    record_sells,
    save_last_orders,
    select_sells,
    submit_fractional_buys,
    submit_fractional_sells,
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

    plan = pd.DataFrame({"target_weight": [1.0], "dollars": [50.0]}, index=["AAPL"])
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
        FakeRobinhood(),
        ["id-filled", "id-legacy", "id-queued", "id-broken"],
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


def _positions() -> pd.DataFrame:
    """One position per situation the take-profit rule has to tell apart."""
    return pd.DataFrame(
        {
            "quantity": [2.0, 0.5, 1.0, 1.0, 1.0, 0.01, 1.0, 1.0],
            "avg_buy_price": [100.0, 200.0, 50.0, 100.0, 100.0, 1.0, 0.0, 10.0],
            "intraday_quantity": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "shares_held_for_sells": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "price": [108.0, 216.0, 53.5, 120.0, 150.0, 50.0, 10.0, math.nan],
            "previous_close": [100.0, 210.0, 50.0, 110.0, 140.0, 49.0, 9.0, math.nan],
        },
        index=pd.Index(
            ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA", "F", "FREE", "DEAD"], name="ticker"
        ),
    )


def test_select_sells_triggers_at_the_threshold_and_keeps_the_rest() -> None:
    plan = select_sells(_positions(), take_profit_pct=8.0)

    # sells come first, most gained first; the 8.00% position is included
    assert list(plan[plan["sell"]].index) == ["AAPL", "MSFT"]
    assert plan.loc["GOOG", "skip"] == "below 8%"
    assert plan.loc["TSLA", "skip"] == "bought today"
    assert plan.loc["NVDA", "skip"] == "open sell order"
    assert plan.loc["F", "skip"] == "under $1.00"
    assert plan.loc["FREE", "skip"] == "no cost basis"
    assert plan.loc["DEAD", "skip"] == "no quote"
    # a position is sellable exactly when its skip reason is empty
    assert (plan["sell"] == (plan["skip"] == "")).all()


def test_select_sells_measures_total_and_today_separately() -> None:
    plan = select_sells(_positions(), take_profit_pct=8.0).loc["MSFT"]

    assert plan["value"] == pytest.approx(108.0)
    assert plan["total_return_pct"] == pytest.approx(8.0)
    assert plan["day_change_pct"] == pytest.approx(6 / 210 * 100)


@pytest.mark.parametrize("threshold", [0.0, -5.0, math.inf, math.nan])
def test_select_sells_rejects_unusable_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="take_profit_pct"):
        select_sells(_positions(), take_profit_pct=threshold)


def test_select_sells_everything_sells_what_the_take_profit_would_spare() -> None:
    plan = select_sells(_positions(), take_profit_pct=None)

    # GOOG is 7% up, under the take-profit, and goes too; what cannot be sold cleanly still stays
    assert list(plan[plan["sell"]].index) == ["AAPL", "MSFT", "GOOG"]
    assert plan.loc["GOOG", "skip"] == ""
    assert plan.loc["TSLA", "skip"] == "bought today"
    assert plan.loc["NVDA", "skip"] == "open sell order"
    assert plan.loc["F", "skip"] == "under $1.00"


class _FakeRobinhood:
    """Just the three reads ``load_positions`` makes."""

    def __init__(self, positions, quotes) -> None:
        self.positions = positions
        self.quotes = quotes
        self.account_number_seen = None

    def get_open_stock_positions(self, account_number=None):
        self.account_number_seen = account_number
        return self.positions

    def get_symbol_by_url(self, url):
        return "AAPL"  # the only instrument _POSITION defines

    def get_quotes(self, tickers):
        return self.quotes


_POSITION = {
    "instrument": "https://api.robinhood.com/instruments/a1/",
    "average_buy_price": "100.0000",
    "quantity": "0.123456",
    "intraday_quantity": "0.000000",
    "shares_held_for_sells": "0.250000",
}


def test_load_positions_reads_positions_and_quotes() -> None:
    rh = _FakeRobinhood(
        [_POSITION],
        [
            {
                "symbol": "AAPL",
                "last_trade_price": "108.00",
                "adjusted_previous_close": "100.00",
            }
        ],
    )

    frame = load_positions(rh, account_number="ACCT1")

    assert rh.account_number_seen == "ACCT1"
    assert list(frame.index) == ["AAPL"]
    assert frame.loc["AAPL", "quantity"] == pytest.approx(0.123456)
    assert frame.loc["AAPL", "avg_buy_price"] == pytest.approx(100.0)
    assert frame.loc["AAPL", "shares_held_for_sells"] == pytest.approx(0.25)
    assert frame.loc["AAPL", "price"] == pytest.approx(108.0)
    assert frame.loc["AAPL", "previous_close"] == pytest.approx(100.0)


def test_load_positions_falls_back_to_unadjusted_previous_close() -> None:
    rh = _FakeRobinhood(
        [_POSITION], [{"symbol": "AAPL", "last_trade_price": "5.00", "previous_close": "4.00"}]
    )

    assert load_positions(rh).loc["AAPL", "previous_close"] == pytest.approx(4.0)


def test_load_positions_raises_on_failed_reads_and_accepts_an_empty_portfolio() -> None:
    # the library reports a failed GET as [None]: that must not look like "no positions"
    with pytest.raises(RuntimeError, match="no positions"):
        load_positions(_FakeRobinhood([None], []))
    assert load_positions(_FakeRobinhood([], [])).empty
    with pytest.raises(RuntimeError, match="no quotes"):
        load_positions(_FakeRobinhood([_POSITION], [None]))


class _MidSessionDatetime(datetime):
    """A clock frozen at 15:00 UTC: inside a 14:30-21:00 UTC regular session."""

    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 25, 15, 0, tzinfo=timezone.utc)


def test_market_status_tells_trading_day_from_open_session(monkeypatch) -> None:
    import stonks.robinhood as rb

    class FakeRobinhood:
        def __init__(self, hours) -> None:
            self.hours = hours

        def get_market_today_hours(self, market):
            return self.hours

    monkeypatch.setattr(rb, "datetime", _MidSessionDatetime)
    session = {
        "is_open": True,
        "opens_at": "2026-11-25T14:30:00Z",
        "closes_at": "2026-11-25T21:00:00Z",
    }
    holiday = {"is_open": False, "opens_at": None, "closes_at": None}

    assert market_status(FakeRobinhood(session)) == (True, True)
    assert market_status(FakeRobinhood({**session, "opens_at": "2026-11-25T16:00:00Z"})) == (
        True,
        False,
    )
    assert market_status(FakeRobinhood(holiday)) == (False, False)
    with pytest.raises(RuntimeError, match="no market hours"):
        market_status(FakeRobinhood(None))
    with pytest.raises(RuntimeError, match="no session times"):
        market_status(FakeRobinhood({**holiday, "is_open": True}))


def test_submit_fractional_sells_the_exact_quantity(monkeypatch) -> None:
    _no_sleep(monkeypatch)

    class FakeRobinhood:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def order_sell_fractional_by_quantity(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"id": f"order-{args[0]}", "state": "queued"}

    rh = FakeRobinhood()
    plan = pd.DataFrame({"quantity": [0.123456789, 3.0]}, index=["AAPL", "MSFT"])
    results = submit_fractional_sells(rh, plan, account_number="ABC123")

    assert rh.calls[0][0] == ("AAPL", 0.123457)  # rounded like the library rounds prices
    assert rh.calls[0][1] == {"account_number": "ABC123", "timeInForce": "gfd"}
    assert rh.calls[1][0] == ("MSFT", 3.0)
    assert list(results.columns) == ["quantity", "state", "order_id", "error"]
    assert (results["state"] == "queued").all()


def test_submit_fractional_sells_can_go_on_past_a_failed_order(monkeypatch) -> None:
    _no_sleep(monkeypatch)

    class FakeRobinhood:
        def order_sell_fractional_by_quantity(self, ticker, *args, **kwargs):
            if ticker == "MSFT":
                return {"detail": "rejected"}
            return {"id": f"order-{ticker}", "state": "queued"}

    plan = pd.DataFrame({"quantity": [1.0, 2.0, 3.0]}, index=["AAPL", "MSFT", "GOOG"])

    results = submit_fractional_sells(FakeRobinhood(), plan, stop_on_failure=False)

    # every position is attempted once, so none is left unsold and none is sold twice
    assert list(results["state"]) == ["queued", "error", "queued"]
    assert results.loc["MSFT", "error"] == "rejected"
    assert results.loc["GOOG", "order_id"] == "order-GOOG"


def test_submit_fractional_sells_stops_after_ambiguous_failure(monkeypatch) -> None:
    _no_sleep(monkeypatch)

    class FakeRobinhood:
        def __init__(self) -> None:
            self.calls = 0

        def order_sell_fractional_by_quantity(self, *args, **kwargs):
            self.calls += 1
            return None

    rh = FakeRobinhood()
    plan = pd.DataFrame({"quantity": [1.0, 1.0]}, index=["AAPL", "MSFT"])
    results = submit_fractional_sells(rh, plan, verbose=False)

    assert rh.calls == 1
    assert results.loc["AAPL", "state"] == "error"
    assert results.loc["MSFT", "state"] == "not_attempted"


def test_last_sells_roundtrip(tmp_path, monkeypatch) -> None:
    import stonks.robinhood as rb
    from stonks.config import Settings

    monkeypatch.setattr(rb, "Settings", lambda: Settings(cache_dir=tmp_path))
    results = pd.DataFrame(
        {"quantity": [0.5], "state": ["queued"], "order_id": ["s-1"], "error": [""]},
        index=pd.Index(["AAPL"], name="ticker"),
    )

    save_last_orders(results, "sell")

    loaded = load_last_orders("sell")
    assert loaded is not None and loaded.loc["AAPL", "order_id"] == "s-1"
    assert loaded.loc["AAPL", "quantity"] == pytest.approx(0.5)
    assert load_last_orders("buy") is None  # buys and sells are tracked apart


def test_login_explains_missing_credentials_when_it_cannot_prompt(monkeypatch) -> None:
    import robin_stocks.robinhood as rh

    def prompt_hits_closed_stdin(*args, **kwargs):
        raise EOFError

    monkeypatch.setattr(rh, "login", prompt_hits_closed_stdin)
    with pytest.raises(RuntimeError, match="ROBINHOOD_USERNAME"):
        login()


def _submission() -> pd.DataFrame:
    """Four sell orders as a run sees them after its status check: two filled, one queued, one rejected."""
    return pd.DataFrame(
        {
            "value": [108.0, 216.0, 53.5, 20.0],
            "state": ["unconfirmed", "unconfirmed", "unconfirmed", "error"],
            "order_id": ["a", "b", "c", ""],
            "error": ["", "", "", "rejected"],
            "live_state": ["filled", "filled", "queued", ""],
            "filled_dollars": [107.9, None, None, None],
        },
        index=pd.Index(["AAPL", "MSFT", "GOOG", "F"], name="ticker"),
    )


def _ledger_in(tmp_path, monkeypatch) -> None:
    import stonks.robinhood as rb
    from stonks.config import Settings

    monkeypatch.setattr(rb, "Settings", lambda: Settings(cache_dir=tmp_path))


def test_record_sells_counts_only_what_robinhood_says_filled(tmp_path, monkeypatch) -> None:
    _ledger_in(tmp_path, monkeypatch)

    proceeds = record_sells("take-profit", _submission())

    # the filled order that reported no amount counts for what the position was worth
    assert proceeds == pytest.approx(107.9 + 216.0)
    assert cash_to_rebuy() == (pytest.approx(323.9), 1)
    events = json.loads(ledger_path().read_text(encoding="utf-8"))
    assert [(event["kind"], event["submitted"], event["filled"]) for event in events] == [
        ("take-profit", 3, 2)
    ]


def test_cash_to_rebuy_runs_from_the_last_rebuy_and_keeps_what_it_left(
    tmp_path, monkeypatch
) -> None:
    _ledger_in(tmp_path, monkeypatch)
    assert cash_to_rebuy() == (0.0, 0)  # nothing has been sold yet

    record_sells("take-profit", _submission())
    record_sells("everything", _submission())
    assert cash_to_rebuy() == (pytest.approx(2 * 323.9), 2)

    record_buy(available=647.8, budget=500.0, submitted=420.0)
    assert cash_to_rebuy() == (pytest.approx(647.8 - 420.0), 0)  # capped and unsent money waits

    record_sells("take-profit", _submission())
    assert cash_to_rebuy() == (pytest.approx(227.8 + 323.9), 1)


def test_cash_to_rebuy_tolerates_a_missing_or_damaged_ledger(tmp_path, monkeypatch) -> None:
    _ledger_in(tmp_path, monkeypatch)
    assert cash_to_rebuy() == (0.0, 0)
    ledger_path().write_text("{not json", encoding="utf-8")
    assert cash_to_rebuy() == (0.0, 0)
