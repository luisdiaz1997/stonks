"""Offline CLI tests with mocked Robinhood calls (no network, no orders)."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _ledger_stays_out_of_the_repository(monkeypatch, tmp_path) -> None:
    """Every test keeps its ledger in a temporary directory, never under the repo's data/."""
    import stonks.robinhood as rb
    from stonks.config import Settings

    monkeypatch.setattr(rb, "Settings", lambda: Settings(cache_dir=tmp_path))


def test_portfolio_execute_prints_live_order_status(monkeypatch, tmp_path) -> None:
    import time as time_mod

    import stonks.cli as cli_mod
    import stonks.portfolio as pf
    import stonks.robinhood as rb
    import stonks.universe as uni

    weights = pd.DataFrame({"weight": [0.6, 0.4]}, index=pd.Index(["AAPL", "MSFT"], name="ticker"))
    monkeypatch.setattr(pf, "optimize_portfolio", lambda **kwargs: weights)
    monkeypatch.setattr(
        uni,
        "load_universe",
        lambda *args, **kwargs: pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "name": ["Apple Inc.", "Microsoft Corp."]}
        ),
    )

    class FakeRobinhood:
        pass

    saved: dict = {}

    def fake_submit(rh, plan, account_number=None, **kwargs):
        rows = [
            {
                "ticker": ticker,
                "dollars": float(row["dollars"]),
                "state": "unconfirmed",
                "order_id": f"id-{ticker}",
                "error": "",
            }
            for ticker, row in plan.iterrows()
        ]
        return pd.DataFrame(rows).set_index("ticker")

    monkeypatch.setattr(rb, "login", lambda: FakeRobinhood())
    monkeypatch.setattr(rb, "account_buying_power", lambda rh: (1000.0, "ACCT1"))
    monkeypatch.setattr(rb, "submit_fractional_buys", fake_submit)
    monkeypatch.setattr(
        rb,
        "fetch_order_states",
        lambda rh, order_ids, **kwargs: {
            oid: {"state": "filled", "filled_dollars": 5.0} for oid in order_ids
        },
    )
    monkeypatch.setattr(
        rb, "save_last_orders", lambda results: saved.update(frame=results) or tmp_path / "f.json"
    )
    monkeypatch.setattr(time_mod, "sleep", lambda seconds: None)

    result = CliRunner().invoke(
        cli_mod.cli,
        ["portfolio", "--top-n", "2", "--execute", "--amount", "50"],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "Live order status" in result.output
    assert "$5.00" in result.output
    # the submission is persisted before the status check runs
    assert set(saved["frame"].index) == {"AAPL", "MSFT"}
    assert (saved["frame"]["order_id"] != "").all()


def _positions() -> pd.DataFrame:
    """AAPL reached an 8% take-profit; MSFT has not."""
    return pd.DataFrame(
        {
            "quantity": [2.0, 1.0],
            "avg_buy_price": [100.0, 50.0],
            "intraday_quantity": [0.0, 0.0],
            "shares_held_for_sells": [0.0, 0.0],
            "price": [108.0, 53.5],
            "previous_close": [100.0, 50.0],
        },
        index=pd.Index(["AAPL", "MSFT"], name="ticker"),
    )


def _mock_robinhood(monkeypatch, market=(True, True)) -> dict:
    """Replace every broker call ``stonks sell`` makes, recording what it submits."""
    import time as time_mod

    import stonks.robinhood as rb

    calls: dict = {}

    def fake_submit(rh, plan, account_number=None, **kwargs):
        calls["plan"] = plan
        calls["account_number"] = account_number
        calls["stop_on_failure"] = kwargs.get("stop_on_failure", True)
        return pd.DataFrame(
            {
                "quantity": plan["quantity"],
                "value": plan["value"],
                "state": "queued",
                "order_id": [f"id-{ticker}" for ticker in plan.index],
                "error": "",
            },
            index=plan.index,
        )

    def fake_save(results, side="buy"):
        calls["saved"], calls["side"] = results, side

    monkeypatch.setattr(rb, "login", lambda: object())
    monkeypatch.setattr(rb, "account_buying_power", lambda rh: (1000.0, "ACCT1"))
    monkeypatch.setattr(rb, "market_status", lambda rh: market)
    monkeypatch.setattr(rb, "load_positions", lambda rh, account_number=None: _positions())
    monkeypatch.setattr(rb, "submit_fractional_sells", fake_submit)
    monkeypatch.setattr(
        rb,
        "fetch_order_states",
        lambda rh, order_ids, **kwargs: {oid: {"state": "filled"} for oid in order_ids},
    )
    monkeypatch.setattr(rb, "save_last_orders", fake_save)
    monkeypatch.setattr(time_mod, "sleep", lambda seconds: None)
    return calls


def _ticker_rows(output: str) -> dict[str, str]:
    return {
        words[0]: line
        for line in output.splitlines()
        if (words := line.split()) and words[0] in {"AAPL", "MSFT"}
    }


def test_sell_is_a_dry_run_without_execute(monkeypatch) -> None:
    import stonks.cli as cli_mod

    calls = _mock_robinhood(monkeypatch)

    result = CliRunner().invoke(cli_mod.cli, ["sell"])

    assert result.exit_code == 0, result.output
    rows = _ticker_rows(result.output)
    assert rows["AAPL"].endswith("SELL")
    assert rows["MSFT"].endswith("below 8%")
    assert "Dry run: no orders were submitted." in result.output
    assert "plan" not in calls  # nothing reached the broker


def test_sell_execute_submits_only_triggered_positions_after_confirmation(monkeypatch) -> None:
    import stonks.cli as cli_mod

    calls = _mock_robinhood(monkeypatch)

    declined = CliRunner().invoke(cli_mod.cli, ["sell", "--execute"], input="n\n")
    assert "Canceled; no orders were submitted." in declined.output
    assert "plan" not in calls

    accepted = CliRunner().invoke(cli_mod.cli, ["sell", "--execute"], input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert list(calls["plan"].index) == ["AAPL"]  # MSFT, below the threshold, is left alone
    assert calls["account_number"] == "ACCT1"
    assert calls["stop_on_failure"] is True  # a failed take-profit sale stops the run
    assert calls["side"] == "sell"
    assert "Live order status" in accepted.output


def test_sell_yes_skips_the_prompt_only_beside_execute(monkeypatch) -> None:
    import stonks.cli as cli_mod

    calls = _mock_robinhood(monkeypatch)

    unattended = CliRunner().invoke(cli_mod.cli, ["sell", "--execute", "--yes"])
    assert unattended.exit_code == 0, unattended.output
    assert "Sell these 1 positions in Robinhood?" not in unattended.output
    assert list(calls["plan"].index) == ["AAPL"]

    never_alone = CliRunner().invoke(cli_mod.cli, ["sell", "--yes"])
    assert never_alone.exit_code == 2
    assert "--yes requires --execute" in never_alone.output


@pytest.mark.parametrize("market", [(False, False), (True, False)])
def test_sell_execute_places_nothing_when_the_market_is_closed(monkeypatch, market) -> None:
    import stonks.cli as cli_mod

    calls = _mock_robinhood(monkeypatch, market=market)

    result = CliRunner().invoke(cli_mod.cli, ["sell", "--execute", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Market closed: no orders were submitted" in result.output
    assert "plan" not in calls


def test_orders_reports_buy_and_sell_submissions(monkeypatch, tmp_path) -> None:
    import stonks.cli as cli_mod
    import stonks.robinhood as rb
    from stonks.config import Settings

    monkeypatch.setattr(rb, "Settings", lambda: Settings(cache_dir=tmp_path))
    index = pd.Index(["AAPL"], name="ticker")
    buys = pd.DataFrame({"dollars": [216.0], "order_id": ["buy-1"]}, index=index)
    sells = pd.DataFrame({"quantity": [2.0], "order_id": ["sell-1"]}, index=index)
    rb.save_last_orders(buys, "buy")
    rb.save_last_orders(sells, "sell")
    monkeypatch.setattr(rb, "login", lambda: object())
    monkeypatch.setattr(
        rb,
        "fetch_order_states",
        lambda rh, order_ids, **kwargs: {"sell-1": {"state": "filled", "filled_dollars": 216.0}},
    )

    result = CliRunner().invoke(cli_mod.cli, ["orders"])

    assert result.exit_code == 0, result.output
    assert "Buy orders:" in result.output and "Sell orders:" in result.output
    assert "sell-1" in result.output


def test_cli_imports_without_torch() -> None:
    """The broker commands run on small servers where torch is not installed."""
    import subprocess
    import sys

    blocked = "import sys; sys.modules['torch'] = None; import stonks.cli"
    done = subprocess.run([sys.executable, "-c", blocked], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


def test_sell_everything_sells_what_the_take_profit_would_spare(monkeypatch) -> None:
    import stonks.cli as cli_mod
    import stonks.robinhood as rb

    calls = _mock_robinhood(monkeypatch)

    shown = CliRunner().invoke(cli_mod.cli, ["sell", "--everything"])
    assert shown.exit_code == 0, shown.output
    rows = _ticker_rows(shown.output)
    assert rows["AAPL"].endswith("SELL") and rows["MSFT"].endswith("SELL")  # MSFT is only 7% up
    assert "the take-profit is not used" in shown.output
    assert "plan" not in calls

    done = CliRunner().invoke(cli_mod.cli, ["sell", "--everything", "--execute", "--yes"])
    assert done.exit_code == 0, done.output
    assert list(calls["plan"].index) == ["AAPL", "MSFT"]
    assert calls["stop_on_failure"] is False  # one rejected order must not leave the rest unsold
    assert "Counted for the next re-buy: $269.50 filled." in done.output  # 2 x $108 and $53.50
    assert rb.cash_to_rebuy() == (pytest.approx(269.5), 1)


def test_portfolio_save_writes_a_pick_and_never_touches_robinhood(monkeypatch, tmp_path) -> None:
    import stonks.cli as cli_mod
    import stonks.portfolio as pf
    import stonks.robinhood as rb
    import stonks.universe as uni

    weights = pd.DataFrame({"weight": [0.6, 0.4]}, index=pd.Index(["AAPL", "MSFT"], name="ticker"))
    monkeypatch.setattr(pf, "optimize_portfolio", lambda **kwargs: weights)
    monkeypatch.setattr(
        uni,
        "load_universe",
        lambda *args, **kwargs: pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "name": ["Apple Inc.", "Microsoft Corp."]}
        ),
    )

    def no_broker() -> None:
        raise AssertionError("portfolio --save must not touch Robinhood")

    monkeypatch.setattr(rb, "login", no_broker)
    target = tmp_path / "picks" / "tuesday.json"

    result = CliRunner().invoke(
        cli_mod.cli,
        ["portfolio", "--top-n", "2", "--gamma", "40", "-k", "25", "--save", str(target)],
    )
    assert result.exit_code == 0, result.output
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["weights"] == {"AAPL": 0.6, "MSFT": 0.4}
    assert saved["settings"]["gamma"] == 40.0 and saved["settings"]["K"] == 25
    assert f"stonks rebuy --pick {target}" in result.output

    for clashing in (["--execute"], ["--execute", "--amount", "5"]):
        refused = CliRunner().invoke(cli_mod.cli, ["portfolio", "--save", str(target), *clashing])
        assert refused.exit_code == 2
        assert "--save cannot be combined" in refused.output


def _pick(tmp_path, age_hours: float = 0.0) -> Path:
    """A recommendation as `portfolio --save` writes it, saved that many hours ago."""
    saved_at = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    path = tmp_path / "tuesday.json"
    path.write_text(
        json.dumps(
            {
                "saved_at": saved_at.isoformat(),
                "settings": {"top_n": 500, "months": 3, "gamma": 40.0, "K": 25},
                "weights": {"AAPL": 0.75, "MSFT": 0.25},
            }
        ),
        encoding="utf-8",
    )
    return path


def _record_a_sale(proceeds: float) -> None:
    import stonks.robinhood as rb

    rb.record_sells(
        "everything",
        pd.DataFrame(
            {
                "value": [proceeds],
                "order_id": ["s1"],
                "live_state": ["filled"],
                "filled_dollars": [proceeds],
            },
            index=pd.Index(["AAPL"], name="ticker"),
        ),
    )


def _mock_rebuy(monkeypatch, buying_power: float = 1000.0, market=(True, True)) -> dict:
    """Replace every broker call `stonks rebuy` makes, recording what it submits."""
    import time as time_mod

    import stonks.robinhood as rb

    calls: dict = {}

    def fake_submit(rh, plan, account_number=None, **kwargs):
        calls["plan"] = plan
        calls["account_number"] = account_number
        return pd.DataFrame(
            {
                "dollars": plan["dollars"],
                "state": "unconfirmed",
                "order_id": [f"id-{ticker}" for ticker in plan.index],
                "error": "",
            },
            index=plan.index,
        )

    monkeypatch.setattr(rb, "login", lambda: object())
    monkeypatch.setattr(rb, "account_buying_power", lambda rh: (buying_power, "ACCT1"))
    monkeypatch.setattr(rb, "market_status", lambda rh: market)
    monkeypatch.setattr(rb, "submit_fractional_buys", fake_submit)
    monkeypatch.setattr(
        rb,
        "fetch_order_states",
        lambda rh, order_ids, **kwargs: {oid: {"state": "filled"} for oid in order_ids},
    )
    monkeypatch.setattr(
        rb, "save_last_orders", lambda results, side="buy": calls.setdefault("saved", results)
    )
    monkeypatch.setattr(time_mod, "sleep", lambda seconds: None)
    return calls


def test_rebuy_is_a_dry_run_and_is_sized_by_what_was_sold(monkeypatch, tmp_path) -> None:
    import stonks.cli as cli_mod

    _record_a_sale(230.0)
    calls = _mock_rebuy(monkeypatch)

    result = CliRunner().invoke(cli_mod.cli, ["rebuy", "--pick", str(_pick(tmp_path))])

    assert result.exit_code == 0, result.output
    assert "Sells since the last re-buy: $230.00 from 1 event(s)." in result.output
    assert "totaling $230.00" in result.output
    assert "Dry run: no orders were submitted." in result.output
    assert "plan" not in calls


def test_rebuy_execute_buys_what_was_sold_and_closes_the_cycle(monkeypatch, tmp_path) -> None:
    import stonks.cli as cli_mod
    import stonks.robinhood as rb

    _record_a_sale(230.0)
    _record_a_sale(45.0)  # a take-profit sale during the week counts too
    calls = _mock_rebuy(monkeypatch)

    result = CliRunner().invoke(
        cli_mod.cli, ["rebuy", "--pick", str(_pick(tmp_path)), "--execute", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert calls["plan"]["dollars"].sum() == pytest.approx(275.0)
    assert calls["account_number"] == "ACCT1"
    assert calls["saved"]["order_id"].astype(str).ne("").all()
    assert (
        "Put on the ledger: $275.00 sent to Robinhood, $0.00 carried to the next re-buy."
        in result.output
    )
    assert rb.cash_to_rebuy() == (0.0, 0)  # closed until the next sale


def test_rebuy_never_exceeds_the_buying_power_and_carries_the_rest(monkeypatch, tmp_path) -> None:
    import stonks.cli as cli_mod
    import stonks.robinhood as rb

    _record_a_sale(2300.0)
    calls = _mock_rebuy(monkeypatch, buying_power=1000.0)

    result = CliRunner().invoke(
        cli_mod.cli, ["rebuy", "--pick", str(_pick(tmp_path)), "--execute", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert calls["plan"]["dollars"].sum() == pytest.approx(1000.0)
    assert "Capped at the $1,000.00 of buying power" in result.output
    assert rb.cash_to_rebuy() == (pytest.approx(1300.0), 0)  # waits rather than being borrowed


def test_rebuy_buys_nothing_when_nothing_was_sold(monkeypatch, tmp_path) -> None:
    import stonks.cli as cli_mod

    calls = _mock_rebuy(monkeypatch)

    result = CliRunner().invoke(
        cli_mod.cli, ["rebuy", "--pick", str(_pick(tmp_path)), "--execute", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert "Nothing to re-buy: the ledger holds $0.00 from 0 sell event(s)" in result.output
    assert "plan" not in calls


@pytest.mark.parametrize("market", [(False, False), (True, False)])
def test_rebuy_execute_places_nothing_when_the_market_is_closed(
    monkeypatch, tmp_path, market
) -> None:
    import stonks.cli as cli_mod

    _record_a_sale(230.0)
    calls = _mock_rebuy(monkeypatch, market=market)

    result = CliRunner().invoke(
        cli_mod.cli, ["rebuy", "--pick", str(_pick(tmp_path)), "--execute", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert "Market closed: no orders were submitted" in result.output
    assert "plan" not in calls


def test_rebuy_refuses_a_recommendation_that_is_not_from_today(monkeypatch, tmp_path) -> None:
    import stonks.cli as cli_mod

    _record_a_sale(230.0)
    calls = _mock_rebuy(monkeypatch)

    result = CliRunner().invoke(
        cli_mod.cli, ["rebuy", "--pick", str(_pick(tmp_path, age_hours=20)), "--execute", "--yes"]
    )

    assert result.exit_code != 0
    assert "20.0 hours old" in result.output
    assert "plan" not in calls


def test_rebuy_yes_requires_execute(monkeypatch, tmp_path) -> None:
    import stonks.cli as cli_mod

    refused = CliRunner().invoke(cli_mod.cli, ["rebuy", "--pick", str(_pick(tmp_path)), "--yes"])

    assert refused.exit_code == 2
    assert "--yes requires --execute" in refused.output
