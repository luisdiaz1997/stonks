"""Offline CLI tests with mocked Robinhood calls (no network, no orders)."""

import pandas as pd
import pytest
from click.testing import CliRunner


def test_portfolio_execute_prints_live_order_status(monkeypatch, tmp_path) -> None:
    import time as time_mod

    import stonks.cli as cli_mod
    import stonks.portfolio as pf
    import stonks.robinhood as rb
    import stonks.universe as uni

    weights = pd.DataFrame(
        {"weight": [0.6, 0.4]}, index=pd.Index(["AAPL", "MSFT"], name="ticker")
    )
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
        return pd.DataFrame(
            {
                "quantity": plan["quantity"],
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
