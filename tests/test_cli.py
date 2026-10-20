"""Offline CLI tests with mocked Robinhood calls (no network, no orders)."""

import pandas as pd
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
