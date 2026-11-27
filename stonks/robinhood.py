"""Robinhood execution helpers.

The optimizer produces target weights.  This module converts those weights to
dollar-denominated fractional buy orders and submits them through
``robin_stocks``.  Buying never sells or rebalances existing positions; selling
is a separate decision, :func:`select_sells`, that exits a position once its
total return reaches a take-profit threshold.

``robin_stocks`` uses Robinhood's private API and may stop working when that
API changes.  No order is submitted by the CLI without explicit confirmation.

Selling and buying again are kept apart on purpose. Every sell the jobs make is
put on a small ledger (:func:`record_sells`), and ``stonks rebuy`` buys a saved
recommendation with what that ledger holds since its previous run
(:func:`cash_to_rebuy`), so a cycle of selling and re-buying moves the same
money around and borrows nothing more.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings


def allocate_fractional_buys(
    weights: pd.Series,
    budget: float,
    min_order_dollars: float = 1.0,
) -> pd.DataFrame:
    """Allocate ``budget`` across weights as whole-cent fractional buy orders.

    When the budget is too small to give every recommendation Robinhood's
    minimum fractional order, the lowest-weight names are removed until every
    remaining order meets the minimum.  The remaining weights are renormalized.
    """
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be a positive finite number")
    if not math.isfinite(min_order_dollars) or min_order_dollars < 1.0:
        raise ValueError("min_order_dollars must be at least 1.00")

    clean = pd.to_numeric(weights, errors="coerce")
    clean.index = clean.index.map(lambda value: str(value).strip().upper())
    clean = clean.groupby(level=0).sum()
    clean = clean[clean.notna() & clean.map(math.isfinite) & (clean > 0)]
    clean = clean.sort_values(ascending=False)
    if clean.empty:
        raise ValueError("weights must contain at least one positive value")

    budget_cents = math.floor(budget * 100 + 1e-9)
    minimum_cents = math.ceil(min_order_dollars * 100 - 1e-9)
    if budget_cents < minimum_cents:
        raise ValueError(
            f"budget must be at least ${minimum_cents / 100:.2f} for a fractional order"
        )

    # Keep the largest possible prefix for which every proportional order meets
    # the broker minimum. This behaves sensibly for many equal, small weights.
    active = clean.copy()
    while len(active) > 1:
        raw_cents = budget_cents * active / active.sum()
        if raw_cents.iloc[-1] >= minimum_cents:
            break
        active = active.iloc[:-1]

    raw_cents = budget_cents * active / active.sum()
    cents = raw_cents.map(math.floor).astype(int)

    # Give the remaining cents to the largest fractional remainders.  This
    # keeps the total exactly at the requested cent-denominated budget.
    remainder = budget_cents - int(cents.sum())
    if remainder:
        priority = (raw_cents - cents).sort_values(ascending=False).index
        cents.loc[priority[:remainder]] += 1

    plan = pd.DataFrame(
        {
            "target_weight": active / active.sum(),
            "dollars": cents / 100.0,
        }
    )
    plan.index.name = "ticker"
    return plan.sort_values("dollars", ascending=False)


def login() -> Any:
    """Authenticate with Robinhood, using environment variables when present.

    ``robin_stocks`` securely prompts for missing credentials and persists its
    session under ``~/.tokens``.  Passwords are never accepted as CLI options.
    """
    import robin_stocks.robinhood as rh

    try:
        result = rh.login(
            username=os.getenv("ROBINHOOD_USERNAME"),
            password=os.getenv("ROBINHOOD_PASSWORD"),
            mfa_code=os.getenv("ROBINHOOD_MFA_CODE"),
        )
    except EOFError as exc:
        # The library prompts for missing credentials, which fails when stdin
        # is closed (cron, systemd): say why instead of showing a traceback.
        raise RuntimeError(
            "Robinhood credentials are missing: set ROBINHOOD_USERNAME and "
            "ROBINHOOD_PASSWORD in the environment"
        ) from exc
    if not isinstance(result, dict) or not result.get("access_token"):
        raise RuntimeError("Robinhood login failed")
    return rh


def account_buying_power(rh: Any) -> tuple[float, str | None]:
    """Return the account's current buying power and account number."""
    profile = rh.load_account_profile()
    if not isinstance(profile, dict):
        raise RuntimeError("Robinhood returned an invalid account profile")
    try:
        buying_power = float(profile["buying_power"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Robinhood account profile has no valid buying power") from exc
    return buying_power, profile.get("account_number")


def _utc(stamp: Any) -> datetime | None:
    """Parse a Robinhood timestamp such as ``2026-11-25T14:30:00Z`` (UTC if naive)."""
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def market_status(rh: Any, market: str = "XNYS") -> tuple[bool, bool]:
    """Return ``(is_trading_day, regular_session_open_now)`` for ``market``.

    Robinhood's calendar is the source of truth for holidays, so a scheduled
    run on a weekday the exchange is closed simply has nothing to do.
    """
    hours = rh.get_market_today_hours(market)
    if not isinstance(hours, dict):
        raise RuntimeError("Robinhood returned no market hours")
    opens, closes = _utc(hours.get("opens_at")), _utc(hours.get("closes_at"))
    if hours.get("is_open") and (opens is None or closes is None):
        raise RuntimeError("Robinhood market hours have no session times")
    now = datetime.now(timezone.utc)
    return bool(hours.get("is_open")), bool(opens and closes and opens <= now < closes)


def _number(value: Any) -> float:
    """Robinhood sends numbers as strings; anything unreadable becomes NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_positions(rh: Any, account_number: str | None = None) -> pd.DataFrame:
    """Return every open stock position with its quote, indexed by ticker.

    One positions call, one batched quote call and one instrument lookup per
    position.  The library's ``build_holdings`` is avoided on purpose: it makes
    several requests per ticker and silently drops any position that errors.

    A failed read raises instead of looking like an empty portfolio, so a
    scheduled run can never mistake an outage for "nothing to sell".
    """
    positions = rh.get_open_stock_positions(account_number=account_number)
    if not isinstance(positions, list) or positions == [None]:  # [None] = failed GET
        raise RuntimeError("Robinhood returned no positions")

    rows = []
    for position in positions:
        if not isinstance(position, dict):
            continue
        instrument = str(position.get("instrument") or "")
        ticker = rh.get_symbol_by_url(instrument) if instrument else ""
        rows.append(
            {
                # an unresolvable instrument keeps its id, so it is still reported
                "ticker": str(ticker).strip().upper() or instrument.rstrip("/").rpartition("/")[2],
                "quantity": _number(position.get("quantity")),
                "avg_buy_price": _number(position.get("average_buy_price")),
                "intraday_quantity": _number(position.get("intraday_quantity")),
                "shares_held_for_sells": _number(position.get("shares_held_for_sells")),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "quantity",
            "avg_buy_price",
            "intraday_quantity",
            "shares_held_for_sells",
        ],
    ).set_index("ticker")
    frame["price"] = math.nan
    frame["previous_close"] = math.nan
    if frame.empty:
        return frame

    quotes = rh.get_quotes(frame.index.tolist())
    if not isinstance(quotes, list) or quotes == [None]:
        raise RuntimeError("Robinhood returned no quotes")
    by_symbol = {
        str(quote.get("symbol")).strip().upper(): quote
        for quote in quotes
        if isinstance(quote, dict)
    }

    def quoted(ticker: str, key: str) -> float:
        quote = by_symbol.get(ticker)
        return _number(quote.get(key)) if quote else math.nan

    def previous_close(ticker: str) -> float:
        """Yesterday's close, adjusted for splits and dividends, when given."""
        adjusted = quoted(ticker, "adjusted_previous_close")
        return quoted(ticker, "previous_close") if math.isnan(adjusted) else adjusted

    frame["price"] = [quoted(ticker, "last_trade_price") for ticker in frame.index]
    frame["previous_close"] = [previous_close(ticker) for ticker in frame.index]
    return frame


def select_sells(positions: pd.DataFrame, take_profit_pct: float | None = None) -> pd.DataFrame:
    """Mark the positions to sell.

    With ``take_profit_pct``, they are the positions whose total return
    reached it. Without it, every position is to go. The return is the price
    against Robinhood's average buy price — the position's *total* return
    since purchase, not today's move (both are kept in the frame so the
    numbers can be checked against the app).  A position is selected only
    when it can be sold cleanly, whichever rule is asked for; every other
    position stays in the frame with a ``skip`` reason, so a dry run shows
    all it examined:

    * ``bought today`` — selling shares bought the same day is a day trade;
    * ``open sell order`` — some shares are already reserved for a sale;
    * ``under $1.00`` — too small for a fractional order;
    * ``below X%`` — has not reached the threshold (only with one).

    Returns the positions plus ``value``, ``total_return_pct``,
    ``day_change_pct``, ``sell`` and ``skip`` (empty when ``sell`` is True),
    sells first.
    """
    if take_profit_pct is not None and not (math.isfinite(take_profit_pct) and take_profit_pct > 0):
        raise ValueError("take_profit_pct must be a positive finite number")

    plan = positions.copy()
    plan["value"] = plan["quantity"] * plan["price"]
    plan["total_return_pct"] = (plan["price"] / plan["avg_buy_price"] - 1.0) * 100.0
    plan["day_change_pct"] = (plan["price"] / plan["previous_close"] - 1.0) * 100.0

    def skip_reason(row: pd.Series) -> str:
        if math.isnan(row["price"]):
            return "no quote"
        if not row["avg_buy_price"] > 0:  # zero or missing: no cost basis to measure against
            return "no cost basis"
        if row["intraday_quantity"] > 0:
            return "bought today"
        if row["shares_held_for_sells"] > 0:
            return "open sell order"
        if row["value"] < 1.0:
            return "under $1.00"
        if take_profit_pct is not None and row["total_return_pct"] < take_profit_pct:
            return f"below {take_profit_pct:g}%"
        return ""

    plan["skip"] = [skip_reason(row) for _, row in plan.iterrows()]
    plan["sell"] = plan["skip"] == ""
    return plan.sort_values(["sell", "total_return_pct"], ascending=[False, False])


def _submit_orders(
    rh: Any,
    plan: pd.DataFrame,
    place: Callable[[str, pd.Series], Any],
    describe: Callable[[pd.Series], str],
    order_interval: float = 2.0,
    verbose: bool = True,
    stop_on_failure: bool = True,
) -> pd.DataFrame:
    """Submit every row of ``plan`` through ``place(ticker, row)``.

    The one order loop behind buy and sell plans; :func:`submit_fractional_buys`
    documents the throttling and the stop-on-failure policy. With
    ``stop_on_failure`` off, a failed order is recorded and the loop goes on
    to the next row, for a plan that has to reach every position.
    """
    if not math.isfinite(order_interval) or order_interval < 0:
        raise ValueError("order_interval must be a non-negative finite number")

    results: list[dict[str, Any]] = []
    stopped = False
    for i, (ticker, row) in enumerate(plan.iterrows()):
        planned = {"ticker": ticker, **row.to_dict()}
        if stopped:
            results.append({**planned, "state": "not_attempted", "order_id": "", "error": ""})
            continue

        try:
            raw_response = place(ticker, row)
        except Exception as exc:  # noqa: BLE001 - recorded per order below
            raw_response = {"detail": f"request failed: {exc}"}

        response = raw_response if isinstance(raw_response, dict) else {}
        error = response.get("detail") or response.get("non_field_errors")
        if not error and not response.get("id"):
            error = "Robinhood returned no order confirmation"
        state = response.get("state", "error" if error else "submitted")
        results.append(
            {
                **planned,
                "state": state,
                "order_id": response.get("id", ""),
                "error": str(error) if error else "",
            }
        )
        if verbose:
            print(
                f"[stonks] order {i + 1}/{len(plan)}: {ticker} {describe(row)} -> {state}",
                flush=True,
            )

        if error and stop_on_failure:
            stopped = True
        elif i < len(plan) - 1:
            time.sleep(order_interval)
    columns = ["ticker", *plan.columns, "state", "order_id", "error"]
    return pd.DataFrame(results, columns=columns).set_index("ticker")


def submit_fractional_buys(
    rh: Any,
    plan: pd.DataFrame,
    account_number: str | None = None,
    order_interval: float = 2.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Submit every row in a confirmed fractional-buy plan.

    Submissions are throttled: Robinhood rate-limits (HTTP 429) bursts of
    order POSTs — an unthrottled ~55-order plan was cut off after ~20 — so
    we wait ``order_interval`` seconds between orders. ``robin_stocks``
    swallows 429 responses into a ``None`` return, making rate-limit hits
    indistinguishable from ambiguous failures, so we prevent them rather
    than detect them.

    On any failed/ambiguous order the loop still stops immediately (an
    earlier request may have reached Robinhood even if its response did
    not) and the remaining rows are returned with state ``"not_attempted"``
    so callers can see exactly what was skipped.
    """

    def place(ticker: str, row: pd.Series) -> Any:
        return rh.order_buy_fractional_by_price(
            ticker,
            float(row["dollars"]),
            account_number=account_number,
            timeInForce="gfd",
            extendedHours=False,
        )

    results = _submit_orders(
        rh, plan, place, lambda row: f"${float(row['dollars']):,.2f}", order_interval, verbose
    )
    return results[["dollars", "state", "order_id", "error"]]


def submit_fractional_sells(
    rh: Any,
    plan: pd.DataFrame,
    account_number: str | None = None,
    order_interval: float = 2.0,
    verbose: bool = True,
    stop_on_failure: bool = True,
) -> pd.DataFrame:
    """Submit every row in a confirmed sell plan as a fractional market sell.

    Each order sells the position's exact share count, so nothing is left
    behind: positions bought in dollars are fractional, and converting a
    dollar amount back into shares would strand a remainder.  Orders are GFD
    regular-hours sells under the same throttle and stop-on-failure policy as
    :func:`submit_fractional_buys`. ``stop_on_failure`` off is for selling a
    whole account, where one rejected order should not leave the rest unsold;
    every position is still attempted once, so nothing is ever sold twice.
    """

    def place(ticker: str, row: pd.Series) -> Any:
        return rh.order_sell_fractional_by_quantity(
            ticker,
            round(float(row["quantity"]), 6),
            account_number=account_number,
            timeInForce="gfd",
        )

    results = _submit_orders(
        rh,
        plan,
        place,
        lambda row: f"{float(row['quantity']):.6f} shares (sell)",
        order_interval,
        verbose,
        stop_on_failure,
    )
    return results[["quantity", "state", "order_id", "error"]]


def fetch_order_states(
    rh: Any, order_ids: list[str], poll_interval: float = 0.5
) -> dict[str, dict[str, Any]]:
    """Fetch the live state of submitted orders by id (read-only GETs).

    Returns ``{order_id: {"state": ..., "filled_dollars": ...}}`` where
    ``filled_dollars`` comes from the executed notional (or price x quantity
    for non-fractional) and is ``None`` until the order executes.
    """
    states: dict[str, dict[str, Any]] = {}
    for i, order_id in enumerate(order_ids):
        try:
            info = rh.get_stock_order_info(order_id)
        except Exception as exc:  # noqa: BLE001 - recorded per order below
            info = None
        info = info if isinstance(info, dict) else {}
        filled = None
        notional = info.get("executed_notional")
        if isinstance(notional, dict) and notional.get("amount") is not None:
            try:
                filled = float(notional["amount"])
            except (TypeError, ValueError):
                filled = None
        elif info.get("average_price") and info.get("quantity"):
            try:
                filled = float(info["average_price"]) * float(info["quantity"])
            except (TypeError, ValueError):
                filled = None
        states[order_id] = {"state": info.get("state", "unknown"), "filled_dollars": filled}
        if poll_interval > 0 and i < len(order_ids) - 1:
            time.sleep(poll_interval)
    return states


# File and size column of each side's most recent --execute submission.
_LAST_ORDERS = {
    "buy": ("last_robinhood_orders.json", "dollars"),
    "sell": ("last_robinhood_sells.json", "quantity"),
}


def last_orders_path(side: str = "buy") -> Path:
    """Where the most recent --execute submission of ``side`` orders is persisted."""
    return Settings().cache_dir / _LAST_ORDERS[side][0]


def save_last_orders(results: pd.DataFrame, side: str = "buy") -> Path:
    """Persist a submission's tickers/sizes/order ids for later re-checking."""
    path = last_orders_path(side)
    frame = results.reset_index()[["ticker", _LAST_ORDERS[side][1], "order_id"]]
    payload = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "orders": frame.to_dict(orient="records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_last_orders(side: str = "buy") -> pd.DataFrame | None:
    """Load a side's previous submission, or None if nothing was saved yet."""
    path = last_orders_path(side)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload["orders"]).set_index("ticker")
    except (ValueError, KeyError, TypeError):
        return None
    return frame if not frame.empty else None


def ledger_path() -> Path:
    """Where the jobs record what they sold and re-bought: a JSON list of events, newest last."""
    return Settings().cache_dir / "sell_ledger.json"


def _ledger() -> list[dict[str, Any]]:
    path = ledger_path()
    if not path.exists():
        return []
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return (
        [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
    )


def _append_ledger(event: dict[str, Any]) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(_ledger() + [event], indent=2), encoding="utf-8")
    os.replace(part, path)


def record_sells(kind: str, results: pd.DataFrame) -> float:
    """Put a submission of sell orders on the ledger; return what it counts as bringing in.

    Only orders Robinhood reports as filled count, and only for what they
    filled: a re-buy must never exceed the cash the sells really produced, so
    an order that is still queued, unconfirmed or unreadable counts for
    nothing. A filled order with no reported amount counts for what the
    position was worth when the run looked at it. ``kind`` names the rule
    that sold, for whoever reads the ledger.
    """
    placed = results["order_id"].astype(str) != ""
    states = (
        results["live_state"] if "live_state" in results else pd.Series("", index=results.index)
    )
    filled = placed & states.isin(["filled", "partially_filled"])
    dollars = (
        pd.to_numeric(results["filled_dollars"], errors="coerce")
        if "filled_dollars" in results
        else pd.Series(math.nan, index=results.index)
    )
    if "value" in results:
        dollars = dollars.fillna(pd.to_numeric(results["value"], errors="coerce"))
    proceeds = float(dollars[filled].fillna(0.0).sum())
    _append_ledger(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": "sell",
            "kind": kind,
            "submitted": int(placed.sum()),
            "filled": int(filled.sum()),
            "proceeds": round(proceeds, 2),
        }
    )
    return proceeds


def record_buy(available: float, budget: float, submitted: float) -> None:
    """Put a re-buy on the ledger.

    ``submitted`` is what it actually sent to Robinhood. Whatever of
    ``available`` was not sent, because Robinhood's buying power capped the
    run or an order failed, is carried to the next re-buy rather than
    forgotten.
    """
    _append_ledger(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": "buy",
            "available": round(available, 2),
            "budget": round(budget, 2),
            "submitted": round(submitted, 2),
            "carried": round(available - submitted, 2),
        }
    )


def cash_to_rebuy() -> tuple[float, int]:
    """What the sells since the last re-buy brought in, and how many sell events that is.

    Anything the last re-buy left undeployed is included, so no proceeds are
    ever dropped by a run that was cut short.
    """
    since: list[dict[str, Any]] = []
    for event in reversed(_ledger()):
        since.append(event)
        if event.get("event") == "buy":
            break
    sells = [event for event in since if event.get("event") == "sell"]
    total = sum(float(event.get("proceeds") or 0.0) for event in sells)
    last_buy = next((event for event in since if event.get("event") == "buy"), None)
    if last_buy:
        total += float(last_buy.get("carried") or 0.0)
    return round(total, 2), len(sells)
