"""Robinhood execution helpers.

The optimizer produces target weights.  This module converts those weights to
dollar-denominated fractional buy orders and submits them through
``robin_stocks``.  It is intentionally buy-only: existing positions are not
sold or rebalanced.

``robin_stocks`` uses Robinhood's private API and may stop working when that
API changes.  No order is submitted by the CLI without explicit confirmation.
"""

from __future__ import annotations

import json
import math
import os
import time
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

    result = rh.login(
        username=os.getenv("ROBINHOOD_USERNAME"),
        password=os.getenv("ROBINHOOD_PASSWORD"),
        mfa_code=os.getenv("ROBINHOOD_MFA_CODE"),
    )
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
    if not math.isfinite(order_interval) or order_interval < 0:
        raise ValueError("order_interval must be a non-negative finite number")

    results: list[dict[str, Any]] = []
    stopped = False
    for i, (ticker, row) in enumerate(plan.iterrows()):
        if stopped:
            results.append(
                {
                    "ticker": ticker,
                    "dollars": float(row["dollars"]),
                    "state": "not_attempted",
                    "order_id": "",
                    "error": "",
                }
            )
            continue

        try:
            raw_response = rh.order_buy_fractional_by_price(
                ticker,
                float(row["dollars"]),
                account_number=account_number,
                timeInForce="gfd",
                extendedHours=False,
            )
        except Exception as exc:  # noqa: BLE001 - recorded per order below
            raw_response = {"detail": f"request failed: {exc}"}

        response = raw_response if isinstance(raw_response, dict) else {}
        error = response.get("detail") or response.get("non_field_errors")
        if not error and not response.get("id"):
            error = "Robinhood returned no order confirmation"
        state = response.get("state", "error" if error else "submitted")
        results.append(
            {
                "ticker": ticker,
                "dollars": float(row["dollars"]),
                "state": state,
                "order_id": response.get("id", ""),
                "error": str(error) if error else "",
            }
        )
        if verbose:
            print(
                f"[stonks] order {i + 1}/{len(plan)}: {ticker} "
                f"${float(row['dollars']):,.2f} -> {state}",
                flush=True,
            )

        if error:
            stopped = True
        elif i < len(plan) - 1:
            time.sleep(order_interval)
    return pd.DataFrame(results).set_index("ticker")


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


def last_orders_path() -> Path:
    """Where the most recent --execute submission is persisted."""
    return Settings().cache_dir / "last_robinhood_orders.json"


def save_last_orders(results: pd.DataFrame) -> Path:
    """Persist a submission's tickers/dollars/order ids for later re-checking."""
    path = last_orders_path()
    frame = results.reset_index()[["ticker", "dollars", "order_id"]]
    payload = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "orders": frame.to_dict(orient="records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_last_orders() -> pd.DataFrame | None:
    """Load the previous submission, or None if nothing was saved yet."""
    path = last_orders_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload["orders"]).set_index("ticker")
    except (ValueError, KeyError, TypeError):
        return None
    return frame if not frame.empty else None
