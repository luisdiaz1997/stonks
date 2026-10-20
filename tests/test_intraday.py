"""Offline tests for the constrained intraday experiment."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stonks.intraday import (
    BacktestConfig,
    OpeningRangeBreakout,
    normalize_bars,
    run_backtest,
)


def _bars(
    closes: list[float],
    *,
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    index = pd.date_range("2026-08-03 09:30", periods=len(closes), freq="5min")
    opens = opens or closes
    highs = highs or [max(open_, close) + 0.1 for open_, close in zip(opens, closes)]
    lows = lows or [min(open_, close) - 0.1 for open_, close in zip(opens, closes)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [10_000] * len(closes),
        },
        index=index,
    )


def test_normalize_bars_localizes_and_filters_extended_hours() -> None:
    frame = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [101.0, 101.0, 101.0],
            "Low": [99.0, 99.0, 99.0],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [1, 1, 1],
        },
        index=pd.to_datetime(
            ["2026-08-03 09:25", "2026-08-03 09:30", "2026-08-03 16:00"]
        ),
    )

    result = normalize_bars(frame)

    assert len(result) == 1
    assert result.index[0].hour == 9
    assert result.index.tz is not None
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]


def test_opening_range_signal_enters_on_next_bar() -> None:
    closes = [100.0] * 6 + [101.0, 101.0, 101.0]
    opens = [100.0] * 6 + [100.5, 101.0, 101.0]
    highs = [100.2] * 6 + [101.2, 103.5, 101.2]
    lows = [99.8] * 6 + [100.4, 100.5, 100.5]
    frame = _bars(closes, opens=opens, highs=highs, lows=lows)
    config = BacktestConfig(slippage_bps=0)

    result = run_backtest(frame, config=config)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_time"].time().isoformat() == "10:05:00"
    assert trade["exit_reason"] == "take_profit"
    assert trade["pnl"] > 0
    assert result.metrics["trade_count"] == 1


def test_same_bar_stop_and_target_uses_stop_pessimistically() -> None:
    closes = [100.0] * 6 + [101.0, 101.0]
    opens = [100.0] * 6 + [100.5, 101.0]
    highs = [100.2] * 6 + [101.2, 104.0]
    lows = [99.8] * 6 + [100.4, 98.0]
    frame = _bars(closes, opens=opens, highs=highs, lows=lows)

    result = run_backtest(frame, config=BacktestConfig(slippage_bps=0))

    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"
    assert result.trades.iloc[0]["pnl"] < 0


def test_position_is_forced_closed_on_last_available_bar() -> None:
    closes = [100.0] * 6 + [101.0, 101.1, 101.2]
    opens = [100.0] * 6 + [100.5, 101.0, 101.1]
    highs = [100.2] * 6 + [101.2, 101.3, 101.4]
    lows = [99.8] * 6 + [100.4, 100.8, 100.9]
    frame = _bars(closes, opens=opens, highs=highs, lows=lows)

    result = run_backtest(frame, config=BacktestConfig(slippage_bps=0))

    assert result.trades.iloc[0]["exit_reason"] == "session_close"
    assert result.trades.iloc[0]["exit_time"] == normalize_bars(frame).index[-1]
    assert math.isclose(result.equity_curve.iloc[-1], result.metrics["final_equity"])


def test_small_account_cap_is_structural() -> None:
    with pytest.raises(ValueError, match="account_cap"):
        BacktestConfig(initial_cash=20.01)
    with pytest.raises(ValueError, match="trade_notional"):
        BacktestConfig(trade_notional=20.01)

    custom_limit = BacktestConfig(
        initial_cash=50.0,
        account_cap=50.0,
        trade_notional=10.0,
    )
    assert custom_limit.account_cap == 50.0
    assert custom_limit.trade_notional == 10.0


def test_strategy_rejects_unreasonable_opening_window() -> None:
    with pytest.raises(ValueError, match="opening_minutes"):
        OpeningRangeBreakout(opening_minutes=1)
