"""Event-ordered intraday backtesting with fractional-share fills."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import time

import pandas as pd

from .data import normalize_bars
from .strategy import OpeningRangeBreakout


@dataclass(frozen=True)
class BacktestConfig:
    """Hard limits and fill assumptions for the small-account experiment."""

    initial_cash: float = 20.0
    account_cap: float = 20.0
    trade_notional: float = 5.0
    min_order_dollars: float = 1.0
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    daily_loss_limit: float = 0.25
    max_trades_per_day: int = 1
    slippage_bps: float = 5.0
    sell_fee_bps: float = 0.0
    fractional_precision: int = 6
    flatten_time: time = time(15, 50)

    def __post_init__(self) -> None:
        finite_positive = {
            "initial_cash": self.initial_cash,
            "account_cap": self.account_cap,
            "trade_notional": self.trade_notional,
            "min_order_dollars": self.min_order_dollars,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "daily_loss_limit": self.daily_loss_limit,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.initial_cash > self.account_cap:
            raise ValueError("initial_cash cannot exceed account_cap")
        if self.trade_notional > self.account_cap:
            raise ValueError("trade_notional cannot exceed account_cap")
        if self.min_order_dollars > self.trade_notional:
            raise ValueError("min_order_dollars cannot exceed trade_notional")
        if self.max_trades_per_day < 1:
            raise ValueError("max_trades_per_day must be at least 1")
        if not math.isfinite(self.slippage_bps) or self.slippage_bps < 0:
            raise ValueError("slippage_bps must be a non-negative finite number")
        if not math.isfinite(self.sell_fee_bps) or self.sell_fee_bps < 0:
            raise ValueError("sell_fee_bps must be a non-negative finite number")
        if not 0 <= self.fractional_precision <= 9:
            raise ValueError("fractional_precision must be between 0 and 9")


@dataclass(frozen=True)
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    quantity: float
    entry_price: float
    exit_price: float
    cost: float
    proceeds: float
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass(frozen=True)
class BacktestResult:
    config: BacktestConfig
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict[str, float | int]


def _round_quantity_down(quantity: float, precision: int) -> float:
    scale = 10**precision
    return math.floor(quantity * scale + 1e-12) / scale


def _metrics(initial_cash: float, equity: pd.Series, trades: list[Trade]) -> dict[str, float | int]:
    final_equity = float(equity.iloc[-1]) if not equity.empty else initial_cash
    running_peak = equity.cummax() if not equity.empty else pd.Series([initial_cash])
    drawdown = equity / running_peak - 1 if not equity.empty else pd.Series([0.0])
    pnls = [trade.pnl for trade in trades]
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = -sum(value for value in pnls if value < 0)
    return {
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "total_pnl": final_equity - initial_cash,
        "total_return_pct": (final_equity / initial_cash - 1) * 100,
        "trade_count": len(trades),
        "win_rate_pct": (sum(value > 0 for value in pnls) / len(pnls) * 100) if pnls else 0.0,
        "average_trade_pnl": (sum(pnls) / len(pnls)) if pnls else 0.0,
        "max_drawdown_pct": float(drawdown.min() * 100),
        "profit_factor": (
            (gross_profit / gross_loss) if gross_loss else (math.inf if gross_profit else 0.0)
        ),
    }


def run_backtest(
    bars: pd.DataFrame,
    *,
    strategy: OpeningRangeBreakout | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Simulate a long-only strategy without carrying positions overnight.

    Entry signals are observed at a bar close and filled at the following bar's
    open.  Stop-loss and take-profit collisions are resolved pessimistically in
    favor of the stop.  The final available bar of every session is always an
    exit, even when data is incomplete.
    """
    bars = normalize_bars(bars)
    strategy = strategy or OpeningRangeBreakout()
    config = config or BacktestConfig()
    signals = strategy.entry_signals(bars).reindex(bars.index, fill_value=False)

    cash = config.initial_cash
    all_trades: list[Trade] = []
    equity_points: dict[pd.Timestamp, float] = {}
    slippage = config.slippage_bps / 10_000
    sell_fee = config.sell_fee_bps / 10_000

    for _, day in bars.groupby(bars.index.date, sort=True):
        quantity = 0.0
        entry_price = 0.0
        entry_cost = 0.0
        entry_time: pd.Timestamp | None = None
        pending_entry = False
        day_realized = 0.0
        day_trade_count = 0

        for offset, (timestamp, bar) in enumerate(day.iterrows()):
            is_last_bar = offset == len(day) - 1

            # A completed signal can only affect the next bar.  Process the
            # pending entry before observing the current bar's signal.
            if (
                pending_entry
                and quantity == 0
                and timestamp.time() < config.flatten_time
                and day_trade_count < config.max_trades_per_day
                and day_realized > -config.daily_loss_limit
            ):
                fill_price = float(bar["open"]) * (1 + slippage)
                notional = min(config.trade_notional, cash)
                proposed_quantity = _round_quantity_down(
                    notional / fill_price, config.fractional_precision
                )
                proposed_cost = proposed_quantity * fill_price
                if proposed_cost >= config.min_order_dollars:
                    quantity = proposed_quantity
                    entry_price = fill_price
                    entry_cost = proposed_cost
                    entry_time = timestamp
                    cash -= entry_cost
                    day_trade_count += 1
                pending_entry = False

            if quantity > 0:
                stop_price = entry_price * (1 - config.stop_loss_pct)
                take_price = entry_price * (1 + config.take_profit_pct)
                exit_reason: str | None = None
                raw_exit_price = 0.0
                if float(bar["low"]) <= stop_price:
                    exit_reason = "stop_loss"
                    raw_exit_price = stop_price
                elif float(bar["high"]) >= take_price:
                    exit_reason = "take_profit"
                    raw_exit_price = take_price
                elif timestamp.time() >= config.flatten_time or is_last_bar:
                    exit_reason = "session_close"
                    raw_exit_price = float(bar["close"])

                if exit_reason is not None:
                    exit_price = raw_exit_price * (1 - slippage)
                    gross_proceeds = quantity * exit_price
                    proceeds = gross_proceeds * (1 - sell_fee)
                    pnl = proceeds - entry_cost
                    cash += proceeds
                    day_realized += pnl
                    trade = Trade(
                        entry_time=entry_time if entry_time is not None else timestamp,
                        exit_time=timestamp,
                        quantity=quantity,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        cost=entry_cost,
                        proceeds=proceeds,
                        pnl=pnl,
                        return_pct=(proceeds / entry_cost - 1) * 100,
                        exit_reason=exit_reason,
                    )
                    all_trades.append(trade)
                    quantity = 0.0
                    entry_price = 0.0
                    entry_cost = 0.0
                    entry_time = None

            marked_position = quantity * float(bar["close"])
            equity_points[timestamp] = cash + marked_position

            if (
                bool(signals.loc[timestamp])
                and quantity == 0
                and not is_last_bar
                and timestamp.time() < config.flatten_time
            ):
                pending_entry = True

    equity = pd.Series(equity_points, name="equity", dtype=float).sort_index()
    trade_columns = list(Trade.__dataclass_fields__)
    trade_frame = pd.DataFrame([asdict(trade) for trade in all_trades], columns=trade_columns)
    return BacktestResult(
        config=config,
        trades=trade_frame,
        equity_curve=equity,
        metrics=_metrics(config.initial_cash, equity, all_trades),
    )
