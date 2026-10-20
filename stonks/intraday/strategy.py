"""Transparent baseline strategies for intraday experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta

import pandas as pd

from .data import normalize_bars


@dataclass(frozen=True)
class OpeningRangeBreakout:
    """Signal once when price closes above the day's opening range.

    The opening range consists of bars whose timestamps are from 09:30 through
    ``opening_minutes`` after the open.  A signal uses a completed bar's close;
    the backtester can only enter on the following bar.

    This is an educational baseline, not an assertion of profitable edge.
    """

    opening_minutes: int = 30

    def __post_init__(self) -> None:
        if not 5 <= self.opening_minutes <= 120:
            raise ValueError("opening_minutes must be between 5 and 120")

    def entry_signals(self, bars: pd.DataFrame) -> pd.Series:
        bars = normalize_bars(bars)
        signals = pd.Series(False, index=bars.index, name="entry_signal")
        opening_end = (
            pd.Timestamp.combine(pd.Timestamp.today(), time(9, 30))
            + timedelta(minutes=self.opening_minutes)
        ).time()

        for _, day in bars.groupby(bars.index.date, sort=True):
            opening = day[day.index.time < opening_end]
            candidates = day[day.index.time >= opening_end]
            if opening.empty or candidates.empty:
                continue
            opening_high = float(opening["high"].max())
            crossed = candidates["close"] > opening_high
            if bool(crossed.any()):
                first_timestamp = crossed[crossed].index[0]
                signals.loc[first_timestamp] = True
        return signals
