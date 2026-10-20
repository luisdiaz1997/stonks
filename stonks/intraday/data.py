"""Intraday OHLCV data loading and normalization."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import Settings

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,15}$")


def normalize_bars(
    frame: pd.DataFrame,
    *,
    timezone: str = "America/New_York",
    regular_hours_only: bool = True,
) -> pd.DataFrame:
    """Return clean, chronological OHLCV bars in the market timezone.

    Naive timestamps are interpreted as market-local time.  Aware timestamps
    are converted.  Rows with invalid prices are discarded, as are duplicate
    timestamps (the final observation wins).
    """
    if frame is None or frame.empty:
        raise ValueError("bars must not be empty")

    bars = frame.copy()
    bars.columns = [str(column).strip().lower() for column in bars.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"bars are missing required columns: {', '.join(missing)}")

    try:
        index = pd.DatetimeIndex(pd.to_datetime(bars.index, errors="raise"))
    except Exception as exc:
        raise ValueError("bars must have a datetime-like index") from exc
    if index.tz is None:
        index = index.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    else:
        index = index.tz_convert(timezone)
    bars.index = index
    bars.index.name = "timestamp"

    bars = bars.loc[:, list(REQUIRED_COLUMNS)]
    for column in REQUIRED_COLUMNS:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna().sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    bars = bars[
        (bars["open"] > 0)
        & (bars["high"] > 0)
        & (bars["low"] > 0)
        & (bars["close"] > 0)
        & (bars["volume"] >= 0)
        & (bars["high"] >= bars[["open", "close", "low"]].max(axis=1))
        & (bars["low"] <= bars[["open", "close", "high"]].min(axis=1))
    ]
    if regular_hours_only:
        bars = bars.between_time("09:30", "15:59", inclusive="both")
    if bars.empty:
        raise ValueError("no valid intraday bars remain after normalization")
    return bars


def _flatten_yfinance_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance's version-dependent single-ticker column layout."""
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    ticker = ticker.upper()
    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level).map(str)
        if ticker in set(value.upper() for value in values):
            matches = [
                value
                for value in frame.columns.levels[level]
                if str(value).upper() == ticker
            ]
            if matches:
                return frame.xs(matches[0], axis=1, level=level, drop_level=True)
    # A single-ticker response may use a redundant level whose exact spelling
    # differs. Dropping the level with one value recovers the OHLCV fields.
    for level in range(frame.columns.nlevels - 1, -1, -1):
        if len(frame.columns.get_level_values(level).unique()) == 1:
            return frame.droplevel(level, axis=1)
    raise RuntimeError("could not identify OHLCV columns in yfinance response")


def download_bars(
    ticker: str,
    *,
    period: str = "30d",
    interval: str = "5m",
    force_refresh: bool = False,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Download and cache regular-session OHLCV bars for one ticker.

    This uses yfinance for research data only.  It is not an execution-quality
    real-time feed, and callers must not use it as the sole price source for a
    live broker order.
    """
    ticker = ticker.strip().upper()
    interval = interval.strip().lower()
    if not _TICKER_RE.fullmatch(ticker):
        raise ValueError(f"invalid ticker {ticker!r}")
    if interval not in INTRADAY_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(INTRADAY_INTERVALS)}")

    settings = settings or Settings()
    cache_key = f"intraday_{ticker.replace('.', '_')}_{interval}_{period}"
    cache_path: Path = settings.prices_cache_path(cache_key)
    if cache_path.exists() and not force_refresh:
        return normalize_bars(pd.read_parquet(cache_path))

    import yfinance as yf

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"no intraday data downloaded for {ticker}")
    bars = normalize_bars(_flatten_yfinance_columns(raw, ticker))
    bars.to_parquet(cache_path)
    return bars
