"""The price-data wrapper.

Public entry point: :func:`get_prices` -> an **N×T** matrix (N tickers as rows,
T time points as columns) of stock prices, with the time resolution of ``T``
configurable via ``interval``.

Two ways to choose which stocks:
  * ``tickers=[...]``  -- explicit list (fast, no ranking). Good for quick tests.
  * ``top_n=N``        -- the N most-liquid US common stocks, ranked by average
                         traded dollar volume over the window.

Everything is cached to parquet so repeat calls are instant.

Note: yfinance intraday intervals (``1m``..``90m``) are only available for the
last 60 days; daily/weekly/monthly work over long ``period`` values.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from .config import Settings
from .universe import load_universe

# Valid yfinance bar intervals. Intraday is limited to ~60 days of history.
VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"}
VALID_FIELDS = {"open", "high", "low", "close", "volume"}

_INTRADAY = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}

# How stale a cached frame may be before we re-download, by interval.
# (weekends/holidays make daily need ~1 day of slack.)
_TOLERANCE_DAYS = {"1d": 1, "5d": 4, "1wk": 4, "1mo": 5, "3mo": 10}


def _chunked(seq: list, n: int) -> Iterable[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _is_stale(frame: pd.DataFrame, interval: str) -> bool:
    """True if the cached frame's last bar is older than the interval allows."""
    if frame is None or frame.empty:
        return True
    try:
        last = pd.Timestamp(frame.index.max())
    except Exception:
        return True
    today = pd.Timestamp.now().normalize()
    if interval in _INTRADAY:
        return last < today  # need same-day data
    tol = pd.Timedelta(days=_TOLERANCE_DAYS.get(interval, 1))
    return last < (today - tol)


def _download_raw(
    tickers: list[str], period: str, interval: str, chunk_size: int
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Bulk-download OHLCV in chunks.

    Returns ``(fields, failed)`` where ``fields`` maps each yfinance field name
    (``Open/High/Low/Close/Volume``) to a date×ticker DataFrame, and ``failed``
    lists tickers whose chunk blew up.
    """
    import yfinance as yf

    fields: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    total = max(1, math.ceil(len(tickers) / chunk_size))

    for chunk in tqdm(_chunked(list(tickers), chunk_size), total=total, desc="download"):
        try:
            data = yf.download(
                chunk,
                period=period,
                interval=interval,
                auto_adjust=True,  # Close is split+dividend adjusted
                progress=False,
                threads=True,
            )
        except Exception:
            failed.extend(chunk)
            continue

        if data is None or data.empty:
            failed.extend(chunk)
            continue

        # Single-ticker responses come back with flat columns; normalize.
        if not isinstance(data.columns, pd.MultiIndex):
            if len(chunk) != 1:
                failed.extend(chunk)
                continue
            data.columns = pd.MultiIndex.from_product([data.columns, [chunk[0]]])

        for field_name in data.columns.get_level_values(0).unique():
            sub = data[field_name].copy()
            sub.columns = [str(c) for c in sub.columns]
            fields[field_name] = fields.get(field_name, pd.DataFrame()).join(sub, how="outer")

        time.sleep(0.3)  # be polite

    return fields, failed


def _full_close_volume(
    period: str, interval: str, settings: Settings, force_refresh: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Close + Volume for the whole universe, cached (the expensive part).

    Returned (and cached) as date×ticker frames so they can be reused across
    different ``top_n`` / ``field`` selections without re-downloading.
    """
    close_path = settings.prices_cache_path(f"full_{interval}_{period}_close")
    volume_path = settings.prices_cache_path(f"full_{interval}_{period}_volume")

    if not force_refresh and close_path.exists() and volume_path.exists():
        close = pd.read_parquet(close_path)
        volume = pd.read_parquet(volume_path)
        if not _is_stale(close, interval):
            return close, volume

    tickers = load_universe(settings)["ticker"].tolist()
    fields, failed = _download_raw(tickers, period, interval, settings.chunk_size)
    close = fields.get("Close", pd.DataFrame())
    volume = fields.get("Volume", pd.DataFrame())
    if not close.empty:
        close.to_parquet(close_path)
    if not volume.empty:
        volume.to_parquet(volume_path)
    if failed:
        print(f"[stonks] {len(failed)} tickers failed to download (ignored).")
    return close, volume


def _select_top_n(close: pd.DataFrame, volume: pd.DataFrame, top_n: int) -> list[str]:
    """Rank tickers by average dollar volume (close*volume) and take the top N."""
    dvol = (close * volume).mean(axis=0).dropna()
    return dvol.sort_values(ascending=False).head(top_n).index.tolist()


def _to_nt_matrix(field_frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Reshape a date×ticker frame to N×T (tickers × dates), ordered by ``tickers``."""
    mat = field_frame.T.reindex(tickers)
    mat.index.name = "ticker"
    return mat


def _cache_key(
    mode: str, interval: str, period: str, field: str, top_n: int | None, tickers: list[str] | None
) -> str:
    parts = ["prices", mode, interval, period, field]
    if mode == "ranked":
        parts.append(f"top{top_n}")
    else:
        digest = hashlib.sha1(",".join(sorted(tickers)).encode()).hexdigest()[:10]
        parts.append(digest)
    return "_".join(parts)


def get_prices(
    top_n: int = 1500,
    tickers: list[str] | None = None,
    period: str = "1y",
    interval: str = "1d",
    field: str = "close",
    force_refresh: bool = False,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Fetch stock prices as an **N×T** matrix (N tickers × T time points).

    Args:
        top_n: Number of most-liquid common stocks to keep (ignored if
            ``tickers`` is given).
        tickers: Explicit ticker list. When provided, ranking is skipped and
            only these are downloaded (fast).
        period: History window, e.g. ``"1y"``, ``"5y"``, ``"max"``.
        interval: Bar resolution (``"1d"``, ``"1wk"``, ``"1mo"`` for long
            history; ``"1m"``..``"90m"`` intraday, last 60 days only).
        field: Which OHLCV field to return: ``open/high/low/close/volume``.
        force_refresh: Bypass cache and re-download.
        settings: Override default :class:`Settings`.

    Returns:
        DataFrame with shape (N tickers, T dates): tickers as the row index,
        dates as columns. ``field="close"`` (default) is split/dividend adjusted.
    """
    settings = settings or Settings()
    field = field.lower()
    interval = interval.lower()

    if field not in VALID_FIELDS:
        raise ValueError(f"field must be one of {sorted(VALID_FIELDS)}, got {field!r}")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {sorted(VALID_INTERVALS)}, got {interval!r}")

    mode = "tickers" if tickers is not None else "ranked"
    key = _cache_key(mode, interval, period, field, top_n, tickers)
    cache_path = settings.prices_cache_path(key)

    # Serve from cache unless stale / forced. Cache is stored date×ticker.
    if not force_refresh and cache_path.exists():
        cached = pd.read_parquet(cache_path)  # date × ticker
        if not _is_stale(cached, interval):
            return _to_nt_matrix(cached, list(cached.columns))

    field_up = field.capitalize()  # yfinance field names: Close, Volume, ...

    if tickers is not None:
        fields_dl, failed = _download_raw(list(tickers), period, interval, settings.chunk_size)
        if field_up not in fields_dl or fields_dl[field_up].empty:
            raise RuntimeError(f"No data downloaded for tickers {tickers!r}")
        src = fields_dl[field_up]
        chosen = [t for t in tickers if t in src.columns]
        frame = src[chosen]
    else:
        close, volume = _full_close_volume(period, interval, settings, force_refresh)
        if close.empty:
            raise RuntimeError("Universe download produced no close data.")
        top = _select_top_n(close, volume, top_n)
        if field in ("close", "volume"):
            frame = (close if field == "close" else volume)[top]
        else:
            fields_dl, _ = _download_raw(top, period, interval, settings.chunk_size)
            frame = fields_dl.get(field_up, pd.DataFrame())[top]

    # Cache as date×ticker (uniform staleness checks), return transposed N×T.
    if not frame.empty:
        frame.to_parquet(cache_path)
        print(f"[stonks] cached {frame.shape[1]} tickers × {frame.shape[0]} bars -> {cache_path.name}")
    return _to_nt_matrix(frame, list(frame.columns))
