"""Configuration for the stonks data wrapper.

A single :class:`Settings` dataclass holds the knobs the rest of the package
reads (cache location, download period/interval, universe size). Keeping them
in one place makes the public :func:`stonks.get_prices` wrapper easy to reason
about and easy to override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_cache_dir() -> Path:
    """Repo-level ``data/`` directory (sibling of the ``stonks`` package)."""
    return Path(__file__).resolve().parent.parent / "data"


@dataclass
class Settings:
    """Runtime settings for data download and caching.

    Attributes:
        cache_dir: Where downloaded data lives (regenerable, gitignored).
        period: yfinance history window, e.g. ``"1y"``, ``"5y"``, ``"max"``.
        interval: Bar resolution. Daily/weekly/monthly (``"1d"``/``"1wk"``/
            ``"1mo"``) work over long ``period`` values; intraday
            (``"1m"``,``"2m"``,``"5m"``,``"15m"``,``"30m"``,``"60m"``,``"90m"``)
            is limited to the last 60 days by yfinance.
        top_n: How many of the most-liquid common stocks to keep when no
            explicit ticker list is given.
        chunk_size: Tick per bulk yfinance request. Keeps us polite and lets a
            failed chunk be retried independently.
        universe_refresh_days: Re-download the NASDAQ Trader universe if the
            cached copy is older than this.
    """

    cache_dir: Path = field(default_factory=_default_cache_dir)
    period: str = "1y"
    interval: str = "1d"
    top_n: int = 1500
    chunk_size: int = 100
    universe_refresh_days: int = 1

    def universe_dir(self) -> Path:
        """Directory for cached universe lists."""
        path = self.cache_dir / "universe"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_subdir(self) -> Path:
        """Directory for cached price matrices."""
        path = self.cache_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prices_cache_path(self, key: str) -> Path:
        """Path for a price-matrix cache file named ``key``."""
        return self.cache_subdir() / f"{key}.parquet"
