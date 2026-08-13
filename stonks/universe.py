"""Stock universe acquisition.

The "popular" universe is built from NASDAQ Trader's public symbol directory
(``nasdaqlisted.txt`` + ``otherlisted.txt``), which lists every US-listed
security. We filter it down to common stocks (dropping ETFs, test issues,
preferreds, warrants, rights, funds, SPACs). Liquidity ranking happens later in
:mod:`stonks.download` once we have prices.

This is not the Russell 1000 — that constituent list isn't reliably available
for free. Ranking all common stocks by traded dollar volume gives us a
comparable, fully reproducible "most popular N" set.
"""

from __future__ import annotations

import io
import re
import time
import urllib.request

import pandas as pd

from .config import Settings

# NASDAQ Trader symbol directory. Pipe-delimited text files, refreshed daily.
_BASE_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/"
_NASDAQ_LISTED = "nasdaqlisted.txt"
_OTHER_LISTED = "otherlisted.txt"

# A desktop browser UA avoids being filtered as a bot.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Exchange codes in otherlisted.txt -> human-readable venue. Unknown codes are
# passed through unchanged.
_EXCHANGE_MAP = {
    "N": "NYSE",
    "A": "NYSE American",
    "P": "NYSE Arca",
    "Z": "CBOE",
    "V": "IEX",
    "Q": "NASDAQ",
}

# Tokens in a security name that flag non-common / non-operational securities.
_DROP_TOKENS = (
    "preferred",
    "rights",
    "warrant",
    "fund",
    "acquisition corp",  # SPACs
    "units",
)

# A plausible ticker symbol: 1-8 uppercase chars, dots/dashes allowed.
_TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,7}$")


def _fetch_txt(filename: str, settings: Settings, force_refresh: bool = False) -> str:
    """Fetch a NASDAQ Trader text file, caching the raw text to disk.

    Reuses the cached file unless ``force_refresh`` or it is older than
    ``settings.universe_refresh_days``.
    """
    path = settings.universe_dir() / filename
    if path.exists() and not force_refresh:
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days <= settings.universe_refresh_days:
            return path.read_text(encoding="utf-8")

    req = urllib.request.Request(_BASE_URL + filename, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - trusted URL
        text = resp.read().decode("utf-8", "ignore")

    path.write_text(text, encoding="utf-8")
    return text


def _read_pipe(text: str) -> pd.DataFrame:
    """Read a pipe-delimited NASDAQ Trader file into a string DataFrame.

    The files have a header row and end with a stray pipe (plus a footer line),
    which produces a trailing all-empty column. The header and that empty column
    are dropped here so callers can index fields by position.
    """
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, header=None)
    df = df.dropna(how="all", axis=1)  # drop the trailing empty column
    # Drop the header row if present (first cell is "Symbol" / "ACT Symbol").
    if len(df) > 0:
        first = str(df.iloc[0, 0]).strip().lower()
        if first in ("symbol", "act symbol", "act symbol"):
            df = df.iloc[1:].reset_index(drop=True)
    return df


def _parse_combined(nasdaq_text: str, other_text: str) -> pd.DataFrame:
    """Combine both NASDAQ Trader files into one frame.

    Returns columns: ``[ticker, name, exchange, is_etf, is_test]``.
    Pure function (no I/O) so it can be unit-tested with synthetic input.
    """
    rows: list[dict] = []

    # nasdaqlisted.txt columns:
    # Symbol|Security Name|Market Category|Test Issue|Financial Status|
    # Round Lot Size|ETF|NextShares
    nq = _read_pipe(nasdaq_text)
    if len(nq.columns) >= 7:
        for _, r in nq.iterrows():
            rows.append(
                {
                    "ticker": str(r.iloc[0]).strip(),
                    "name": str(r.iloc[1]).strip(),
                    "exchange": "NASDAQ",
                    "is_etf": str(r.iloc[6]).strip().upper(),
                    "is_test": str(r.iloc[3]).strip().upper(),
                }
            )

    # otherlisted.txt columns:
    # ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|
    # Test Issue|NASDAQ Symbol
    ot = _read_pipe(other_text)
    if len(ot.columns) >= 7:
        for _, r in ot.iterrows():
            ex_code = str(r.iloc[2]).strip().upper()
            rows.append(
                {
                    "ticker": str(r.iloc[0]).strip(),
                    "name": str(r.iloc[1]).strip(),
                    "exchange": _EXCHANGE_MAP.get(ex_code, ex_code or "UNKNOWN"),
                    "is_etf": str(r.iloc[4]).strip().upper(),
                    "is_test": str(r.iloc[6]).strip().upper(),
                }
            )

    return pd.DataFrame(rows, columns=["ticker", "name", "exchange", "is_etf", "is_test"])


def filter_common_stocks(combined: pd.DataFrame) -> pd.DataFrame:
    """Filter a combined frame down to operating common stocks.

    Keeps ETF/test flags == 'N', names containing "common stock", dropping
    names that mention preferreds / warrants / rights / funds / SPACs / units.
    Returns ``[ticker, name, exchange]`` deduped, ticker validated.
    """
    df = combined.copy()
    df["name_lc"] = df["name"].str.lower()

    mask = (
        (df["is_etf"] == "N")
        & (df["is_test"] == "N")
        & df["name_lc"].str.contains("common stock", na=False)
    )
    for token in _DROP_TOKENS:
        mask &= ~df["name_lc"].str.contains(token, na=False)

    df = df[mask].copy()

    # Validate tickers and clean up.
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df = df[df["ticker"].str.match(_TICKER_RE).fillna(False)]
    df["exchange"] = df["exchange"].fillna("UNKNOWN")

    df = df.drop_duplicates(subset="ticker", keep="first")
    return df[["ticker", "name", "exchange"]].sort_values("ticker").reset_index(drop=True)


def list_all_common_stocks(
    settings: Settings | None = None, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch and filter the full list of US-listed common stocks."""
    settings = settings or Settings()
    nasdaq_text = _fetch_txt(_NASDAQ_LISTED, settings, force_refresh=force_refresh)
    other_text = _fetch_txt(_OTHER_LISTED, settings, force_refresh=force_refresh)
    return filter_common_stocks(_parse_combined(nasdaq_text, other_text))


def load_universe(
    settings: Settings | None = None, force_refresh: bool = False
) -> pd.DataFrame:
    """Return the common-stock universe, cached to parquet for reuse.

    The cache lives at ``<cache_dir>/universe/all_common_stocks.parquet`` and is
    refreshed when older than ``settings.universe_refresh_days`` or when
    ``force_refresh`` is set.
    """
    settings = settings or Settings()
    cache_path = settings.universe_dir() / "all_common_stocks.parquet"

    if cache_path.exists() and not force_refresh:
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days <= settings.universe_refresh_days:
            return pd.read_parquet(cache_path)

    df = list_all_common_stocks(settings, force_refresh=force_refresh)
    df.to_parquet(cache_path, index=False)
    return df
