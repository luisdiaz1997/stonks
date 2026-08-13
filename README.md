# stonks

Simple, exploratory algorithms for stocks and investments.

A learning playground for experimenting with market-related ideas — backtests,
signals, portfolio heuristics, and anything else worth trying out. **No financial
advice**; just code to read, run, and tinker with.

## Installation

From source (editable, recommended while developing):

```bash
git clone https://github.com/luisdiaz1997/stonks.git
cd stonks
pip install -e ".[dev]"
```

Minimal install (runtime deps only):

```bash
pip install -e .
```

## Usage

The package currently provides one thing: `get_prices`, a thin wrapper that
returns a clean **N×T** matrix (N tickers × T time points) of stock prices.

```python
from stonks import get_prices

# The 1500 most-liquid US common stocks, 1y of daily close prices.
prices = get_prices(top_n=1500, period="1y", interval="1d", field="close")
print(prices.shape)  # (1500 tickers, ~252 trading days)

# Or a handful of specific tickers (fast — skips ranking).
prices = get_prices(tickers=["AAPL", "MSFT", "GOOG"], period="1y")

# `interval` sets the resolution of T: daily, weekly, monthly, or intraday.
weekly = get_prices(tickers=["AAPL"], period="2y", interval="1wk", field="volume")
```

Or from the command line:

```bash
python -m stonks fetch --tickers AAPL,MSFT,GOOG --period 1y --interval 1d
python -m stonks fetch --top-n 40 --period 1y --field close
```

Everything is cached under `data/` (gitignored), so repeat calls are instant.

### Where the data comes from

- **Prices**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance),
  split/dividend adjusted.
- **Universe**: the full list of US-listed common stocks is built from NASDAQ
  Trader's public symbol directory, then ranked by average traded dollar volume
  to keep the most-liquid `top_n`. This is **not** the Russell 1000 (that list
  isn't reliably free) but yields a comparable, reproducible "most popular N".

> **Survivorship bias:** the universe reflects *currently listed* stocks, so
> delisted/bankrupt names are absent. Fine for learning and exploration; real
> backtests will look better than reality.

### Valid `interval` values

`1d`, `5d`, `1wk`, `1mo`, `3mo` work over long `period`s. Intraday
(`1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`) is limited to the last ~60 days by
yfinance. `field` is one of `open`/`high`/`low`/`close`/`volume`.

## Development

Run tests:

```bash
pytest
```

Format and check types:

```bash
black .
isort .
mypy stonks
```

## License

[MIT](LICENSE)
