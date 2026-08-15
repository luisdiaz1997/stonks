# stonks

Simple, exploratory algorithms for stocks and investments.

A learning playground for experimenting with market-related ideas — backtests,
signals, portfolio heuristics, and anything else worth trying out. **No financial
advice**; just code to read, run, and tinker with.

## Installation

From PyPI (distribution name `stonker`, import name `stonks`):

```bash
pip install stonker
```

From source (editable, recommended while developing):

```bash
git clone https://github.com/luisdiaz1997/stonks.git
cd stonks
pip install -e ".[dev]"
```

Extras: `pip install "stonker[notebooks]"` (jupyter/matplotlib/seaborn) or
`"stonker[all]"`.

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

Or from the command line (a `stonks` console command is installed, plus
`python -m stonks`):

```bash
stonks fetch --tickers AAPL,MSFT,GOOG --period 1y --interval 1d
stonks fetch --top-n 40 --period 1y --field close
stonks fetch --help
```

Generate a portfolio recommendation without placing any orders:

```bash
stonks portfolio --top-n 500 --months 3 --gamma 50 -k 25
```

To opt into Robinhood execution, add `--execute`. The CLI authenticates, shows
the exact dollar orders, and asks for confirmation; answering no submits
nothing. Orders are buy-only fractional market orders and do not sell or
rebalance existing holdings. Use `--amount` to invest less than the full buying
power reported by Robinhood.

```bash
stonks portfolio --top-n 500 --months 3 --gamma 50 -k 25 --execute --amount 500
```

`robin_stocks` securely prompts for credentials when it has no cached session.
Alternatively, set `ROBINHOOD_USERNAME`, `ROBINHOOD_PASSWORD`, and optionally
`ROBINHOOD_MFA_CODE` in the environment. Never pass a password as a command-line
argument or commit credentials to this repository. `robin_stocks` relies on an
unofficial Robinhood API, so execution can break when Robinhood changes it.

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
