# Quickstart

## The data layer

`get_prices` returns an **N×T matrix** (N tickers × T dates) of adjusted close
prices for the most-liquid US common stocks (or any ticker list you pass):

```python
from stonks import get_prices, to_returns

prices = get_prices(top_n=500, period="2y", interval="1d", field="close")
returns = to_returns(prices)   # daily simple returns, N x (T-1)
```

The universe comes from NASDAQ Trader's symbol directory, filtered to common
stocks and ranked by average traded dollar volume. Everything is cached under
`data/`, so repeat calls are instant.

## The portfolio

One call builds a long-only mean-variance portfolio with a factor-model
covariance:

```python
from stonks import optimize_portfolio

weights = optimize_portfolio(top_n=500, months=3, gamma=50.0, K=25)
# DataFrame: ticker -> weight, non-negative, summing to 1
```

See {doc}`bayesian_factor_model` for what this does mathematically.

## The CLI

```bash
stonks fetch --tickers AAPL,MSFT --period 1y
stonks portfolio --top-n 500 --months 3 --gamma 50 -k 25
```

`stonks portfolio` prints the recommended holdings with weights and company
names. Add `--execute` (opt-in) to place buy-only fractional orders via
Robinhood, with an explicit confirmation prompt before anything is submitted.
