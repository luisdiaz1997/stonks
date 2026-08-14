# stonks — project summary & current state (handoff)

Exploratory stock/investment algorithms. Python package + notebooks + CLI.
**Not financial advice.** Repo: github.com/luisdiaz1997/stonks.

## Stack & environment
- Python >=3.11; deps: torch, numpy, pandas, yfinance, click, pyarrow, tqdm.
- Conda env `stonks` (Python 3.13) has `pip install -e ".[notebooks]"` (jupyter) + pytest.
  Binary: `/Users/luisfcd/opt/miniconda3/envs/stonks/bin/stonks`. Editable install — code changes are live.
- Jupyter runs on port 8888, notebook-dir = repo root.
- 17 offline tests (`pytest`): universe parsing/filtering, returns, factor covariance, optimizer.

## Data layer
- `get_prices(top_n, tickers=None, period, interval, field)` → **N×T matrix** (rows=tickers, cols=dates),
  adjusted close by default. Cached as parquet under `data/` (gitignored, regenerable).
- Universe: NASDAQ Trader symbol directory (`nasdaqlisted.txt`+`otherlisted.txt`), filtered to common
  stocks (drops ETFs/test issues/preferreds/warrants/rights/funds/SPACs) → ~4000 names, ranked by
  average dollar volume → top_n. NOT Russell 1000 (not reliably free); survivorship-biased by construction.
- `to_returns(prices, log=False)` → daily simple returns `P_t/P_{t-1} − 1`, N×(T−1).
  Covariance/PCA is done on **returns, not prices** (prices are non-stationary levels; returns are the
  stationary increments — the object we model).

## The core pipeline (`stonks/portfolio.py` → `optimize_portfolio`)
1. Slice last `months` months from cached 2y daily prices, `dropna` → returns matrix R (N stocks × T days).
2. **Factor-model covariance** (the key idea): SVD of R (non-demeaned — deliberate, documented choice),
   rank-K truncation `X = U_K S_K V_K^T` = denoised returns. Then
   `Sigma = (X_c X_c^T)/(T−1) + diag(Psi)`, where
   `Psi = clamp(diag(sample cov) − diag(low-rank cov), 0, None)` = per-stock variance the K factors
   don't explain (idiosyncratic). This is low-rank signal + diagonal noise: **full-rank, PD,
   well-conditioned even at N ≫ T** (the raw sample covariance there has rank T−1 and cond ~1e20).
   Truncating small singular values = Marchenko-Pastur denoising.
3. **Optimization**: maximize `mu'w − (γ/2) w'Σw` subject to `w ≥ 0, Σw = 1` (long-only, fully
   invested — enforced *structurally* by `w = softmax(z)`, so the outer problem is unconstrained
   gradient ascent on z with torch/Adam; **no Σ⁻¹ needed**, works with any Σ).
   - **Non-convex in z** (many local optima; a single init is arbitrary — e.g. one seed gave 6 holdings,
     another 46). So: `n_restarts` (default 10) seeded inits (`seed+r`, deterministic), keep the restart
     with the **highest final objective** (gradient ascent). `return_history=True` returns per-iteration
     objective curves for plotting.
   - `lr=0.05` default: lr=0.5 made ~3/10 restarts overshoot (curves drop from peak); 0.05 ascends
     monotonically to the same optimum.
4. Output: ticker→weight DataFrame (≥ min_weight, renormalized to sum 1).

### Knobs
- `gamma` — risk aversion. Higher → more diversified, lower vol, lower return. Low γ (1–5) = few
  concentrated names (dangerous: γ=1 lost −28% OOS).
- `K` — number of SVD factors. **Double duty: denoising AND concentration dial.** Larger K → smaller
  Psi → weaker ridge → fewer, more concentrated holdings. Too large re-absorbs MP noise (Σ → raw sample).
  OOS denoising was best at small K (k=3–5 min-variance Sharpe ~6.7 vs 2.0 raw); **K=25 is the current
  operating sweet spot** chosen for a usable, concentrated-but-diversified book in the 3m/500-stock window.
- `months`, `top_n`, `lr`, `n_restarts`, `min_weight`.

## Hard constraint: LONG-ONLY (no shorting)
The trading app doesn't allow short positions. Everything production-facing is the softmax simplex
(w ≥ 0, Σw = 1). A gross-exposure long/short variant (`‖w‖₁ = 1` via `w = z/‖z‖₁`) was explored in
`portfolio_short_and_long_stocks.ipynb` but is not the focus.

## Empirical findings (honest, from the notebooks)
- PCA on **prices**: PC1 = 98.3% = the 2y trend (useless). PCA on **returns**: PC1 ~84% = market factor;
  nearest-neighbours in PCA space recover sector pairs (JPM~BAC, WMT~COST, AMAT~LRCX).
- **OOS harness** (`portfolio_oos_test`, 6mo train / 3mo test, ~452 stocks): 1/N equal-weight beats every
  μ-using mean-variance portfolio on risk-adjusted return (Sharpe ~2.5; γ-sweep all worse, γ=1 → −28%).
  **Chasing the noisy μ̂ loses out-of-sample — the bottleneck is μ̂, not the optimizer.**
- Σ-only constructions (min-variance, risk parity 1/σ, factor-model Σ) beat 1/N on Sharpe by taking
  ~half the vol for similar return. The factor-model Σ gave min-variance Sharpe ~6.7 vs ~2.0 raw Σ.
- Sparsity lessons: **L1 is a no-op on the simplex** (‖w‖₁ ≡ Σw ≡ 1); concentration reward λ‖w‖² works
  but OOS profit is non-monotonic in λ; top-L selection loses to 1/N at every L. Conclusion: **use K as
  the sparsity dial** — no extra machinery.
- Max-ratio (tangency μ'w/w'Σw) + jitter (Σ+λI = ridge) explored in `max_ratio_portfolio.ipynb`;
  closed-form Σ⁻¹μ overfits even jittered; long-only softmax ratio is the robust version.

## CLI (click)
- `stonks fetch --tickers AAPL,MSFT --period 1y` / `--top-n 40` — fetch N×T matrix.
- `stonks portfolio --top-n 500 --months 3 --gamma 50 -k 25 --n-restarts 10 --lr 0.05`
  → prints recommended holdings (ticker, weight, company name), weights sum to 1.
  `--force-refresh` re-downloads prices (cache may be a day stale).

## Layout
- `stonks/`: config.py (Settings), universe.py, download.py (get_prices), returns.py, portfolio.py
  (optimize_portfolio), cli.py. Exports: get_prices, to_returns, optimize_portfolio, load_universe, Settings.
- `notebooks/`: simple_PCA, PCA_on_returns, mean_variance_portfolio, portfolio_short_window
  (thin wrapper over optimize_portfolio; restarts + training-curve plots), portfolio_oos_test,
  portfolio_factor_model, portfolio_short_and_long_stocks, max_ratio_portfolio.
- `tests/`: offline, synthetic data — no network.

## Known limitations / next steps
- Survivorship bias (today's listings only). Single OOS window → **walk-forward evaluation** is the
  obvious next step. μ̂ still unused productively (Black-Litterman / shrinkage would be principled).
- yfinance unofficial + rate-limited; intraday limited to last ~60 days.
- In-sample recommendations only; validate before believing any Sharpe.
