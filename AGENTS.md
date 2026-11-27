# stonks — project summary & current state (handoff)

Exploratory stock/investment algorithms. Python package + notebooks + CLI.
**Not financial advice.** Repo: github.com/luisdiaz1997/stonks.
- **Published on PyPI as `stonker`** (import name `stonks`): https://pypi.org/project/stonker/

## Stack & environment
- Python >=3.11; deps: torch, numpy, pandas, yfinance, click, pyarrow, tqdm, robin_stocks.
- Conda env `stonks` (Python 3.13) has `pip install -e ".[notebooks]"` (jupyter) + pytest.
  Binary: `/Users/luisfcd/opt/miniconda3/envs/stonks/bin/stonks`. Editable install — code changes are live.
- Jupyter runs on port 8888, notebook-dir = repo root.
- 50 offline tests (`pytest`): universe parsing/filtering, returns, factor covariance, optimizer,
  Robinhood allocation + take-profit sells, CLI (mocked broker — no network).

## Data layer
- `get_prices(top_n, tickers=None, period, interval, field)` → **N×T matrix** (rows=tickers, cols=dates),
  adjusted close by default. Cached as parquet under `data/` (gitignored, regenerable).
- Universe: NASDAQ Trader symbol directory (`nasdaqlisted.txt`+`otherlisted.txt`), filtered to common
  stocks (drops ETFs/test issues/preferreds/warrants/rights/funds/SPACs) → ~4000 names, ranked by
  average dollar volume → top_n. NOT Russell 1000 (not reliably free); survivorship-biased by construction.
- `to_returns(prices, log=False)` → daily simple returns `P_t/P_{t-1} − 1`, N×(T−1).
  Covariance/PCA is done on **returns, not prices** (prices are non-stationary levels; returns are the
  stationary increments — the object we model).

## Mathematical formulation

**Objects.** Prices $P\in\mathbb{R}^{N\times T}$ (adjusted close, N tickers × T days). Daily simple returns
$$R_{it} = \frac{P_{i,t}}{P_{i,t-1}} - 1 \;\in\; \mathbb{R}^{N\times(T-1)},$$
the stationary increments we model (never price levels).

**Factor model / denoising.** SVD $R = U S V^\top$; rank-K truncation
$$X = U_K S_K V_K^\top = \arg\min_{\mathrm{rank}(M)\le K} \|R - M\|_F \quad\text{(Eckart–Young)}.$$
(The code SVDs non-demeaned $R$ — a deliberate, documented choice.) With $\hat\mu = X\mathbf{1}/T$ (row means) and $X_c$ the row-centered $X$:
$$\hat C = \tfrac{1}{T-1}R_c R_c^\top \;(\text{sample cov, rank}\le\min(N,T{-}1)),\qquad
\hat C_K = \tfrac{1}{T-1}X_c X_c^\top \;(\text{rank-}K),$$
$$\Psi = \operatorname{clip}\!\big(\operatorname{diag}(\hat C) - \operatorname{diag}(\hat C_K),\, 0,\, \infty\big),\qquad
\boxed{\;\Sigma = \hat C_K + \operatorname{diag}(\Psi)\;}$$
i.e. low-rank signal + diagonal idiosyncratic noise (the $\Sigma = BB^\top + \Psi$ statistical factor model; equivalently probabilistic PCA). Full-rank and PD even at $N\gg T$, where $\hat C$ is singular (rank $T{-}1$, cond $\sim10^{20}$). Truncation rationale: Marchenko–Pastur — for a pure-noise correlation matrix with $q=N/T$, eigenvalues lie in $[(1-\sqrt q)^2,(1+\sqrt q)^2]$; small singular values are indistinguishable from noise. (Notebooks also scale eigenvector loadings by $\sqrt N$ so noise loadings are $\sim\mathcal N(0,1)$.)

**Optimization.** Long-only mean-variance:
$$\max_{w}\; \hat\mu^\top w - \tfrac{\gamma}{2}\, w^\top \Sigma w \quad \text{s.t.} \quad w \ge 0,\;\; \mathbf{1}^\top w = 1,$$
solved by the parametrization $w = \mathrm{softmax}(z)$, which satisfies both constraints **for every** $z$ ⇒ unconstrained gradient **ascent** on $z$ (Adam; only forward products $w^\top\Sigma w$, never $\Sigma^{-1}$). The objective in $z$ is **non-convex** (many local optima) ⇒ $n$ restarts from seeded inits, keep $w^{(r^*)}$ with $r^* = \arg\max_r f(w^{(r)})$ (deterministic: restart $r$ uses seed $+r$).

Reference closed forms (used in notebooks, not in production — they need $\Sigma^{-1}$):
unconstrained $w^* = \tfrac{1}{\gamma}\Sigma^{-1}\hat\mu$; budget-constrained
$w^* = \tfrac{1}{\gamma}\big(\Sigma^{-1}\hat\mu - \lambda \Sigma^{-1}\mathbf{1}\big)$,
$\lambda = \tfrac{\mathbf{1}^\top\Sigma^{-1}\hat\mu - \gamma}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}$;
tangency (max ratio $\hat\mu^\top w / w^\top\Sigma w$) direction $w \propto \Sigma^{-1}\hat\mu$.

**Sparsity geometry.** On the simplex $\|w\|_1 \equiv \mathbf{1}^\top w \equiv 1$ (constant ⇒ L1 penalty is a no-op). Concentration varies via the Herfindahl $H(w)=\sum_i w_i^2 \in [1/N, 1]$; $\operatorname{diag}(\Psi)$ acts as a weighted ridge on $w$, so larger $K$ (smaller $\Psi$) concentrates. Long/short variant (explored, unused): $\|w\|_1 = 1$ with signed $w = z/\|z\|_1$.

**Evaluation.** Portfolio daily return $r_{p,t} = w^\top R_t$; compounded profit $\prod_t(1+r_{p,t}) - 1$; annualization ×252 (trading days) for return, ×$\sqrt{252}$ for vol; Sharpe $= \bar r_p / \mathrm{std}(r_p)\cdot\sqrt{252}$.

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
     objective curves for plotting; `keep_restarts=True` also returns every restart's weights.
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
`portfolio_short_and_long_stocks.ipynb` but is not the focus. Selling (`stonks sell`) only ever exits a
long position; nothing goes short.

## Take-profit sells (`stonks/robinhood.py` → `stonks sell`)
- Rule: sell the **whole position** once its *total* return (price vs Robinhood `average_buy_price`)
  reaches `--take-profit` (default 8%). Not today's move — both percentages are printed for cross-checking
  against the app. Checked **once per run** (not continuously); the shipped timer runs every 15 minutes through
  the session, so a spike above the threshold between two runs is missed (it was hourly until 2026-11-27,
  when a position crossed +10% between two hourly checks and had to be sold by hand).
- `select_sells()` is the pure decision; `load_positions()` does one positions GET + one batched quotes GET
  (the library's `build_holdings()` silently drops positions that error — avoided). A failed read **raises**
  so a scheduled run can't mistake an outage for "nothing to sell".
- Skips (kept in the table with a reason): `bought today` (day trade), `open sell order`
  (`shares_held_for_sells`), `under $1.00`, `no cost basis`, `no quote`.
- Execution: `order_sell_fractional_by_quantity(ticker, quantity)` — exact share count, GFD, regular hours
  only (market closed → `--execute` places nothing). Same order loop as buys (`_submit_orders`): 2s
  throttle, stop at first failed/ambiguous order, rest reported `not_attempted`. Broker state makes reruns
  idempotent (a sold position is gone; a pending one has `shares_held_for_sells`).
- Dry run by default; `--execute` confirms; `--execute --yes` is the unattended mode. Submissions persisted to
  `data/last_robinhood_{orders,sells}.json`; `stonks orders` re-checks both.
- Server: `deploy/systemd/` timer every 15 minutes, 06:32–12:47 `America/Los_Angeles` (open is 06:30 PT; first
  run is at :32 because quotes can lag the opening seconds; two `OnCalendar` lines so nothing fires before the
  open), dry run until `--execute --yes` is added as a
  drop-in (`deploy/systemd/live/override.conf`). torch is imported lazily (inside `_optimize_long_only`) so
  `sell`/`orders` run without it (`pip install --no-deps` + pandas, robin_stocks, click, tqdm).
  `deploy/README.md` is the step-by-step deployment guide.
- **Backtests** (`notebooks/take_profit_*.py`; results and figures in `docs/take_profit.rst`): $1,000 a month
  into the optimizer's picks every other Monday, selling at +X%. γ=40 with the normal picks leads in both spans.
  Held accounts: 8–15% is a plateau (+36.9% to +42.5% over 2y; never selling +29.1%). That lead depends on the end
  date: over the 2y the +10% rule was behind never selling at 19 of 25 month-ends (by 20% at the end of May 2026)
  and went ahead in the last month; the second year alone ended level (+14.7% against +13.4%). 19% of positions are never
  sold (median −9.2% at the end). The rule's gains are short-term and realized every year, so after tax
  never selling can beat the +10% rule unless the leftover losers are sold too; the size of that depends on
  the investor's rates (`take_profit_taxes.py`). Flipping μ and selling everything every buy day are on the
  same page. Caveats: the 500-stock list is ranked with hindsight, one mostly-rising stretch of market, no
  trading costs.
- **Stop-loss on top of the +10% rule** (`notebooks/stop_loss_heatmap.py`, sells checked at the close): for γ=40
  no level from 1% to 50% beat having no stop over 2y (+40.1%): +20.5% at a 5% stop, +28.9% at 10%, +32.2% at
  20%, +39.2% at 30%. A stop did help the flipped picks, which are not run. Not on the docs page.
- **"Buy the dip" picks** (`dip4` on the backtest script: the usual objective on the three months before the last
  month, plus the flipped one on the last month, its covariance at a tenth of γ; that length and that tenth are
  `DIP_LAST` and `DIP_WEIGHT`, not varied yet).
  γ=40, held, the last row of `take_profit_heatmap.py`. Over 2y it is below the normal picks at every threshold
  from 1% to 50%: best +36.1% at 3% against +42.5% at 13%, +19.0% against +40.1% at the live +10%, never selling
  +22.1% against +29.1%, and behind at all 25 month-ends. The latest year alone favours it, but only with a tight
  sell: +18.3% against +14.7% at +10% and +22.1% against +13.5% at +8%, behind from 13% up. Concentrated (11–13
  names a buy, the top four a third of the money) and about 25% down in its last month. The figure on the docs
  page still has eight rows; the one in `outputs/` has nine.
- **One sum in from the start** (`notebooks/lump_sum.py`; the simulator's `lump` argument, which also
  brought `daily` values and a buy day with nothing to deploy being skipped): $24,000 in on 2024-11-04,
  everything sold and re-bought every other Monday. γ=40 normal: +214% with no sell inside the two weeks,
  +143% with +10%, and the row rises with the threshold, so never selling inside them is best; +34% at 6
  months, +149% at 12, worst fall −21%. γ=160 +57%, flipped +19%, dip +41%. The same 500 equal-weight made
  +62% and SPY +35%; one-year sums from 26 start days ran +32% to +149% and beat SPY in all 26, which is
  more likely the hindsight-ranked list at work than a real edge. Not on the docs page, and not a forecast.
- **K on one sum** (`notebooks/lump_sum_k.py`; `lump_sum_paths.py` draws every threshold's path for one
  setting): $24,000 in on 2024-11-04, everything sold and re-bought every other Monday, γ=40, no sell inside
  the two weeks: K=5 +99% (88 names a buy, worst fall −9%), K=10 +95%, K=15 +116%, K=20 +145%, K=25 +214%,
  K=30 +234%, K=35 +222%, K=40 +222% (19 names, −20%). At every K the sell inside the two weeks only costs,
  and the loosest thresholds come closest to none. At γ=160 the same account makes +56.5% with none and +20%
  to +56% across thresholds, below the equal-weight basket of the same 500 (+62%). The picks cache is now
  written as each recommendation arrives, `WORKERS` sets the pool size, and K travels with its task. Same
  caveats as the rest: not a forecast.
- **Rhythm on one sum** (`notebooks/lump_sum_cadence.py`; `_buy_days` now knows "weekly" and "bimonthly", and the
  weekly picks cache starts from the every-other-Monday one): γ=40, K=25, $24,000 in on 2024-11-04, everything
  sold and re-bought at every buy day, no sell inside the period: weekly +259%, every two weeks +214%, monthly
  +284%, every two months +303%. With a +10% sell inside the period: +184%, +143%, +103%, +56%, because the
  proceeds wait for the next re-buy; at +1%: +64%, +51%, +17%, +4%. Worst fall: −20%, −21%, −18%, −31%.
- **A four-month window, same account** (`lump_sum_cadence.py 2 40 25 4`): with no sell inside the period weekly
  +150%, every two weeks +167%, monthly +135%, every two months +268%, so below the three-month window in three
  of the four rhythms; with +10%: +104%, +104%, +41%, +37%. With a two-month window (`... 2 40 25 2`): +162%,
  +166%, +210%, +300%; with +10%: +121%, +93%, +84%, +81%. So of two, three and four months the usual three did
  best at every rhythm with no sell inside the period, and its worst falls were the shallowest. The window
  length travels with each task (`months`) and is in the picks cache's name (`_m2`, `_m4`).
- **Speed**: the backtest module caps each process's maths helper threads at one (set before numpy and torch
  load). Without it, workers running side by side each spun up their own team and took 60 s to minutes a job
  where a lone one takes 22; with it, eight side by side take about 35 s each. None of this is on the docs
  page, and none of it is a forecast.
- **Every setting travels with its task** (`_recommend(i, gamma, flip, dip, weight, last)`; a worker keeps none).
  macOS starts worker processes fresh, so
  before this a worker took γ and the flip from the command line's first arguments, not from what a script had
  set on the module: on a cache miss, `take_profit_heatmap.py` and the tax and path scripts would have computed
  picks at the wrong γ and unflipped, under the intended cache name. The cached picks were checked afterwards by
  rebuilding one buy day per setting with the settings passed in; they match.
- The job went live at 10% on 2026-11-24, and made its first live sale on 2026-11-25 (GKOS at +11.2%, the
  order filled). The hourly runs never failed to authenticate in their first three days.
  **Re-login is not unattended**: a fresh sign-in triggers an app-approval prompt (confirmed on the
  server), so a run that finds the session expired waits until systemd kills it (10 min) and sells nothing.

## Weekly cycle: sell everything on Monday, buy it back on Tuesday
- `stonks sell --everything [--execute] [--yes]` is the sell loop with the threshold removed. The skips stay
  (bought today, open sell order, under $1.00, no cost basis, no quote), so the three dust positions under a
  dollar never go. It passes `stop_on_failure=False`: a rejected order is reported and the run goes on to the
  next position, so one rejection cannot leave most of the account unsold (each position is still attempted
  once). The take-profit run keeps stopping at its first failure.
- Every live `stonks sell` run, take-profit or everything, puts its **filled** orders on `data/sell_ledger.json`
  (`record_sells`): a queued, unconfirmed or rejected order counts for nothing, a filled order with no reported
  amount for what the position was worth. `stonks rebuy --pick FILE [--execute] [--yes]` buys a recommendation
  saved by `stonks portfolio --save FILE` (refused when more than 12 h old) with `cash_to_rebuy()`: the ledger
  since the last re-buy plus what that re-buy left undeployed, capped at Robinhood's buying power
  (`record_buy`). So the cycle moves the same money and borrows nothing more; only Monday's proceeds would leave
  the +10% sales of the week piling up as cash and the invested amount shrinking every week. Money moved by
  hand is never re-bought.
- Server: `stonks-weekly-sell` Mon 12:50 PT, `stonks-weekly-pick` Tue 05:15 PT, `stonks-weekly-buy` Tue 06:36
  PT (all in `deploy/systemd/`). The pick runs from a **second
  install**, `/root/gitclones/stonks-opt`, with CPU torch and yfinance; the live install has neither. Measured
  there with a week-old price cache: 14 minutes, 316 MB of process memory, the box down to 71 MB free. Its unit
  caps it at 430 MB and makes it the first process the kernel would kill. It must never overlap the sell job's
  minutes, which is why it starts 75 minutes before the open.
- Status checks for more than ten orders poll a second apart, not half, to stay inside the rate limit.

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
- **Restart dispersion is ~zero** (`portfolio_hyperparam_sweep`, 30 restarts at lr=0.005, whole γ×K grid):
  every config converges to the same portfolio across restarts (the objective is strictly concave in w
  with the PD factor-model Σ), so sweep differences are γ/K/window, not seed luck.
- **Walk-forward** (`portfolio_walk_forward`: 6 rolling 3m-train / 1m-test windows over the cached 2y of
  top-500 returns, full γ×K grid, 5 restarts): the single-window sweep winner (K=40, γ=5, +17.8% in Sept)
  ranks only 5th of 30 by mean with a −28.0% worst month. The K=40 row dominates the table and γ acts as
  its tail-risk dial (γ=40: +4.2% mean / −1.8% worst / best Sharpe; γ=10: +5.2% / −14.8%). Anchor
  K=25/γ=40 is the most stable but lower mean; 1/N mean +1.05%.

## CLI (click)
- `stonks fetch --tickers AAPL,MSFT --period 1y` / `--top-n 40` — fetch N×T matrix.
- `stonks portfolio --top-n 500 --months 3 --gamma 50 -k 25 --n-restarts 10 --lr 0.05`
  → prints recommended holdings (ticker, weight, company name), weights sum to 1.
  `--force-refresh` re-downloads prices (cache may be a day stale).
- `stonks portfolio ... --execute [--amount 500]` — buy the recommendation in Robinhood (fractional market
  buys by dollars, 2s throttle, live confirmation; never sells).
- `stonks sell --take-profit 8 [--execute] [--yes]` — take-profit sells, dry run by default.
- `stonks orders` — live status of the most recent buy and sell submissions (read-only).

## Layout
- `stonks/`: config.py (Settings), universe.py, download.py (get_prices), returns.py, portfolio.py
  (optimize_portfolio), robinhood.py (login, buy/sell submission, select_sells), cli.py. Exports: get_prices,
  to_returns, optimize_portfolio, load_universe, Settings.
- `deploy/`: `README.md` (step-by-step deployment guide) and `systemd/` with `stonks-sell.{service,timer}`
  for the scheduled server run (dry run as shipped) and `live/override.conf`, the drop-in that turns it live.
- `notebooks/`: simple_PCA, PCA_on_returns, mean_variance_portfolio, portfolio_short_window
  (thin wrapper over optimize_portfolio; restarts + training-curve plots), portfolio_oos_test,
  portfolio_factor_model, portfolio_short_and_long_stocks, max_ratio_portfolio,
  portfolio_hyperparam_sweep (γ×K grid, restart statistics), portfolio_walk_forward (rolling validation),
  and the take-profit backtests as scripts: `take_profit_backtest.py` (the simulator; the others import it),
  `take_profit_heatmap.py`, `take_profit_paths.py`, `take_profit_taxes.py`, `stop_loss_heatmap.py` (the
  simulator's `stop` argument), `pick_trends.py` (the three ways of picking drawn on one day, for
  `docs/models.rst`), `lump_sum.py` (one sum in from the start, everything sold and re-bought every buy
  day; the simulator's `lump` argument), `lump_sum_paths.py` (every sell threshold's path for one setting)
  and `lump_sum_k.py` (the same for several K; `pick K` computes one K's picks with a single worker, so
  several K can run side by side) and `lump_sum_cadence.py` (weekly, every other Monday, monthly and every two
  months on one set of Monday picks, at any window length). They download their prices on first use; nothing is
  hard-coded to one machine.
- **Not on `master`, on purpose**: the intraday experiment (`stonks/intraday/`, `tests/test_intraday.py`,
  the `stonks intraday` CLI group and the README's intraday section) lives only on
  `experiment/intraday-backtest`, which is `master` plus that experiment and nothing else.
- `tests/`: offline, synthetic data — no network.
- `docs/`: Sphinx sources for Read the Docs (built from `master`). `api.rst` is the API reference;
  `take_profit.rst` is the backtest results page, with its figures in `docs/_static/`;
  `models.rst` is the three ways of picking (normal, flipped, dip), each drawn on one day at γ=20, 40 and 160
  in `docs/_static/picks_*.png` by `notebooks/pick_trends.py` (copy them over from `outputs/` after rerunning
  it; the day shown is then the latest close, so the page's text keeps no numbers from one day);
  `lump_sum.rst` is the one-sum page (which picks, K, rhythm, window length, other start days), with its figures
  `docs/_static/lump_sum_*.png` copied from `outputs/` after rerunning the `lump_sum_*.py` scripts; it states
  first that most of the gain is likely the hindsight-ranked list;
  `deployment.rst` includes `deploy/README.md` verbatim (through `myst-parser`, so the two cannot drift).

## Known limitations / next steps
- Survivorship bias (today's listings only). Walk-forward evaluation exists (`portfolio_walk_forward`),
  but its training windows overlap, the config is chosen with hindsight, and it covers a single regime.
  μ̂ still unused productively (Black-Litterman / shrinkage would be principled).
- yfinance unofficial + rate-limited; intraday limited to last ~60 days.
- In-sample recommendations only; validate before believing any Sharpe.
