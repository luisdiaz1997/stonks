"""Long-only mean-variance portfolio optimization with a factor-model covariance.

The pipeline from ``notebooks/portfolio_short_window``:

1. fetch prices, slice the last ``months`` months, compute daily returns;
2. denoise the returns matrix with a rank-``K`` SVD;
3. build a factor-model covariance ``Sigma = low-rank signal + diag(idiosyncratic)``
   — full-rank and well-conditioned even when ``N > T``;
4. solve the long-only mean-variance problem
   ``max mu'w - (gamma/2) w'Sigma w`` with ``w = softmax(z)`` (so ``w >= 0``,
   ``sum(w) = 1``).

``K`` is both the denoising knob and the concentration dial: larger ``K`` shrinks
the idiosyncratic ``Psi`` (weaker ridge), so the dense portfolio naturally holds
fewer, more concentrated names.

No financial advice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .download import get_prices
from .returns import to_returns


def _factor_covariance(returns: pd.DataFrame, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Factor-model (mu, Sigma) from a returns frame via a rank-K SVD.

    ``Sigma = (low-rank signal covariance) + diag(idiosyncratic variance)``,
    where the idiosyncratic variance is the per-stock variance the K factors
    fail to explain (clamped to >= 0).
    """
    R = returns.to_numpy(dtype=float)
    N, T = R.shape
    K = max(1, min(K, N, T))
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    X = (U[:, :K] * S[:K]) @ Vt[:K]          # rank-K denoised returns
    mu = X.mean(axis=1)
    Xc = X - mu[:, None]
    Rc = R - R.mean(axis=1, keepdims=True)
    tm1 = max(T - 1, 1)
    Psi = np.clip((Rc ** 2).sum(axis=1) / tm1 - (Xc ** 2).sum(axis=1) / tm1, 0.0, None)
    Sigma = (Xc @ Xc.T) / tm1 + np.diag(Psi)
    return mu, Sigma


def _optimize_long_only(
    mu: np.ndarray,
    Sigma: np.ndarray,
    gamma: float,
    iters: int = 8000,
    lr: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Maximize ``mu'w - (gamma/2) w'Sigma w`` with ``w = softmax(z)``.

    The softmax keeps ``w >= 0`` and ``sum(w) = 1`` (long-only, fully invested)
    by construction, so the outer problem is unconstrained ascent on ``z``.
    """
    torch.manual_seed(seed)
    N = len(mu)
    mu_t = torch.tensor(mu, dtype=torch.float64)
    Sig_t = torch.tensor(Sigma, dtype=torch.float64)
    z = torch.randn(N, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        w = torch.softmax(z, dim=0)
        loss = -(mu_t @ w - 0.5 * gamma * w @ Sig_t @ w)
        loss.backward()
        opt.step()
    return torch.softmax(z, dim=0).detach().numpy()


def optimize_portfolio(
    top_n: int = 500,
    months: int = 3,
    interval: str = "1d",
    gamma: float = 20.0,
    K: int = 10,
    period: str = "2y",
    field: str = "close",
    min_weight: float = 1e-3,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build a long-only mean-variance portfolio using a factor-model covariance.

    Args:
        top_n: number of most-liquid common stocks.
        months: window length (sliced from the cached ``period`` prices).
        interval: bar resolution.
        gamma: risk aversion (higher -> more diversified / lower vol).
        K: number of SVD factors; also the concentration dial.
        period: cache source to slice ``months`` from (default ``"2y"``).
        field: OHLCV field used to compute returns.
        min_weight: drop weights below this, then renormalize to sum to 1.
        force_refresh: bypass the price cache.

    Returns:
        DataFrame indexed by ticker with a ``weight`` column, sorted descending;
        weights are non-negative and sum to 1.
    """
    prices = get_prices(
        top_n=top_n, period=period, interval=interval,
        field=field, force_refresh=force_refresh,
    )
    last = pd.Timestamp(prices.columns[-1])
    prices_win = prices.loc[:, prices.columns >= last - pd.DateOffset(months=months)]
    returns = to_returns(prices_win).dropna()

    mu, Sigma = _factor_covariance(returns, K)
    w = _optimize_long_only(mu, Sigma, gamma)

    weights = pd.Series(w, index=returns.index)
    weights = weights[weights >= min_weight]
    weights = (weights / weights.sum()).rename("weight")
    return weights.sort_values(ascending=False).to_frame()
