"""Offline tests for portfolio helpers (synthetic data, no network)."""

import numpy as np
import pandas as pd

from stonks.portfolio import _factor_covariance, _optimize_long_only


def test_factor_covariance_shape_and_pd() -> None:
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.normal(size=(20, 60)))  # 20 stocks, 60 days
    mu, Sigma = _factor_covariance(R, K=5)
    assert mu.shape == (20,)
    assert Sigma.shape == (20, 20)
    assert np.allclose(Sigma, Sigma.T)               # symmetric
    assert np.linalg.matrix_rank(Sigma) == 20         # full rank (signal + diagonal)
    assert np.linalg.eigvalsh(Sigma).min() > 0        # positive-definite


def test_optimize_long_only_feasible() -> None:
    rng = np.random.default_rng(1)
    n = 15
    mu = rng.normal(0.001, 0.0005, n)
    Sigma = np.cov(rng.normal(size=(n, 100)))
    w, hist = _optimize_long_only(mu, Sigma, gamma=20.0)
    assert w.shape == (n,)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)        # fully invested
    assert (w >= -1e-9).all()                          # non-negative (long-only)
    assert len(hist) == 1 and hist[0].ndim == 1        # one history per restart


def test_higher_gamma_diversifies() -> None:
    """More risk aversion -> weights spread out (higher entropy / more holdings)."""
    rng = np.random.default_rng(2)
    n = 30
    mu = rng.normal(0.001, 0.0005, n)
    Sigma = np.cov(rng.normal(size=(n, 100)))
    w_low, _ = _optimize_long_only(mu, Sigma, gamma=2.0)
    w_high, _ = _optimize_long_only(mu, Sigma, gamma=200.0)
    holdings_low = int((w_low > 1e-3).sum())
    holdings_high = int((w_high > 1e-3).sum())
    assert holdings_high >= holdings_low               # higher gamma -> at least as many holdings


def test_multi_restart_best_of_n() -> None:
    """n_restarts runs -> n histories, best final objective kept, deterministic."""
    rng = np.random.default_rng(3)
    n = 25
    mu = rng.normal(0.001, 0.0005, n)
    Sigma = np.cov(rng.normal(size=(n, 100)))
    w_a, hist_a = _optimize_long_only(mu, Sigma, gamma=20.0, n_restarts=3, iters=500)
    w_b, hist_b = _optimize_long_only(mu, Sigma, gamma=20.0, n_restarts=3, iters=500)
    assert len(hist_a) == 3                            # one history per restart
    assert np.allclose(w_a, w_b)                       # deterministic (seeded restarts)
    assert np.isclose(w_a.sum(), 1.0, atol=1e-6)
    # the returned weights must achieve the best final objective across restarts
    finals = [h[-1] for h in hist_a]
    obj = float(mu @ w_a - 0.5 * 20.0 * w_a @ Sigma @ w_a)
    assert np.isclose(obj, max(finals), rtol=1e-9)
