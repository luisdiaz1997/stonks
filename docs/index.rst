Welcome to stonks' documentation!
=================================

stonks is a Python package of simple, exploratory algorithms for stocks and
investments: a market-data wrapper, a factor-model covariance estimator, and a
long-only mean-variance portfolio optimizer with a CLI.

.. note::
   This project is licensed under the MIT License. **Not financial advice.**

Installation
------------

Install from PyPI (distribution name ``stonker``, import name ``stonks``):

.. code-block:: bash

   pip install stonker

Or install from source:

.. code-block:: bash

   git clone https://github.com/luisdiaz1997/stonks.git
   cd stonks
   pip install -e .

Quick Start
-----------

.. code-block:: python

   from stonks import get_prices, to_returns, optimize_portfolio

   # N x T matrix of adjusted close prices (rows = tickers, cols = dates)
   prices = get_prices(top_n=500, period="2y", interval="1d", field="close")

   # Daily simple returns, N x (T-1)
   returns = to_returns(prices)

   # Long-only mean-variance portfolio with a factor-model covariance
   weights = optimize_portfolio(top_n=500, months=3, gamma=50.0, K=25)

Or from the command line:

.. code-block:: bash

   stonks fetch --tickers AAPL,MSFT --period 1y
   stonks portfolio --top-n 500 --months 3 --gamma 50 -k 25

Mathematical Formulation
-------------------------

Model
~~~~~

Daily (simple) returns of :math:`N` stocks over :math:`T` days follow a
:math:`K`-factor model

.. math::

   r = Bz + \epsilon, \qquad z \sim \mathcal{N}(0, I_K), \qquad
   \epsilon \sim \mathcal{N}(0, D),

with :math:`B \in \mathbb{R}^{N \times K}` the factor loadings and
:math:`D = \operatorname{diag}(\sigma_1^2, \dots, \sigma_N^2)` the
**non-isotropic** independent per-stock idiosyncratic variances. The latent
state :math:`z` is unknown; we observe each stock's sample mean return
:math:`y_i` over the training window, which averages the noise by :math:`T`:

.. math::

   y = Bz + \bar\epsilon, \qquad \bar\epsilon \sim \mathcal{N}(0, D/T).

Conjugate Posterior
~~~~~~~~~~~~~~~~~~~

Everything is linear-Gaussian, so the posterior :math:`p(z \mid y)` is exactly
Gaussian (closed form, no VI):

.. math::

   S_z^{-1} = I_K + T\, B^\top D^{-1} B, \qquad
   m_z = T\, S_z\, B^\top D^{-1} y.

The update is :math:`K \times K` with :math:`K \ll N` — the whole Bayesian
inference happens in the small factor space. Pushing forward through the
linear map :math:`\mu = Bz`:

.. math::

   m_\mu = B\, m_z, \qquad S_\mu = B\, S_z\, B^\top,

where :math:`S_\mu` has rank :math:`\le K` **by construction** — the Bayesian
update cannot destroy the low-rank-plus-diagonal structure. The law of total
variance then gives the predictive covariance of returns:

.. math::

   \operatorname{Cov}(r \mid y) = D + B\, S_z\, B^\top .

Portfolio Objective
~~~~~~~~~~~~~~~~~~~

The **principal objective** is the long-only mean-variance problem: given
expected returns :math:`\mu` and a covariance :math:`\Sigma` of the daily
returns :math:`r`,

.. math::

   \boxed{\;\max_{w}\; r^\top w - \frac{\gamma}{2}\, w^\top \Sigma w
   \qquad \text{s.t.} \qquad w \ge 0,\;\; \mathbf{1}^\top w = 1.\;}

The first term is expected portfolio return, the second penalizes portfolio
variance with risk aversion :math:`\gamma`. It is solved via the softmax
parametrization :math:`w = \mathrm{softmax}(z_w)`, which enforces both
constraints structurally, so the outer problem is unconstrained gradient ascent
(needing only forward products :math:`w^\top \Sigma w`, never
:math:`\Sigma^{-1}`).

Instantiating :math:`\mu` and :math:`\Sigma` with the Bayesian posterior
(:math:`m_\mu = B m_z`, :math:`D + B S_z B^\top`) gives the objective the
model actually optimizes:

.. math::

   U(w) = w^\top B\, m_z - \frac{\gamma}{2}\, w^\top
   \left(D + B\, S_z\, B^\top\right) w.

See :doc:`bayesian_factor_model` for the full derivation, the prior/estimate
interpolation limits, and how the estimator below relates to it.

Practical Estimation
~~~~~~~~~~~~~~~~~~~~

The code estimates :math:`B` and :math:`D` from the training window by a
rank-:math:`K` SVD :math:`R = U S V^\top` (Marchenko-Pastur denoising):
:math:`X = U_K S_K V_K^\top` gives the low-rank signal covariance
:math:`\hat C_K = \tfrac{1}{T-1} X_c X_c^\top`, and the idiosyncratic diagonal
is **what the factors fail to explain**:

.. math::

   \Sigma = \hat C_K + \operatorname{diag}(\Psi), \qquad
   \Psi = \operatorname{clip}\!\big(\operatorname{diag}(\hat C) -
   \operatorname{diag}(\hat C_K),\ 0,\ \infty\big),

which is full-rank and positive-definite even when :math:`N \gg T` (where the
sample covariance is singular). The conjugate posterior update above is the
natural closed-form upgrade of the current plug-in estimator.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   bayesian_factor_model
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
