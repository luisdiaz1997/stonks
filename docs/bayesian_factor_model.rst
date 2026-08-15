The Bayesian Factor Model: Theory and Practice
==============================================

This page derives the model behind :func:`stonks.optimize_portfolio` in full: a
linear-Gaussian factor model with a conjugate Bayesian treatment of the factor
posterior, and how the *practical* estimator in the code (non-isotropic
idiosyncratic noise via :math:`\operatorname{diag}(C_{\text{full}} -
C_{\text{approx}})`) relates to it.

Setup: what we're conditioning on
---------------------------------

Same generative model as the :doc:`index`, but now :math:`z` is genuinely
unknown and we observe data to learn about it. Let :math:`y_i` be the sample
mean return of stock :math:`i` over the :math:`T`-period training window:

.. math::

   y = Bz + \bar\epsilon, \qquad z \sim \mathcal{N}(0, I_K), \qquad
   \bar\epsilon \sim \mathcal{N}(0, D/T).

The :math:`/T` is because :math:`y` is a sample mean of :math:`T` draws of
:math:`\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)` — averaging :math:`T`
i.i.d. draws divides the variance by :math:`T`. :math:`B` and :math:`D` are
the already-fitted factor loadings / idiosyncratic variances, treated as known
(fixed) here.

Step 1 — this is standard Bayesian linear regression
-----------------------------------------------------

:math:`z` and :math:`y` are jointly Gaussian (a linear transform of a Gaussian
plus independent Gaussian noise), so :math:`p(z \mid y)` is exactly Gaussian —
closed form, no VI needed. The conjugate result:

.. math::

   S_z^{-1} = I_K + T\, B^\top D^{-1} B, \qquad
   m_z = T\, S_z\, B^\top D^{-1} y.

:math:`S_z^{-1}` is :math:`K \times K` — cheap to invert since :math:`K \ll N`.
This is exactly why factor models are nice computationally: the whole Bayesian
update happens in the small :math:`K`-dimensional space, not the big
:math:`N`-dimensional one.

Step 2 — push the posterior on z forward to the expected returns
------------------------------------------------------------------

Since :math:`\mu = Bz` is a linear map of :math:`z`:

.. math::

   m_\mu = \E[\mu \mid y] = B\, m_z, \qquad
   S_\mu = \operatorname{Cov}(\mu \mid y) = B\, S_z\, B^\top .

Note :math:`S_\mu` is automatically **rank** :math:`\le K` — it is :math:`B`
(an :math:`N \times K` matrix) sandwiching a :math:`K \times K` matrix. Unlike
a Gaussian-process version where the posterior over :math:`\mu` would be
generically full-rank and dense, building the model this way means the
low-rank-plus-diagonal structure cannot be wrecked by the update — it is
low-rank by construction.

Step 3 — total covariance of :math:`r` given the data
-----------------------------------------------------

By the law of total variance,

.. math::

   \operatorname{Cov}(r \mid y)
   = \underbrace{\E_{z\mid y}\!\left[\operatorname{Cov}(r \mid z)\right]}_{=\,D}
   + \underbrace{\operatorname{Cov}_{z\mid y}\!\left(\E[r \mid z]\right)}_{=\,B S_z B^\top}
   = D + B\, S_z\, B^\top ,

using :math:`\operatorname{Cov}(r \mid z) = D`: conditional on :math:`z` the
only remaining randomness is the idiosyncratic noise :math:`\epsilon`,
independent of :math:`z`, so it doesn't depend on which value of :math:`z` we
condition on and the outer expectation does nothing to it.

Step 4 — plug into the objective
--------------------------------

.. math::

   U(w) = w^\top B\, m_z - \frac{\gamma}{2}\, w^\top
   \left(D + B\, S_z\, B^\top\right) w .

The first term is nonzero — :math:`m_z \ne 0` because the data :math:`y`
pulled the posterior away from the zero prior mean. And notice the structure:
this is **not** the same as the point-estimate plug-in
:math:`\hat\Sigma = BB^\top + D`. The :math:`BB^\top` term has been replaced by
:math:`B S_z B^\top` — your **certainty about** :math:`z` (the inverse of
:math:`S_z`) directly controls how much of the raw :math:`BB^\top` risk
survives into the objective.

The two limits (sanity checks)
------------------------------

- **Lots of data** (:math:`T` large relative to prior uncertainty):
  :math:`S_z \to` small, :math:`B S_z B^\top` shrinks — the model becomes
  *confident*, approaching the point-estimate factor model with :math:`B m_z`
  as the return forecast.
- **Little data** (:math:`T` small): :math:`S_z \to I_K` (falls back to the
  prior), so :math:`B S_z B^\top \to BB^\top`, recovering exactly the plug-in
  factor covariance.

So the Bayesian version **smoothly interpolates between "trust the prior" and
"trust the estimate"** as a function of how much data you have — a tradeoff
that is invisible in the point-estimate formulation.

In theory vs in practice: the noise covariance :math:`D`
--------------------------------------------------------

**In theory** (textbook), one often writes isotropic noise
:math:`\epsilon \sim \mathcal{N}(0, \sigma^2 I_N)`. That assumption is
*limited*: it says every stock has the same idiosyncratic variance, which is
wildly false — a volatile semiconductor name and a staple consumer stock
differ by an order of magnitude in independent risk. To model **independent
per-stock variance** we need the non-isotropic diagonal

.. math::

   D = \operatorname{diag}(\sigma_1^2, \dots, \sigma_N^2).

Conjugacy survives: :math:`D` enters the posterior only through
:math:`D^{-1}`, and the formulas above are unchanged — each stock simply gets
its own precision :math:`T / \sigma_i^2` in the factor update.

**In practice** (this repo), :math:`B` and :math:`D` are estimated from the
training window by a rank-:math:`K` SVD of the returns matrix (see
``stonks.portfolio._factor_covariance``):

1. :math:`R = U S V^\top`; keep the top :math:`K` factors:
   :math:`X = U_K S_K V_K^\top` — the Eckart-Young optimal rank-:math:`K`
   approximation. Truncation drops the small singular values that
   Marchenko-Pastur random-matrix theory says are indistinguishable from pure
   noise when :math:`q = N/T` is not small.
2. The low-rank signal covariance is
   :math:`\hat C_K = \tfrac{1}{T-1} X_c X_c^\top` (the ":math:`BB^\top`" term,
   loadings :math:`B = U_K S_K / \sqrt{T-1}`).
3. The **idiosyncratic diagonal is what the factors fail to explain**:

   .. math::

      D = \operatorname{diag}(\Psi), \qquad
      \Psi = \operatorname{clip}\!\big(\operatorname{diag}(\hat C) -
      \operatorname{diag}(\hat C_K),\ 0,\ \infty\big),

   i.e. the full sample covariance's diagonal minus the low-rank
   approximation's diagonal, **clamped at zero**. The clamp guards against the
   (non-demeaned) SVD occasionally over-explaining a stock's variance and
   producing a negative residual; with a demeaned SVD the projection is
   orthogonal and the residual is non-negative automatically.

The final covariance used by the optimizer is

.. math::

   \Sigma = \hat C_K + \operatorname{diag}(\Psi)

(low-rank signal + diagonal idiosyncratic), which is **full-rank and
positive-definite even at** :math:`N \gg T`, where the raw sample covariance
has rank :math:`T-1` and condition number :math:`\sim 10^{20}`.

Status in the code
------------------

The current pipeline is the **plug-in** version: :math:`\hat\mu` is the row
means of the denoised returns and :math:`\Sigma = \hat C_K +
\operatorname{diag}(\Psi)`, optimized long-only with the softmax parametrization
and :math:`n` deterministic restarts keeping the best final objective (the
problem is non-convex in the softmax parameters). The conjugate update of
Steps 1-4 — replacing :math:`\hat\mu` by :math:`B m_z` and :math:`\hat C_K` by
:math:`B S_z B^\top` — is the natural next upgrade and remains closed-form: it
only needs the :math:`K \times K` solve defining :math:`S_z`.

One deliberate simplification: :math:`D` is treated as *known* at its point
estimate. Making :math:`D` itself random (an inverse-Gamma / Wishart-type
prior) is the next natural extension, but it breaks conjugacy and requires
variational inference or a more involved closed form — out of scope for now.

*Not financial advice.*
