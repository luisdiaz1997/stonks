# The Bayesian factor model: theory and practice

This page derives the model behind `stonks.optimize_portfolio`: a linear-Gaussian
factor model with a conjugate Bayesian treatment of the factor posterior, and how
the *practical* estimator in the code (non-isotropic idiosyncratic noise via
$\operatorname{diag}(C_{\text{full}} - C_{\text{approx}})$) relates to it.

## Setup

Returns follow a $K$-factor model

$$
r = Bz + \epsilon, \qquad z \sim \mathcal N(0, I_K), \qquad \epsilon \sim \mathcal N(0, D),
$$

with $r \in \mathbb R^N$ the (mean) returns of $N$ stocks, $B \in \mathbb R^{N\times K}$
the factor loadings, $z$ the latent factor state, and $D = \operatorname{diag}(\sigma_i^2)$
the **independent per-stock idiosyncratic variances**.

We observe, over a $T$-period training window, the sample mean return of each
stock, $y_i$. Averaging $T$ i.i.d. draws divides the noise variance by $T$, so

$$
y = Bz + \bar\epsilon, \qquad \bar\epsilon \sim \mathcal N(0, D/T).
$$

$B$ and $D$ are treated as known (already fitted); $z$ is unknown and inferred.

## Step 1 — conjugate posterior on the factors

$z$ and $y$ are jointly Gaussian (linear map of a Gaussian plus independent
Gaussian noise), so the posterior $p(z \mid y)$ is exactly Gaussian — closed
form, no variational inference needed:

$$
S_z^{-1} = I_K + T\, B^\top D^{-1} B, \qquad
m_z = T\, S_z\, B^\top D^{-1} y .
$$

Note the computational structure: $S_z^{-1}$ is $K \times K$ with $K \ll N$, so
the whole Bayesian update happens in the **small** factor space — this is
precisely why factor models are nice to be Bayesian about.

## Step 2 — posterior on the expected returns $\mu = Bz$

$\mu$ is a *linear* map of $z$, so its posterior is Gaussian too:

$$
m_\mu = \mathbb E[\mu \mid y] = B\, m_z, \qquad
S_\mu = \operatorname{Cov}(\mu \mid y) = B\, S_z\, B^\top .
$$

$S_\mu$ is **rank $\le K$ by construction** (an $N \times K$ matrix
sandwiching a $K\times K$ one) — the Bayesian update cannot destroy the
low-rank-plus-diagonal structure, which is exactly what we want.

## Step 3 — predictive covariance of returns

By the law of total variance,

$$
\operatorname{Cov}(r \mid y)
= \underbrace{\mathbb E_{z\mid y}\!\left[\operatorname{Cov}(r \mid z)\right]}_{=\,D}
+ \underbrace{\operatorname{Cov}_{z\mid y}\!\left(\mathbb E[r \mid z]\right)}_{=\,B S_z B^\top}
= D + B S_z B^\top ,
$$

where $\operatorname{Cov}(r \mid z) = D$ because, conditional on $z$, the only
remaining randomness is the idiosyncratic noise $\epsilon$ (independent of $z$),
so the outer expectation leaves it unchanged.

## Step 4 — the portfolio objective

Plugging the posterior mean and predictive covariance into mean-variance:

$$
U(w) = w^\top B\, m_z - \frac{\gamma}{2}\, w^\top \left(D + B S_z B^\top\right) w ,
$$

solved long-only ($w \ge 0$, $\mathbf 1^\top w = 1$) via the softmax
parametrization $w = \mathrm{softmax}(z_w)$ with (multi-restart) gradient
ascent. The return term is nonzero — the data $y$ pulled $m_z$ away from the
zero prior mean.

### The two limits (sanity checks)

Compare with the point-estimate plug-in covariance $\hat\Sigma = BB^\top + D$
(up to the $1/(T{-}1)$ normalization used in the code):

- **Lots of data** ($T$ large): $S_z \to$ small, so $B S_z B^\top$ shrinks — the
  model becomes *confident*, and the objective approaches the point-estimate
  factor model with the posterior mean $B m_z$ as its return forecast.
- **Little data** ($T$ small): $S_z \to I_K$ (falls back to the prior), so
  $B S_z B^\top \to BB^\top$ — recovering exactly the plug-in covariance.

So the Bayesian version **smoothly interpolates between "trust the prior" and
"trust the estimate"** as a function of how much data you have — a tradeoff
that is invisible in the point-estimate formulation.

## In theory vs in practice: the noise covariance $D$

**In theory (textbook),** one often writes isotropic noise $\epsilon \sim \mathcal N(0, \sigma^2 I_N)$.
That assumption is *limited*: it says every stock has the same idiosyncratic
variance, which is wildly false — a volatile semiconductor name and a staple
consumer stock have different independent risk by an order of magnitude. To
model **independent per-stock variance** we need the non-isotropic diagonal

$$
D = \operatorname{diag}(\sigma_1^2, \dots, \sigma_N^2).
$$

Conjugacy survives: $D$ appears only through $D^{-1}$, and the posterior
formulas above are unchanged — each stock simply gets its own precision
$T/\sigma_i^2$ in the factor update.

**In practice (this repo),** $B$ and $D$ are estimated from the training window
by a rank-$K$ SVD of the returns matrix $R$ (see
`stonks.portfolio._factor_covariance`):

1. $R = U S V^\top$, keep the top $K$ factors: $X = U_K S_K V_K^\top$
   (the Eckart–Young optimal rank-$K$ approximation — Marchenko–Pastur
   denoising: the discarded small singular values are indistinguishable from
   noise when $N/T$ is not small).
2. The low-rank signal covariance is $\hat C_K = \tfrac{1}{T-1} X_c X_c^\top$
   (this is the "$BB^\top$" term, loadings $B = U_K S_K / \sqrt{T-1}$).
3. The **idiosyncratic diagonal is what the factors fail to explain**:

$$
D = \operatorname{diag}(\Psi), \qquad
\Psi = \operatorname{clip}\!\Big(\operatorname{diag}\big(\hat C\big) - \operatorname{diag}\big(\hat C_K\big),\ 0,\ \infty\Big),
$$

   i.e. the full sample covariance's diagonal minus the low-rank approximation's
   diagonal, **clamped at zero**. The clamp guards against the (non-demeaned)
   SVD occasionally over-explaining a stock's variance and producing a negative
   residual; with a demeaned SVD the projection is orthogonal and the residual
   is non-negative automatically.

The final covariance used by the optimizer is

$$
\Sigma = \hat C_K + \operatorname{diag}(\Psi)
\qquad\text{(low-rank signal + diagonal idiosyncratic, full-rank and PD even at } N \gg T\text{)}.
$$

## Status in the code

The current pipeline is the **plug-in** version: $\hat\mu$ = row means of the
denoised returns, $\Sigma = \hat C_K + \operatorname{diag}(\Psi)$, long-only
softmax optimization with $n$ deterministic restarts keeping the best final
objective. The conjugate update of Steps 1–4 (replacing $\hat\mu$ by $B m_z$ and
$\hat C_K$ by $B S_z B^\top$) is the natural next upgrade and is closed-form:
it only needs the $K \times K$ solve defining $S_z$.

One deliberate simplification: $D$ is treated as *known* at its point estimate.
Making $D$ random (inverse-Gamma / Wishart-type prior) is the next natural
extension, but it breaks conjugacy and requires VI or a more involved closed
form — out of scope for now.

*Not financial advice.*
