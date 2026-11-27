Three ways of picking
=====================

The optimizer looks at the last three months of daily returns and gives every
stock an expected return and a risk. It then spreads the money over the stocks
so as to maximize

.. math::

   \mu^\top w - \frac{\gamma}{2}\, w^\top \Sigma w,
   \qquad w \ge 0, \quad \mathbf{1}^\top w = 1,

where :math:`\mu` is each stock's expected return, :math:`\Sigma` the
covariance from the factor model and :math:`w` the weights. The expected
return is the part that can be changed, and each change gives a different way
of picking:

- **Normal** leaves the formula as it is. It favors stocks that have been
  going up.
- **Flipped** negates :math:`\mu`. It favors stocks that have been going down.
- **Dip** adds a second, flipped objective that looks only at the last month.
  It favors stocks that were going up and have just fallen.

:math:`\gamma` is the same dial in all three. A low :math:`\gamma` makes a high
expected return worth a lot of risk, so the money goes to a few names. A high
:math:`\gamma` makes risk expensive, so it spreads over many names whose moves
offset each other.

The figures below show each way on one day, from closes up to the date in the
title, with K=25 and the 500 most liquid stocks as in the backtests. Each line
is one stock picked, followed from four months before that day and thick in
proportion to its weight; the black line is the picks' weighted average.
Beside each chart are the largest weights and what each of those stocks had
done over the three months the optimizer saw. The charts are here to show what
each way looks for. They are not recommendations.

Normal
------

The expected return of a stock is its mean daily return over the last three
months. A stock that rose has a high one, so the optimizer puts most of the
money in what has been going up. This is what ``stonks portfolio`` does.

.. figure:: /_static/picks_normal.png
   :width: 100%
   :target: /_static/picks_normal.png
   :alt: Three charts, one for each of gamma 20, 40 and 160, of the normal picks on 2026-11-24. Each line is one stock's return from four months before, with the first month shaded because the optimizer did not look at it. Most lines rise over the last three months and a few fall. The top four names carry 53, 45 and 35 percent of the money as gamma rises.

What the figure shows:

- **Most of what it picked rose over the three months, but not all.** The
  expected return is not the three-month gain itself. It is the gain left after
  the factor model has smoothed the returns, and a stock whose moves offset the
  others can earn a place through the risk term.
- **A higher γ spreads the money.** The top four names carry about half of it
  at γ=20 and γ=40 and about a third at γ=160. The picks as a whole had also
  risen less at γ=160, because the optimizer is paying more attention to risk.

Flipped
-------

Negating the expected return turns the optimizer around: a stock that fell has
the highest score. This is the simplest way to buy the dip. It asks only that a
stock ended the three months down, not that it was rising before, so it picks
a long decline as readily as a pullback.

.. figure:: /_static/picks_flipped.png
   :width: 100%
   :target: /_static/picks_flipped.png
   :alt: Three charts, one for each of gamma 20, 40 and 160, of the flipped picks on 2026-11-24. The lines end lower than they started over the last three months, most of them falling in the second half of the window. At gamma 160 some of the picks are up.

What the figure shows:

- **Every name picked at γ=20 and γ=40 ended the three months down.** Several
  had risen earlier in the window before falling. The flipped objective does not
  tell those apart from stocks that fell the whole time.
- **A higher γ makes it less strict.** At γ=160 some of the picks are up over
  the three months: with risk that expensive, the optimizer takes stocks that
  offset each other over stocks that merely fell the most.

Buy the dip
------------

Two objectives added together. The first is the normal one on the three months
before the last month. The second is the flipped one on the last month alone:

.. math::

   (\mu_{\text{rise}} - \mu_{\text{fall}})^\top w
   - \frac{\gamma}{2}\, w^\top
     \left(\Sigma_{\text{rise}} + 0.1\,\Sigma_{\text{fall}}\right) w

A stock that rose for three months scores well on the first and a stock that
fell in the last month scores well on the second, so what it likes best is a
stock that did both. The last month's covariance enters the risk penalty with
a tenth of γ. It looks back four months, where the other two look back three.

.. figure:: /_static/picks_dip.png
   :width: 100%
   :target: /_static/picks_dip.png
   :alt: Three charts, one for each of gamma 20, 40 and 160, of the buy-the-dip picks on 2026-11-24. Blue lines rise for most of the four months and fall after the dashed line that marks the last month. Orange lines fall the whole time, and there are more of them at higher gamma.

What the figure shows:

- **The blue lines rise for most of the chart and fall in the last month.**
  Those are the pullbacks it is built for.
- **The orange lines fell the whole time.** The last month's fall counts in
  full whether or not the stock rose before, so a stock in a long decline can
  still be picked. There are more of them at the higher levels of γ, where the
  risk term does more of the choosing.

How the three did
-----------------

The test on the :doc:`backtests page <take_profit>`: $1,000 a month into the
picks every other Monday for two years, positions held until the sell rule
fires, sold cash reused, valued on 2026-11-02. γ=40 for all three.

.. list-table::
   :header-rows: 1

   * - Picks
     - Never sell
     - Sell at +10%
     - Best threshold
   * - Normal
     - 29.1%
     - 40.1%
     - 42.5%, at 13%
   * - Flipped
     - 23.3%
     - 9.6%
     - 17.5%, at 49%
   * - Dip
     - 22.1%
     - 19.0%
     - 36.1%, at 3%

The normal picks led over the two years. The dip picks beat the flipped picks
at 49 of the 50 thresholds, but only the latest of the two years favored them
over the normal picks, and there only with a sell at 3% to 11%. The heatmap on
the backtests page predates the dip picks and covers the normal and flipped
picks only; its caveats apply to all three.

Reproducing the figures
-----------------------

.. code-block:: bash

   python notebooks/pick_trends.py

writes the three figures to ``outputs/``. It computes its nine
recommendations on first use, about half a minute each, and caches them under
``data/``, which git ignores.

Not financial advice.
