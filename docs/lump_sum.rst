.. _lump_sum:

One sum in from the start
=========================

The :doc:`backtests page <take_profit>` puts a fixed sum in at every buy day.
This page asks what happens when the money is there from the beginning, as it
would be for someone with savings and nothing to add. The whole of it goes in
on the first buy day, nothing follows, and from then on the account only
changes through what it sells.

The results are large, and they are consistent across everything that was
varied. Read :ref:`what that does and does not show <lump-sum-limit>` alongside
them: most of the gain is very likely an artifact of how the test is built.

The test
--------

$24,000 goes in on 2024-11-04 and the account is valued on 2026-11-02. The
stocks, K=25 and the recommendations are the backtests page's own. At every
buy day the whole account is sold, winners and losers alike, and that day's
recommendation is bought with all of it. Inside each stretch between two buy
days a position is also sold once its return reaches a threshold, and what a
sale brings in waits for the next buy day. Sells are checked at the close,
fills are at the closing price, and nothing is charged for trading or for tax.

"Nothing sold inside the stretch" is the account with only the first rule.
"Worst fall" is the deepest drop from an earlier close over the two years.

Which picks
-----------

γ=40, everything sold and re-bought every other Monday, two years later:

.. list-table::
   :header-rows: 1

   * - Picks
     - Nothing sold inside the stretch
     - Sell at +10% inside
     - Worst fall
   * - Normal
     - +214%
     - +143%
     - −21%
   * - Buy the dip
     - +41%
     - +46%
     - −26%
   * - Flipped
     - +19%
     - +12%
     - −18%
   * - The same 500 stocks, equal weight
     - +62%
     -
     - −20.5%
   * - SPY, bought once and held
     - +35%
     -
     - −19%

- **The normal picks lead at every threshold**, and the dip picks lead the
  flipped picks at 49 of the 50.
- **A sell inside the stretch acts differently on each.** It costs the normal
  and the flipped picks and helps the dip picks a little.
- **The dip picks end slightly ahead of SPY, and late.** They trailed it for
  the first 18 months.

Concentration: K
----------------

K is the number of factors and, through it, the concentration dial: a small K
gives a book of many names, a large K a book of few. γ=40, everything sold and
re-bought every other Monday. Each line is one sell threshold, from 1% (dark)
to 50% (light); black is nothing sold inside the stretch.

.. figure:: /_static/lump_sum_k.png
   :width: 100%
   :target: /_static/lump_sum_k.png
   :alt: Eight panels, one for each K from 5 to 40, of the account's return from the day the sum went in, 2024-11-04, to 2026-11-02. Each panel holds fifty lines, one per sell threshold, rising from zero to between 100 and 235 percent. The black line, nothing sold inside the stretch, is highest in every panel.

.. list-table::
   :header-rows: 1

   * - K
     - Names a buy
     - Nothing sold inside the stretch
     - Sell at +10% inside
     - Sell at +1% inside
     - Worst fall
   * - 5
     - 88
     - +99%
     - +76%
     - +25%
     - −9%
   * - 10
     - 57
     - +95%
     - +66%
     - +24%
     - −12%
   * - 15
     - 37
     - +116%
     - +79%
     - +38%
     - −15%
   * - 20
     - 28
     - +145%
     - +98%
     - +42%
     - −18%
   * - 25
     - 24
     - +214%
     - +143%
     - +51%
     - −21%
   * - 30
     - 22
     - +234%
     - +157%
     - +55%
     - −23%
   * - 35
     - 20
     - +222%
     - +130%
     - +52%
     - −21%
   * - 40
     - 19
     - +222%
     - +139%
     - +59%
     - −20%

From K=5 to K=25 the gain rose from +99% to +214% and the worst fall from 9%
to 21%. Beyond K=25 both flattened. At every K, selling inside the stretch
only cost, and the loosest thresholds came closest to not selling.

The rhythm
----------

Weekly, every other Monday, monthly and every two months, all starting on the
same day, with the usual three-month window. The lines are as above.

.. figure:: /_static/lump_sum_rhythm.png
   :width: 100%
   :target: /_static/lump_sum_rhythm.png
   :alt: Four panels, weekly, every two weeks, monthly and every two months, of the account's return over two years, each with fifty lines, one per sell threshold, and a black line for nothing sold inside the stretch. The black lines end at 259, 214, 284 and 303 percent. In the monthly and two-monthly panels the tighter thresholds stay far below the black line.

.. list-table::
   :header-rows: 1

   * - Rhythm
     - Re-buys in two years
     - Nothing sold inside the stretch
     - Sell at +10% inside
     - Sell at +1% inside
     - Worst fall
   * - Weekly
     - 104
     - +259%
     - +184%
     - +64%
     - −20%
   * - Every other Monday
     - 52
     - +214%
     - +143%
     - +51%
     - −21%
   * - Monthly
     - 24
     - +284%
     - +103%
     - +17%
     - −18%
   * - Every two months
     - 12
     - +303%
     - +56%
     - +4%
     - −31%

- **With nothing sold inside the stretch, the rhythm makes little
  difference.** All four end between +214% and +303%, and the every-other-Monday
  rhythm used elsewhere on this site is the lowest of the four.
- **With a sell inside the stretch, the rhythm matters.** The proceeds wait
  for the next re-buy, so the slower the rhythm the longer they sit idle. At
  +10% the order is reversed, and at +1% the monthly and two-monthly accounts
  end at +17% and +4%.

The window length
-----------------

The optimizer usually looks back three months. Two and four were tried on the
same account.

Nothing sold inside the stretch:

.. list-table::
   :header-rows: 1

   * - Rhythm
     - Two months
     - Three months
     - Four months
   * - Weekly
     - +162%
     - +259%
     - +150%
   * - Every other Monday
     - +166%
     - +214%
     - +167%
   * - Monthly
     - +210%
     - +284%
     - +135%
   * - Every two months
     - +300%
     - +303%
     - +268%

Sell at +10% inside the stretch:

.. list-table::
   :header-rows: 1

   * - Rhythm
     - Two months
     - Three months
     - Four months
   * - Weekly
     - +121%
     - +184%
     - +104%
   * - Every other Monday
     - +93%
     - +143%
     - +104%
   * - Monthly
     - +84%
     - +103%
     - +41%
   * - Every two months
     - +81%
     - +56%
     - +37%

.. figure:: /_static/lump_sum_rhythm_2months.png
   :width: 100%
   :target: /_static/lump_sum_rhythm_2months.png
   :alt: The same four panels as the rhythm figure, with a two-month window. The black lines end at 162, 166, 210 and 300 percent.

.. figure:: /_static/lump_sum_rhythm_4months.png
   :width: 100%
   :target: /_static/lump_sum_rhythm_4months.png
   :alt: The same four panels as the rhythm figure, with a four-month window. The black lines end at 150, 167, 135 and 268 percent.

With nothing sold inside the stretch, the three-month window did best at every
rhythm. It was the setting in use before any of these tests, not one picked
after seeing them. The two-month window also fell harder along the way: 28%
and 27% on the two faster rhythms, against about 20% at three months. The
two-monthly rhythm fell about a third at every window.

Other start days
----------------

With one sum, the day it goes in matters, so the same rules were started again
from other days: one-year sums, each begun on a different buy day of the first
year. The start days overlap heavily and all fall in the same two years, so
this shows whether a result hangs on its start date, not whether it holds in
another market.

The window, on the weekly rhythm, from 52 start days. "Best of the three"
counts the start days on which a window ended ahead of the other two.

Nothing sold inside the stretch:

.. list-table::
   :header-rows: 1

   * - Window
     - Median
     - Worst to best
     - Best of the three
   * - Two months
     - +76%
     - +19% to +139%
     - 0 of 52
   * - Three months
     - +108%
     - +41% to +158%
     - 52 of 52
   * - Four months
     - +80%
     - +35% to +120%
     - 0 of 52

Sell at +10% inside the stretch:

.. list-table::
   :header-rows: 1

   * - Window
     - Median
     - Worst to best
     - Best of the three
   * - Two months
     - +64%
     - +16% to +106%
     - 0 of 52
   * - Three months
     - +87%
     - +28% to +122%
     - 50 of 52
   * - Four months
     - +64%
     - +29% to +103%
     - 2 of 52

SPY over the same years had a median of +19%.

The ways of picking, every other Monday, from 26 start days.

Nothing sold inside the stretch:

.. list-table::
   :header-rows: 1

   * - Picks
     - Median
     - Worst to best
     - Ahead of SPY
   * - Normal
     - +96%
     - +32% to +149%
     - 26 of 26
   * - Buy the dip
     - +21%
     - −8% to +36%
     - 11 of 26
   * - Flipped
     - +12%
     - −8% to +23%
     - 1 of 26

Sell at +10% inside the stretch:

.. list-table::
   :header-rows: 1

   * - Picks
     - Median
     - Worst to best
     - Ahead of SPY
   * - Normal
     - +67%
     - +24% to +97%
     - 26 of 26
   * - Buy the dip
     - +25%
     - +1% to +41%
     - 14 of 26
   * - Flipped
     - +9%
     - −10% to +18%
     - 0 of 26

SPY's median over those years was +20%, and its worst start +14%.

.. _lump-sum-limit:

What the figures do and do not show
-----------------------------------

They show that, inside this test, the results do not hang on one choice:

- the normal picks lead at every sell threshold;
- with nothing sold inside the stretch, every rhythm and every window ended
  between +135% and +303%;
- the three-month window beat two and four months from 52 of 52 start days.

They do not show that the gains are real. Consistent inside a test is not the
same as robust. The test has features that give large, steady gains whatever
the rules are:

- **A stock list chosen with hindsight.** The 500 stocks are the most liquid
  at the end of the period, so the list already contains the period's winners
  and none of the stocks that failed. An equal-weight basket of it, re-bought
  every other Monday, made +62% against SPY's +35%. At γ=160 the picks made
  +57%, less than that basket. Every window beat SPY from all 52 start days,
  and the normal picks from all 26 of theirs.
- **One stretch of market.** Two years, mostly rising.
- **No trading costs.** This account sells everything 12 to 104 times in two
  years. At 0.5% a side the normal picks' +214% becomes about +89%, and the
  dip and flipped picks end at −15% and −28%.
- **Settings kept because they did well.** γ=40 and K=25 stayed because they
  led earlier sweeps.
- **Many cells.** Eight values of K, four rhythms, three windows and fifty
  thresholds; the best of that many is always optimistic.

The test that would settle it is the same one with the stock list chosen as of
each date, from prices that include stocks later delisted. It has not been
run.

Reproducing it
--------------

Four scripts in ``notebooks/`` produce everything on this page from the
cached prices of the backtests page. Each computes the recommendations it
lacks on first use: 52 buy days take about seven minutes with eight workers,
and the weekly rhythm needs twice that.

.. code-block:: bash

   # the three ways of picking, SPY, and one-year sums from every start day
   python notebooks/lump_sum.py 2
   # K = 5 to 40, each with every sell threshold's path
   python notebooks/lump_sum_k.py 2 40
   # the four rhythms at each window; the last argument is the window in months.
   # The three-month run also prints the start-day comparison of the windows.
   python notebooks/lump_sum_cadence.py 2 40 25 2
   python notebooks/lump_sum_cadence.py 2 40 25 3
   python notebooks/lump_sum_cadence.py 2 40 25 4
   # one setting with every sell threshold's path, labelled where they end
   python notebooks/lump_sum_paths.py 2 160

The scripts write to ``outputs/``. The figures on this page are copies of
theirs in ``docs/_static/``.

Not financial advice.
