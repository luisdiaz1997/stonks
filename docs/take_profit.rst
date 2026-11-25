Take-profit backtests
=====================

``stonks sell`` exits a position once its total return reaches a threshold.
The backtests on this page ask what one and two years of that would have done
when the buys are the optimizer's own recommendations. They describe two recent
stretches of history. They are not evidence about the future, and
:ref:`what they leave out <take-profit-caveats>` matters.

The test
--------

Every other Monday the production optimizer is run with its usual settings
(``K=25``, a 3-month window, the 500 most liquid stocks) and the day's share of
$1,000 a month is invested, along with whatever earlier sells brought in. One
year buys $12,000 from November 2025; two years buy $24,000 from November 2024.
Both are valued on 2026-11-02.

The account follows the live job where it matters:

- a stock bought more than once is one position with a blended average cost,
  and the whole position is sold once that blended return reaches the threshold;
- a position bought that day is not checked until the next day;
- proceeds wait for the next buy day.

Nothing is sold at a loss: whatever never reaches the threshold is kept. Prices
are total-return prices, with dividends folded in. Fills are at the closing
price, and nothing is charged for trading or for tax unless the text says so.

Result by sell threshold
------------------------

.. figure:: /_static/take_profit_sweep_two_years.png
   :width: 100%
   :target: /_static/take_profit_sweep_two_years.png
   :alt: Heatmap of the two-year return for sell thresholds from 1% to 50%, for four optimizer settings, each with positions held until sold and with everything sold at every buy day.

   Two years, $24,000 in. Each cell is the profit as a percentage of the money
   put in. Click the image to read every number; the table below has the main
   ones.

Four settings appear: the normal optimizer at γ=160 and at γ=40, and the same
two with the expected return negated ("flipped"), which makes the optimizer buy
the stocks that fell over its window instead of the ones that rose. Each comes
in two versions. In the *held* version a position stays until the rule sells
it. In the *rebalanced* version every buy day starts by selling the whole
account and the new recommendation is bought with all of it.

Two-year return on $24,000:

.. list-table::
   :header-rows: 1

   * - Setting
     - Never sell
     - +8%
     - +10%
     - +15%
     - Best threshold
   * - γ=160, normal, held
     - 24.0%
     - 25.4%
     - 25.5%
     - 25.0%
     - 31%, at 27.0%
   * - γ=160, flipped, held
     - 20.4%
     - 15.3%
     - 16.0%
     - 16.6%
     - 40%, at 19.2%
   * - γ=40, normal, held
     - 29.1%
     - 37.7%
     - 40.1%
     - 36.9%
     - 13%, at 42.5%
   * - γ=40, flipped, held
     - 23.3%
     - 7.3%
     - 9.6%
     - 8.5%
     - 49%, at 17.5%
   * - γ=160, normal, rebalanced
     - 22.1%
     - 19.8%
     - 22.1%
     - 20.2%
     - 45%, at 22.7%
   * - γ=160, flipped, rebalanced
     - 16.3%
     - 13.3%
     - 15.7%
     - 16.6%
     - 12%, at 17.9%
   * - γ=40, normal, rebalanced
     - 61.6%
     - 41.4%
     - 46.8%
     - 47.2%
     - 45%, at 63.0%
   * - γ=40, flipped, rebalanced
     - 17.1%
     - 12.0%
     - 14.8%
     - 16.0%
     - 12%, at 19.8%

What it shows:

- **γ=40 with the normal picks came first** in both ways of managing the
  account. Held and never sold it made +29.1%, against +24.0% for an equal-weight
  basket of the same 500 stocks and +16.6% for SPY with the same contributions.
- **For held accounts the threshold is a plateau.** At γ=40 every threshold
  from 8% to 15% lands between +36.9% and +42.5%, so the differences between
  neighbouring cells are not a reason to prefer one.
- **Flipping falls behind over two years, and the rule makes it worse.** For the
  flipped accounts, never selling beats every threshold up to about 40%.
- **Selling everything every buy day gives the highest number,** +61.6% at γ=40
  with no take-profit. It relied on the first year: on its own, the second year
  gave +10.2%, behind holding at +13.4%.

Over one year, the second of the two, flipped and normal were close: at +10%
with positions held they made +12.5% and +14.7%.

What the rule leaves behind
---------------------------

The rule sells winners and never sells losers, so it is worth looking at what
accumulates. The figure lines up every position at its own day zero: γ=40,
normal picks, sell at +10%, two years.

.. figure:: /_static/take_profit_position_paths.png
   :width: 100%
   :target: /_static/take_profit_position_paths.png
   :alt: Four panels, one per six months of buys, each showing every position's return from the day it was bought. Blue lines are sold at +10%, mostly within a few weeks. Orange lines are never sold and mostly drift below zero.

   Each line is one position, from the day the stock was first bought until the
   day it was sold or the span ends. A stock that is re-bought after a sale
   counts as a new position. Click the image to enlarge.

- **About one in five positions is never sold:** 138 of 711, or 19%, were still
  held at the end of the two years.
- **Sales happen early.** The median time from first buy to sale is 18 trading
  days, with half of the sales between 7 and 55 days.
- **What does not sell early tends to sink.** The never-sold positions end at a
  median of −9.2%, and a quarter of them are below −30%.

.. list-table:: Positions first bought in each six months
   :header-rows: 1

   * - Bought
     - Positions
     - Sold at +10%
     - Never sold
   * - Nov 2024 to Apr 2025
     - 190
     - 91%
     - 9%
   * - May to Oct 2025
     - 160
     - 93%
     - 7%
   * - Nov 2025 to Apr 2026
     - 172
     - 80%
     - 20%
   * - May to Oct 2026
     - 189
     - 60%
     - 40%

The last row is mostly a matter of time: those positions had weeks to months to
reach +10%, not years. Counted by time held instead, 54% are still unsold after
21 trading days, 32% after 63, 19% after 126, and 7% after 252.

A position is also re-bought whenever the optimizer picks a stock it already
holds, which happened to 38% of them. That resets the average cost: buying more
of a loser lifts its plotted return without any price move, and buying more of
a winner pulls it down.

Taxes and dividends
-------------------

None of the figures above include tax. The rule realizes a gain every time it
fires, almost always within a year, so those gains are taxed as ordinary income
in the year they occur, while never selling realizes nothing until the position
is sold. How much that narrows the rule's lead, or reverses it, depends
entirely on the investor's rates. ``take_profit_taxes.py`` estimates it for any
pair of short-term and long-term rates, tracking every purchase as its own
lot, and also shows the effect of selling the positions left below cost, whose
losses offset the rule's gains.

Dividends are small for these picks. Paid as cash, they came to $243 over the
two years when nothing is sold and $357 to $402 at the thresholds, under 1.2%
of the final value.

.. _take-profit-caveats:

What the backtests leave out
----------------------------

- **One stretch of market.** The two spans overlap and were mostly rising.
- **A stock list chosen with hindsight.** The 500 stocks are ranked by trading
  volume over the period itself, so stocks that rose and stayed liquid are
  over-represented. That flatters every stock-picking row; SPY is the only
  comparison it does not.
- **Trading costs.** Fills are at the closing price with no bid-ask spread,
  which matters most for selling everything every buy day.
- **Hourly checks.** The live job checks hourly. The backtests sell at the
  close, or at the exact threshold the moment a day's high reaches it; at γ=40
  the two differ by up to about $1,700 over two years.
- **When tax is paid.** The bill on the first year's gains would leave less to
  invest in the second.
- **Many variants.** Hundreds of combinations were tried. The best cell of that
  many is always optimistic.
- **Concentration.** The γ=40 book holds about 24 names per buy, so a few stocks
  decide the result.

Reproducing it
--------------

Four scripts in ``notebooks/`` run the whole thing. They download their prices
and compute the optimizer's recommendations on first use, which takes about
four minutes per setting and span; after that everything is cached under
``data/``, which git ignores.

.. code-block:: bash

   # one year, γ=40: the full report with every comparison
   python notebooks/take_profit_backtest.py 40 biweekly
   # the same over two years
   STONKS_BACKTEST_SPAN=2 python notebooks/take_profit_backtest.py 40 biweekly
   # the heatmap on this page
   python notebooks/take_profit_heatmap.py 2
   # the position paths on this page
   python notebooks/take_profit_paths.py 2 40 10
   # taxes
   python notebooks/take_profit_taxes.py 2 40

Not financial or tax advice.
