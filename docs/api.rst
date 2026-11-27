API Reference
=============

This page contains the API reference documentation for stonks.

Main Interface
--------------

.. autofunction:: stonks.portfolio.optimize_portfolio

Data Layer
----------

.. autofunction:: stonks.download.get_prices

.. autofunction:: stonks.returns.to_returns

Universe
---------

.. autofunction:: stonks.universe.load_universe

.. autofunction:: stonks.universe.list_all_common_stocks

Configuration
-------------

.. autoclass:: stonks.config.Settings
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Robinhood Execution
-------------------

Buying turns target weights into dollar orders; ``stonks sell`` exits positions
once their total return reaches a threshold, or all of them with
``--everything``; ``stonks rebuy`` buys a saved recommendation with what the
sells brought in. All are dry runs until the CLI's ``--execute`` is given. See
:doc:`deployment` for running them on a schedule.

.. autofunction:: stonks.robinhood.allocate_fractional_buys

.. autofunction:: stonks.robinhood.submit_fractional_buys

Take-Profit Sells
-----------------

.. autofunction:: stonks.robinhood.load_positions

.. autofunction:: stonks.robinhood.select_sells

.. autofunction:: stonks.robinhood.market_status

.. autofunction:: stonks.robinhood.submit_fractional_sells

Order Status
------------

.. autofunction:: stonks.robinhood.fetch_order_states

.. autofunction:: stonks.robinhood.load_last_orders


Selling and Buying Again
------------------------

.. autofunction:: stonks.robinhood.record_sells

.. autofunction:: stonks.robinhood.record_buy

.. autofunction:: stonks.robinhood.cash_to_rebuy
