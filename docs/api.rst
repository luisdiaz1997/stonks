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

.. autofunction:: stonks.robinhood.allocate_fractional_buys
