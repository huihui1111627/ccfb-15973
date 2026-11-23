"""Spare-parts demand forecasting and inventory strategy for data-centre fleets.

The whole world -- servers, parts, failures, procurement, transport -- is
simulated (see :mod:`spareparts.fleet`); the estimation, policy and decision
layers only see the CMMS-style records the simulator would have produced up to
the planning date.

Typical use::

    python3 -m spareparts.acceptance      # build results (data/results.json)
    python3 -m spareparts.server 8000     # dashboard on http://127.0.0.1:8000
"""
__version__ = "1.0.0"
