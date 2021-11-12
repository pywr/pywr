Aggregated parameters
---------------------

Basic usage
===========

An aggregated parameter returns the values of its child parameters aggregated using an *aggregation function*. The following aggregation functions are available: sum, minimum, maximum, mean (value only), median (value only), product (value only), any (index only), all (index only), custom.

There are two kinds of aggregated parameter: ``AggregatedParameter`` and ``AggregatedIndexParameter``, referred to as "value" and "index" in this document. A value aggregated parameter aggregates the values of it's children, while the index aggregated parameter does the same for the index of ``IndexParameter``.

An example of creating an ``AggregatedParameter`` in Python is given below. The value of the aggregated parameter is the product of the baseline and profile values, e.g. in January the value is 5.0 * 0.8 = 4.0.

.. code-block:: python

    baseline = ConstantParameter(5.0)
    profile = MonthlyParameter(
        [0.8, 0.8, 0.8, 0.8, 1.1, 1.1, 1.1, 1.1, 0.8, 0.8, 0.8, 0.8]
    )
    agg = AggregatedParameter(
        parameters=[baseline, profile],
        agg_func="product"
    )

An example of creating an ``AggregatedParameter`` in JSON is given below. This parameter aggregates three parameters, one of which is a constant while the other two are references to parameters named in the ``"parameters"`` section.

.. code-block:: javascript

    {
        "type": "aggregated",
        "agg_func": "product",
        "parameters": [
            104.7
            "monthly_demand_profile",
            "demand_saving_factor"
        ]
    }

Aggregated parameters can be used to build up complex functions. A more detailed explanation of the above example can be found in :ref:`demand_saving`.

Sum, Min, Max, Product
======================

The ``"sum"``, ``"min"``, ``"max"`` and ``"product"`` aggregation functions are available to both aggregated value and aggregated index parameters.

Mean and Median
===============

The ``"mean"`` and ``"median"`` aggregation functions are only available for aggregated value parameters, as they could return non-integer values.

Any and All
===========

The ``any`` and ``all`` aggregation functions behave like their NumPy equivalents, ``numpy.any`` and ``numpy.all`` , returning 0 or 1 depending on whether any or all of the child values are truthy (i.e. non-zero).

An example use is an index parameter which is "on" when any reservoir in a group is below its control curve.

Custom aggregation functions
============================

Custom aggregation functions can be used via the Python API only. The function is called with the values of the individual parameters and should return the aggregated value. For example, the following aggregation function returns the 25th percentile of the values:

.. code-block:: python

    func = lambda x: np.percentile(x, 25)
    parameter = AggregatedParameter(parameters, agg_func=func)

MaxParameter, MinParameter and NegativeParameter, NegativeMaxParameter
======================================================================

The Max/Min/Negative parameters are optimised aggregation functions for some common operations, which aggregate a single parameter and a constant.

The examples below compare the "max" aggregation function in ``AggregatedParameter`` to the ``MaxParameter``. The JSON required is shorter, arguably more readable and quicker to evaluate.

.. code-block:: javascript

    {
        "type": "aggregated",
        "agg_func": "max",
        "parameters": [
            "another_parameter",
            0.0
        ]
    }

.. code-block:: javascript

    {
        "type": "max",
        "parameter": "another_parameter",
        "threshold": 0.0
    }

An example use of these functions is to handle the net inflow timeseries for a reservoir, which includes both positive flows (net gain) and negative flows (net evaporation / leakage). If the original parameter is given as *X*, the positive component can be achieved using ``max(X, 0)`` and attached to an ``Input`` node. The negative component needs to be made positive, as ``Outputs`` require positive flows, using ``max(negative(X))``. This setup is shown in JSON below.

.. code-block:: javascript

    "original": ...

    "inflow": {
        "type": "max",
        "parameter": "original",
        "threshold": 0.0
    }
    
    "evaporation": {
        "type": "max",
        "parameter": {
            "type": "negative",
            "parameter": "original"
        }
        "threshold": 0.0
    }

The pattern above was common enough to warrant the creation of the ``NegativeMaxParameter``.

.. code-block:: javascript

    "evaporation": {
        "type": "negativemax",
        "parameter": "original",
        "threshold": 0.0
    }
