# Aggregated

## Basic usage
An aggregated parameter returns the values of its child parameters and aggregates them using an 
*aggregation function*. For example, if you want to scale a profile, you can multiply the value, 
returned by a monthly profile parameter, with a constant parameter.

The following aggregation functions are available: 

- sum
- minimum
- maximum
- mean (value only)
- median (value only)
- product (value only)
- any (index only)
- all (index only)
- custom function

Pywr offers two kinds of aggregated parameter: [pywr.parameters.AggregatedParameter][] and  [pywr.parameters.AggregatedIndexParameter][], 
referred to as "value" and "index" in this section. A value-aggregated parameter takes and aggregates the values (or `float`)
returned by the child parameters, while an index-aggregated parameter does the same but aggregated the indexes (or `int`)
returned by the [pywr.parameters.IndexParameter][] children.

An example of a [pywr.parameters.AggregatedParameter][] is given below in Python, where the value of
the aggregated parameter `agg` is the product of a constant parameter (`baseline`) and a monthly profile (`profile`).

```python
from pywr.core import Model
from pywr.parameters import ConstantParameter, MonthlyProfileParameter, AggregatedParameter

model = Model()
baseline = ConstantParameter(model=model, value=5.0)
profile = MonthlyProfileParameter(
    model=model,
    values=[0.8, 0.8, 0.8, 0.8, 1.1, 1.1, 1.1, 1.1, 0.8, 0.8, 0.8, 0.8]
)
agg = AggregatedParameter(
    model=model,
    parameters=[baseline, profile],
    agg_func="product"
)
```
The new `agg` parameter will return `5.0 * 0.8 = 4.0` in any time step in January.

An example in JSON format is given below, where the aggregated parameter "Scaled profile" aggregates three parameters, one of
which is a constant, while the other two are references to parameters named in the `"parameters"` section of the JSON
document.

```json
{
  "Scaled profile": {
    "type": "aggregated",
    "agg_func": "product",
    "parameters": [
      104.7,
      "monthly_demand_profile",
      "demand_saving_factor"
    ]
  }
}
```

Aggregated parameters can be used to build up complex functions. A more detailed explanation of the 
above example can be found in :ref:`demand_saving`.

## Sum, Min, Max, Product
The `"sum"`, `"min"`, `"max"` and `"product"` aggregation functions are available to both aggregated-value and aggregated-index parameters.

## Mean and Median
The `"mean"` and `"median"` aggregation functions are only available for aggregated-value parameters, as these functions
could return non-integer values and cannot be therefore applied to parameters that return an index (or `int`).

## Any and All
The `"any"` and `"all"` aggregation functions behave like their NumPy equivalents, `numpy.any` and `numpy.all`, 
returning `0` or `1` depending on whether any or all of the child values are truthy (i.e. non-zero). These two
functions can only be applied to aggregated-index parameters (i.e. those that inherit 
from [pywr.parameters.IndexParameter][], such as [pywr.parameters.ControlCurveIndexParameter][]). 

The example below shows an index parameter which is "on" (i.e. returns an index of `1`) when all reservoirs are 
below their control curves (a flat line at 50% storage for "Small storage" and a flat line at 30% for "Large storage").

```python
from pywr.core import Model
from pywr.nodes import Storage
from pywr.parameters import ControlCurveIndexParameter, ConstantParameter, AggregatedParameter

model = Model()

storage_1 = Storage(model=model, max_volume=10, name="Small storage")
# this returns 0 if the storage is above the curve, 1 is below
rule_curve_1_index = ControlCurveIndexParameter(
    model=model,
    storage_node=storage_1, 
    control_curves=[
        ConstantParameter(model=model, value=0.5)
    ]
)

storage_2 = Storage(model=model, max_volume=300, name="Large storage")
rule_curve_2_index = ControlCurveIndexParameter(
    model=model,
    storage_node=storage_2, 
    control_curves=[
        ConstantParameter(model=model, value=0.3)
    ]
)

agg = AggregatedParameter(
    model=model,
    parameters=[rule_curve_1_index, rule_curve_2_index],
    agg_func="all"
)
```

## Custom aggregation functions
Custom aggregation functions can be used via the **Python API only**. The function is called with the values 
of the individual parameters and should return one aggregated value. For example, the following aggregation
function returns the 25<sup>th</sup> percentile of the values:

```python
import numpy as np
from pywr.parameters import AggregatedParameter

# x is a list of `float`s
func = lambda x: np.percentile(x, 25)
parameter = AggregatedParameter(parameters=[...], agg_func=func)
```

## Max, Min, Negative and NegativeMax Parameters
The Max/Min/Negative parameters are optimised aggregation functions for some common operations, 
which aggregate a single parameter and a constant.

The examples below compare the "max" aggregation function in [pywr.parameters.AggregatedParameter][] to the
[pywr.parameters.MaxParameter][]. 
The JSON required is shorter, arguably more readable and quicker to evaluate.

```json
{
    "type": "aggregated",
    "agg_func": "max",
    "parameters": [
        "another_parameter",
        1.0
    ]
}
```

```json
{
    "type": "max",
    "parameter": "another_parameter",
    "threshold": 1.0
}
```

An example use of these functions is to handle the net inflow timeseries for a reservoir, which includes both
positive flows (net gain) and negative flows (net evaporation / leakage). If the original parameter is given 
as *X*, the positive component can be achieved using `max(X, 0)` and attached to an `Input` node. 
The negative component needs to be made positive, as `Outputs` require positive flows, 
using `max(negative(X))`. This setup is shown in JSON below.

```json
{
  "parameters": {
    "original": {
      ...
    },
    "inflow": {
      "type": "max",
      "parameter": "original",
      "threshold": 0.0
    },
    "evaporation": {
      "type": "max",
      "parameter": {
        "type": "negative",
        "parameter": "original"
      },
      "threshold": 0.0
    }
  }
}
```

The pattern above was common enough to warrant the creation of the [pywr.parameters.NegativeMaxParameter][]:

```json
{
  "parameters": {
    "evaporation": {
      "type": "negativemax",
      "parameter": "original",
      "threshold": 0.0
    }
  }
}
```