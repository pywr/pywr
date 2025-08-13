# Interpolated

If you a function with discrete data points of x and y coordinates, Pywr
implements some parameters that let you interpolate over the function using
the value from another parameter or volume or node's flow.

Suppose you have a set of points expressing the relationship between a
reservoir surface area and its volume, and you want to find the surface area
corresponding to the storage being modelled at each timestep to calculate
evaporative losses. Or you want to calculate operation costs, but the relationship
between a node's flow and cost is non-linear. With these set of parameters you can
achieve these calculations.

## Interpolated parameter

The [pywr.parameters.InterpolatedParameter][] allows interpolating a parameter
value over a user-defined function.

In the example below, the monthly profile `"P1"` values are interpolated using a piecewise
linear relationship over the exponential function x<sup>3</sup>: 

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.parameters import MonthlyProfileParameter, InterpolatedParameter
    x = np.arange(0, 22, 2)
    y = np.exp(x / 3.0)

    model = Model()
    p1 = MonthlyProfileParameter(
        model=model, 
        values=[0, 1, 5.6, 9.2, 11.45, 8.7, 5, 4, 10.1, 19.1, 3.2, 13.1],
        name="P1"
     )
    p2 = InterpolatedParameter(
        model=model,
        parameter=p1,
        name="P2",
        x=x,
        y=y,
    )
    ```

=== "JSON"
    ```json
    {
        "P1": {
            "type": "MonthlyProfileParameter",
            "values": [0, 1, 5.6, 9.2, 11.45, 8.7, 5, 4, 10.1, 19.1, 3.2, 13.1]
        },
        "P2": {
            "type": "InterpolatedParameter",
            "parameter": "P1",
            "x": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "y": [1.0, 1.94773404, 3.79366789, 7.3890561, 14.3919161, 28.03162489, 54.59815003, 106.3426754, 207.12724889, 403.42879349, 785.77199423]
        }
    }
    ```

The parameter `"P2"` when recorder will output the following values for each month

<p align="center">

```python exec="1" html="1"
from io import StringIO
import numpy as np
import pandas as pd

from pywr.core import Model
from pywr.nodes import Input, Output
from pywr.recorders import NumpyArrayParameterRecorder
import matplotlib.pyplot as plt
from pywr.parameters import MonthlyProfileParameter, InterpolatedParameter

model = Model()
node = Input(model, max_flow=5, name="input")
node.connect(Output(model, cost=-100, name="output"))

x = np.arange(0, 22, 2)
y = np.exp(x / 3.0)

m_x = [0, 1, 5.6, 9.2, 11.45, 8.7, 5, 4, 10.1, 19.1, 3.2, 13.1]
p1 = MonthlyProfileParameter(
    model=model,
    values=m_x,
    name="p1",
)
p2 = InterpolatedParameter(model=model, parameter=p1, x=x, y=y)
r = NumpyArrayParameterRecorder(model, p2)
model.run()

del model
df = r.to_dataframe()
df1 = pd.Series(index=range(1, 13))
for month in df1.index:
    df1[month] = df[df.index.month == month].values[0]
df1.index = m_x

_, ax = plt.subplots()
df1.plot(color="k", style=".", ax=ax)
ax.plot(x, y, "b-")
ax.set_xlabel("x value")
ax.set_ylabel("y value")
ax.legend(["Interpolated", "Exponential"])

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```    

</p>

!!!waring "Out of bounds"
    By default, the parameter raises an error any time the
    interpolation is attempted on a value outside the range of x. You can change
    this behaviour as explained in the section below.

### Change the interpolation options
Pywr uses the [interp1 function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)
in the Scipy package to perform the interpolation. You can change the interpolation optiond using
the `interp_kwargs` parameter. This accepts a dictionary with the parameterd of the interp1 function.

For example, if you want to change the interpolation method to `cubic` and prevent the
function to raise an error when the interpolation is attempted outside the x bounds, you can use:

```json
{
    "P1": {
        "type": "MonthlyProfileParameter",
        "values": [0, 1, 5.6, 9.2, 11.45, 8.7, 5, 4, 10.1, 19.1, 3.2, 13.1]
    },
    "P2": {
        "type": "InterpolatedParameter",
        "parameter": "P1",
        "x": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "y": [1.0, 1.94773404, 3.79366789, 7.3890561, 14.3919161, 28.03162489, 54.59815003, 106.3426754, 207.12724889, 403.42879349, 785.77199423],
        "interp_kwargs": {
          "kind": "cubic",
          "bounds_error": false
        }
    }
}
```

## Interpolated volume
To interpolate over a reservoir absolute volume, you can use the [pywr.parameters.InterpolatedVolumeParameter][].
The parameter uses the `volumes` parameter to specify the x coordinates, the `values`
parameter for the y values and the `storage` to provide the node:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import InterpolatedParameter

    model = Model()
    storage = Storage(model=model, max_volume=500, name="Storage")
    p2 = InterpolatedVolumeParameter(
        model=model,
        node=storage,
        volumes=[100, 200, 340, 450, 510],
        values=[0.1, 0.32, 0.98, 1.2, 2.5],
        interp_kwargs={"kind": "quadratic"},
        name="My parameter"
    )
    ```

=== "JSON"

    ```json
    {
        "My parameter": {
            "type": "InterpolatedVolumeParameter",
            "node": "Storage",
            "volumes": [100, 200, 340, 450, 510],
            "values": [0.1, 0.32, 0.98, 1.2, 2.5],
            "interp_kwargs": {"kind": "quadratic"}
        }
    }
    ```

In this parameter, the x and y coordinates can also be loaded from an
external file if your model is implemented in a JSON document:

```json
{
    "My parameter": {
        "type": "InterpolatedVolumeParameter",
        "node": "Storage",
        "volumes": {
            "table": "Storage table",
            "column": "Volume"
        },
        "values": {
            "table": "Storage table",
            "column": "Area"
        },
        "interp_kwargs": {"kind": "quadratic"}
    }
}
```

!!!warning "Volume"
    The interpolation is performed using the absolute volume and not the
    relative storage.


## Interpolated flow
To interpolate using flow data, you can use the [pywr.parameters.InterpolatedFlowParameter][].
The x coordinates can be provided using the `flows` parameter and the node is
given using the `node` parameter:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import InterpolatedFlowParameter

    model = Model()
    link = Link(model=model, max_flow=20, name="Link")
    parameter = InterpolatedFlowParameter(
        model=model,
        node=link,
        flows=[0, 5, 10, 20],
        values=[0, 10, 30, -5]
    )
    ```

=== "JSON"

    ```json
    {
        "My parameter": {
            "type": "InterpolatedFlowParameter",
            "node": "Link",
            "flows": [0, 5, 10, 20],
            "values": [0, 10, 30, -5]
        }
    }
    ```

## Interpolated quadrature
The [pywr.parameters.InterpolatedQuadratureParameter][] integrate a discrete function
of x and y points between a lower and upper bound from two parameter values

$$\int_{lower parameter}^{upper parameter} f(x)dx$$

Suppose you have this model:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.parameters import (
        MonthlyProfileParameter,
        InterpolatedQuadratureParameter
    )

    
    x = range(0, 22, 2)
    y = [v**2 for v in x]

    model = Model()
    p1 = MonthlyProfileParameter(
        model=model,
        values=m_x,
        name="P1",
    )
    InterpolatedQuadratureParameter(
        model=model,
        upper_parameter=p1,
        x=x,
        y=y,
        name="P2"
    )
    ```

=== "JSON"

    ```json
    {
        "P1": {
            "type": "MonthlyProfileParameter",
            "values": [0, 1, 5.6, 9.2, 11.45, 8.7, 5, 4, 10.1, 19.1, 3.2, 13.1]
        },
        "P2": {
            "type": "InterpolatedQuadratureParameter",
            "upper_parameter": "P1",
            "x": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "y": [0, 4, 16, 36, 64, 100, 144, 196, 256, 324, 400]
        }
    }
    ```

In August, when `P1` returns `4`, `P2` will calculate the following:

$$\int_0^4 f(x)dx$$

which returns `24`. When the `lower_parameter` is not given, this
defaults to `0`. In the example, when the `x` value needed for the integration is not available
in the given `x` values, the value for `y` is interpolated.

!!!info "Integration"
    The integration is performed using [scipy.integrate.quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#quad)