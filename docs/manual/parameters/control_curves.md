# Control curves
In water resource management, control curves define thresholds for reservoir storage, which trigger different
operational responses. These responses can range from restricting abstractions 
to augmenting downstream flows. 

Pywr provides a flexible and intuitive way to implement complex operational rules on reservoirs through its control curve parameters.
By linking these curves to other parameters, you can create dynamic rules without writing any bespoke Python code.

In Pywr there are four built-in control curve parameters: [pywr.parameters.ControlCurveParameter][],
[pywr.parameters.ControlCurveIndexParameter][],
[pywr.parameters.ControlCurveInterpolatedParameter][] and [pywr.parameters.ControlCurvePiecewiseInterpolatedParameter][].
These are explained in detail in the next sections.

## Control curve parameter
The [pywr.parameters.ControlCurveParameter][] is the simplest control curve parameter and is particularly 
useful for binary decisions, such as switching between normal and drought operation modes, or activating a specific 
water transfer based on reservoir levels.

Suppose you need to implement the following control curves which set the maximum abstraction rate at the drawoff tower. If the storage
is above all curves, the rate is `10`, when below the first line `6`, then `2` and `0` when below the last control
curve:

<p align="center">

```python exec="1" html="1"
from io import StringIO

import matplotlib.pyplot as plt
from calendar import month_abbr

c1 = [0.8, 0.73, 0.7, 0.62, 0.6, 0.55, 0.6, 0.65, 0.72, 0.75, 0.83, 0.8]
c2 = [0.6, 0.52, 0.5, 0.45, 0.4, 0.35, 0.4, 0.43, 0.51, 0.55, 0.63, 0.6]
x = [month_abbr[i] for i in range(1, 13)]

plt.figure()
plt.plot(x, c1, color='#1f77b4')
plt.plot(x, c2, color='#1f77b4')
plt.plot([0.22] * 13, color='#1f77b4')

text_props = dict(fontsize=12, color="g", xycoords="axes fraction")
plt.annotate("10", (0.48, 0.8), **text_props)
plt.annotate("6", (0.5, 0.46), **text_props)
plt.annotate("2", (0.5, 0.26), **text_props)
plt.annotate("0", (0.5, 0.14), **text_props)
plt.xlim(x[0], x[-1])
plt.ylim(0, 1)

plt.ylabel("Storage")
plt.xlabel("Time")

buffer = StringIO()
plt.savefig(buffer, format="svg")
print(buffer.getvalue())
```
</p>

This behaviour is implemented as follows:

=== "Python"
    ```python  
    from pywr.core import Model
    from pywr.parameters import MonthlyProfileParameter, ControlCurveParameter, ConstantParameter
    
    model = Model()
    line_1 = MonthlyProfileParameter(
        model=model,
        values=[0.8, 0.73, 0.7, 0.62, 0.6, 0.55, 0.6, 0.65, 0.72, 0.75, 0.83, 0.8],
        name="Top line"
    )
    line_2 = MonthylProfileParameter(
        model=model,
        values=[0.6, 0.52, 0.5, 0.45, 0.4, 0.35, 0.4, 0.43, 0.51, 0.55, 0.63, 0.6],
        name="Second line"
    )
    line_3 = ConstantParameter(model=model, value=0.3, name="Last line")

    storage_node = Storage(
        model=model,
        name="Reservoir",
        max_volume=100,
        initial_volume=100
    )
    intake = Link(model, name="Intake", cost=-20)
    intake.max_flow = ControlCurveParameter(
        model=model,
        name="Abstraction",
        storage_node=storage_node,
        control_curves=[line_1, line_2, line_3],
        values=[10, 6, 2, 0]
    )
    ```

=== "JSON"
    ```json
    {
        "parameters": {
            "Top line": {
                "type": "MonthlyProfileParameter",
                "values": [0.8, 0.73, 0.7, 0.62, 0.6, 0.55, 0.6, 0.65, 0.72, 0.75, 0.83, 0.8],
            },
            "Second line": {
                "type": "MonthlyProfileParameter",
                "values": [0.6, 0.52, 0.5, 0.45, 0.4, 0.35, 0.4, 0.43, 0.51, 0.55, 0.63, 0.6],
            },
            "Last line": {
                "type": "ConstantParameter",
                "value": 0.3
            },
            "Abstraction": {
                "type": "ControlCurveParameter",
                "storage_node": "Reservoir",
                "control_curves": ["Top line", "Second line", "Last line"],
                "values": [10, 6, 2, 0]
            }
       }
    }
    ```

The parameter works by performing a simple comparison at each timestep:

- it checks the current proportional volume of a specified storage node. 
- It compares this volume to the value of the top control curve. 
- If the storage volume is greater than or equal to the control curve value, it returns
the first value from a list of user-defined values. 
- If the storage volume is less than the control curve value, it returns the 
second value from the list.
- And so on.

Some **important tips** you need to remember when you implement the parameter:

- you can use any profile parameter you want.
- There is no limit on the number of control curves you can use.
- Each curve must provide the relative storage between 0 and 1 (not the absolute reservoir volume)
- The curves must be sorted in descending order in the `control_curves` attribute with the first parameter
being the top line and the last the bottom line.
- The number of `values` must equal the number of control curves plus 1.

### Omit `values`
If you do not provide `values`, this parameter returns an integer which is zero, if the storage
is above the first control curve. The integer increments by one for each control curve 
the `storage_node` node is below.

### Use parameters instead of values
If you want the number the parameter returns from `values` to be variable, you can use the
`parameters` argument, which accept a list of parameters. As for constant values, the number 
of parameters must equal the number of control curves plus 1. In the example below, the first two
abstraction rates change monthly using a [pywr.parameters.MonthlyProfileParameter][]:

=== "Python"
    ```python  
    from pywr.core import Model
    from pywr.parameters import MonthlyProfileParameter, ControlCurveParameter, ConstantParameter
    model = Model()

    ....

    rate_above_line_1 = MonthlyProfileParameter(
        model=model,
        values=[9, 9, 10, 8, 8, 9, 8, 10, 8, 8, 9, 8],
        name="Rate 1"
    )
    rate_below_line_1 = MonthlyProfileParameter(
        model=model,
        values=[7, 7, 6, 6, 7, 7, 7, 7, 6, 7, 6, 6],
        name="Rate 2"
    )
    rate_below_line_2 = ConstantParameter(
        model=model,
        value=3,
        name="Rate 3"
    )
    rate_below_line_3 = ConstantParameter(
        model=model,
        value=0,
        name="Rate 5"
    )
    intake.max_flow = ControlCurveParameter(
        model=model,
        name="Abstraction",
        storage_node=storage_node,
        control_curves=[line_1, line_2, line_3],
        parameters=[rate_above_line_1, rate_below_line_1, rate_below_line_2, rate_below_line_3]
    )
    ```

=== "JSON"
    ```json
    {
        "parameters": {
            "Rate 1": {
                "type": "MonthlyProfileParameter",
                "values": [9, 9, 10, 8, 8, 9, 8, 10, 8, 8, 9, 8]
            },
            "Rate 2": {
                "type": "MonthlyProfileParameter",
                "values": [7, 7, 6, 6, 7, 7, 7, 7, 6, 7, 6, 6]
            },
            "Rate 3": {
                "type": "ConstantParameter",
                "value": 3
            },
            "Rate 4": {
                "type": "ConstantParameter",
                "value": 0
            },
            "Abstraction": {
                "type": "ControlCurveParameter",
                "storage_node": "Reservoir",
                "control_curves": ["Top line", "Second line", "Last line"],
                "parameters": ["Rate 1", "Rate 2", "Rate 3", "Rate 4"]
            }
       }
    }
    ```

## Interpolate values
If you want the `values` to linearly change with the reservoir storage within each control curve band, you
can use the [pywr.parameters.ControlCurveInterpolatedParameter][]. This is normally used if you need
to [set a cost on a reservoir](../key_concepts/costs.md#two-interpolated-costs).

This parameter configuration is similar to the [ControlCurveParameter](#control-curve-parameter), but the number of `values`
or `parameters` you need to provide is equal to the number of control curves plus 2:

=== "Python"
    ```python  
    from pywr.core import Model
    from pywr.parameters import ControlCurveInterpolatedParameter
    
    ...
    intake.max_flow = ControlCurveInterpolatedParameter(
        model=model,
        name="Abstraction",
        storage_node=storage_node,
        control_curves=[line_1, line_2, line_3],
        values=[10, 8, 6, 2, 0]
    )
    ```

=== "JSON"
    ```json
    {
        "parameters": {
            "Abstraction": {
                "type": "ControlCurveInterpolatedParameter",
                "storage_node": "Reservoir",
                "control_curves": ["Top line", "Second line", "Last line"],
                "values": [10, 8, 6, 2, 0]
            }
       }
    }
    ```


The above operates as follows:

- if the storage is between 100% and `"Top line"`, the value is linearly interpolated between `10` and `8`. At 100% the value is `10` 
and when the storage is equal to the control curve `8`.
- if the storage is between `"Top line"` and `"Second line"`, the value is linearly interpolated between `8` and `6`. The value 
at `"Second line"` is `6`.
- if the storage is between `"Second line"` and `"Last line"`, the value is linearly interpolated between `6` and `2`.
- if the storage is below `"Last line"`, the value is interpolated between `2` and `0`. The value when the storage is 
empty is `0`.

Parameters can be used in place of constant values using the `parameters` argument.

## Piecewise interpolated values
A more refined approach can be used with the [pywr.parameters.ControlCurvePiecewiseInterpolatedParameter][], which
interpolates between two or more pairs of `values`:

=== "Python"
    ```python  
    from pywr.core import Model
    from pywr.parameters import ControlCurvePiecewiseInterpolatedParameter
    
    ...
    intake.max_flow = ControlCurvePiecewiseInterpolatedParameter(
        model=model,
        name="Abstraction",
        storage_node=storage_node,
        control_curves=[line_1, line_2],
        values=[
            [-0.1, -1],
            [-100, -200],
            [-200, -500],
        ]
    )
    ```

=== "JSON"
    ```json
    {
        "parameters": {
            "Abstraction": {
                "type": "ControlCurvePiecewiseInterpolatedParameter",
                "storage_node": "Reservoir",
                "control_curves": ["Top line", "Last line"],
                "values": [
                    [-0.1, -1],
                    [-100, -200],
                    [-200, -500]
                ]
            }
       }
    }
    ```

In the code above:

- the first pair is used between the maximum storage and the first control curve. `-0.1` is used if the storage
is at 100%, `-1` at `"Top line"`. Any storage value in-between is linearly interpolated between the two values.
- The next pair (`-100` and `-200`) is used between the first control curve and second control curve.
- and so on until the last pair is used between the last control curve and the storage reaches 0.

The number of pairs to provide is equal to the number of control curves plus 2. 

!!!warning "Parameters not supported"
    You cannot use parameters to specify the value pairs.

### Adjust the min and max
By default the parameter assumes that the first value in the first pair (`-0.1` in the example above) refers to 100%.
You can change the storage considered the top of the upper curve (or the lower curve) using the `maximum` 
(or `minimum`) parameter :

=== "Use maximum"
    ```json
    {
        "parameters": {
            "Abstraction": {
                "type": "ControlCurvePiecewiseInterpolatedParameter",
                "storage_node": "Reservoir",
                "control_curves": ["Top line", "Second line", "Last line"],
                "values": [
                    [-0.1, -1],
                    [-100, -200],
                    [-200, -500]
                ],
                "maximum": 0.9
            }
       }
    }
    ```
=== "Use minimum"
    ```json
    {
        "parameters": {
            "Abstraction": {
                "type": "ControlCurvePiecewiseInterpolatedParameter",
                "storage_node": "Reservoir",
                "control_curves": ["Top line", "Second line", "Last line"],
                "values": [
                    [-0.1, -1],
                    [-100, -200],
                    [-200, -500]
                ],
                "minimum": 0.1
            }
       }
    }
    ```

In the first example, when the storage is above `"Top line"`, the interpolation is performed between 90% and the
control curve. If the storage is above 90%, then `-0.1` is returned.
Similarly, for the `minimum`, when the storage is below the last curve, the interpolation is performed between the 
last control line and 10%. If the storage is below 10%, then `-500` is returned.

## Control curve index
The [pywr.parameters.ControlCurveIndexParameter][] returns the index or band the reservoir node is in with respect 
to the control curves. This parameter is meant to be used
in conjunction with a parameter that uses an index parameter to return a value. For example, if you need to apply
a different set of rules on different nodes using the same set of control curves, instead of using multiple
`ControlCurveParameter`s using the same curves but returning different values, you can couple
the `ControlCurveIndexParameter` with multiple `IndexedArrayParameter`s. This makes the model more
efficient and easy to read:

=== "Python"
    ```python  
    from pywr.core import Model
    from pywr.parameters import ControlCurveIndexParameter, IndexedArrayParameter
    
    ...
    cc_index = ControlCurveIndexParameter(
        model=model,
        name="Abstraction",
        storage_node=storage_node,
        control_curves=[line_1, line_2],
        name="CC index"
    )
    intake1.max_flow = IndexedArrayParameter(
        model=model,
        index_parameter=cc_index,
        params=[0.5, 0.9, 1.4],
        name="Abstraction 1"
    )
    intake2.max_flow = IndexedArrayParameter(
        model=model,
        index_parameter=cc_index,
        params=[2, 1, 0],
        name="Abstraction 2"
    )
    ```

=== "JSON"
    ```json
    {
        "parameters": {
            "CC index": {
                "type": "ControlCurveIndexParameter",
                "storage_node": "Reservoir",
                "control_curves": ["Top line", "Last line"]
            },
            "Abstraction 1": {
                "type": "IndexedArrayParameter",
                "index_parameter": "CC index",
                "params": [0.5, 0.9, 1.4],
            },
            "Abstraction 2": {
                "type": "IndexedArrayParameter",
                "index_parameter": "CC index",
                "params": [2, 1, 0],
            }
       }
    }
    ```

When the reservoir is :

- above the first curve, it returns `0`.  `"Abstraction 1"` returns `0.5` and `"Abstraction 2"` returns `2`.
- Below the first curve, it returns `1`.  `"Abstraction 1"` returns `0.9` and `"Abstraction 2"` returns `1`.
- Below the second curve, it returns `2`. `"Abstraction 1"` returns `1.4` and `"Abstraction 2"` returns `0`.

The maximum index equal the number of control curves in `control_curves`. 