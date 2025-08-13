# Threshold
Threshold parameters implement simple if-else conditions and
return one of two values depending on a value calculated
by Pywr during the simulation (such as the current volume, or node's flow, or parameter or recorder
value) by assessing:

\begin{cases}
B & \text{if $Value$ predicate_sign $Threshold$} \\[2ex]
A & \text{otherwise}
\end{cases} 

where in the predicate `Value` `predicate_sign` `Threshold`:

- `Value` is the current value
- `predicate_sign` is one of the inequality signs ("<", ">", "<=", ">=") or the equal ("=") sign
- `Threshold` an user-defined threshold `Value` is compared against.

When the predicate evaluates false, the parameter returns `A`, otherwise, when true, it
return` B`. 

These sets of parameters can be used to model, for example, reductions in licensed abstractions
based on a river level or hands-off flow conditions. For example, if a license needs to reduce an intake 
`max_flow` from `5` to `1` when a river flow is below `1000`, the condition above can be written
as:

\begin{cases}
1 & \text{if $River$ <= 1000} \\[2ex]
5 & \text{otherwise}
\end{cases} 

## Simple thresholds

### Node threshold
The [pywr.parameters.NodeThresholdParameter][] reads the previous flow in a node and compares
it against a user-defined threshold. In the example below the flow of `"River"` is
compared against the constant threshold `1000`. When the previous-day flow of `"River"` is
less than `1000`, the `max_flow` on the `"Abstraction"` node is limited to `2`, 
otherwise it is set to `10`.

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import StorageThresholdParameter

    model = Model()
    river = Link(model=model, name="River")
    parameter = NodeThresholdParameter(
        model=model, 
        node=river, 
        predicate="LT", 
        values=[10, 2], 
        threshold=1000,
        name="Daily river-dependant license"
    )
    intake = Link(model=model, max_flow=parameter, name="Abstraction")

    ```

=== "JSON"

    ```json
    {
        "nodes": [
            {
                "type": "link",
                "name": "River"
            },
            {
                "type": "link",
                "name": "Abstraction",
                "max_flow": {
                    "type": "NodeThresholdParameter",
                    "node":"River",    
                    "predicate": "LT", 
                    "values": [10, 2], 
                    "threshold": 1000,
                }
            }
        ],

    }
    ```

The predicate value can be on of the following:

- `LT` or `<` for less than;
- `GT` or `>` for greater than;
- `EQ` or `=` for equality;
- `LE` or `<=` for less and equal to;
- `GE` or `>=` for greater and equal to.

The `threshold` can also be dynamic and you can use any parameter type. For example, if the threshold
is determined from an inflow file, scaled by 10%, you can combined the [pywr.parameters.DataFrameParameter][]
with a [pywr.parameters.ConstantParameter][]:


=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import StorageThresholdParameter, AggregatedParameter

    model = Model()
    river = DataFrameParameter(model=model, name="Inflow", dataframe=...)
    scaling_factor = ConstantParameter(model, value=0.1)
    threshold = AggregatedParameter(
        model=model, 
        parameters=[river, scaling_factor],
        agg_func="product", 
        name="Threshold value"
    )
    NodeThresholdParameter(
        model=model, 
        node=river, 
        predicate="LT", 
        values=[10, 2], 
        threshold=threshold,
        name="Daily river-dependant license"
    )
    ```

=== "JSON"

    ```json
    {
        "Inflow":  {
            "type": "dataframe",
            "table": "Inflow file",
            "column": "River flow"
        }
        "Threshold value": {
            "type": "AggregatedParameter",
            "parameters": ["Inflow", 0.1],
            "agg_func": "product"
        },
        "Daily river-dependant license": {
            "type": "NodeThresholdParameter",
            "node": "River",    
            "predicate": "LT", 
            "values": [10, 2], 
            "threshold": "Threshold value",
        }
    }
    ```

!!!warning "Initial condition"
    On the first day of the model run, the calculated flow will not have a value for
    the previous day. In this case, the predicate evaluates to True.

### Storage threshold
The [pywr.parameters.StorageThresholdParameter][] behaves exactly like the parameter above,
but instead of a flow value, it reads the absolute volume of a storage node. 

In the example below the storage in the node `"Reservoir"` is compared against the constant threshold `0.5`. When
the storage is less than `0.5`, `0` is returned, otherwise `10` is returned. 


=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import StorageThresholdParameter

    model = Model()
    node = Storage(
        model=model, 
        max_volume=100,
        name="Reservoir",
        initial_volume=100
    )
    StorageThresholdParameter(
        model=model, 
        storage=node, 
        predicate="LT", 
        values=[10, 0], 
        threshold=0.5,
        name="My parameter"
    )
    ```

=== "JSON"

    ```json
    {
        "My parameter": {
            "type": "StorageThresholdParameter",
            "storage":"Reservoir",    
            "predicate": "LT", 
            "values": [10, 0], 
            "threshold": 0.5,
        }
    }
    ```

### Parameter threshold
The [pywr.parameters.ParameterThresholdParameter][] behaves as the [node threshold](#node-threshold),
but instead of a flow value, it reads the value from another parameter. 


=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ControlCurveParameter, ParameterThresholdParameter

    model = Model()
    control_curve = ControlCurveParameter(model, name="Control curve", ...)
    ParameterThresholdParameter(
        model=model, 
        param=control_curve, 
        predicate="LT", 
        values=[10, 0], 
        threshold=0.5,
        name="My parameter"
    )
    ```

=== "JSON"

    ```json
    {
        "My parameter": {
            "type": "ParameterThresholdParameter",
            "parameter": "Control curve",    
            "predicate": "LT", 
            "values": [10, 0], 
            "threshold": 0.5,
        }
    }
    ```

### Recorder threshold
The [pywr.parameters.RecorderThresholdParameter][] behaves as the [node threshold](#node-threshold),
but instead of a flow value, it reads the value from a recorder at the previous timestep: 

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.recorders import RollingMeanFlowNodeRecorder
    from pywr.parameters import RecorderThresholdParameter

    model = Model()
    parameter = RollingMeanFlowNodeRecorder(
        model=model, 
        node=model.nodes["Link"],
        timesteps=5,
        name="My recorder"
    )
    RecorderThresholdParameter(
        model=model, 
        param=parameter, 
        predicate="LT", 
        values=[10, 0], 
        threshold=0.5,
        name="Threshold"
    )
    ```

=== "JSON"

    ```json
    {
        "Threshold": {
            "type": "RecorderThresholdParameter",
            "parameter": "My recorder",    
            "predicate": "LT", 
            "values": [10, 0], 
            "threshold": 0.5,
        }
    }
    ```

!!!warning "Recorder type"
    This parameter only accepts recorders of [`NumpyArray*Recorder`](../recorders/memory/numpy.md) type
    or recorders implementing the `data` property. If you get an error, it is likely the recorder type you
    are using is not compatible.


## Time-based thresholds
### Current year
The [pywr.parameters.CurrentYearThresholdParameter][] works as the above parameters, but
the threshold is compared against the timestep year. In the example below the parameter 
`"Threshold"` returns `0.1` when the year is above 1967, `0.6` otherwise:


=== "Python"
    ```python
    from pywr.core import Model
    from pywr.parameters import CurrentYearThresholdParameter

    model = Model()
    CurrentYearThresholdParameter(
        model=model,
        predicate=">",
        values=[0.6, 0.1],
        threshold=1967,
        name="Threshold"
    )
    ```

=== "JSON"

    ```json
    {
        "Threshold": {
            "type": "CurrentYearThresholdParameter",
            "predicate": "GT",
            "values": [0.6, 0.1],
            "threshold": 1967,
        }
    }
    ```

### Ordinal day
The [pywr.parameters.CurrentOrdinalDayThresholdParameter][] compares the threshold against
the current [proleptic Gregorian ordinal](https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar).
The Proleptic Gregorian ordinal gives the number of days elapsed from the date 1/1/0001. You can
use this parameter to change a value when a specified date is passed. 

In the following example the parameter `"Threshold"` returns `0.1` when the date is above 1/1/1976 (ordinal is `721354`), 
`0.6` otherwise:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.parameters import CurrentOrdinalDayThresholdParameter

    model = Model()
    CurrentOrdinalDayThresholdParameter(
        model=model,
        predicate=">",
        values=[0.6, 0.1],
        threshold=721354,
        name="Threshold"
    )
    ```

=== "JSON"

    ```json
    {
        "Threshold": {
            "type": "CurrentOrdinalDayThresholdParameter",
            "predicate": "GT",
            "values": [0.6, 0.1],
            "threshold": 721354,
        }
    }
    ```

!!!info "Ordinal day calculation"
    To calculate the ordinal day, you can use the `datetime` Python library. To get the
    integer for 1/1/1976:

    ```python
    import datetime
    print(datetime.date(1976, 1, 1).toordinal())
    ```

## Multiple thresholds
These parameters can read a node's flow or a parameter value and
are of type [pywr.parameters.IndexParameter][], which means they do not return a value, 
but a zero-based index or position of the threshold being crossed. For example:

- if only one threshold is supplied, then the index returned is either 0 (when the flow or parameter 
value is above or equal to the threshold) or 1 (when below). 
- For two thresholds, the index is either 0 (when above both), 1 (in between), or 2 (below
both), and so on.

The parameter always returns the index of the first threshold the node flow is above, and
the maximum returned index is the number of supplied thresholds.

### Node
You can check a node's flow against multiple thresholds by using [pywr.parameters.MultipleThresholdIndexParameter][].
Similarly to the [`NodeThresholdParameter`](#node-threshold), the flow is the one calculated
at the previous timestep. Consider the following example where the `max_flow` of the `"input"`
node is the index of the model timestep and the thresholds are given in descending order:

=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Input
    from pywr.parameters import ArrayIndexedParameter

    model = Model()
    node = Input(
        max_flow=ArrayIndexedParameter(
            model, 
            values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        ), 
        name="input"
    )
    parameter = MultipleThresholdIndexParameter(
        model=model,
        name="My parameter",
        node=node,
        thresholds=[10, 5, 2]
    )
    ```

=== "JSON"

    ```json
    {
        "nodes": [
            {
                "type": "input",
                "name": "input",
                "max_flow": {
                    "type": "ArrayIndexedParameter",
                    "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                }
            }
        ],
        "parameters": {
            "My parameter": {
                "type": "MultipleThresholdIndexParameter",
                "node": "input",    
                "thresholds": [10, 5, 2]
            }
        }
    }
    ```

If you run the model for 13 timesteps, the parameter when recorder will return the following 
indexes:

| Timestep      | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---------------|---|---|---|---|---|---|---|---|---|----|----|----|----|
| Previous flow | 0 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8  | 9  | 10 | 11 |
| Index         | 3 | 3 | 3 | 2 | 2 | 2 | 1 | 1 | 1 | 1  | 1  | 0  | 0  |

In the first three timesteps, as no threshold is crossed, the index is `3`. When the
first threshold `2` is crossed at timestep `4`, the parameter returns `2`, the index of the last
threshold in the array. At timestep `6`, the model has crossed the second
threshold, which is reported at timestep `7`, and so on. When the flow is above
the last threshold, `0` is returned.

!!!info "Threshold type"
    For the `thresholds` you can either use constant number as shown in the
    example or any [pywr.parameters.Parameter].

The result changes if you were to sort the thresholds in reverse order (i.e. `[2, 5, 10]`):

| Timestep      | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---------------|---|---|---|---|---|---|---|---|---|----|----|----|----|
| Previous flow | 0 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8  | 9  | 10 | 11 |
| Index         | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0  | 0  | 0  | 0  |

For timesteps less than `3`, the flow is below all thresholds and index `3` is returned.
At timestep `4`, yesterday's flow is `2` and the first threshold at position `0` is crossed.
When the second threshold is crossed at timestep `7`, the parameter still returns `0` as it 
**always** returns the first threshold being triggered (i.e. `2`).

!!!success "Threshold sorting"
    You should sort the thresholds in descending order to obtain when any threshold
    is crossed as the parameter **always** returns the first threshold being crossed. 

### Parameter
The [pywr.parameters.MultipleThresholdParameterIndexParameter][] work exactly
as the [MultipleThresholdIndexParameter](#node), but it uses a parameter value
calculated for the current timestep:

=== "Python"
    ```python
    from pywr.core import Model
    import numpy as np
    from pywr.nodes import Input
    from pywr.parameters import MonthlyProfileParameter, MultipleThresholdParameterIndexParameter

    model = Model()
    parameter = MonthlyProfileParameter(
        model, 
        values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
        name="Profile"
    )
    MultipleThresholdParameterIndexParameter(
        model=model,
        name="My parameter",
        parameter=parameter,
        thresholds=[10, 5, 2]
    )
    ```

=== "JSON"
    ```json
    {
        "parameters": {
            "Profile": {
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            },
            "My parameter": {
                "type": "MultipleThresholdParameterIndexParameter",
                "parameter": "Profile",    
                "thresholds": [10, 5, 2]
            }
        }
    }
    ```
