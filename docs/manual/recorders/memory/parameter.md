# Parameters
The following recorders store data and metrics about the value of a parameter.

## Mean value
The [pywr.recorders.MeanParameterRecorder][] calculates the mean value of a
[pywr.parameters.Parameter][] over a simulation. 
An optional factor can be provided to apply a linear scaling of the values.

### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| param      | The parameter instance to use to get the value.                                             | Yes      |               |
| factor     | The scaling factor for the values of `param`.                                               | No       | 1             |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example
In the following example a parameter `"Daily river-dependant license"` set the maximum abstraction rate
based on a river flow. The `MeanParameterRecorder` recorders the mean maximum rate the model
can use over the model run:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import NodeThresholdParameter
    from pywr.recorders import MeanParameterRecorder

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
    MeanParameterRecorder(
        model=model,
        name="Mean max abstraction",
        param=parameter
    )
    ```

=== "JSON"
    ```json
    {

        "Daily river-dependant license": {
            "type": "NodeThresholdParameter",
             "node": "River",
            "predicate": "LT",
            "values": [10, 2],
            "threshold": 1000,
        },
        "Mean max abstraction": {
            "type": "MeanParameterRecorder",
            "parameter": "Daily river-dependant license"
        }
    }
    ```


## Rolling value
The [pywr.recorders.RollingWindowParameterRecorder][] works in a similar way to the
previous parameter, but it calculates the rolling value over the last N timesteps
provided in the `window` parameter. The metric to use over the rolling window can be provided in `temporal_agg_func`.

### Available key options

| Name              | Description                                                                                 | Required | Default value |
|-------------------|---------------------------------------------------------------------------------------------|----------|---------------|
| param             | The parameter instance to use to get the value.                                             | Yes      |               |
| window            | The number of timestep to use to calculate the rolling window.                              | Yes      |               |
| temporal_agg_func | The function to use to aggregate the values of `param` over the rolling window.             | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example
In the following example a parameter `"Daily river-dependant license"` set the maximum abstraction rate
based on a river flow. The `RollingWindowParameterRecorder` recorders the mean maximum rate over
a four-day window:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import NodeThresholdParameter
    from pywr.recorders import RollingWindowParameterRecorder

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
    RollingWindowParameterRecorder(
        model=model,
        name="Mean rolling abstraction",
        param=parameter,
        window=4,
        temporal_agg_func="mean"
    )
    ```

=== "JSON"
    ```json
    {

        "Daily river-dependant license": {
            "type": "NodeThresholdParameter",
             "node": "River",
            "predicate": "LT",
            "values": [10, 2],
            "threshold": 1000,
        },
        "Mean rolling abstraction": {
            "type": "RollingWindowParameterRecorder",
            "parameter": "Daily river-dependant license",
            "window": 4,
            "temporal_agg_func": "mean"
        }
    }
    ```

Note that `temporal_agg_func` could have been omitted, as its value defaults to true.

## Total value
The [pywr.recorders.TotalParameterRecorder][] calculates the total value of a [pywr.parameters.Parameter][]
over a simulation. An optional factor can be provided to apply a linear scaling of the values.

### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| param      | The parameter instance to use to get the value.                                             | Yes      |               |
| factor     | The scaling factor for the values of `param`.                                               | No       | 1             |
| integrate  | Whether to multiply the value by the time-step length in days during the summation.         | No       | False         |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example
In the following example a parameter `"Daily river-dependant license"` set the maximum abstraction rate
based on a river flow. The `TotalParameterRecorder` recorders the sum of the maximum rates
the model can use over the model run and scale them by 0.1:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import NodeThresholdParameter
    from pywr.recorders import TotalParameterRecorder

    model = Model()
    river = Link(model=model, name="River")
    parameter = ParameterThresholdParameter(
        model=model,
        node=river,
        predicate="LT",
        values=[10, 2],
        threshold=1000,
        name="Daily river-dependant license"
    )
    TotalParameterRecorder(
        model=model,
        name="Mean max abstraction",
        param=parameter,
        factor=0.1
    )
    ```

=== "JSON"
    ```json
    {

        "Daily river-dependant license": {
            "type": "NodeThresholdParameter",
             "node": "River",
            "predicate": "LT",
            "values": [10, 2],
            "threshold": 1000,
        },
        "Mean max abstraction": {
            "type": "TotalParameterRecorder",
            "parameter": "Daily river-dependant license",
            "factor": 0.1
        }
    }
    ```
