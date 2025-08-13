# Counter with threshold
These recorders count the number of times an index from an `IndexParameter` exceeds a 
given threshold with different frequencies.

## Timestep counter
The [pywr.recorders.TimestepCountIndexParameterRecorder][] counts the number of timesteps an
index parameter is equal or exceeds a threshold for each scenario. When the model runs on a
daily timestep, this returns the number of days.

### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| parameter  | The index parameter instance to record.                                                     | Yes      |               |
| threshold  | The threshold to compare the parameter to.                                                  | Yes      |               |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example
To recorder the number of days a control curve, defined as [pywr.parameters.ControlCurveIndexParameter][],
is crossed, you can use:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ControlCurveIndexParameter, ConstantParameter
    from pywr.recorders import TimestepCountIndexParameterRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    parameter = ControlCurveIndexParameter(
        model=model,
        name="Rule curve index",
        storage_node=storage,
        control_curves=[ConstantParameter(model, 0.76)],
    )
    TimestepCountIndexParameterRecorder(
        model=model,
        name="Control curve counter",
        threshold=1,
        parameter=parameter,
    )
    ```

=== "JSON"
    ```json
    {
        "Control curve counter": {
            "type": "TimestepCountIndexParameterRecorder",
            "parameter": "Rule curve index",
            "threshold": 1
        }
    }
    ```

To access the counter for each scenario, you can call:

```python
import numpy as np

print(np.array(model.recorders["Control curve counter"].values()))
```

## Annual counter
The [pywr.recorders.AnnualCountIndexParameterRecorder][] works similarly to the previous
recorder, but it counts the number of years.


### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| parameter  | The index parameter instance to record.                                                     | Yes      |               |
| threshold  | The threshold to compare the parameter to.                                                  | Yes      |               |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example
To recorder the number of years a control curve, defined as [pywr.parameters.ControlCurveIndexParameter][],
is crossed, you can use:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ControlCurveIndexParameter, ConstantParameter
    from pywr.recorders import AnnualCountIndexParameterRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    parameter = ControlCurveIndexParameter(
        model=model,
        name="Rule curve index",
        storage_node=storage,
        control_curves=[ConstantParameter(model, 0.76)],
    )
    AnnualCountIndexParameterRecorder(
        model=model,
        name="Control curve counter",
        threshold=1,
        param=parameter,
    )
    ```

=== "JSON"
    ```json
    {
        "Control curve counter": {
            "type": "AnnualCountIndexParameterRecorder",
            "parameter": "Rule curve index",
            "threshold": 1
        }
    }
    ```

To access the counter for each scenario, you can call:

```python
import numpy as np

print(np.array(model.recorders["Control curve counter"].values()))
```

TODO
## Annual counter for multiple parameters
The [pywr.recorders.AnnualCountIndexThresholdRecorder][] counts, for each scenario, the number of times a list of index
parameters exceeds a threshold in each year. If multiple parameters exceed the threshold in one
timestep, then the counter is only incremented once.

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| parameters        | The list index parameter instances or strings (for JSON) to record.                                              | Yes      |               |
| threshold         | The threshold to compare the parameters to.                                                                      | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `aggregated_value()`.                                                   | No       | false         |

The recorder also allows excluding specific months and including a range of dates within a calendar year
to which the parameter exceedence is counted. Both the exclusion of months and the inclusion of dates can
simultaneously be provided. The intersection of excluded months with a range of dates will result
in the day not counting any exceedences. The dates can be controlled using the following additional options:


| Name               | Description                                                                                                          | Required | Default value |
|--------------------|----------------------------------------------------------------------------------------------------------------------|----------|---------------|
| exclude_months     | Optional list of month numbers to exclude from the count.                                                            | No       | None          |
| include_from_month | Start month to specify a range of dates to include in the count. The period to count is inclusive of the start date. | No       | None          |
| include_from_day   | Same as above but this specifies the start day.                                                                      | No       | None          |
| include_to_month   | End month to specify a range of dates to include in the count. The period to count is inclusive of the end date.     | No       | None          |
| include_to_day     | Same as above but this specifies the end day.                                                                        | No       | None          |

When these options are not provided, no time filter is applied.

!!! warning "Inclusion dates"
    If you intend to use the `include_from_` options, both the `include_from_month` and `include_from_day` must be supplied, otherwise the recorder 
    will assume that this is not used and default to the 1st Jan.

    If you intend to use the `include_to_` options, both the `include_to_month` and `include_to_day` must be supplied, otherwise the recorder 
    will assume that this is not used and default to the 31st Dec.

### Example
The following example shows how to record the number of times any of two control curves are crossed. The control curves
are crossed when the [pywr.parameters.ControlCurveIndexParameter][] returns `1`, which is the value set in the
`threshold` option. The counter is only increased between 1st April and 15th August.

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ControlCurveIndexParameter, MonthlyProfileParameter
    from pywr.recorders import AnnualCountIndexThresholdRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    line_1 = MonthlyProfileParameter(
        model=model,
        values=[0.8, 0.73, 0.7, 0.62, 0.6, 0.55, 0.6, 0.65, 0.72, 0.75, 0.83, 0.8],
        name="First line"
    )
    line_2 = MonthylProfileParameter(
        model=model,
        values=[0.6, 0.52, 0.5, 0.45, 0.4, 0.35, 0.4, 0.43, 0.51, 0.55, 0.63, 0.6],
        name="Second line"
    )
    parameter1 = ControlCurveIndexParameter(
        model=model,
        name="Rule curve 1 index",
        storage_node=storage,
        control_curves=[line_1],
    )
    parameter2 = ControlCurveIndexParameter(
        model=model,
        name="Rule curve 2 index",
        storage_node=storage,
        control_curves=[line_2],
    )
    AnnualCountIndexThresholdRecorder(
        model=model,
        name="Control curve value",
        parameters=[parameter1, parameter2],
        threshold=1,
        include_from_month=4,
        include_from_day=1,
        include_to_day=15,
        include_to_month=8,
    )
    ```

=== "JSON"
    ```json
    {
        "parameters": {
            "First line": {
                "type": "MonthlyProfileParameter",
                "values": [0.8, 0.73, 0.7, 0.62, 0.6, 0.55, 0.6, 0.65, 0.72, 0.75, 0.83, 0.8]
            },
            "Second line": {
                "type": "MonthlyProfileParameter",
                "values": [0.6, 0.52, 0.5, 0.45, 0.4, 0.35, 0.4, 0.43, 0.51, 0.55, 0.63, 0.6]
            },
            "Rule curve 1 index": {
                "type": "ControlCurveIndexParameter",
                "control_curves": ["First line"],
                "storage_node": "Reservoir"
            },
            "Rule curve 2 index": {
                "type": "ControlCurveIndexParameter",
                "control_curves": ["Second line"],
                "storage_node": "Reservoir"
            }
        },
        "recorders": {
            "Control curve value": {
                "type": "AnnualCountIndexThresholdRecorder",
                "parameters": ["Rule curve 1 index", "Rule curve 2 index"],
                "threshold"=1,
                "include_from_month": 4,
                "include_from_day": 1,
                "include_to_day": 15,
                "include_to_month": 8
            }
        }
    }
    ```

### Access the data
To access the counters for each year and scenario, you can call:
```python
import numpy as np

print(model.recorders["Control curve counter"].to_dataframe())
```

This will return a `DataFrame` where the index contains the years and the columns the 
number of times the parameter indexes are exceeded. Each column refers to a different 
scenario.

To access the aggregated counter for each scenario, you can call:

```python
import numpy as np

print(np.array(model.recorders["Control curve counter"].values()))
```
The method returns the counters aggregated using the function given in `temporal_agg_func`,
which defaults to `"mean"`.
