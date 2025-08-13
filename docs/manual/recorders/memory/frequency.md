# Frequency
These recorders allow measuring frequency metrics on flows, storage and deficit.

## Flow duration curve
The [pywr.recorders.FlowDurationCurveRecorder][] calculates a flow duration curve for each scenario from a node's flow.

### Available key options

| Name                    | Description                                                                                                                                                                                                            | Required | Default value |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node                    | Node instance to record.                                                                                                                                                                                               | Yes      |               |
| percentiles             | The list of percentiles to use in the calculation of the flow duration curve. Values must be in the range 0-100.                                                                                                       | Yes      |               |
| factor                  | The scaling factor for the values of `param`.                                                                                                                                                                          | No       | 1             |
| temporal_agg_func       | An aggregation function used to aggregate the FDCs over the percentiles when computing a value per scenario in the `value()` method. This can be used to return, for example, the median exceeded flow for a scenario. | No       | "mean"        |
| agg_func                | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                                                                                                                            | No       | "mean"        |
| ignore_nan              | A flag to ignore NaN values when calling `aggregated_value()`.                                                                                                                                                         | No       | false         |

### Example
To calculate the curve for a node representing a river, you can use:

=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.recorders import FlowDurationCurveRecorder

    model = Model()
    demand = Link(model, name="River")
    FlowDurationCurveRecorder(
        model=model,
        name="Flow duration curve",
        node=demand,
        percentiles=np.arange(1, 101, 0.5),
        temporal_agg_func="mean"
    )
    ```

=== "JSON"
    ```json
    {
        "Flow duration curve": {
            "type": "FlowDurationCurveRecorder",
            "node": "River",
            "percentiles": [1, 5, 20, 40, 60, 80, 100],
            "temporal_agg_func": "mean"
        }
    }
    ```

### Access the FDC
To access the flow duration curve stored by this recorder, you can use the `fdc` property:

```python
model.recorders["Flow duration curve"].fdc
```

which returns an array of values for each `percentiles`; or you can access the `DataFrame` object using:

```python
import matplotlib.pyplot as plt

fdc = model.recorders["Flow duration curve"].to_dataframe()
fdc.plot()
plt.show()
```

The index will contain the `percentiles` and each column the flow duration curve value for each scenario.

### Seasonal curve
Pywr also implements the [pywr.recorders.SeasonalFlowDurationCurveRecorder][] to recorder the FDC
for a given season by providing a list of months using the `months` parameter:

=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.recorders import SeasonalFlowDurationCurveRecorder
    
    model = Model()
    demand = Link(model, name="River")
    SeasonalFlowDurationCurveRecorder(
        model=model,
        name="Flow duration curve",
        node=demand,
        months=[7, 8, 9],
        percentiles=np.arange(1, 101, 0.5),
        temporal_agg_func="mean"
    )
    ```

=== "JSON"
    ```json
    {
        "Flow duration curve": {
            "type": "SeasonalFlowDurationCurveRecorder",
            "node": "River",
            "percentiles": [1, 5, 20, 40, 60, 80, 100],
            "temporal_agg_func": "mean",
            "months": [7, 8, 9]
        }
    }
    ```

## Flow duration curve variation
The [pywr.recorders.FlowDurationCurveDeviationRecorder][] calculates the flow duration curves for each scenario and then
returns their deviations from an upper and lower target FDCs using the following steps.

For each percentile, the recorder calculates the difference between the flow duration curve
of a node and a user-defined upper (`upper_target_fdc`) and/or lower target (`lower_target_fdc`) curves divided
by the target.

For the upper target, the deviation for one scenario is calculated as:

    (fdc[k] - upper_target_fdc[k]) / upper_target_fdc[k]

where `k` is the percentile.

For the upper target, the deviation for one scenario is calculated as:

    (lower_target_fdc[k] - fdc[k]) / lower_target_fdc[k]

If you provide one target curve, the deviation is calculated only using the provided target. If you provide both the lower and upper
target curves, the overall deviation is the worst of the upper and lower difference.
The deviation is positive if the node's FDC is above the upper target or below the lower
target. If the FDC falls between the upper and lower targets, the deviation is zero.

If you are running scenarios in the model, by default the FDC is calculated for all scenario combinations and
the size of the targets must match the number of scenarios (see table below). If you want to calculate the curve for one
scenario only, you can use the `scenario` option; in this case the size of the targets must match the size of the scenario.

### Available key options

| Name              | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Required | Default value |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | Node instance to record.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Yes      |               |
| percentiles       | The list of percentiles to use in the calculation of the flow duration curve. Values must be in the range 0-100.                                                                                                                                                                                                                                                                                                                                                                                                               | Yes      |               |
| scenario          | The scenario instance (in Python) or name (in JSON document) to use to calculate the FDC. When provided, the FDC is calculate only using the data for a specific scenario.                                                                                                                                                                                                                                                                                                                                                     | No       | None          |
| lower_target_fdc  | The lower FDC against which the scenario FDCs are compared. When `scenario` is `None`, this can be a 1D array of size equal to `percentiles` or a 2D array where the shape is (scenario_combination_size, percentile_size). If `scenario` is given, then this must be a 2D array where the shape is (scenario_size, percentile_size). If this is not provided, then deviations from a lower target FDC are recorded as 0.0. If targets are loaded from an external file, this needs to be indexed using the percentile values. | No       | None          |
| upper_target_fdc  | Same as above, but this refers to the upper target.                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | No       | None          |
| factor            | The scaling factor for the values of `param`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | No       | 1             |
| temporal_agg_func | An aggregation function used to aggregate the FDCs over the percentiles when computing a value per scenario in the `value()` method. This can be used to return, for example, the median exceeded flow for a scenario.                                                                                                                                                                                                                                                                                                         | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                                                                                                                                                                                                                                                                                                                                                                                                                                    | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `aggregated_value()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | No       | false         |

You will get an error:

- if the `lower_target_fdc` or `upper_target_fdc` is not provided or when the two arguments
  do not match the length of `percentiles`.
- When the first dimension of `lower_target_fdc` or `upper_target_fdc` does not match the scenario
  size (when `scenario` is given) or the model scenario combinations.

### Example
In the example below, we calculate the deviation from a lower target:

=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import FlowDurationCurveDeviationRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    FlowDurationCurveDeviationRecorder(
        model=model,
        name="FDC deviation",
        node=demand,
        percentiles=np.array([1, 5, 20, 40, 60, 80, 100]),
        lower_target_fdc=np.array([100, 80, 56, 51, 43, 23, 12]),
        temporal_agg_func="mean"
    )
    ```

=== "JSON"
    ```json
    {
        "FDC deviation": {
            "type": "FlowDurationCurveDeviationRecorder",
            "node": "Demand",
            "percentiles": [1, 5, 20, 40, 60, 80, 100],
            "lower_target_fdc": [100, 80, 56, 51, 43, 23, 12],
            "temporal_agg_func": "mean"
        }
    }
    ```

You can access the deviations using the `.to_dataframe()` method.

## Storage duration curve
To calculate the curve on a storage node, you can use the [pywr.recorders.StorageDurationCurveRecorder][] which works
exactly as the [node FDC](#flow-duration-curve):

=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import StorageDurationCurveRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        max_volume=500,
        cost=-20.0,
        initial_volume_pc=0.8
    )
    StorageDurationCurveRecorder(
        model=model,
        name="FDC storage",
        node=storage,
        percentiles=np.arange(1, 101, 0.5),
        temporal_agg_func="mean"
    )
    ```

=== "JSON"
    ```json
    {
        "FDC storage": {
            "type": "StorageDurationCurveRecorder",
            "node": "Reservoir",
            "percentiles": [1, 5, 20, 40, 60, 80, 100],
            "temporal_agg_func": "mean"
        }
    }
    ```

By default, the recorder uses the absolute volume; if you want to use the relative
volume (between 0 and 1), you can set the `proportional` option to `true`.

## Deficit frequency
The [pywr.recorders.DeficitFrequencyNodeRecorder][] returns the frequency of failure to meet a node's `max_flow`.
The deficit is calculated as the difference between the value in the node's `max_flow`
attribute and the flow allocated in `flow` at each timestep:

    deficit = max_flow - actual_flow

When this is not zero, the recorder internal counter increases by one (timestep). At the end
of the run, this number is divided by the total number of timesteps to return a frequency.

### Available key options

| Name              | Description                                                                                                                                                                                                            | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | Node instance to record.                                                                                                                                                                                               | Yes      |               |
| factor            | The scaling factor for the values of `param`.                                                                                                                                                                          | No       | 1             |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                                                                                                                            | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `aggregated_value()`.                                                                                                                                                         | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import DeficitFrequencyNodeRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    DeficitFrequencyNodeRecorder(
        model=model,
        name="Demand deficit frequency",
        node=demand
    )
    ```

=== "JSON"
    ```json
    {
        "Demand deficit frequency": {
            "type": "DeficitFrequencyNodeRecorder",
            "node": "Demand"
        }
    }
    ```

## Kernel Density estimation
Pywr implements two recorders to fit the Gaussian [Kernel Density Estimation (KDE)](https://en.wikipedia.org/wiki/Kernel_density_estimation)
to a time-series of storage node's volume. These are: [pywr.recorders.GaussianKDEStorageRecorder][] and
[pywr.recorders.NormalisedGaussianKDEStorageRecorder][].

The KDE is used to estimate the probability density function of the storage
time-series to return the probability of being at or below a specified target volume. This can be used, for example, to ensure
a certain return period is met for a specific storage (i.e., emergency or dead storage) during an optimisation.

By default the KDE is fitted to the timeseries of the recorder relative storage.
The user can also specify an optional resampling method, for example, to calculate the density function of the
annual minimum storage. This relies on the Pandas aggregation methods.

### Available key options

| Name             | Description                                                                                                                                                                                                                                                                                                  | Required | Default value |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node             | Storage instance or name to record.                                                                                                                                                                                                                                                                          | Yes      |               |
| target_volume_pc | The proportional target volume for which a probability of being at or lower is estimated.                                                                                                                                                                                                                    | Yes      |               |
| resample_freq    | The resampling frequency (such as "D" for day, "M" for month, etc.) used by prior to distribution fitting. When omitted, no resampling is performed. See [this page](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases) for a list of available resampling frequencies. | No       |               |
| resample_func    | The resampling function (such as "mean", "min", etc.) used prior to distribution fitting.  See [this page](https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats) for a list of the function names you can use.                                | No       | 1             |
| use_reflection   | Whether to reflect the PDF at the upper and lower bounds. See [below](#reflection)                                                                                                                                                                                                                           | No       | True          |

!!!warning "Resampling"
    If you want to resample, you must provide both the `resample_freq` and `resample_func` parameters, otherwise
    resampling is skipped.

### Example
To calculate the KDE for a storage node to target 30% of its annual minimum volume, you can use:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import GaussianKDEStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        max_volume=500,
        cost=-20.0,
        initial_volume_pc=0.8
    )
    GaussianKDEStorageRecorder(
        model=model,
        name="KDE storage",
        node=storage,
        resample_freq="Y",
        resample_func="min",
        target_volume_pc=0.3
    )
    ```

=== "JSON"
    ```json
    {
        "KDE storage": {
            "type": "GaussianKDEStorageRecorder",
            "node": "Reservoir",
            "resample_freq": "Y",
            "resample_func": "min",
            "target_volume_pc": 0.3
        }
    }
    ```

### Access the PDF
To access the probability density function, you can use the `to_dataframe()` method:

```python
import matplotlib.pyplot as plt

pdf = model.recorders["KDE storage"].to_dataframe()
pdf.plot()
plt.show()
```

The `DataFrame` will contain the storage values between 0 and 1 in the index and the
probabilities between 0 and 1 in the values.

### Access the target probability
To access the probability of the target storage (defined in `target_volume_pc`), you can
call:

```python
probability = model.recorders["KDE storage"].aggregated_value()
print(probability)
```

You can then convert this to return period or another metric of interest.

### Reflection
By default, the KDE is reflected at the upper (100%) and lower bounds (0%). This is done to
correct the density function near its boundaries. For a detail examination, see [this page](https://aakinshin.net/posts/kde-bc-reflection/).

To disable reflection, you can set the `use_reflection` to `False`.

### Normalised KDE
The [pywr.recorders.NormalisedGaussianKDEStorageRecorder][] works in a similar way, but the
volume is normalised relative to a user-defined control curve. The data is normalised such that values of 1, 0 and
-1 align with full, at the control curve, and empty volumes respectively.

For example, to normalise the storage with respect to a flat line at 30%, you can use:

=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ConstantParameter
    from pywr.recorders import NormalisedGaussianKDEStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        max_volume=500,
        cost=-20.0,
        initial_volume_pc=0.8
    )
    NormalisedGaussianKDEStorageRecorder(
        model=model,
        name="KDE storage",
        node=storage,
        resample_freq="Y",
        parameter=ConstantParameter(model, 0.8),
        target_volume_pc=0.3
    )
    ```

=== "JSON"
    ```json
    {
        "KDE storage": {
            "type": "NormalisedGaussianKDEStorageRecorder",
            "node": "Reservoir",
            "resample_freq": "Y",
            "target_volume_pc": 0.3,
            "parameter": 0.8
        }
    }
    ```
   