# Numpy array (timeseries)
In Pywr, NumPy array recorders leverage the [numpy library](http://numpy.org) to store timeseries data from your simulation.
They store data in memory-efficient NumPy arrays, which can be easily accessed and manipulated for post-processing 
and visualization.

This section provides an overview of the available NumPy array recorders, complete with their input parameters 
and some practical examples.

!!!info "Common implementation"
    All these recorders use the same method to read the data, which is **explained in the first section only**.
    Make sure to read this first; the remaining sections only include a brief description about the
    recorder setup.


## Node's flow
To record the time series of a node's flow, you can use the [pywr.recorders.NumpyArrayNodeRecorder][].

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | The node to record as instance or node name from JSON                                                            | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| factor            | A factor can be provided to scale the total flow (e.g. for calculating operational costs).                       | No       | 1             |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                      | No       | false         |

### Example
#### Recorder setup
In the following example, a model has two nodes where the output node demand is a
scenario parameter with two levels of demand. The recorder at the end is setup to
save the demanded flow.

=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model, Scenario
    from pywr.nodes import Input, Output
    from pywr.timestepper import Timestepper
    from pywr.parameters import MonthlyProfileParameter, ConstantScenarioParameter
    from pywr.recorders import NumpyArrayNodeRecorder
      
    model = Model(start="2016-01-01", end="2019-12-31", timestep=7)
    scenario = Scenario(
        model=model,
        name="Demand", 
        size=2,
        ensemble_names=["Low demand", "High demand"]
    )

    input_node = Input(
        model=model,
        name="Input",
        max_flow=MonthlyProfileParameter(
            model=model, 
            # random input data
            values=[3.26, 5.02, 9.89, 7.54, 3.16, 4.35, 1.51, 8.05, 0.93, 0.29, 9.49, 0.32]
        )
    )
    demand = Output(
        model,
        name="Demand",
        max_flow=ConstantScenarioParameter(
            model=model,
            scenario=scenario,
            values=[2.0, 4.0]
        ),
        cost=-20.0,
    )
    input_node.connect(demand)  

    NumpyArrayNodeRecorder(
        model=model,
        name="Demanded flow", 
        node=demand,
        temporal_agg_func="sum"
    )
    model.run()
    ```

=== "JSON"

    ```json
      {
        "metadata": {
            "title": "Test model",
            "minimum_version": "0.1"
        },
        "timestepper": {
            "start": "2016-01-01",
            "end": "2019-12-31",
            "timestep": 7
        },
        "scenarios": [
            {
                "name": "Demand", 
                "ensemble_names": ["Low demand", "High demand"],
                "size": 2
            },
        ],
        "nodes": [
            {
                "name": "Input",
                "type": "Input",
                "max_flow"= {
                    "type": "MonthlyProfileParameter",
                    "values": [3.26, 5.02, 9.89, 7.54, 3.16, 4.35, 1.51, 8.05, 0.93, 0.29, 9.49, 0.32]
                }
            },
            {
                "name": "Demand",
                "type": "Output",
                "max_flow": {
                    "type": "ConstantScenarioParameter",
                    "scenario": ,
                    "values": [2.0, 4.0]
                }
            }
        ],
        "edges": [
            ["Input", "Demand"]
        ],
        "recorders": {
            "Demanded flow": {
                "type": "NumpyArrayNodeRecorder",
                "node": "Demand",
                "temporal_agg_func": "sum"
            }
        }
    }
    ```

#### Access the data using
You can access the stored data in the following ways

##### the `data` property
This property returns a numpy array with the timeseries. Its shape is 209 x 2, where 209 is the
number of model timesteps and 2 the number of model scenarios. To access the data you can:

```python
# access the recorder instance
rec = model.recorders["Demanded flow"]

# print all data
print(rec.data)

# slice the data. Get data at first timestep for all scenarios
print(rec.data[0, :]) # >>> [2.   3.26]
# or for the first scenario only
print(rec.data[0, 0]) # >>> 2.0
```

##### the `to_dataframe()` method
To convert the internal data to a Pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html),
you can use:

```python
# access the recorder instance
import matplotlib.pyplot as plt

rec = model.recorders["Demanded flow"]
# read the data
df = rec.to_dataframe()

# plot the data for all scenarios
df.plot()
plt.show()
```

This will output the following table and show a chart with two lines:

    Demand     Low demand High demand
    2016-01-01       2.00        3.26
    2016-01-08       2.00        3.26
    2016-01-15       2.00        3.26
    2016-01-22       2.00        3.26
    2016-01-29       2.00        3.26
    ...               ...         ...
    2019-11-29       2.00        4.00
    2019-12-06       0.32        0.32
    2019-12-13       0.32        0.32
    2019-12-20       0.32        0.32
    2019-12-27       0.32        0.32
    
    [209 rows x 2 columns]

!!!info "Model"
    You can also call the method on the model instance. Pywr will collect all the results in
    one `DataFrame` object for all recorders with a `to_dataframe()` method.
    
    ```python
    print(model.to_dataframe())
    ```

##### the `values()` method
You can call `values()` on the recorder instance to aggregate the data temporally. Since
the `temporal_agg_func` option is set to `sum`, the method will return the total delivered
flow for the two scenarios:

```python
import numpy as np
rec = model.recorders["Demanded flow"]
values = rec.values()

# the method returns a memory view for efficiency
print(np.asarray(values)) # >>> [332.81 585.95]

# access the first total flow in the memory view
print(values[0]) # >>> 332.81
```

As the method returns a [Memory view](https://cython.readthedocs.io/en/stable/src/userguide/memoryviews.html), if you
want to print the full content, you need to convert the data using the `np.asarray` function.

##### the `aggregated_value()` method
To combine the data between the two scenarios, you can call the `aggregated_value()` method. Because
the `agg_func` parameter was omitted from the recorder's configuration and this defaults to `mean`, the mean
total delivered flow will be returned:

```python
rec = model.recorders["Demanded flow"]
print(rec.aggregated_value()) # >>> 459.379
```

## Storage
To record the time series of the relative or absolute volume of a storage node, you can use the [pywr.recorders.NumpyArrayStorageRecorder][].

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | The node to record as instance or node name from JSON                                                            | Yes      |               |
| proportional      | Whether to record proportional [0, 1.0] or absolute storage volumes.                                             | No       | False         |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                      | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import NumpyArrayStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    NumpyArrayStorageRecorder(
        model=model,
        name="Relative storage",
        node=storage,
        proportional=True,
        temporal_agg_func="sum"
    )
    ```

=== "JSON"
    ```json
    {
        "Relative storage": {
            "type": "NumpyArrayStorageRecorder",
            "node": "Reservoir",
            "temporal_agg_func": "sum",
            "proportional": true,
        }
    }
    ```

## Storage level
To record the time series of the level in a storage node, you can use the [pywr.recorders.NumpyArrayLevelRecorder][].

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | The node to record as instance or node name from JSON                                                            | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                      | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import NumpyArrayLevelRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    NumpyArrayLevelRecorder(
        model=model,
        name="Level",
        node=storage,
        temporal_agg_func="sum"
    )
    ```

=== "JSON"
    ```json
    {
        "Level": {
            "type": "NumpyArrayLevelRecorder",
            "node": "Reservoir",
            "temporal_agg_func": "sum",
        }
    }
    ```

## Storage area
To record the time series of the area in a storage node, you can use the [pywr.recorders.NumpyArrayAreaRecorder][].

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | The node to record as instance or node name from JSON                                                            | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                      | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import NumpyArrayAreaRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    NumpyArrayAreaRecorder(
        model=model,
        name="Area",
        node=storage,
        temporal_agg_func="sum"
    )
    ```

=== "JSON"
    ```json
    {
        "Area": {
            "type": "NumpyArrayAreaRecorder",
            "node": "Reservoir",
            "temporal_agg_func": "sum",
        }
    }
    ```
## Parameter value
To record the time series of a parameter values, you can use the [pywr.recorders.NumpyArrayParameterRecorder][].

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| param             | The parameter to record as instance or name from JSON                                                            | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                      | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ControlCurveParameter, ConstantParameter
    from pywr.recorders import NumpyArrayParameterRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    parameter = ControlCurveParameter(
        model=model,
        name="Rule curve position",
        storage_node=storage,
        values=[1.0, 45.0, 90.0],
        control_curves=[ConstantParameter(model, 0.76), ConstantParameter(model, 0.56)],
    )
    NumpyArrayParameterRecorder(
        model=model,
        name="Control curve value",
        param=parameter,
        temporal_agg_func="sum"
    )
    ```

=== "JSON"
    ```json
    {
        "Control curve value": {
            "type": "NumpyArrayParameterRecorder",
            "param": "Rule curve position",
            "temporal_agg_func": "sum"
        }
    }
    ```

## Index parameter
To record the time series of the index of a parameter index, you can use the [pywr.recorders.NumpyArrayIndexParameterRecorder][].

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| param             | The index parameter to record as instance or name from JSON                                                      | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                      | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ControlCurveIndexParameter, ConstantParameter
    from pywr.recorders import NumpyArrayIndexParameterRecorder

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
        control_curves=[ConstantParameter(model, 0.76), ConstantParameter(model, 0.56)],
    )
    NumpyArrayIndexParameterRecorder(
        model=model,
        name="Control curve value",
        param=parameter,
        temporal_agg_func="sum"
    )
    ```

=== "JSON"
    ```json
    {
        "Control curve value": {
            "type": "NumpyArrayIndexParameterRecorder",
            "param": "Rule curve index",
            "temporal_agg_func": "sum",
        }
    }
    ```

## Daily profile parameter
The [pywr.recorders.NumpyArrayDailyProfileParameterRecorder][] is designed to recorder parameters that define a daily profile (with 366
values). It records the value from the profile corresponding to the day of the year at each timestep.

### Available key options

| Name              | Description                                                                                                     | Required | Default value |
|-------------------|-----------------------------------------------------------------------------------------------------------------|----------|---------------|
| param             | The parameter to record as instance or name from JSON                                                      | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                     | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                     | No       | false         |


## Node's deficit
To record the time series of a node's deficit, you can use the [pywr.recorders.NumpyArrayNodeDeficitRecorder][].

The deficit is calculated as the difference between the value in the node's [`max_flow`](../../../../api/nodes/core/#pywr.core.Node.max_flow)
attribute and the flow allocated during the time-step in [`flow`](../../../../api/nodes/core/#pywr.core.Node.flow):

    deficit = max_flow - actual_flow

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | The node to record as instance or node name from JSON                                                            | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                      | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import NumpyArrayNodeDeficitRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    NumpyArrayNodeDeficitRecorder(
        model=model,
        name="Demand deficit",
        node=demand,
        temporal_agg_func="sum"
    )
    ```

=== "JSON"
    ```json
    {
        "Demand deficit": {
            "type": "NumpyArrayNodeDeficitRecorder",
            "node": "Demand",
            "temporal_agg_func": "sum"
        }
    }
    ```

## Node's supply ratio
To record the time series of a node's supply ratio, you can use the [pywr.recorders.NumpyArrayNodeSuppliedRatioRecorder][].

The supply ratio is calculated as the ratio of the flow allocated during the time-step in [`flow`](../../../../api/nodes/core/#pywr.core.Node.flow)
and the node's [`max_flow`](../../../../api/nodes/core/#pywr.core.Node.max_flow) attribute:

    supply_ratio = actual_flow / max_flow

If the node's `max_flow` returns zero, then the ratio is recorded as `1.0`.

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | The node to record as instance or node name from JSON                                                            | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                      | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import NumpyArrayNodeSuppliedRatioRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    NumpyArrayNodeSuppliedRatioRecorder(
        model=model,
        name="Demand supply ratio",
        node=demand,
        temporal_agg_func="sum"
    )
    ```

=== "JSON"
    ```json
    {
        "Demand supply ratio": {
            "type": "NumpyArrayNodeSuppliedRatioRecorder",
            "node": "Demand",
            "temporal_agg_func": "sum"
        }
    }
    ```

## Node's curtailment ratio
To record the time series of a node's curtailment ratio, you can use the [pywr.recorders.NumpyArrayNodeCurtailmentRatioRecorder][].

The curtailment ratio is calculated as one minus the ratio of the flow allocated during the time-step in [`flow`](../../../../api/nodes/core/#pywr.core.Node.flow)
and the [`max_flow`](../../../../api/nodes/core/#pywr.core.Node.max_flow) attribute:

    curtailment_ratio = 1 - actual_flow / max_flow

If the node's `max_flow` returns zero, then the curtailment ratio is recorded as `0.0`.

### Available key options

| Name              | Description                                                                                                      | Required | Default value |
|-------------------|------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | The node to record as instance or node name from JSON                                                            | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over time when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                      | No       | "mean"        |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                      | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import NumpyArrayNodeCurtailmentRatioRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    NumpyArrayNodeCurtailmentRatioRecorder(
        model=model,
        name="Demand curtailment ratio",
        node=demand,
        temporal_agg_func="sum"
    )
    ```

=== "JSON"
    ```json
    {
        "Demand curtailment ratio": {
            "type": "NumpyArrayNodeCurtailmentRatioRecorder",
            "node": "Demand",
            "temporal_agg_func": "sum"
        }
    }
    ```

## Hydropower production
You can use the [pywr.recorders.HydropowerRecorder][] to calculate the timeseries of the
power production using the hydropower equation:

P = q *  C<sub>F</sub> * ρ * g * H * δ  * C<sub>E</sub>

where:

- `P` is the hydropower production.
- `q` is the turbine flow.
- C<sub>F</sub> is a coefficient to convert the flow unit. Use the `flow_unit_conversion` parameter to convert `q`
    from units of m<sup>3</sup> day<sup>-1</sup> to those used by the model.
- C<sub>E</sub> is a coefficient to convert the energy unit.
- `ρ` is the water density.
- `g` is the gravitational acceleration (9.81 m s<sup>-2</sup>).
- `H` is the turbine head. If `water_elevation` is given, then the head is the difference between `water_elevation`
    and `turbine_elevation`. If `water_elevation` is not provided, then the head is simply `turbine_elevation`.
- `δ` is the turbine efficiency.

### Available key options

| Name                      | Description                                                                                                                                                                     | Required | Default value         |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|-----------------------|
| node                      | The node to record as instance or node name from JSON                                                                                                                           | Yes      |                       |
| turbine_elevation         | A number to set the elevation of the turbine itself. The difference between the `water_elevation` and this value gives the working head of the turbine (`H`).                   | No       | 0                     |
| water_elevation_parameter | This an optional parameter to set the elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives the working head of the turbine. | No       | 0                     |
| efficiency                | The efficiency of the turbine (`δ`).                                                                                                                                            | No       | 1                     |
| flow_unit_conversion      | The factor (C<sub>F</sub>) used to transform the units of flow to be compatible with the equation above. This should convert flow to units of `m^3/day`                         | No       | 1                     |
| energy_unit_conversion    | The factor (C<sub>E</sub>) used to transform the units of total energy.                                                                                                         | No       | `1e-6` to return `MJ` |
| temporal_agg_func         | An aggregation function used to aggregate over years when computing a value per scenario in the `value()` method.                                                               | No       | "mean"                |
| agg_func                  | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                                                                                     | No       | "mean"                |
| factor                    | A factor can be provided to scale the total flow (e.g. for calculating operational costs).                                                                                      | No       | 1                     |
| ignore_nan                | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                                                                                     | No       | false                 |


### Example
To recorder the power production you can use:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.recorders import HydropowerRecorder

    model = Model()
    node = Link(model=model, name="Release")
    HydropowerRecorder(
        model=model,
        node=node,
        efficiency=0.98,
        turbine_elevation=10.3,
        name="Power"
    )
    ```

=== "JSON"
    ```json
    {
        "Power": {
            "type": "HydropowerRecorder",
            "node": "Release",
            "efficiency": 0.98,
            "turbine_elevation": 10.3,
        }
    }
    ```