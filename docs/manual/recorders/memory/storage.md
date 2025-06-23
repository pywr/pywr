# Storage
The following recorders store data and metrics about the volume of a storage node.

## Current volume
To get the volume for the current timestep, you can use the [pywr.recorders.StorageRecorder][].  Note that this does not 
return a timeseries, but returns the latest flow(s) when the `.value()` method is called.


### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| node       | The node to record as instance or node name from JSON                                       | Yes      |               |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import StorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    StorageRecorder(
        model=model,
        name="Last volume",
        node=storage
    )
    ```

=== "JSON"
    ```json
    {
        "Last volume": {
            "type": "StorageRecorder",
            "node": "Reservoir"
        }
    }
    ```

## Minimum storage
To recorder the minimum volume in a storage node during a simulation, you can use the
[pywr.recorders.MinimumVolumeStorageRecorder][].


### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| node       | The node to record as instance or node name from JSON                                       | Yes      |               |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import MinimumVolumeStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    MinimumVolumeStorageRecorder(
        model=model,
        name="Min volume",
        node=storage
    )
    ```

=== "JSON"
    ```json
    {
        "Min volume": {
            "type": "MinimumVolumeStorageRecorder",
            "node": "Reservoir"
        }
    }
    ```

## Minimum threshold volume
To check whether the absolute volume in a storage node during a simulation
falls below a particular volume threshold, you can use the [pywr.recorders.MinimumThresholdVolumeStorageRecorder][]. 
This returns a value of `1.0`, for each scenario, when the absolute volume is less than or equal to the threshold
at any time-step during the simulation. Otherwise, it will return zero.

### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| node       | The node to record as instance or node name from JSON                                       | Yes      |               |
| threshold  | The storage threshold.                                                                      | Yes      |               |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |


### Example
=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import MinimumThresholdVolumeStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        initial_volume_pc=1,
        max_volume=50,
    )
    MinimumThresholdVolumeStorageRecorder(
        model=model,
        name="Threshold",
        threshold=10,
        node=storage
    )
    ```
=== "JSON"
    ```json
    {
        "Threshold": {
            "type": "MinimumThresholdVolumeStorageRecorder",
            "node": "Reservoir",
            "threshold": 10
        }
    }
    ```