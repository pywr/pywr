# Flow
The following recorders store data and metrics about the flow of a node.

## Current flow
To get the flow for the current timestep, you can use the [pywr.recorders.NodeRecorder][].  Note that this does not 
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
    from pywr.nodes import Output
    from pywr.recorders import NodeRecorder

    model = Model()
    demand = Output(model, name="Demand", max_flow=5, cost=-20.0)
    ...

    model.step()
    rec = NodeRecorder(
        model=model,
        name="Demanded flow",
        node=demand
    )
    print(rec.value())
    ```

=== "JSON"
    ```json
    {
        "Demanded flow": {
            "type": "NodeRecorder",
            "node": "Demand"
        }
    }
    ```

## Mean flow
To record the mean flow for a node at the end of the simulation for each scenario, you can use the
[pywr.recorders.MeanFlowNodeRecorder][]. A factor can also be provided to scale the mean flow. 

### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| node       | The node to record as instance or node name from JSON                                       | Yes      |               |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| factor     | A factor can be provided to scale the total flow (e.g. for calculating operational costs).  | No       | 1             |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example
If you want to calculate the mean operational costs at the end of the simulation you can use:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import MeanFlowNodeRecorder

    model = Model()
    node = Output(model, name="Demand", max_flow=5)
    MeanFlowNodeRecorder(
        model=model,
        node=node,
        factor=100,
        name="Total cost"
    )
    ```

=== "JSON"
    ```json
    {
        "Total cost": {
            "type": "MeanFlowNodeRecorder",
            "factor": 100,
            "node": "Demand"
        }
    }
    ```

## Total flow
The [pywr.recorders.TotalFlowNodeRecorder][] returns the total flow for a node at the end of the simulation for each
scenario. You can also scale the flow using a factor. 

### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| node       | The node to record as instance or node name from JSON                                       | Yes      |               |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| factor     | A factor can be provided to scale the total flow (e.g. for calculating operational costs).  | No       | 1             |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example
To calculate the total operational cost of supply:


=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import TotalFlowNodeRecorder

    model = Model()
    node = Output(model, name="Demand", max_flow=5)
    TotalFlowNodeRecorder(
        model=model,
        node=node,
        factor=100,
        name="Total cost"
    )
    ```

=== "JSON"
    ```json
    {
        "Total cost": {
            "type": "TotalFlowNodeRecorder",
            "factor": 100,
            "node": "Demand"
        }
    }
    ```

## Annual total flow
If you want to calculate the total flow in each year across a list of nodes for each scenario, you can use the 
[pywr.recorders.AnnualTotalFlowRecorder][]. 
A list of factors can also be provided to scale the total flow for each node, for example, if you want to calculate
the operational costs in each year and scenario.

### Available key options

| Name              | Description                                                                                                       | Required | Default value |
|-------------------|-------------------------------------------------------------------------------------------------------------------|----------|---------------|
| node              | The node to record as instance or node name from JSON                                                             | Yes      |               |
| temporal_agg_func | An aggregation function used to aggregate over years when computing a value per scenario in the `value()` method. | No       | "mean"        |
| agg_func          | An aggregation function used to aggregate over scenario in the `aggregated_value()` method.                       | No       | "mean"        |
| factor            | A factor can be provided to scale the total flow (e.g. for calculating operational costs).                        | No       | 1             |
| ignore_nan        | A flag to ignore NaN values when calling `value()` or `aggregated_value()`.                                       | No       | false         |

### Example
In the following example, we record the total operational cost of supply by multiply the flow of `"Demand 1"` by 100 £/Ml
and `"Demand 2"` by 12 £/Ml:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import AnnualTotalFlowRecorder

    model = Model()
    node1 = Output(model, name="Demand 1", max_flow=5)
    node2 = Output(model, name="Demand 2", max_flow=3.4)
    AnnualTotalFlowRecorder(
        model=model,
        nodes=[node1, node2],
        factors=[100, 12],
        name="Total costs"
    )
    ```

=== "JSON"
    ```json
    {
        "Total costs": {
            "type": "AnnualTotalFlowRecorder",
            "factors": [100, 12],
            "nodes": ["Demand 1, "Demand 2"]
        }
    }
    ```

The easiest way of getting the recorder's data is by using the `to_dataframe()` method:


=== "Python"
    ```python
    ...
    recorder = AnnualTotalFlowRecorder(
        model=model,
        nodes=[node1, node2],
        factors=[100, 12],
        name="Total costs"
    )
    print(recorder.to_dataframe())
    ```
=== "From JSON"
    ```python
    from pywr.core import Model
    
    model = Model.load("my_model.json")
    model.run()
    print(model.recorders["Total costs"].to_dataframe())
    ```

This will print a DataFrame structured like this:

| Year | Scenario 1 | Scenario 2 |
|------|------------|------------|
| 1961 | 11000      | 10020      |
| 1962 | 8700       | 7630       |
| .... |            |            |
| 2022 | 231000     | 210000     |

where the column names refer to the scenario names. If you do run any scenario, the table will have only one column.

Data can be aggregated for each scenario, to calculate, for example, the maximum cost using the `values()` method:

```python
print(model.recorders["Total costs"].values())
```

Which returns an array with the aggregated metric. The array has a size equal to the number of scenarios. To change
the aggregation metric, you can set the `temporal_agg_func` parameter in the recorder's configuration (this defaults to
`mean`).

To aggregate, first by year using `temporal_agg_func`, and then by scenario, using `agg_func` you can call the 
`aggregated_value()` method:

```python
print(model.recorders["Total costs"].aggregated_value())
```

## Total deficit
To recorder the total  deficit at the end of the run for a node you can use the [pywr.recorders.TotalDeficitNodeRecorder][].
The deficit is calculated as the difference between the value in the node's `max_flow` attribute and the flow allocated at
each timestep.


### Available key options

| Name       | Description                                                                                 | Required | Default value |
|------------|---------------------------------------------------------------------------------------------|----------|---------------|
| node       | The node to record as instance or node name from JSON                                       | Yes      |               |
| agg_func   | An aggregation function used to aggregate over scenario in the `aggregated_value()` method. | No       | "mean"        |
| factor     | A factor can be provided to scale the total flow (e.g. for calculating operational costs).  | No       | 1             |
| ignore_nan | A flag to ignore NaN values when calling `aggregated_value()`.                              | No       | false         |

### Example

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Output
    from pywr.recorders import TotalDeficitNodeRecorder

    model = Model()
    demand = Output(
        model,
        name="Demand",
        max_flow=5,
        cost=-20.0,
    )
    TotalDeficitNodeRecorder(
        model=model,
        name="Demand total deficit",
        node=demand
    )
    ```

=== "JSON"
    ```json
    {
        "Demand total deficit": {
            "type": "TotalDeficitNodeRecorder",
            "node": "Demand"
        }
    }
    ```