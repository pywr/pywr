# Storage

## Storage node

| What it does         | <span style="font-weight:normal;">This node storage water into the network.</span>                                 |
|----------------------|--------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node from reservoirs, service reservoirs or any other point in the network where you want to store water. |
| **Pywr class**       | [pywr.nodes.Storage][]                                                                                             |

### Available key options

| Name              | Description                                                                                                                                                                                                                                                                                                       | Required | Default value |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| initial_volume_pc | Set the current relative initial volume, between 0 and 1                                                                                                                                                                                                                                                          | No       | 0             |
| initial_volume    | Specify the initial volume in absolute terms.                                                                                                                                                                                                                                                                     | No       | 0             |
| cost              | The cost of net flow in to the storage node. I.e. a positive cost penalises increasing volume by giving a benefit to negative net flow (release), and a negative cost penalises decreasing volume by giving a benefit to positive net flow (inflow). See the [cost page](../key_concepts/costs.md#reservoirs) for more detail. | No       | 0             |
| max_volume        | The maximum volume of the storage.  This can be a constant or a Parameter.                                                                                                                                                                                                                                        | No       | 0             |
| min_volume        | The minimum volume of the storage.  When set, Pywr will not use this volume. Use this option to model, for example, dead storage.                                                                                                                                                                                 | No       | 0             |

!!!warning "Initial volume"
    1. The `initial_volume` and initial_volume_pc` are both required if `max_volume is a [pywr.parameters.Parameter][]
    because the parameter will not be evaluated at the first time-step.
    2. If both `initial_volume` and `initial_volume_pc` are given and `max_volume` is not a [pywr.parameters.Parameter][], then the absolute
    value is ignored.

### Example
This is an example of a storge node with a cost controlled by a rule curve:
```json
{
  "name": "Reservoir",
  "type": "Storage",
  "max_volume": 1000,
  "cost": {
    "type": "ControlCurve",
    "storage_node": "Reservir",
    "control_curves": [
      {
        "type": "Constant",
        "value": 0.5
      }
    ],
    "values": [-0.1, -100]
  }
}
```

## Reservoir

| What it does         | <span style="font-weight:normal;">This node is a [pywr.nodes.Storage][]  with with one control curve.</span>    |
|----------------------|-----------------------------------------------------------------------------------------------------------------|
| **Pywr class**       | [pywr.domains.river.Reservoir][]                                                                                |
    

### Available key options
On top of the options provided by [storage node section](#storage), this node has the following additional options:

| Name             | Description                                                                                                                                                                                                                      | Required | Default value |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| control_curve    | This can be a number (to assume a flat control curve) or a Parameter that returns the control curve position, as relative volume of fill for the given timestep. When omitted, the control curve defaults to 1 (full reservoir). | No       | 1             |
| above_curve_cost | The cost when the reservoir is above the control curve. When omitted, the reservoir cost defaults to `cost` and the control curve is ignored.                                                                                    | No       | -             |
| cost             | The cost when the reservoir is below the control curve.                                                                                                                                                                          | No       | 0             |

### Example
```json
{
    "name": "Reservoir",
    "type": "Reservoir",
    "control_curve": 0.4,
    "above_curve_cost": -0.1,
    "cost": -100
}
```

## Virtual storages (licenses)
These nodes are not physical storages, but conceptual ones that calculates a "virtual" storage level based on the sum 
of flows over a defined period. These storage nodes cannot be connected to any other nodes. They start
from a maximum volume, and the storage decreases based on the amount of water delivered by other nodes in the network. When
the storage over a specified time window runs out, the tracked nodes cannot deliver any flow anymore.


### Annual storage

| What it does         | <span style="font-weight:normal;">This node tracks the amount of water delivered by other nodes and resets on an annual basis using the calendar or financial year. </span> |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node to implement annual licenses, for example for abstraction points; when the annual license volume runs out, the abstraction cannot take place.                 |
| **Pywr class**       | [pywr.nodes.AnnualVirtualStorage][]                                                                                                                                         |

#### Available key options

| Name                    | Description                                                                                                                                                                   | Required | Default value |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|--------------|
| nodes                   |         List of inflow/outflow nodes that affect the storage volume.                                                                                                                  | Yes      |              |
| initial_volume_pc       | Set the initial relative initial volume, between 0 and 1                                                                                                                      | No       | 0            |
| initial_volume          | Specify the initial volume in absolute terms.                                                                                                                                 | No       | 0            |
| max_volume              | The maximum volume of the storage.  This can be a constant or a Parameter.                                                                                                    | No       | 0            |
| min_volume              | The minimum volume the storage is allowed to reach.  This can be a constant or a Parameter.                                                                                   | No       | 0            |
| reset_day               | The day of the month (1-31) to reset the volume to the initial value.                                                                                                         | No       | 1            |
| reset_month             | The month of the year (1-12) to reset the volume to the initial value.                                                                                                        | No       | 1            |
| reset_to_initial_volume | Reset the volume to the initial volume instead of maximum volume each year.                                                                                                   | No       | false        |
| factors                 | List of factors to multiply each node's flow in `nodes` by. Positive factors remove water from the storage, negative factors remove it. When omitted, the flow is not scaled. | No       | -            |

!!!warning "Initial volume"
    1. The `initial_volume` and initial_volume_pc` are both required if `max_volume is a [pywr.parameters.Parameter][]
    because the parameter will not be evaluated at the first time-step.
    2. If both `initial_volume` and `initial_volume_pc` are given and `max_volume` is not a [pywr.parameters.Parameter][], then the absolute
    value is ignored.

!!!danger "Cost"
     Although you can set the cost of flow into/out the storage, this property is not currently 
    respected (see issue [#242](https://github.com/pywr/pywr/issues/242)) and the node will raise
    an exception when initialised.

#### Example
In this example, the license tracks the flow of one node and resets using the financial year to `10`; when the
model starts, the node can abstract up to `5` in the first year:

```json
{
    "name": "Annual license",
    "type": "AnnualVirtualStorage",
    "reset_month": 4,
    "initial_volume": 5,
    "max_volume": 10,
    "nodes": ["Track flow"]
}
```


### Monthly storage

| What it does         | <span style="font-weight:normal;">This node tracks the amount of water delivered by other nodes and resets after a certain number of months. </span> |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node to implement monthly licenses.                                                                                                         |
| **Pywr class**       | [pywr.nodes.MonthlyVirtualStorage][]                                                                                                                 |


#### Available key options

| Name                    | Description                                                                                                                                                                   | Required | Default value |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| nodes                   | List of inflow/outflow nodes that affect the storage volume.                                                                                                                  | Yes      |               |
| initial_volume_pc       | Set the initial relative initial volume, between 0 and 1                                                                                                                      | No       | 0             |
| initial_volume          | Specify the initial volume in absolute terms.                                                                                                                                 | No       | 0             |
| max_volume              | The maximum volume of the storage.  This can be a constant or a Parameter.                                                                                                    | No       | 0             |
| min_volume              | The minimum volume the storage is allowed to reach.  This can be a constant or a Parameter.                                                                                   | No       | 0             |
| months                  | The number of months after which the storage volume resets.                                                                                                                   | No       | 1             |
| initial_months          | The number of months into the reset period the storages is at when the model run starts.                                                                                      | No       | 0             |
| reset_to_initial_volume | Reset the volume to the initial volume instead of maximum volume each year.                                                                                                   | No       | false         |
| factors                 | List of factors to multiply each node's flow in `nodes` by. Positive factors remove water from the storage, negative factors remove it. When omitted, the flow is not scaled. | No       | -             |


!!!warning "Initial volume"
    1. The `initial_volume` and initial_volume_pc` are both required if `max_volume is a [pywr.parameters.Parameter][]
    because the parameter will not be evaluated at the first time-step.
    2. If both `initial_volume` and `initial_volume_pc` are given and `max_volume` is not a [pywr.parameters.Parameter][], then the absolute
    value is ignored.

!!!danger "Cost"
     Although you can set the cost of flow into/out the storage, this property is not currently 
    respected (see issue [#242](https://github.com/pywr/pywr/issues/242)) and the node will raise
    an exception when initialised.

#### Example
In this example, the license tracks the flow of one node and resets every `3` months to `10`:

```json
{
    "name": "Monthly license",
    "type": "MonthlyVirtualStorage",
    "month": 3,
    "max_volume": 10,
    "nodes": ["Track flow"]
}
```


### Seasonal storage

| What it does         | <span style="font-weight:normal;">This virtual storage tracks the used volume only over certain periods. The `reset_day` and `reset_month` parameters indicate when the node starts operating and the `end_day` and `end_month` when it stops operating. For the period when the node is not operating, the volume of the node remains unchanged and the node does not apply any constraints to the model.</span> |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node to represent licences that are only enforced during specified periods.                                                                                                                                                                                                                                                                                                                              |
| **Pywr class**       | [pywr.nodes.SeasonalVirtualStorage][]                                                                                                                                                                                                                                                                                                                                                                             |


#### Available key options

| Name                    | Description                                                                                                                                                                   | Required | Default value |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| nodes                   | List of inflow/outflow nodes that affect the storage volume.                                                                                                                  | Yes      |               |
| initial_volume_pc       | Set the initial relative initial volume, between 0 and 1                                                                                                                      | No       | 0             |
| initial_volume          | Specify the initial volume in absolute terms.                                                                                                                                 | No       | 0             |
| max_volume              | The maximum volume of the storage.  This can be a constant or a Parameter.                                                                                                    | No       | 0             |
| min_volume              | The minimum volume the storage is allowed to reach.  This can be a constant or a Parameter.                                                                                   | No       | 0             |
| reset_day               | The day of the month (1-31) when the node starts operating and its volume is reset to the initial value or maximum volume.                                                    | Yes      |               |
| reset_month             | The month of the year (1-12) when the node starts operating and its volume is reset to the initial value or maximum volume.                                                   | Yes      |               |
| end_day                 | The day of the month (1-31) when the node stops operating.                                                                                                                    | No       | 31            |
| end_month               | The month of the year (1-12) when the node stops operating.                                                                                                                   | No       | 12            |
| reset_to_initial_volume | Reset the volume to the initial volume instead of maximum volume each year.                                                                                                   | No       | false         |
| factors                 | List of factors to multiply each node's flow in `nodes` by. Positive factors remove water from the storage, negative factors remove it. When omitted, the flow is not scaled. | No       | -             |

!!!warning "Initial volume"
    1. The `initial_volume` and initial_volume_pc` are both required if `max_volume is a [pywr.parameters.Parameter][]
    because the parameter will not be evaluated at the first time-step.
    2. If both `initial_volume` and `initial_volume_pc` are given and `max_volume` is not a [pywr.parameters.Parameter][], then the absolute
    value is ignored.

!!!danger "Cost"
     Although you can set the cost of flow into/out the storage, this property is not currently 
    respected (see issue [#242](https://github.com/pywr/pywr/issues/242)) and the node will raise
    an exception when initialised.


#### Example
In this example, the license is active between October and March and not active between April and September:

```json
{
    "name": "Seasonal license",
    "type": "SeasonalVirtualStorage",
    "reset_day": 1,
    "reset_month": 10,
    "end_day": 31,
    "end_month": 3,
    "max_volume": 10,
    "nodes": ["Track flow"]
}
```

### Rolling storage

| What it does         | <span style="font-weight:normal;">This node ensures that the nodes associated with the storage deliver a flow equal to the rolling mean over a number of `timesteps` or `days`. </span> |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **When is it used?** | Use this node to implement a rolling licenses over a certain fixed period.                                                                                                              |
| **Pywr class**       | [pywr.nodes.RollingVirtualStorage][]                                                                                                                                                    |


#### Available key options

| Name              | Description                                                                                                                                                                   | Required | Default value |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| nodes             | List of inflow/outflow nodes that affect the storage volume.                                                                                                                  | Yes      |               |
| initial_volume_pc | Set the initial relative initial volume, between 0 and 1                                                                                                                      | No       | 0             |
| initial_volume    | Specify the initial volume in absolute terms.                                                                                                                                 | No       | 0             |
| max_volume        | The maximum volume of the storage.  This can be a constant or a Parameter.                                                                                                    | No       | 0             |
| min_volume        | The minimum volume the storage is allowed to reach.  This can be a constant or a Parameter.                                                                                   | No       | 0             |
| timesteps         | The number of timesteps to apply to the rolling storage over.                                                                                                                 | No       | 0             |
| days              | The number of days to apply the rolling storage over. Specifying a number of days (instead of a number of timesteps) is only valid when the model runs at a daily timestep.   | No       | -             |
| factors           | List of factors to multiply each node's flow in `nodes` by. Positive factors remove water from the storage, negative factors remove it. When omitted, the flow is not scaled. | No       | -             |


!!!warning "Initial volume"
    1. The `initial_volume` and initial_volume_pc` are both required if `max_volume is a [pywr.parameters.Parameter][]
    because the parameter will not be evaluated at the first time-step.
    2. If both `initial_volume` and `initial_volume_pc` are given and `max_volume` is not a [pywr.parameters.Parameter][], then the absolute
    value is ignored.

!!!danger "Cost"
    Although you can set the cost of flow into/out the storage, this property is not currently 
    respected (see issue [#242](https://github.com/pywr/pywr/issues/242)) and the node will raise
    an exception when initialised.

#### Example
In the following example node `A` delivers flow to a `B`:

```json
{
  "nodes": [
    {
      "type": "Input",
      "name": "A",
      "max_flow": 10
    },
    {
      "type": "Output",
      "name": "B",
      "max_flow": 10,
      "cost": -10
    }
  ],
  "edges": [["A", "B"]],
  "parameters": {
    "RollingVirtualStorage": {
      "type": "RollingVirtualStorage",
      "timesteps": 3,
      "max_volume": 17,
      "initial_volume_pc": 1,
      "nodes": [
        "B"
      ]
    }
  }
}
```

The flow delivered is:

| Timestep | 1  | 2   | 3   | 4  | 5   | 6   |
|----------|----|-----|-----|----|-----|-----|
| Flow     | 10 | 7.0 | 0.0 | 10 | 7.0 | 0.0 |

Which ensures that the rolling mean flow over 3 days is `17/3=5.666`. On the third day, the solver delivers no flow as
the virtual node is empty. The usage over the three-day period is still `17`.

#### Initial utilisation
The amount of water being used in the license, which sets the amount of water the model can use, is set to the following
amount, if the initial volume of the storage (`initial_volume` or `initial_volume_pc`) is less than the maximum volume (`max_volume`):

    (max volume - initial volume) / (timesteps - 1)

This utilisation is assumed to occur equally across each timestep of the rolling period. The virtual storage is then used
until a full rolling period is completed. At this point, the available volume will be based on the previous utilisation 
of the storage during the model run. 

In the example above, if `initial_volume_pc` is set to `0.8`, then the delivered flow will be:

| Timestep | 1  | 2   | 3   | 4  | 5   | 6   |
|----------|----|-----|-----|----|-----|-----|
| Flow     | 10 | 5.3 | 1.7 | 10 | 5.3 | 1.7 |


The rolling mean flow over 3 days is still `17/3=5.666` and the usage over the three-day period is still `17`.

However, before the period starts at timestep 4, the utilisation is set to `(17 - 17*0.8)/2 = 1.7`.