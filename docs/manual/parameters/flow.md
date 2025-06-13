# Flow
This class of parameters measures a node's flow or a derived metric.

## Current flow
To get a node's flow, you can use the [pywr.parameters.FlowParameter][]. For example,
if you need to set an abstraction limit based on a reservoir release:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import FlowParameter, ConstantParameter, AggregatedParameter
    
    model = Model()
    node = Link(model=model, name="Reservoir release node")
    release = FlowParameter(model=model, node=node name="Released") 
    scaling_factor = ConstantParameter(model=model, value=0.5, name="Scaling factor")
    AggregatedParameter(
        model=model, 
        parameters=[release, scaling_factor],
        agg_func="product",
        name="Max abstraction"
    )
    ```

=== "JSON"
    ```json
    {
        "Released": {
            "type": "FlowParameter",
            "node": "Reservoir release node"
        },
        "Scaling factor": {
            "type": "ConstantParameter",
            "value": 0.5
        },
        "Max abstraction": {
            "type": "AggregatedParameter",
            "parameters": ["Released", "Scaling factor"],
            "agg_func": "product",
        }
    }
    ```

!!!warning
    This parameter provides the flow at the previous time-step as
    the solution for the current timestep is not known yet.


## Flow delay
The [pywr.parameters.FlowDelayParameter][] returns the delayed flow for a node after a given number of timesteps 
or days. For example, if you need to release water from a reservoir, but you need to account
for a three-day lag before it reaches a component downstream, you can use:

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import FlowDelayParameter
    
    model = Model()
    node = Link(model=model, name="Release")
    FlowDelayParameter(model=model, node=node, days=3, name="Release with lag")
    ```

=== "JSON"
    ```json
    {
        "Release with lag": {
            "type": "FlowDelayParameter",
            "node": "Release",
            "days": 3
        }
    }
    ```

The lag can be provided using one of the following arguments:

-  `timesteps`: this is the number of model timesteps. For example, if this is `1` and you are running the model
with a weekly timestep, then the delay equals to seven days.
- `days`: this is the number of days to delay the flow. You can use this
only if the number of days is exactly divisible by the total number of model timesteps.

When the model starts and before the model reaches the delay time, the parameter returns `0` flow by
default. If you want to change this behaviour, you can set the `initial_flow` property.
This is the flow value to return for the initial model timesteps prior to any delayed flow being available. This
value is constant across all delayed timesteps and any model scenarios.

## Rolling mean
To measure a node's mean flow of a Node for the previous `N` timesteps or days,
you can use the [pywr.parameters.RollingMeanFlowNodeParameter][]:


=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import RollingMeanFlowNodeParameter
    
    model = Model()
    node = Link(model=model, name="WTW")
    RollingMeanFlowNodeParameter(
        model=model, 
        node=node,
        days=3,
        name="Rolling mean"
    )
    ```

=== "JSON"
    ```json
    {
        "Rolling mean": {
            "type": "RollingMeanFlowNodeParameter",
            "node": "WTW",
            "days": 3
        }
    }
    ```

Similarly to the [flow delay](#flow-delay), you can express the rolling period using:

- `days` as the number of days to calculate the mean flow for. This is converted into a number of timesteps
internally provided that the timestep is a number of days.
- `timesteps` as the number of timesteps to calculate the mean flow for. If `days` is provided, then `timesteps` is ignored.

Before the rolling period is reached, the parameter returns `0` flow. If you want to
chnage the flow to use in the first timesteps, before any flows have been recorded,
you can set the `initial_flow` argument.
        
## Hydropower target
The [pywr.parameters.HydropowerTargetParameter][] calculates the flow required to generate a given hydropower production 
target `P`. It is intended to be used on a node representing a turbine where a particular production target
is required at each time-step. The parameter uses the following (hydropower) equation to calculate
the flow `q` required to produce `P`:

q = P / (C<sub>E</sub> * ρ * g * H * δ * C<sub>F</sub>)

where:

- `q` is the flow needed to achieve `P`.
- `P` is the desired hydropower production target.
- C<sub>E</sub> is a coefficient to convert the energy unit.
- `ρ` is the water density.
- `g` is the gravitational acceleration (9.81 m s<sup>-2</sup>).
- `H` is the turbine head.
- `δ` is the turbine efficiency.
- C<sub>F</sub> is a coefficient to convert the flow unit. Use this to convert `q` from units of m<sup>3</sup> day<sup>-1</sup> to those used by the model.



=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import HydropowerTargetParameter
    
    model = Model()
    node = Link(model=model, name="WTW")
    RollingMeanFlowNodeParameter(
        model=model, 
        target=ConstantParameter(model, value=100),
        min_head=2.3,
        turbine_elevation=10.3,
        name="Power"
    )
    ```

=== "JSON"
    ```json
    {
        "HydropowerTargetParameter": {
            "type": "HydropowerTargetParameter",
            "target": 100,
            "min_head": 2.3,
            "turbine_elevation": 10.3,
        }
    }
    ```

### Available key options

| Name                      | Description                                                                                                                                                                     | Required | Default value         |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|-----------------------|
| target                    | This is the Hydropower production target and must be a parameter. Units should be in units of energy per day.                                                                   | Yes      |                       |
| turbine_elevation         | A number to set the elevation of the turbine itself. The difference between the `water_elevation` and this value gives the working head of the turbine (`H`).                   | No       | 0                     |
| water_elevation_parameter | This an optional parameter to set the elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives the working head of the turbine. | No       | 0                     |
| efficiency                | The efficiency of the turbine (`δ`).                                                                                                                                            | No       | 1                     |
| min_head                  | The minimum head for flow to occur. If the actual head is less than this value, zero flows are returned.                                                                        | No       | 0                     |
| flow_unit_conversion      | The factor (C<sub>F</sub>) used to transform the units of flow to be compatible with the equation above. This should convert flow to units of `m^3/day`                         | No       | 1                     |
| energy_unit_conversion    | The factor (C<sub>E</sub>) used to transform the units of total energy.                                                                                                         | No       | `1e-6` to return `MJ` |


!!!info "Record power"
    If you want to recorder the power, you can use one of the following recorders:
    [pywr.recorders.TotalHydroEnergyRecorder][] or [pywr.recorders.HydropowerRecorder][]