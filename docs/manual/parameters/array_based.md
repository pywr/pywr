# Array based

## Indexed array
The [pywr.parameters.IndexedArrayParameter][] returns a value from an array of parameters, based on the index returned by a
[pywr.parameters.IndexParameter][].

In the example below, when the parameter `"Rule curve position"` (based on the level in the storage node) return:

- `0` (above the 76% curve), `"My parameter"` returns `0.5`
- `1` (between the 76% and 56% curve), `"My parameter"` returns `0.9`.
- `2` (between below the 56% curve), `"My parameter"` returns `1.4`.


=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import (
        ControlCurveIndexParameter, 
        IndexedArrayParameter, 
        ConstantParameter
    )

    model = Model()
    storage_node = Storage(
        model=model,
        name="reservoir",
        max_volume=100, 
        initial_volume=100
    )
    index_parameter = ControlCurveIndexParameter(
        model=model,
        name="Rule curve position",
        storage_node=storage_node,
        control_curves=[ConstantParameter(model, 0.76), ConstantParameter(model, 0.56)],
    )
    parameter = IndexedArrayParameter(
        model=model,
        index_parameter=index_parameter,
        params=[0.5, 0.9, 1.4],
        name="My parameter"
    )
    ```

=== "JSON"
    Both `params` and `parameters` keys are accepted in the JSON format.
    ```json
    {
        "My parameter": {
            "type": "IndexedArrayParameter",
            "index_parameter": {
                "type": "ControlCurveIndexParameter".
                "name": "Rule curve position",
                "storage_node": "reservoir",
                "control_curves": [0.76, 0.56],
            },
            "params": [0.5, 0.9, 1.4],
        }
    }
    ```

The number of parameters in `params` must match the maximum index returned by the `IndexParameter`.

This parameter is particularly useful, if for example, you need to implement complex, multi-zone operational policies 
where different rules apply depending on the reservoir's water level. Instead of using multiple
`ControlCurveParameter`s using the same curves but returning different values, you can couple
the `ControlCurveIndexParameter` with multiple `IndexedArrayParameter`s. This makes the model more
efficient and easy to read. Have a look at the [control curve page](control_curves.md#control-curve-index) for more
details.

## Array indexed
The [pywr.parameters.ArrayIndexedParameter][] returns a value from a given array based on the index of the 
current timestep.

In the following example, the parameter returns `10.1` at the first timestep (index #0),
`12` at the second timestep (index #1) and so on.

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ArrayIndexedParameter
    
    model = Model()
    ArrayIndexedParameter(
        model=model, 
        values=[10.1, 12, 15, 19], 
        name="My parameter"
    )
    ```

=== "JSON"
    ```json
    {
        "My parameter": {
            "type": "ArrayIndexedParameter",
            "values": [10.1, 12, 15, 19], 
        }
    }
    ```

The values in this parameter are constant across all scenarios, and the array length
must equal the number of model timesteps.

## Array indexed scenario
The [pywr.parameters.ArrayIndexedScenarioParameter][] returns a time-varying value
from an array values, based on the index of a the current timestep and scenario. 

In the example below at the first timestep (index #0) and first scenario index (index #0),
`1.0` is returned. At the second timestep (index #1) and second scenario index (index #1),
`-99` is returned.

```python
from pywr.core import Model, Scenario
from pywr.parameters import ConstantParameter, ArrayIndexedScenarioParameter

model = Model()
scenario = Scenario(
    model=model,
    name="Demand", 
    size=2,
    ensemble_names=["Low demand", "High demand"]
)
ArrayIndexedScenarioParameter(
    model=model, 
    values=[[1.0, 2.0], [5.0, -99]], 
    scenario=scenario,
    name="My parameter"
)
```

The array must be 2-dimensional, where the first dimension contains the value for a timestep index and the second dimension the value for the
scenario index.

!!! warning "Python only"
    This parameter cannot be loaded via JSON.

## Array indexed scenario with monthly factors
The [pywr.parameters.ArrayIndexedScenarioMonthlyFactorsParameter][] returns a time varying value taken
from an array indexed using the timestep index; the value is then multiplied by a factor
that changes monthly and per scenario.

In the example below the parameter returns `10*0.28` for the first scenario and month,
`10*0.14` for the second scenario and first month. At the second timestep, the month is February,
and the parameter returns `30*0.3` and `30*0.88` for the first and second scenario respectively.

=== "Python"
    ```python
    from pywr.core import Model, Scenario, Timestepper
    from pywr.parameters import ArrayIndexedScenarioMonthlyFactorsParameter
    
    model = Model()
    model.timestepper = Timestepper("2001-1-1", "2001-3-31", 31)
    scenario = Scenario(
        model=model,
        name="Demand", 
        size=2,
        ensemble_names=["Low demand", "High demand"]
    )
    
    factors = [
        # 12 factors for scenario "Low demand"
        [0.28, 0.3 , 0.72, 0.57, 0.1 , 0.24, 0.91, 0.58, 0.26, 0.79, 0.27, 0.82],
        # 12 factors for scenario "High demand"
        [0.14, 0.88, 0.15, 0.84, 0.93, 0.95, 0.91, 0.27, 0.64, 0.04, 0.76, 0.38]
    ]
    # values for three timesteps
    values = [10, 30, 129]
    ArrayIndexedScenarioMonthlyFactorsParameter(
        model=model, 
        values=values, 
        factors=factors,
        scenario=scenario,
        name="My parameter"
    )
    ```
=== "JSON"
    ```json
    {
        "My parameter": {
            "type": "ArrayIndexedScenarioMonthlyFactorsParameter",
            "values": [10, 30, 129], 
            "factors":  [
                [0.28, 0.3 , 0.72, 0.57, 0.1 , 0.24, 0.91, 0.58, 0.26, 0.79, 0.27, 0.82],
                [0.14, 0.88, 0.15, 0.84, 0.93, 0.95, 0.91, 0.27, 0.64, 0.04, 0.76, 0.38]
            ],
            "scenario": "Demand"
        }
    }
    ```

## Scenario wrapper
Some parameters implement the same version of a parameter but with support for scenarios.
For example for the [pywr.parameters.ConstantParameter][], which returns a time-independent value, Pywr
also implements the [pywr.parameters.ConstantScenarioParameter][], where the value does not change over
time but it changes based on the scenario being run.

However, other parameters (such as the control curves or interpolation parameters) have no such scenario-aware alternative and the [pywr.parameters.ScenarioWrapperParameter][]
solves this issue: this parameter uses a different parameter type depending on the scenario ensemble being modelled.

In the example below, when the model runs the first scenario, the first control curve
parameter `p1` is used, otherwise `p2` is used to assign the cost to the reservoir node.

=== "Python"
    ```python
    from pywr.core import Model, Scenario
    from pywr.nodes import Storage
    from pywr.parameters import (
        ScenarioWrapperParameter, 
        ControlCurveInterpolatedParameter,
        ConstantParameter
    )

    model = Model()
    scenario = Scenario(
        model=model,
        name="Demand",
        size=2,
        ensemble_names=["Low demand", "High demand"]
    )
    storage_node = Storage(
        model=model,
        name="reservoir",
        max_volume=100,
        initial_volume=100
    )
    p1 = ControlCurveInterpolatedParameter(
        name="CC1",
        storage_node=storage_node,
        control_curves=[
            ConstantParameter(model, 0.5),
            ConstantParameter(model, 0.3)
        ],
        values=[0.0, -5.0, -10.0, -20.0]
    )
    p2 = ControlCurveInterpolatedParameter(
        name="CC2",
        storage_node=storage_node,
        control_curves=[
            ConstantParameter(model, 0.3), 
            ConstantParameter(model, 0.1)
        ],
        values=[0.0, -5.0, -10.0, -20.0]
    )
    storage.cost = ScenarioWrapperParameter(
        model=model,
        scenario=scenario,
        parameters[p1, p2],
        name="Cost"
    )
    ```

=== "JSON"
    ```json
    {
        "CC1": {
            "type": "ControlCurveInterpolatedParameter",
            "storage_node": "reservoir",
            "control_curves": [0.5, 0.3],
            "values": [0.0, -5.0, -10.0, -20.0]
        },
        "CC2": {
            "type": "ControlCurveInterpolatedParameter",
            "storage_node": "reservoir",
            "control_curves": [0.3, 0.1],
            "values": [0.0, -5.0, -10.0, -20.0]
        },
        "Cost": {
            "type": "ScenarioWrapperParameter",
            "scenario": "Demand",
            "parameters": ["CC1,", "CC2"]
        }
    }
    ```

!!!warning "Number of parameters"
    The number of parameters must match the number of ensembles in the scenario otherwise
    Pywr will raise an exception.
    
