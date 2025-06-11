# Simple
The simplest parameter you can use are [pywr.parameters.ConstantParameter][], [pywr.parameters.ConstantScenarioParameter][],
and [pywr.parameters.ConstantScenarioIndexParameter][] which return values that are fixed throughout a simulation run, 
either as a single value or as a value dependent on the active scenario.

## Constant
A [pywr.parameters.ConstantParameter][] defines a single, fixed numerical value that does not change during a 
simulation run. It is the simplest way to represent a constant quantity in a Pywr model.

This is useful for defining fixed properties or operational set-points, such as:

- A fixed demand value.
- The maximum capacity of a pipe or canal.
- A constant transfer amount.
- A fixed minimum residual flow requirement.

This example defines a constant parameter named `"Demand"` with a fixed value of 25.5:

=== "Python"
    ```python
    from pywr.model import Model
    from pywr.parameters import ConstantParameter
      
    model = Model()
    urban_demand = ConstantParameter(model, name="Demand", value=25.5)
    ```

=== "JSON"

    ```json
    { 
        "Demand": {
            "type": "ConstantParameter",
            "value": 25.5
        }
    }
    ```

## Constant scenario
A [pywr.parameters.ConstantScenarioParameter][] selects a constant value from a predefined list based on the
currently active scenario. This allows you to create a single model where certain values change depending on the 
scenario being run.

In the following example, when Pywr run the `"Low demand` scenario, it will use `12` for the constant, otherwise
`31`. The number of values in the `values` list must match the scenario size.

=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model, Scenario
    from pywr.parameters import ConstantScenarioParameter
      
    model = Model()
    scenario = Scenario(
        model=model,
        name="Demand", 
        size=2,
        ensemble_names=["Low demand", "High demand"]
    )
    ConstantScenarioParameter(
        model=model,
        name="Pipe capacity",
        scenario=scenario,
        values=np.array([12.0, 31.0]),  
    )
    ```

=== "JSON"

    ```json
      {
        "scenarios": [
            {
                "name": "Demand", 
                "ensemble_names": ["Low demand", "High demand"],
                "size": 2
            },
        ],
        "parameters": {
            "Pipe capacity": {
                "type": "ConstantScenarioParameter",
                "scenario": "Demand",
                "values": [12.0, 31.0]
            }
        }
    }
    ```

## Constant scenario index
The [pywr.parameters.ConstantScenarioIndexParameter][] is similar to [pywr.parameters.ConstantScenarioParameter][], 
but it returns an index (i.e. an integer starting from `0`) instead of a floating point number for the currently active scenario. 

In the following example, when Pywr run the `"Low demand` scenario, it will use `0` as index, otherwise
`3`. The number of values in the `values` list must match the scenario size.

=== "Python"
    ```python
    import numpy as np
    from pywr.core import Model, Scenario
    from pywr.parameters import ConstantScenarioIndexParameter
      
    model = Model()
    scenario = Scenario(
        model=model,
        name="Demand", 
        size=2,
        ensemble_names=["Low demand", "High demand"]
    )
    ConstantScenarioIndexParameter(
        model=model,
        name="Index",
        scenario=scenario,
        values=np.array([1, 3]),  
    )
    ```

=== "JSON"

    ```json
      {
        "scenarios": [
            {
                "name": "Demand", 
                "ensemble_names": ["Low demand", "High demand"],
                "size": 2
            },
        ],
        "parameters": {
            "Index": {
                "type": "ConstantScenarioIndexParameter",
                "scenario": "Demand",
                "values": [1, 3]
            }
        }
    }
    ```