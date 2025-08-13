# Scenarios
There are some situations where you may need to perform multiple simulations of a model by varying 
one or more parameters. For example, you need to change the model demand to understand the 
point of failure, or you need to perturb your input data to run a sensitivity analysis.

Pywr implements a feature called "Scenario" that lets you achieve that. In particular, it lets you:

- run one scenario or a subsample. For example, you are running climate change scenarios (by changing the
inflow data); you can decide to run them all or just a few of them.
- run multiple scenarios. For example, you have one scenario that runs a different climate change dataset, and a second
scenario that runs a different model demand. In this case, Pywr will run all possible combinations of the
two scenarios; or you can decide to run just a subsample of one of them.

The following sections explain how to implement scenarios and how they work.

## Example
The following simple model has a reservoir fed from a catchment node with an inflow file. We create
a scenario where we change the demand on the [pywr.nodes.Output][] node:


=== "Python"
    ```python
    from pywr.core import Model, Scenario
    from pywr.domains.river import Catchment
    from pywr.nodes import Output, Storage
    from pywr.parameters import DataFrameParameter, ConstantScenarioParameter
    
    model = Model()
    scenario = Scenario(
        model=model, name="Demand", size=2, ensemble_names=["Low demand", "High demand"]
    )
    
    inflow = DataFrameParameter.load(
        model, {"url": "my_file.csv", "column": "Data", "index_col": "Timestamp"}
    )
    demand = ConstantScenarioParameter(
        model=model,
        name="Demand scenario",
        scenario=scenario,
        values=np.array([12.0, 31.0]),
    )
    
    i = Catchment(model=model, name="Catchment", flow=inflow)
    storage = Storage(model=model, max_volume=320, initial_volume_pc=1, name="Small storage", cost=-10)
    spill = Output(model=model, name="Spill", cost=100)
    demand = Output(model=model, name="Demand centre", cost=-100, max_flow=demand)
    
    i.connect(storage)
    storage.connect(demand)
    storage.connect(spill)
    ```

=== "JSON"

    ```json
      {
        "metadata": ...
        "timestepper": ...
        "scenarios": [
            {
                "name": "Demand scenario", 
                "ensemble_names": ["Low demand", "High demand"],
                "size": 2
            }
        ],
        "nodes": [
            {
                "type": "catchment",
                "name": "Catchment",
                "flow": {
                    "type": "DataFrameParameter",
                    "url": "my_file.csv", 
                    "column": "Data",
                    "index_col": "Timestamp"
                }
            },
            {
                "type": "Storage",
                "name": "Small storage",
                "cost": -10,
                "max_volume": 320,
                "initial_volume_pc": 1
            },
            {
                "type": "Output",
                "name": "Spill",
                "cost": 100
            },
            {
                "type": "Output",
                "name": "Demand centre",
                "cost": -100,
                "max_flow": {
                    "type": "ConstantScenarioParameter",
                    name="Demand",
                    "scenario": "Demand scenario",
                    "values": [12.0, 31.0]
                }
            }
    
        ],
        "edges": [
            ["Catchment", "Small storage"],
            ["Small storage", "Demand centre"],
            ["Small storage", "Spill"],
        ]
    }
    ```

To run a scenario, you need at least to implement two components:

1. Set up the scenario. This is done by using the `Scenario` class in Python or adding
a new dictionary in the `"scenario"` section of the JSON document. The scenario needs a
name and a size that tells the number of ensembles to run. In this case we are planning on running
two demand figures, therefore the size is set to `2`. You can also set the ensemble names,
however this is optional; when you provide no names, the names is a list of index (from zero to `size`-1). 
These names will appear in the recorder's data to help you identify the exported datasets
2. A parameter that supports scenarios. In this case, we assume the demand is constant and
therefore we use a [ConstantScenarioParameter](./parameters/simple.md#constant-scenario).
The number of values must equal the scenario `size` you set.

When the model runs, Pywr will use the same network structure, but, at each timestep, it
will **independently** solve two problems, one with demand set to `12` and the other set to `31`. Any recorder
you configure will store the results from the two runs. This advantage of this feature is that
Pywr will load your model and data, and build the linear programming matrix **only once**.
The process is shown in [the diagram describing the linear programme problem](key_concepts/problem.md#the-algorithm);
all problems are **not** solved in parallel.

!!!info "Memory efficiency"
    This approach runs the models more efficiently by
    taking advantage of the shared structure of the linear programme. The setup and
    time-stepping process are shared across all scenarios too, which helps 
    reducing overhead. At each time-step, the bounds and objective function
    of the linear programme are updated for each scenario and then solved again. 
    That way, there is no need to store or build separate versions of the 
    linear programme for every scenario.

    This approach will increase the memory requirement because Pywr needs 
    to simultaneously store state variables for each scenario.

## Parameters supporting scenarios
The following parameters support scenarios:

- [ConstantScenarioParameter](./parameters/simple.md#constant-scenario)
- [ConstantScenarioIndexParameter](./parameters/simple.md#constant-scenario-index)
- [ScenarioDailyProfileParameter](./parameters/profiles.md#scenario-profiles)   
- [ScenarioWeeklyProfileParameter](./parameters/profiles.md#scenario-profiles) 
- [ScenarioMonthlyProfileParameter](./parameters/profiles.md#scenario-profiles)
- [ArrayIndexedScenarioParameter](./parameters/array_based.md#array-indexed-scenario)
- [ArrayIndexedScenarioMonthlyFactorsParameter](./parameters/array_based.md#array-indexed-scenario-with-monthly-factors)
- [ScenarioWrapperParameter](./parameters/array_based.md#scenario-wrapper)

## Recorders
As explained in the [recorder pages](./recorders/memory/numpy.md) of this manual,
recorders will save the results for each scenario independently.

For numpy (timeseries)
recorders, the `to_dataframe()` method will return a `DataFrame` object with a
[`MultiIndex`](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html).
For example, if you record the demand node using a [NumpyArrayNodeRecorder](./recorders/memory/numpy.md#nodes-flow),
the method will return a `DataFrame` that will look like this: 

    Demand     Low demand High demand
    2100-01-01       12.0        31.0
    2100-01-02       12.0        31.0
    2100-01-03       12.0        31.0
    2100-01-04       12.0        31.0
    2100-01-05       12.0        31.0
    ...               ...         ...
    2101-12-27       12.0        31.0
    2101-12-28       12.0        31.0
    2101-12-29       12.0        31.0
    2101-12-30       12.0        31.0
    2101-12-31       12.0        31.0

The column index is named after the scenario and the column names are the ensemble names
you set.

If you record the output node's flow and the storage and you call the method on
the model instance, Pywr will combine all the data in one `DataFrame` object that will look like this:

    Recorder      Demand centre            Storage            
    Demand        Low demand High demand   Low demand High demand
    2100-01-01          12.0        31.0        320.0  316.058502
    2100-01-02          12.0        31.0        320.0  320.000000
    2100-01-03          12.0        31.0        320.0  320.000000
    2100-01-04          12.0        31.0        320.0  320.000000

Data are grouped by the node's name and the second level is the ensemble name.


## Slicing (subsample)
You can run a subset of the ensembles instead of running all of them using the `slice` option. This
option needs a [Python slice](https://docs.python.org/3/library/functions.html#slice):

=== "Python"
    ```python
    scenario = Scenario(
        model=model,
        name="Demand",
        size=3,
        ensemble_names=["Low demand", "Medium demand", "High demand"],
        slice=slice(0, 2),
    )
    ```
=== "JSON"
    ```json
    {
        "name": "Demand scenario", 
        "ensemble_names": ["Low demand", "Medium demand", "High demand"],
        "size": 3,
        "slice": [0, 2, null]
    },
    ```

In JSON, the values in the list (`[0, 2, null]`) are directly passed to
the `slice` object as positional arguments. For example, `slice(0, None, 2)`
(without the end option) translate to `[0, null, 2]`. The code above will run 
only the first two ensembles. To run the first and last two, you could use `slice(0, None, 2)`.

You can also set slices on multiple scenarios too:

=== "Python"
    ```python
    Scenario(
        model=model,
        name="scenario A",
        size=0,
        slice=slice(0, None, 2),
    )
    Scenario(
        model=model,
        name="scenario B",
        size=2,
        slice=slice(0, 1, 1),
    )
    ```
=== "JSON"
    ```json
    {
      "scenarios": [
        {
          "name": "scenario A",
          "size": 10,
          "slice": [0, null, 2]
        },
        {
          "name": "scenario B",
          "size": 2,
          "slice": [0, 1, 1]
        }
      ]
    }
    ```

## Custom combinations
If you want to run only specific combinations of the scenarios or
slices of scenarios, you use the `scenario_combinations` option:

=== "Python"
    ```python
    model = Model()
    
    Scenario(
        model=model,
        name="scenario A",
        size=0,
        slice=slice(0, None, 2),
    )
    Scenario(
        model=model,
        name="scenario B",
        size=2,
        slice=slice(0, 1, 1),
    )
    
    # set combinations
    model.scenarios.user_combinations = [
        [0, 0],
        [0, 1],
        [5, 1]
    ]
    ```
=== "JSON"
    ```json
    {
        "scenarios": [
            {
                "name": "scenario A",
                "size": 10,
                "slice": [0, null, 2]
            },
            {
                "name": "scenario B",
                "size": 2,
                "slice": [0, 1, 1]
            }
        ],
        "scenario_combinations": [
            [0, 0],
            [0, 1],
            [5, 1]
        ]
    }
    ```