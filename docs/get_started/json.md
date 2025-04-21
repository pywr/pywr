# Build model in JSON format
JSON (JavaScript Object Notation) is a human-readable and widely used file format that is used by Pywr
to flexibly build and run models. This format offers several advantages:

- You can define nodes, parameters, and edges without writing Python code.
- It allows non-developers or domain experts to review or even edit the model without writing Python.
- It is useful for rapid prototyping and for stakeholders who aren’t coders.
- Tools can generate or modify JSON programmatically (e.g. if you’re building many variations of a model).
- You can keep your model structure/data separate from your Python logic.
- Since JSON isn’t Python-specific, other systems (e.g., web apps or external tools) can generate or analyze models.

## File structure
The overall structure of the model is given below. A description of the contents of each of the
first level items is given in the following sections. The most important are the `nodes` and `edges` sections.

```json
{
    "metadata": {},
    "timestepper": {},
    "solver": {},
    "nodes": {},
    "edges": {},
    "parameters": {},
    "tables": {},
    "recorders": {}
}
```


### Metadata
The metadata section includes information about the model as key-value pairs. 
It is expected as a minimum to include a `title` and `description` and may 
additionally include keys other such as `author`.

```json
{
  "metadata": {
    "title": "Example",
    "description": "An example for the documentation",
    "author": "John Smith",
    "minimum_version": "1.2.5"
  }
}
```

When the `minimum_version` option is provided, Pywr will check that the version of Pywr you have installed is at equal
or greater than the version you specify in the JSON file. This is optional and, when omitted, Pywr will not perform
this check.


!!!info
    The `minimum_version` option is useful, if you want to make sure a modeller runs a newer version of Pywr, for example
    when the model relies on a new feature only available in a more recent version of the software.


### Timestepper
The timestepper defines the period a model is run for and the timestep used. It corresponds directly to the 
[pywr.timestepper.Timestepper][] instance on the model. It has three properties: the start date, end date and timestep.

The example below describes a model that will run from 1st January 2016 to 31st December 2016 using a 7-day timestep.

```json
{
  "timestepper": {
    "start": "2016-01-01",
    "end": "2016-12-31",
    "timestep": 7
  }
}
```

!!! warning "Missing dates"
    If the start or end date is not contained in a timeseries used in the model, Pywr
    will raise an `Exception` and the model will not run.

#### Down-sampling
If a timeseries has got a higher time frequency (or smaller time step) than the one
you specify in the `Timestepper`, Pywr will down-sample the timeseries to a lower
frequency by calculating the average over the new period. For example the following 
timeseries with a frequency of 1 day

| 2015-1-1 | 2015-1-2 | 2015-1-3 | 2015-1-4 | 2015-1-5 | 2015-1-6 | 2015-1-7 |
|----------|----------|----------|----------|----------|----------|----------|
| 1        | 5        | 7        | 1        | 9        | 11       | 32       |

if it is resample with a frequency of 2 days, will be converted to:

| 2015-1-1 | 2015-1-3 | 2015-1-5 |
|----------|----------|----------|
| 3        | 4        | 10       |

#### Up-sampling
If the `Timestepper` frequency is lower than the original timeseries, Pywr will up-sample the 
timeseries by forward filling the values for the new period. For example the following with a
timestep of 2 days

| 2015-1-1 | 2015-1-3 | 2015-1-5 |
|----------|----------|----------|
| 3        | 4        | 10       |

if it is resample with a frequency of 1 day, will be converted to:

| 2015-1-1 | 2015-1-2 | 2015-1-3 | 2015-1-4 | 2015-1-5 | 2015-1-6 | 
|----------|----------|----------|----------|----------|----------|
| 3        | 3        | 4        | 4        | 10       | 10       |



### Solver
The solver section is optional and allow specifying the solver to use along with options to be passed to the solver. 
The only required item is the name of the solver to use.

```json
{
    "solver": {
        "name": "glpk"
    }
}
```

### Nodes
The nodes section describes the nodes in the model. Pywr implements different types of nodes 
with different properties. As a minimum a node must have two properties:
a `name` and a `type`. The `name` uniquely identifies the node into the network, whereas 
the `type` tell Pywr the node you can use (for example if it's an input or a storage node).
The properties of a node can be defined as a simple scalar value (e.g. `"cost": 10.0`) or 
as a [parameter]() which is introduced later in this manual.

!!! info
    If you assign two nodes with the same name, Pywr will raise an `Exception`. A node
    must have a unique name.

There are two fundamental types of node in Pywr: 
[pywr.core.Node][] and [pywr.nodes.Storage][] which have different properties.

#### Non-storage nodes
Any non-storage node in Pywr is based on a basic node called [pywr.nodes.Node][].
The [pywr.nodes.Node][] type and any derived node have a `min_flow`, `max_flow` and `cost` property, 
which have the following default values.

| Property | Default value |
|----------|---------------|
| min_flow | 0             |
| max_flow | Infinite      |
| cost     | 0             |

This is an example of an [pywr.nodes.Input][] being used to represent a groundwater source: 

```json
{
    "nodes": [
        {
            "name": "groundwater",
            "type": "input",
            "max_flow": 23.0,
            "cost": 10.0
        }
    ]
}
```

!!!info "How to set a Node type"
    The value you set in type is the name of the Python class that identifies the node. This is 
    is case-insensitive. So both `input` and `Input` values are accepted. 

The node properties can also be set by using a [pywr.parameters.Parameter][]. A parameter implements
a specific logic and returns a value at each time step. The code above can be rewritten 
using a [pywr.parameters.ConstantParameter][] which returns the same number at each time step:

```json
{
    "nodes": [
        {
            "name": "groundwater",
            "type": "input",
            "max_flow": {
              "type": "constant",
              "value": 23.0
            },
            "cost": {
              "type": "constant",
              "value": 10.0
            }
        }
    ]
}
```

!!! info  "How to set the Parameter type"
    The `type` in the `max_flow` dictionary represent the name of the Python class of the parameter. This
    is case-insensitive and you can omit the suffix `Parameter`. For example, the Python class for a constant
    parameter is `ConstantParameter` and all the following strings for the `type` are recognised:
    "constant", "Constant", "constantparameter" or "ConstantParameter"

Other parameters can also be defined. For example to assign a monthly profile to the `max_flow` property,
you can use a [pywr.parameters.MonthlyProfileParameter][], which returns one of the twelve values depending
on the simulated month:

```json
{
    "nodes": [
        {
            "name": "groundwater",
            "type": "input",
            "max_flow": {
              "type": "monthlyprofile",
              "value": [10, 10, 10, 45, 56, 57, 89, 110, 90, 11, 0, 12]
            },
            "cost": {
              "type": "constant",
              "value": 10.0
            }
        }
    ]
}
```


Other node subtypes provide additional properties; often these correspond directly to the keyword arguments of the 
class. For example, a river gauge which has a soft minimum residual flow (MRF) constraint as demonstrated below. 
The `mrf` property is the minimum residual flow required, the `mrf_cost` is the cost applied to t
hat minimum flow, and the `cost` property is the cost associated with the residual flow.

```json
{
  "nodes": [
    {
      "name": "Teddington GS",
      "type": "rivergauge",
      "mrf": 200.0,
      "cost": 0.0,
      "mrf_cost": -1000.0
    }
  ]
}
```

#### Storage nodes
The [pywr.nodes.Storage][] type have a `max_volume`, `min_volume` and `initial_volume` properties. 
The maximum and initial volumes must be specified, whereas the others have default values.

```json
{
  "nodes": [
    {
        "name": "Big Wet Lake",
        "type": "storage",
        "max_volume": 1000,
        "initial_volume": 700,
        "min_volume": 0,
        "cost": -10.0
    }
  ]
}
```

### Edges
The edges section describes the connections between nodes. As a minimum an edge is defined as a two-item 
list containing the names of the nodes to connect (given in the order corresponding to the direction of flow).
For example to connect the node named `supply` to `intermediate` use:

```json
{
  "edges": [
    ["supply", "intermediate"],
    ["intermediate", "demand"]
  ]
}
```

### Parameters
Sometimes it is convenient to define a [pywr.parameters.Parameter][] used in the model in the ``"parameters"`` 
section instead of inside a node, for instance when the parameter is needed by more than one node. In this case
the parameter is assigned a name, which must be unique:

```json
{
  "nodes": [
    {
        "name": "groundwater",
        "type": "input",
        "max_flow": "gw_flow"
    }
  ],
  "parameters": {
    "gw_flow": {
      "type": "constant",
      "value": 23.0
    }
  }
}
```

#### Non-constant parameters
Parameters can be more complicated than simple scalar values. For instance, a time varying parameter can be
defined using a monthly or daily profile which repeats each year.

```json
{
  "parameters": {
    "mrf_profile": {
      "type": "monthlyprofile",
      "values": [10, 10, 10, 10, 50, 50, 50, 50, 20, 20, 10, 10]
    }
  }
}
```

#### External data
Instead of defining the data inline using the `values` property, external data can be referenced as below. 

```json
{
  "parameters": {
    "catchment_inflow": {
      "type": "dataframe",
      "url": "data/catchmod_outputs_v2.csv",
      "column": "Flow",
      "index_col": "Date",
      "parse_dates": true
    }
  }
}
```

!!! warning
    The URL should be relative to the JSON document and *not* the current working directory.

## Run the model
A Pywr JSON document can be loaded into a [pywr.core.Model][] instance by using the `load` class-method:

```python
from pywr.model import Model
my_model = Model.load('/path/to/my_model.json')
my_model.run()
```

Once a model is loaded, you can get any reference to a model component. For example, if you need to 
get a specific node, you can use the [pywr.core.Model.nodes][] attribute in the model instance:

```python
try:
    node = my_model.nodes["River Thames"]
    print(f"max_flow: {node.max_flow}")
except KeyError:
    print("Not found")
```

!!!warning
    The `nodes` attribute is a Python iterator; if the node name is not available in the
    network, Pywr will raise a `KeyError` exception.

It is also possible to test for node and component membership using their names:

```python
assert "River Thames" in my_model.nodes
assert "Demand" in my_model.parameters
```

