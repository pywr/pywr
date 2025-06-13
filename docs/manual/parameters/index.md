# Introduction
Parameters are a core part of a Pywr water resource model as they can control how water is managed 
and allocated. They are dynamic components that implement specific logics and return one number depending on the
simulation timestep and scenario being processed. For example, they can return the value of a monthly profile, 
depending on the month the simulation is in; or restrict a flow based on the position of a
reservoir with respect to its rule curves.

The next sections and pages explain how to use most of the available parameters in Pywr.
For a comprehensive list of all available parameters, please refer to the
[API reference](../../api/parameters/core.md).

## Parameter types
Pywr includes a rich library of built-in parameter types to handle common modelling tasks. Some
of the most frequently used categories include:

- **constant parameters**: these represent fixed values that do not change throughout the simulation.
- **timeseries parameters**: these read data from an external files, such as CSV, Excel or HDF5 files to provide a timeseries. 
- **profile parameters**: these define a repeating pattern of values, such as a monthly or daily profile.
- **control curve parameters**: these are used to model complex reservoir operating rules, where the output of the parameter depends on the value of another variable, often the storage in a reservoir.
- **aggregated parameters**: these combine the values of other parameters. For instance, the combined inflows from several upstream catchments.
- **conditional parameters**: these return different values based on a set of conditions, allowing for "if-then-else" logic within the model.

You can also write your own parameter to implement specific logics, and this is covered in the
[Custom parameters section](../custom_parameters.md) later in this manual.

The following sections in this chapter explain how to use most of the parameters available in Pywr.
For a comprehensive list of all available parameters, please refer to the
[API reference](../../api/parameters/core.md).

## Define a parameter in JSON
There are two ways you can define a parameter in the JSON document.

### Inline definition
You can define the parameter "inline" and directly attach it to a node's property. For example, if you have a demand
centre with a constant demand of `5`, you can define the node as follows:

```json
{
  "nodes": [
    {
      "name": "Demand centre",
      "type": "Output",
      "max_flow": {
        "type": "ConstantParameter",
        "name": "Demand parameter",
        "value": 5
      }
    }
  ]
}
```

!!!info "Inline name"
    The name property is optional, but it is best practise to define the name in case there is an error
    with loading the parameter.

### In "parameters" section
If the parameter is being used multiple times in your model, it is best to define it in the "parameters" section
of the JSON file:


```json
{
  "nodes": [
    {
      "name": "Demand centre",
      "type": "Output",
      "max_flow": "Demand parameter"
    }
  ],
  "parameters": {
    "Demand parameter": {
        "type": "ConstantParameter",
        "value": 5
      }
  }
}
```

The parameter can then be referenced by its unique name (i.e. `"Demand parameter"` in the example above).

## Values vs. index parameters
Pywr implements two classes of parameters:

- **value parameters**: these return a [floating point number](https://en.wikipedia.org/wiki/Floating-point_arithmetic),
for example the capacity of a pipe.
- **index parameters**: these return an index (i.e. an integer starting from `0`) to express, for example,
the flow band number of an abstraction license or the position of a reservoir with respect to its set of control curves.
These are easily identifiable because they contain the `Indexparameter` suffix in their names. 

It is important to understand this distinction as certain types of parameters or recorders may accept only one
type of parameter.

!!!info "Implementation"
    Value parameters inherit from the [pywr.parameters.Parameter][] class and only implement the 
    class `value()` method, whereas index parameters inherit from the [pywr.parameters.IndexParameter][] class
    and implement the class `index()` method, which always return an integer. If Pywr expects an index parameter,
    but you supply a value parameter, you will get an error.