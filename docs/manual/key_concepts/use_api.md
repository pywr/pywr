# Use API

!!! warning "Before reading this page"
    Before reading this section, make sure you are familiar with how [node](../nodes/input.md), [parameters](../parameters/index.md)
    and [recorders](../recorders/index.md) work.

The pages in this manual explain how to use most of the available Pywr components. However, 
for a comprehensive list of all available components and their options, you may need to refer to the
[API reference page](../../api/nodes/core.md) in this documentation. This page explains how to use the reference
and translate the component options from Python to the JSON format.

If you head to the [constant parameter API page](../../api/parameters/simple/constant.md#constant), and scroll down
to the `__init__` method, the table with the "Parameter" and "Description" columns tells you the parameter you
can use to initialise the parameter:
	

```python
from pywr.core import Model
from pywr.parameters import ConstantParameter

model = Model()
ConstantParameter(model=model, value=1.0, scale=0.5, name="My parameter")
```

Translating this to the JSON format is very easy. Convert each parameter to a string in the parameter dictionary
and assign the type allowed as specified in the table. The code above is converted to:

```json
{
    "My parameter": {
        "type": "ConstantParameter",
        "value": 1.0,
        "scale": 0.5
    }
}
```

Pywr will initialise the class specified in the `"type"` key and will then pass all the remaining key-value pairs
provided in the rest of the dictionary. The value in `"type"` is case-insensitive and the "Parameter" suffix may be
omitted. This would equal to the following Python code:


```python
from pywr.core import Model
import pywr.parameters

model = Model()
parameter_config = {
    "type": "ConstantParameter",
    "value": 1.0,
    "scale": 0.5
}
param_class = parameter_config.pop("type")
getattr(pywr.parameters, param_class)(model=model, **parameter_config)
```

If you take a more complex parameter accepting a parameter, node or recorder instance:
```python
from pywr.core import Model
from pywr.nodes import Storage
from pywr.parameters import ControlCurveParameter, ConstantParameter, MonthlyProfileParameter
model = Model()
line = ConstantParameter(model=model, value=0.3, name="Line")

storage_node = Storage(
    model=model,
    name="Reservoir",
    max_volume=100,
    initial_volume=100
)
values_1 = MonthlyProfileParameter(
    model=model,
    values=[7, 7, 6, 6, 7, 7, 7, 7, 6, 7, 6, 6],
    name="Rate 1"
)
values_2 = ConstantParameter(
    model=model,
    value=2,
    name="Rate 2"
)
ControlCurveParameter(
    model=model,
    name="Abstraction",
    storage_node=storage_node,
    control_curves=[line],
    parameters=[values_1, values_2]
)
```

In the JSON document you would need to replace the parameter, node or recorder instances with their names:

```json
{
    "parameters": {
        "Rate 1": {
            "type": "MonthlyProfileParameter",
            "values": [7, 7, 6, 6, 7, 7, 7, 7, 6, 7, 6, 6]
        },
        "Rate 2": {
            "type": "ConstantParameter",
            "value": 2
        },
        "Abstraction": {
            "type": "ControlCurveParameter",
            "storage_node": "Reservoir",
            "control_curves": ["Line"],
            "parameters": ["Rate 1", "Rate 2"]
        }
   }
}
```