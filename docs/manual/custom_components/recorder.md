# Write a custom recorder
In Pywr you can implement your own recorder when you cannot implement the same logic using Pywr's built-in recorders
or a combination of them.

!!!warning "Before proceeding"
    Make sure you have read the implementation of [custom parameters](./parameters.md), as the same principles
    apply to custom recorders and will not be reiterated here.

## A simple recorder
This section explains how to implement a custom recorder that, at the end of the simulation, returns the minimum 
flow through a node. 

Here is the code; some key sections are explained at the end of the code block:

```python
from pywr.core import Model, Node
from pywr.recorders import Recorder
import numpy as np


class MinFlowNodeRecorder(Recorder):
    """
    This recorder returns min mean flow for a Node at the end of the simulation
    for each scenario.
    """

    def __init__(self, model: Model, node: Node, *args, **kwargs):
        """Initialise the recorder.
        :param model: The model instance.
        :param node: The node instance to recorder the flow of.
        """
        super().__init__(model, *args, **kwargs)
        self._node = node

    def setup(self):
        """Setup the internal variable."""
        self._values = np.zeros(len(self.model.scenarios.combinations))

    def reset(self):
        """Reset the internal variable."""
        self._values[...] = 0.0

    def after(self):
        """Update the min flow for each scenario."""
        self._values = np.minimum(self._node._flow, self._values)

    def values(self) -> np.ndarray:
        """Return the internal values."""
        return self._values

    @classmethod
    def load(cls, model: Model, data: dict):
        """Load the recorder from the data dictionary (i.e. when the
        recorder is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the recorder configuration.

        Returns
        -------
        MinFlowNodeRecorder
            The loaded class.
        """
        return cls(model, model.nodes[data.pop("nodes")], **data)


MinFlowNodeRecorder.register()

```

In pywr, a custom recorder is a class or object, in this example this is called `MinFlowNodeRecorder`. To let
pywr know this is a valid recorder, the class must inherit from the [pywr.recorders.Recorder][] class or
any existing recorder class. The class name should end with the "Recorder" suffix and follow the
[CamelCase](https://en.wikipedia.org/wiki/Camel_case) convection.


### Load the recorder from JSON using the `load()` method
Similarly to [custom parameters](./parameters.md#load-the-parameter-from-json-using-the-load-method),
you can use the `load()` method to load the recorder configuration defined in a JSON document:

```python
@classmethod
def load(cls, model: Model, data: dict):
```

From the `data` dictionary, we extract and remove the value we are interested in (i.e., `"node"`)
and we initialise the class with the mandatory arguments.

### Additional keyword arguments
It is important that, any additional argument your class does not directly use, these are passed
to the class constructor via `**kwargs`, as these options are passed to the parent class `Recorder`.
If you do not implement this mechanism, your recorder will not work as objective or constraint in an optimisation 
problem. Also, any option that handles the value aggregation between scenarios when `aggregated_value()` is
called (such as `agg_func`), will be ignored.

### The `values()` method
When you write a custom recorder, the only function
you need to **at least** implement is the `values()` method, which should return one number per
model scenario as array. In the example, the function returns the internal data stored
in the `_values` class attribute. The array has already the correct shape: it contains one
value per model scenario.

#### Temporal aggregation
If your recorder stored a timeseries, similarly to a `NumpyArray*Recorder`, or data with a different frequency,
for example annual data, you should aggregate the data temporally in the `values()` method. The best way to implement
this is the following:

```python
from typing import Callable
from pywr.core import Model
from pywr.recorders import Recorder, Aggregator

class MyRecorder(Recorder):
    
    def __init__(self, model: Model, temporal_agg_func: str | Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        
    def values(self):
        """
        Compute a value for each scenario using `temporal_agg_func`.

        Returns
        -------
        Iterable[float]
            A memory view of the values.
        """
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)
```

- In the `__init__` method accepts an argument called `temporal_agg_func`; the value of this
argument is [the same used for `agg_func`](../recorders/memory/index.md#aggregation-functions).
- In the body, create a class attribute (in the example is private and called `_temporal_aggregator`),
which initialises the [pywr.recorders.Aggregator][]. This is a class Pywr implements to
handle any kind of data aggregation.
- In the `values()` method call the `*aggregate_2d` method on the data to aggregate
based on the user's choice of `temporal_agg_func`. The function automatically
handles time data with scenarios and will return an array of the correct shape. 
Note that the function also handle `NaNs` using the `ignore_nan` class attribute of the
[pywr.recorders.Recorder][] class.

### The `aggregated_value()` method
The [pywr.recorders.Recorder][] class, your recorder should inherit from, already supports and implements data 
aggregation by scenario when the `aggregated_value()` method is called; there is no need to re-implement it.
Your recorder will automatically handle the `agg_func` and `ignore_nan` parameters.

### Events
In the above example, the data is collected using three events functions: `setup()`, `reset()`
and `after()`. Recorders also support an additional event called `finish()`, which is explained
later.

#### `setup()`
The `setup()` method is called once at the start of a model run. You can call this method
to initialise internal variables. In the example, we initialise the variable
 responsible for storing data with a numpy array of zeros. This has a size
equal to the number of model scenario.

#### `reset()`
The `reset()` method is called at the start of every model run. It is important you reset
any internal variable; otherwise your array may maintain the values from a previous run.
In the example, we set all the data back to zeros.

#### `after()`
This method is called after each timestep, when the solver is finished. This is the
part of the recorder where you are likely to collect the data. In the example, we apply
the `np.minimum` function to calculate the minimum between any existing values and the 
new flow data in `_node._flow`. The numpy function will calculate the minimum for each scenario,
therefore returning an array with the correct shape.

### `finish()`
The `finish()` method is called when the simulation is complete. In this method, you can
perform additional calculations or modifications of your data. 

## Improving performance with Cython
Converting your parameter to Cython follows the same principles described
in the [custom parameter page](./parameters.md#improving-performance-with-cython). You also
need to change the signature of the methods as follows:

```cython
from pywr.recorders cimport Recorder

cdef class MyRecorderRecorder(Recorder):

    def __init(model, *args, **kwargs):
        ...
        
    cpdef double[:] values(self) except *:
        ...
    
    cpdef setup(self):
        super().setup()
        ...
    
    cpdef reset(self):
        super().reset()
        ...
    
    cpdef before(self):
        ...
    
    cpdef after(self):
        ...
    
    cpdef finish(self):
        ...

MyRecorder.register()
```

Remember to call the method in the parent class for `setup()` and `reset()`.