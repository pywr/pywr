# Build model in Python


The simplest example has two nodes: an [pywr.nodes.Input][] which adds flow 
to the network and an [pywr.nodes.Output][] which removes flow from the network.

```python
from pywr.core import Model, Input, Output

# create a model (including an empty network)
model = Model()

# create two nodes: a supply, and a demand
supply = Input(model, name='supply')
demand = Output(model, name='demand')

# create a connection from the supply to the demand
supply.connect(demand)
```

## Set constraints
While technically valid, this model isn't very interesting because we haven't set any constraints or
costs on flows in the network.

Let's add some flow constraints to the problem:

```python
# set maximum flows
supply.max_flow = 10.0
demand.max_flow = 6.0
```

The default minimum flow for a node is zero, so we don't need to set it explicitly.

The model still doesn't do anything as it's missing costs for flow through the nodes. If the cost
of supply is less than the benefit received from satisfying demand, flow in the network 
will occur (within the models constraints).

```python
# set cost (+ve) or benefit (-ve)
supply.cost = 3.0
demand.cost = -100.0
```
## Set the start and end date
Next we need to tell the model how long to run for. As an example, we'll use a daily timestep for all of 2015.

```python
import datetime
import pandas
from pywr.core import Timestepper

model.timestepper = Timestepper(
    pandas.to_datetime('2015-01-01'),  # first day
    pandas.to_datetime('2015-12-31'),  # last day
    datetime.timedelta(1)  # interval
)
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


## Record the flow
In order to capture the output from the model we need to use a recorder, such as the 
[pywr.recorders.NumpyArrayNodeRecorder[].] Records are explained in detail in the 
[recorder](./) section of this manual; a recorder simply saves the flow at each time-step
in memory.

```python
from pywr.recorders import NumpyArrayNodeRecorder

recorder = NumpyArrayNodeRecorder(model, supply)
```

Finally we are ready to run our model:

```python
# lets get this party started!
model.run()
```

We can check the result for the first timestep by accessing the recorder's data property:

```python
scenario = 0
timestep = 0
print(recorder.data[scenario][timestep])  # prints 6.0
```

The result of this example model is trivial: the supply exceeds the demand, so the maximum 
flow at the demand is the limiting factor.

## The final script
```python
from pywr.core import Model, Input, Output
from pywr.recorders import NumpyArrayNodeRecorder
from pywr.core import Timestepper

import datetime
import pandas

# create a model (including an empty network)
model = Model()

# create two nodes: a supply, and a demand
supply = Input(model, name='supply')
demand = Output(model, name='demand')

# create a connection from the supply to the demand
supply.connect(demand)

supply.max_flow = 10.0
demand.max_flow = 6.0

# set cost (+ve) or benefit (-ve)
supply.cost = 3.0
demand.cost = -100.0

model.timestepper = Timestepper(
    pandas.to_datetime('2015-01-01'),  # first day
    pandas.to_datetime('2015-12-31'),  # last day
    datetime.timedelta(1)  # interval
)

recorder = NumpyArrayNodeRecorder(model, supply)

model.run()

scenario = 0
timestep = 0
print(recorder.data[scenario][timestep])  # prints 6.0
```