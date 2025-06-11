# Save results to files
Pywr can export the timeseries of the model results into two different formats:

- as CSV file, using the [pywr.recorders.CSVRecorder][]
- as [PyTable](http://pytables.org) HDF file, using the [pywr.recorders.TablesRecorder][]

## CSV recorder
The basic configuration for this recorder needs:

- the path to the CSV file where to save the data. This must be an existing path.
- the list of nodes to export.

For example:

```json
{
  "recorders": {
    "CSV recorder": {
      "type": "CSVRecorder",
      "url": "/path/to/file/with/results.csv",
      "nodes": ["Works", "Reservoir"]
    }
  }
}
```

The recorder exports the flow for all nodes listed in the `"nodes"` key; however, if a node is a storage node, its
volume is reordered instead. 

!!!danger "All nodes"
    If you omit the `"nodes"` property, Pywr will export the flow and storage of all nodes. This may slow down the
    model and lead to a large file if your model has a lot of nodes. 

### Data compression
If your CSV file is large, Pywr can compress the file when it is done exporting the data. This is done:

- using the `complib` option. You can use `gzip` or `bzip2` as value to compress the CSV file using the [gzip](https://docs.python.org/3/library/gzip.html)
and [bz2](https://docs.python.org/3/library/bz2.html#module-bz2) library respectively;
- setting a compression level with the `complevel` option. This is an integer from 0 to 9; 1 means fast saving time and
produces the least compression, and 9 is the slowest and produces the most compression. 0 is no compression
at all. When compression is enabled, this option defaults to `9`.

For example to compress the file with the `bz2` library at level `5` use:

```json
{
  "recorders": {
    "CSV recorder": {
      "type": "CSVRecorder",
      "url": "/path/to/file/with/results.csv",
      "nodes": ["Works", "Reservoir"],
      "complib": "bzip2",
      "complevel": 5
    }
  }
}
```

### Scenarios
If you are running scenarios, you can change the scenario to export using the `scenario_index` option. By default, this
is `0`, which means the first scenario is always exported.


For example to export two scenarios into two separate files, you can use:

```json
{
  "recorders": {
    "CSV recorder - scenario 1": {
      "type": "CSVRecorder",
      "url": "/path/to/file/with/results_1.csv",
      "nodes": ["Works", "Reservoir"],
      "scenario_index": 0
    },
    "CSV recorder - scenario 2": {
      "type": "CSVRecorder",
      "url": "/path/to/file/with/results_2.csv",
      "nodes": ["Pumping station", "Reservoir"],
      "scenario_index": 2
    }
  }
}
```

### Additional arguments
The recorder also accepts additional keyword arguments accepted by the [CSV writer](https://docs.python.org/3/library/csv.html#csv.writer)
in the Python's [csv library](https://docs.python.org/3/library/csv.html).  For full details about dialects and formatting parameters, see
the [Dialects and Formatting Parameters section](https://docs.python.org/3/library/csv.html#csv-fmt-params) on
the official Python documentation. 

For example, to change the column delimiter to space use:

```json
{
  "recorders": {
    "CSV recorder": {
      "type": "CSVRecorder",
      "url": "/path/to/file/with/results.csv",
      "nodes": ["Works", "Reservoir"],
      "delimiter": " "
    }
  }
}
```

### Read/plot the data
You can easily parse the exported file using the [Pandas library](http://pandas.pydata.org). The model timestep is saved
in the first column named `Datetime` with the Pandas' default format (`YYYY-mm-dd`). For example, to parse and
plot a timeseries of a specific node or all the data you can use:

```python
import matplotlib.pyplot as plt
import pandas as pd

# read the file and set the date as index
df = pd.read_csv("results.csv", index_col=[0], parse_dates=True)

# plot one node
df["My node"].plot()
plt.show()

# or to plot all the data 
df.plot()
plt.show()

# or to plot all the data in panels/subplots
df.plot(subplots=True)
plt.show()
```

## Table recorder
This recorder saves the model outputs to an [HDF file](https://en.wikipedia.org/wiki/Hierarchical_Data_Format). 
The data for a node is saved in a 2-dimensional array, where each row contains the outputs at each timestep, and the columns contain the results for
each model scenario. This recorder is more powerful than the CSV recorder as:

- it exports all scenarios and allows calculating statistics across multiple scenarios;
- it can export data of nodes (flow or storage) and parameter (a number for parameter's
values or an integer for parameter's indexes);
- it exports information about the scenarios as metadata;
- it can save other information such as the route tables.

### Basic configuration
The basic configuration for this recorder requires:

- the path to the HDF file where to save the data. This must be an existing path.
- the list of nodes to export.

For example to export two nodes:

```json
{
  "recorders": {
    "Table recorder": {
      "type": "TablesRecorder",
      "url": "/path/to/file/with/results.h5",
      "nodes": ["Works", "Reservoir"]
    }
  }
}
```

!!!danger "All nodes"
    If you omit the `"nodes"` property, Pywr will export the flow and storage of all nodes. This may slow down the
    model and lead to a large file if your model has a lot of nodes. 

### Export parameters
This recorder allows you to export parameter values and indexes too using the `"parameters"` key:

```json
{
  "recorders": {
    "Table recorder": {
      "type": "TablesRecorder",
      "url": "/path/to/file/with/results.h5",
      "parameters": ["My control curve", "My index parameter"]
    }
  }
}
```

### Export time
To export the timesteps, you can use the `"time"` option which accepts the HDF node or path name:

```json
{
  "recorders": {
    "Table recorder": {
      "type": "TablesRecorder",
      "url": "/path/to/file/with/results.h5",
      "nodes": ["Link"],
      "time": "/time"
    }
  }
}
```

The root identifier `/` in the string is optional. If omitted, this is automatically prepended. The final file will
contain a new table called `/time` which is an array with as many rows as the number of timesteps. Each row contains 
the following attributes: `year`, `month`, `day` and `index`. The latter represents the timestep index starting from `0`.

### Export additional information
The following options can also be provided to store additional outputs:

- `where` : this is the default path where to create the tables with the data inside the file. This defaults to `/`.
- `scenarios`: this is a string representing the path where to store the scenario table. This table will contain the following
scenario property: `name`, `start`, `stop` and `step`. See the [Scenario page](../scenarios.md) for a description of these properties.
The value of this property defaults to `/scenarios`.
- `routes_flows`: this is a string representing the path (relative to `where`) where to store the routes flow.  When this 
attribute is omitted, no additional data is saved.
- `routes` :  this is a string representing the path where to save the route tables. The value of this property defaults to `/routes`.
- `metadata`: this is a dictionary of user-defined attributes to save in `root._v_attrs`. When this 
attribute is omitted, no additional data is saved.

### Export the results and parse the file
Suppose you have the following simple model with two scenarios with two demand levels on the output node:

```python
from pywr.model import Model
from pywr.core import Scenario, ScenarioCollection
from pywr.nodes import Input, Output, Link
from pywr.parameters import ConstantScenarioParameter, MonthlyProfileParameter
from pywr.recorders import TablesRecorder
import numpy as np

model = Model(start="2016-01-01", end="2019-12-31", timestep=7)

scenario = Scenario(
    model=model,
    name="Demand", 
    size=2,
    ensemble_names=["Low demand", "High demand"]
)

# create three nodes (an input, a link, and an output)
random_data = np.random.rand(12, 1) * 10  # random input data
node_a = Input(
    model=model,
    name="A",
    max_flow=MonthlyProfileParameter(
        model=model, 
        values=random_data[:, 0]
    )
)
node_b = Link(model, name="B", cost=10.0)
# this node has two scenarios with two different demand levels
node_c = Output(
    model,
    name="C",
    max_flow=ConstantScenarioParameter(
        model=model,
        scenario=scenario,
        values=[5.0, 9.0]
    ),
    cost=-20.0,
)

# connect the nodes
node_a.connect(node_b)
node_b.connect(node_c)

# register the recorder - export all nodes 
TablesRecorder(model, h5file="./results.h5", time="/time")

# run the model
result = model.run()
```

After the model runs, the recorder will create the `results.h5` file. To parse the data, you can use the following script;
each line is commented to explain what it does and how to fetch specific data:

```python
import matplotlib.pyplot as plt
import tables
import pandas as pd

# read the file with the tables library
with tables.open_file("results.h5", mode="r") as table:
    # read the time
    time = table.get_node("/time").read()
    # print the first timestep
    t1 = time[0]
    print(t1)  # this will output (1, 0, 1, 2016)

    # unpack the tuple - t1[0] is the day / t[1] the timestep index / t[2] the month / t[3] the year
    year, _, month, day = t1
    print(year, month, day)

    # build the vector of date objects
    time_vector = pd.to_datetime(
        {k: table.get_node("/time").col(k) for k in ("year", "month", "day")}
    )
    print(time_vector)

     # get the numpy array of node A's flow; this is a 2D array
    flow_a = table.get_node("/A").read()
    print(flow_a.shape)  # (209; 2)
    print(flow_a)

    # print the scenario name
    print(table.get_node("/scenarios").read())

    # show a chart with three panels, one for each node. Each panel has got two lines (one per scenario)
    fig, axs = plt.subplots(3, 1)
    for ni, node in enumerate(["A", "B", "C"]):
        axs[ni].plot(time_vector, table.get_node(f"/{node}").read())
        axs[ni].set_title(node)

    plt.show()
```

Alternatively, you can use the utility method available in the recorder which
yields pairs of node names and Pandas' DataFrames:

```python
from pywr.recorders import TablesRecorder
import matplotlib.pyplot as plt

for node, df in TablesRecorder.generate_dataframes("./results.h5"):
    print(node, df)
    df.plot()
    plt.title(node)
    plt.show()
```