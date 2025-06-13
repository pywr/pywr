# Timeseries

Before reading this section, make sure you are familiar with how Pywr
handles [tables and external data](../tables.md#timeseries).

## Dataframe parameter
Timeseries can be loaded using the [pywr.parameters.DataFrameParameter][] parameter, which
internally stored the data as [`DataFrame` objects](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
using the [Pandas library](https://pandas.pydata.org).

### Load a CSV file
If you are writing your model using a JSON document, Pywr will detect the file type you are trying to
load and use the correct Pandas' function to parse your file. To parse a CSV file, you can use:

```json
{
    "type": "dataframe",
    "url": "timeseries1.csv",
    "index_col": "Timestamp",
    "parse_dates": true,
    "column": "Flow"
}
```

A valid timeseries will be loaded from a CSV file, if you supply:

- the column where the date is, using the`"index_col"` key;
- enable data parsing with the `"parse_dates"` key; 
- provide the column with the data using the `"column"` key.

If you are initialising the parameter with Python, you will need to load the `DataFrame` object
yourself:

```python
import pandas as pd
from pywr.core import Model
from pywr.parameters import DataFrameParameter

model = Model()
df = pd.read_csv("timeseries1.csv", index_col="Timestamp", parse_dates=True)
DataFrameParameter(
    model=model,
    name="Inflow",
    dataframe=df["Flow"],
)
```

### Load an Excel file
Reading Excel spreadsheet is also supported; there is no need to parse dates as long as
the cells containing dates are set as a date type:


=== "Python"
    ```python
    import pandas as pd
    from pywr.core import Model
    from pywr.parameters import DataFrameParameter
    
    model = Model()
    df = pd.read_excel("timeseries1.xlsx")
    DataFrameParameter(
        model=model,
        name="Inflow",
        dataframe=df["Flow"],
    )
    ```

=== "JSON"
    ```json
    {
        "type": "dataframe",
        "url": "timeseries1.xlsx",
        "column": "Flow"
    }
    ```


### Load a Pandas' HDF file
File exported using the [`to_hdf`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_hdf.html)
function, can be loaded too:


=== "Python"
    ```python
    import pandas as pd
    from pywr.core import Model
    from pywr.parameters import DataFrameParameter
    
    model = Model()
    df = pd.read_hdf("timeseries1.h5")
    DataFrameParameter(
        model=model,
        name="Inflow",
        dataframe=df["Flow"],
    )
    ```

=== "JSON"
    ```json
    {
        "type": "dataframe",
        "url": "timeseries1.h5",
        "column": "Flow"
    }
    ```

Like the Excel format, you do not need to parse any date because the H5 file already stores dates that
Python can understand.

## Table parameter (PyTables)
The [pywr.parameters.TablesArrayParameter][] provides a powerful and efficient way to
load large numerical datasets into your model from a [Hierarchical Data Format, version 5 (HDF5) file](https://www.hdfgroup.org/solutions/hdf5/)
and is useful when when working with scenario-based analyses. You can store data for
multiple scenarios (e.g., climate change projections, stochastic replicates for different sites) within
the same file and instruct the parameter to select the appropriate dataset based on the active 
scenario in the simulation.

The parameter reads data from a specified table (or "dataset") within the 
HDF5 file at each simulation timestep. The entire dataset is not loaded into memory but it is read only when the data is
needed by the model. 

!!!info "Pandas vs. PyTables"
    There is a difference between reading an HDF5 file with Pandas (using `pandas.read_hdf`) and
    PyTables. Although Pandas internally uses PyTables to handle HDF5 files, Pandas provides only a high-level 
    interface around the PyTables library. Pandas manages the file handling, data type conversions, 
    and structuring of the data into a DataFrame for you. However, Pandas **always returns a `DataFrame` object
    which is loaded entirely into memory**.

    If you rely instead on PyTables directly, it allows more control over the file and its contents, it is **much more
    memory efficient**, but is also a bit more complex to write.    

### File structure
At the top level of every HDF5 file is a root group, designated by a forward slash (/). 
This root group can contain other groups and datasets. Each group, in turn, 
can also hold other groups and datasets, creating a nested, tree-like structure. 

For example, the inflow for a catchment may be stored in a group like this:

    /Inflows/My catchment

where `/Inflow` is the key group containing all your model inflows. With the 
[pywr.parameters.TablesArrayParameter][] you can access this dataset by using two options:

- `where`: this is the path where to read the dataset or group from. This always defaults to "/", but it can be changed.
- `node`: this is the name of the node or group where the inflow data is stored.

In the example above you can use `where="/Inflows` and `node="My catchment"`.

The data in the `My catchment` node must be stored as a 2D array of `np.float32`, 
`np.float64` or `np.int32`. The first dimension of the array (or number of rows)
should match the number of timesteps, whereas the second dimension (or number of
columns) corresponds to the number of scenarios to load (for example, three stochastic replicates).
The next section shows how to properly create a valid HDF file. 

!!!warning "Dataset size"
    A dataset must always be a 2D array otherwise you will get an error when Pywr
    tries loading the data. 

    The number of rows may also be larger than the number of
    timesteps you are running. For example, the dataset may contain 100 years, but you
    run 50 years only. In this scenario, you will get a `UnutilisedDataWarning` which you
    can ignore. However, if the number of rows is less than the number of timesteps, Pywr will
    raise an `IndexError` due to missing data.

### Write the data
#### Using the `h5py` library
The following example script shows how to create a H5 file with dummy data
using the [`h5py` library](https://docs.h5py.org/en/stable/index.html):

```python
import h5py
import numpy as np

from pywr.core import Model
from pywr.nodes import Input, Output
from pywr.parameters import TablesArrayParameter

# write the file. Create random data
file = "data_h5py.h5"
with h5py.File(file, "w") as hf:
    hf.create_dataset("/Inflow/Catchment 1", data=np.random.rand(365, 5), compression="gzip")
    hf.create_dataset("/Inflow/Catchment 2", data=np.random.rand(365, 5), compression="gzip")

# Create a dummy model
model = Model()
node = Input(model, max_flow=5, name="input")
output = Output(model, cost=-100, name="output")
node.connect(output)

catchment_1_data = TablesArrayParameter(
    model, h5file=file, where="/Inflow", node="Catchment 1"
)

# Setup the components
model.setup()

# Read the data
print(
    catchment_1_data.h5store.file.get_node(
        catchment_1_data.where, catchment_1_data.node
    )
)
```

- L10 creates an empty file
- L11 creates a new dataset called `"/Inflow/Catchment 1"`. The parameter `data` in the
function assigns the data. In this case, we generate random data with shape (365, 5), which means
the dataset has 365 timesteps and 5 scenarios. If you had real data, you could import the dataset
from a CSV file. Data is also compressed.
- L12 does the same but for a different dataset.
- L14-18 setup a dummy model
- L20 loads the data using the `TablesArrayParameter` for the first catchment.
- L25 setup the model (to setup the parameter).
- L28-31 prints the data. This is not something you would normally do, but it is just to
show that the reference to the data is stored within the pywr parameter.

#### Using the `table` library (PyTables)
If you require more control on how you store the data, you can use the [`table` package](https://www.pytables.org):

```python
import tables
import numpy as np
from pywr.core import Model
from pywr.nodes import Output, Input
from pywr.parameters import TablesArrayParameter

file = "data_tables.h5"
with tables.open_file(file, mode="w") as f:
    filters = tables.Filters(complevel=5, complib="blosc")

    # create the parent group
    parent_group = f.create_group(where="/", name="Inflow")

    # save the first array or dataset
    arr = f.create_carray(
        where=parent_group,
        name="Catchment 1",
        # store data as float64
        atom=tables.Float64Atom(),
        # compress the data
        filters=filters,
        # assign the d
        obj=np.random.rand(365, 5),
    )

# create a dummy model
model = Model()
node = Input(model, max_flow=5, name="input")
node.connect(Output(model, cost=-100, name="output"))

catchment_1_data = TablesArrayParameter(
    model, h5file=file, where="/Inflow", node="Catchment 1"
)

# Setup the components
model.setup()

# Read the data
print(
    catchment_1_data.h5store.file.get_node(
        catchment_1_data.where, catchment_1_data.node
    ).read()
)
```

### Read the data
The following code is an additional example of how you setup the parameter via
Python or using the JSON document. The configuration will fetch the data
from the `/Intake` dataset as `where` defaults to `/`:


=== "Python"
    ```python
    from pywr.core import Model
    from pywr.parameters import TablesArrayParameter
    
    model = Model()
    TablesArrayParameter(
        model=model,
        name="Data",
        h5file="/path/to/file.h5",
        node="Intake"
    )
    ```

=== "JSON"

    ```json
    {
        "Data": {
            "type": "TablesArrayParameter",
            "url": "/path/to/file.h5",
            "node": "Intake"
        }
    }
    ```

### Use specific scenarios
The parameter also accepts a `scenario` argument which is the scenario instance (or scenario name in JSON)
to use for the data:

```json
{
  "scenarios": [
    {
        "name": "scenario A",
        "size": 10
    }
  ],
  "parameters": {
   " flow": {
        "type": "tablesarray",
        "scenario": "scenario A",
        "checksum": {
            "md5": "0f6c65a36851c89c7c4e63ab1893554b"
        },
        "url" : "timeseries2.h5",
        "node": "timeseries2"
    }
  }
}
```

In the example above, each dataset must contain at least 10 scenarios to run. If the arrays contain more than
10 columns, Pywr will warn you that you may not be using all the data. If the number of columns
is however less than 10, Pywr will raise an `IndexError` due to missing data.

If you specify a scenario slice to run:

```json
{
  "scenarios": [
    {
        "name": "scenario A",
        "size": 400,
        "slice": [0, 1, 2, 10]
    }
  ],
  "parameters": {
   " flow": {
        "type": "tablesarray",
        "scenario": "scenario A",
        "checksum": {
            "md5": "0f6c65a36851c89c7c4e63ab1893554b"
        },
        "url" : "timeseries2.h5",
        "node": "timeseries2"
    }
  }
}
```

Pywr will only run columns `0`, `1`, `2` and `10` from the array out of 400 columns.