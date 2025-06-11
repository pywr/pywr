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
