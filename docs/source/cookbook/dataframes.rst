Using external data
-------------------

It's not always practical or desirable to store all the data required by a model in the JSON document. This is particularly true when working with large amounts of timeseries data, but is also applicable to data better stored in a tabular format.

Many parameter types support loading data from an external file using the ``"url"`` keyword. The data is loaded during the setup/reset process, before the model is run.

Supported formats
=================

Support for the following data formats is available via the `Pandas <http://pandas.pydata.org/pandas-docs/stable/io.html>`_ module.

    * Comma separated values (.csv)
    * Excel (.xls, .xlsx)
    * HDF5 (.h5)

**Warning:** When reading Excel documents formulae are supported but not re-evaluated and links are not updated; the value read is the value present when the document was last saved.

When working with large amounts of timeseries data the `HDF5 format <https://www.hdfgroup.org/why-hdf/>`_ is recommended as it has superior read speeds. Data in an HDF5 file should be in the "fixed" format; this is achieved by passing the ``format="fixed"`` argument to ``DataFrame.to_hdf``. Where data access speed is critical, users are advised to look at the ``TablesArrayParameter`` parameter instead which supports very fast access via `pytables` directly (rather than indirectly via `pandas`).

External data is read using the appropriate ``pandas.read_xxx`` function determined by the file extension (e.g. ``pandas.read_excel`` for xls/xlsx). Keywords that are not recognised by Pywr are passed on to these functions. For example, when reading timeseries data from a CSV you can parse the date strings into pandas timestamps by passing ``parse_dates=True`` (see example below).

Checksums
~~~~~~~~~

Often external boundary condition data is very large in comparison to the model definition (JSON) itself. Model definitions
might be stored in a version control system (e.g. Git) but this may not be suitable for large amounts of binary data.
Users of a model therefore might need to obtain the external data via another means. Tracking revisions of this external
data can become problematic.

To address this `DataFrameParameter` and `TablesArrayParameter` support validating external file checksums before reading
the external data. The example below shows how to define a checksum in the JSON definition of a `DataFrameParameter`.
If the local file does not match the checksum in the JSON definition a `HashMismatchError` is raised. Pywr uses
`hashlib <https://docs.python.org/3/library/hashlib.html>`_ and supports all of its algorithms.

**Note:** The example shows checksums for two different algorithms, but usually one is sufficient.

.. code-block:: javascript

    "max_flow": {
        "type": "dataframe",
        "url" : "timeseries2.csv",
        "checksum": {
            "md5": "a5c4032e2d8f5205ca99dedcfa4cd18e",
            "sha256": "0f75b3cee325d37112687d3d10596f44e0add374f4e40a1b6687912c05e65366"
        }
    }

The author of the external data will need to produce a file checksum to add the JSON definition. The following script
shows how Python can be used to calculate the checksum of a file.

.. code-block:: python

    import hashlib
    md5 = hashlib.md5()
    with open("data.h5", "rb") as f:
        for block in iter(lambda: f.read(8192), ""):
           md5.update(block)
    print(md5.hexdigest())



Examples
========

Timeseries
~~~~~~~~~~

The most common kind of data to store in an external file is timeseries data. For example, the flow used by a ``Catchment`` node.

An example dataset is given below with three columns: a timestamp (used as the index), rainfall and flow.

.. csv-table:: timeseries1.csv
   :header: "Timestamp", "Rainfall", "Flow"
   :widths: 15, 10, 10

   "1910-01-01", 0.0, 23.920
   "1910-01-02", 1.8, 22.140
   "1910-01-03", 5.2, 22.570

The parameter below references the `timeseries1.csv` file in its ``"url"``. The index column is defined by ``"index_col"`` and the data column is defined by the ``"column"`` keyword (in this case, ``"Flow"``). The ``"parse_dates"`` keyword is required in order to parse the dates from strings in the CSV file into pandas date objects.

.. code-block:: javascript

    "flow": {
        "type": "dataframe",
        "url": "timeseries1.csv",
        "index_col": "Timestamp"
        "parse_dates": true,
        "column": "Flow"
    }

If the index column of the dataframe is a timestamp the parameter will support automatic resampling, if required. For example, if the external data is on a daily timestep the model can still be run on a weekly timestep. In this case the data mean flow for the week is used. Only subsampling is currently supported (e.g. you can go from daily to monthly, but not monthly to daily).

Constants
~~~~~~~~~

Constant parameters can also load their data from an external source. This is useful when data with a common theme is stored in a table. For example, the demand for different nodes in the model.

An example is given below with the population and demand for three different cities (not real data!).

.. csv-table:: demands.csv
   :header: "City", "Population", "Demand"
   :widths: 15, 10, 10

   "Oxford", "30,294", 20.3
   "Cambridge", "28,403", 19.4
   "London", "790,930", 520.9

As in the previous example, the filename is passed to the ``"url"`` keyword. The ``"index_col"`` keyword defines which column should be used for the lookups, with the ``"index"`` keyword specifying the lookup key.

.. code-block:: javascript

   "max_flow_oxford": {
       "type": "constant",
       "url": "demands.csv",
       "index_col": "City"
       "index": "Oxford"
       "column": "Demand"
   }

Monthly profiles
~~~~~~~~~~~~~~~~

Monthly profiles can also be loaded from an external data source. Instead of passing a ``"column"`` keyword, the parameter expects the data source to have 12 columns (plus 1 for the index). The names of the columns are not important.

.. csv-table:: demands_monthly.csv
   :header: "City", "Jan", "Feb", "Mar", "...", "Dec"
   :widths: 15, 10, 10, 10, 10, 10

   "Oxford", 23.43, 25.32, 24.24, "...", 21.24
   "Cambridge", 11.23, 14.34, 13.23, "...", 12.23

.. code-block:: javascript

    "max_flow": {
        "type": "monthlyprofile",
        "url": "demands_monthly.csv",
        "index_col": "City",
        "index": "Oxford"
    }

Multi-index
~~~~~~~~~~~

Multi-indexing of dataframes is supported by passing a list to the ``"index_col"`` keyword. Both numeric and string indexes are valid.

.. csv-table:: multiindex_data.csv
    :header: "level", "node", "max_flow", "cost"
    :widths: 10, 15, 10, 10

    0,"demand1",10,-10
    0,"demand2",20,-20
    1,"demand1",100,-100
    1,"demand2",200,-200

.. code-block:: javascript

    {
        "name": "DC1",
        "type": "output",
        "max_flow": {
            "type": "constant",
            "url": "multiindex_data.csv",
            "column": "max_flow",
            "index": [0, "demand1"],
            "index_col": ["level", "node"]
        },
        "cost": {
            "type": "constant",
            "url": "multiindex_data.csv",
            "column": "cost",
            "index": [1, "demand1"],
            "index_col": ["level", "node"]
        }
    }

In the example above, *max_flow* evaluates to 10 and *cost* evaluates to -100.

Tables
======

Each time an external data source is referenced using the ``"url"`` keyword the data is reloaded from disk. If a dataset is going to be used multiple times in a model it can be defined in the ``"tables"`` section of the JSON document. In this way the data will only be loaded once. Parameters can then reference the data using the ``"table"`` keyword instead of the ``"url"`` keyword. Although the index column applied to the data must be defined in the ``"tables"`` section, the index used for each lookup can be different.

An example is given below using the `demands.csv` dataset shown previously. Two constant parameters are defined referencing data in the table.

.. code-block:: javascript

    "parameters": {
        "oxford_demand": {
            "type": "constant",
            "table": "simple_data",
            "column": "Demand",
            "index": "Oxford"
        },
        "cambridge_demand": {
            "type": "constant",
            "table": "simple_data",
            "column": "Demand",
            "index": "Cambridge"
        }
    },
    "tables": {
        "simple_data": {
            "url": "demands.csv",
            "index_col": "City"
        }
    }
