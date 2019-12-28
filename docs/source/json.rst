.. _json-model-format:

JSON model format
-----------------

Overview of document structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to creating models programmatically using the Python API, models can be described in a JSON (JavaScript Object Notation) document [#]_.

The overall structure of the model is given below. A description of the contents of each of the first level items is given in this document. The most important are the `nodes`_ and `edges`_ sections.

.. code-block:: json

    {
        "metadata": {},
        "timestepper": {},
        "solver": {},
        "nodes": {},
        "edges": {},
        "parameters": {}
    }

Some examples of JSON models can be found in the `tests/models <https://github.com/pywr/pywr/tree/master/tests/models>`_ folder.

Metadata
~~~~~~~~

The metadata section includes information about the model as key-value pairs. It is expected as a minimum to include a ``"title"`` and ``"description"`` and may additionally include keys such as ``"author"``.

.. code-block:: json

    {"metadata": {
        "title": "Example",
        "description": "An example for the documentation",
        "author": "John Smith"
    }}

Timestepper
~~~~~~~~~~~

The timestepper defines the period a model is run for and the timestep used. It corresponds directly to the :class:`pywr.core.Timestepper` instance on the model. It has three properties: the start date, end date and timestep.

The example below describes a model that will run from 1st January 2016 to 31st December 2016 using a 7 day timestep.

.. code-block:: json

    {"timestepper": {
        "start": "2016-01-01",
        "end": "2016-12-31",
        "timestep": 7
    }}

Solver
~~~~~~

The solver section contains items to be passed to the solver. The only required item is the name of the solver to use. Other items will be specific to the solver.

.. code-block:: json

    {"solver": {
        "name": "glpk"
    }}

Nodes
~~~~~

The nodes section describes the nodes in the model. As a minimum a node must have a ``name`` and a ``type``. There are two fundamental types of node in Pywr (:class:`pywr.core.Node` and :class:`pywr.core.Storage`) which have different properties.

Where a parameter can be described as a simple scalar value it is sufficient to pass the value directly (e.g. ``"cost": 10.0``). See also the `parameters`_ section for details on defining non-scalar parameters.

Non-storage nodes
=================

The ``Node`` type and it's subtypes have a ``max_flow`` and ``cost`` property, both of which have default values.

.. code-block:: json

    {"nodes": [
        {
            "name": "groundwater",
            "type": "input",
            "max_flow": 23.0,
            "cost": 10.0
        }
    ]}

In addition to the basic ``input``, ``output`` and ``link`` types, subtypes can be created by specifying the appropriate name. Some subtypes will provide additional properties; often these correspond directly to the keyword arguments of the class. For example, a river gauge which has a soft MRF constraint is demonstrated below. The ``"mrf"`` property is the minimum residual flow required, the ``"mrf_cost"`` is the cost applied to that minimum flow, and the ``"cost"`` property is the cost associated with the residual flow.

.. code-block:: json

    {"nodes": [
        {
            "name": "Teddington GS",
            "type": "rivergauge",
            "mrf": 200.0,
            "cost": 0.0,
            "mrf_cost": -1000.0
        }
    ]}

Storage nodes
=============

The ``Storage`` type and it's subtypes have a ``max_volume``, ``min_volume`` and ``initial_volume``, as well as ``num_inputs`` and ``num_outputs``. The maximum and initial volumes must be specified, whereas the others have default values.

.. code-block:: json

    {"nodes": [
        {
            "name": "Big Wet Lake",
            "type": "storage",
            "max_volume": 1000,
            "initial_volume": 700,
            "min_volume": 0,
            "num_inputs": 1,
            "num_outputs": 1,
            "cost": -10.0
        }
    ]}

When defining a storage node with multiple inputs or outputs connections need to be made using the slot notation (discussed in the `edges`_ section).

Edges
~~~~~

The edges section describes the connections between nodes. As a minimum an edge is defined as a two-item list containing the names of the nodes to connect (given in the order corresponding to the direction of flow), e.g.:

.. code-block:: json

    {"edges": [
        ["supply", "intermediate"],
        ["intermediate", "demand"]
    ]}

Additionally the to and from slots can be specified. For example the code below connects `reservoirA` slot 2 to `reservoirB` slot 3.

.. code-block:: json

    {"edges": [
        ["reservoirA", "reservoirB", 2, 3]
    ]}

Parameters
~~~~~~~~~~

Sometimes it is convenient to define a ``Parameter`` used in the model in the ``"parameters"`` section instead of inside a node, for instance if the parameter is needed by more than one node.

.. code-block:: json

    {
        "nodes": [
            {
                "name": "groundwater",
                "type": "input",
                "max_flow": "gw_flow"
            }
        ],
        "parameters": [
            {
                "name": "gw_flow",
                "type": "constant",
                "value": 23.0
            }
        ]
    }

Parameters can be more complicated than simple scalar values. For instance, a time varying parameter can be defined using a monthly or daily profile which repeats each year.

.. code-block:: json

    {"parameters": [
        {
            "name": "mrf_profile",
            "type": "monthlyprofile",
            "values": [10, 10, 10, 10, 50, 50, 50, 50, 20, 20, 10, 10]
        }
    ]}

Instead of defining the data inline using the ``"values"`` property, external data can be referenced as below. The URL should be relative to the JSON document *not* the current working directory.

.. code-block:: json

    {"parameters": [
        {
            "name": "catchment_inflow",
            "type": "dataframe",
            "url": "data/catchmod_outputs_v2.csv",
            "column": "Flow",
            "index_col": "Date",
            "parse_dates": true
        }
    ]}


Loading a JSON document
~~~~~~~~~~~~~~~~~~~~~~~

A Pywr JSON document can be loaded into a `Model` instance by using the `Model.load` class-method:

.. code-block:: python

    from pywr.model import Model
    my_model = model.load('/path/to/my_model.json')
    my_model.run()

Once a model is loaded if a reference to an actual node is required, using .get ...

.. code-block:: python

    node = my_model.nodes.get("River Thames")
    if node:
        print(f"max_flow: {node.max_flow}")
    else:
        print("Not found")

... or try-except is preferable to avoid searching twice.

.. code-block:: python

    try:
        node = model.nodes["River Thames"]
    except KeyError:
        print("Not found")
    else:
        print(f"max_flow: {node.max_flow}")

It is also possible to test for node and component membership using their names:

.. code-block:: python

    assert "River Thames" in model.nodes
    assert "Demand" in model.parameters


Debugging and syntax errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The JSON format is not sensitive to white space but is otherwise quite strict. When the `json` module fails to parse a document an exception will be raised. The exception includes a (somewhat cryptic) description of the problem and usefully includes a line number (see example below).

.. code-block:: pycon

    >>> model = Model.loads(data)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/Users/snorf/Desktop/pywr/pywr/core.py", line 316, in loads
        data = json.loads(data)
      File "/Users/snorf/miniconda3/envs/pywr/lib/python3.4/json/__init__.py", line 318, in loads
        return _default_decoder.decode(s)
      File "/Users/snorf/miniconda3/envs/pywr/lib/python3.4/json/decoder.py", line 343, in decode
        obj, end = self.raw_decode(s, idx=_w(s, 0).end())
      File "/Users/snorf/miniconda3/envs/pywr/lib/python3.4/json/decoder.py", line 359, in raw_decode
        obj, end = self.scan_once(s, idx)
    ValueError: Expecting property name enclosed in double quotes: line 17 column 9 (char 372)

Common mistakes when writing JSON documents "by hand" include:

 * Trailing commas at the end of a list (``["like", "this",]``)
 * Strings not enclosed in quotes (``name`` instead of ``"name"``)

Footnotes
~~~~~~~~~

.. [#] In fact the model can be represented as a hierarchy of basic Python types, which can be conveniently parsed from a JSON document. Alternative formats are possible; for example, a YAML (Yet Another Markup Language) document as it can be translated to/from JSON losslessly.
