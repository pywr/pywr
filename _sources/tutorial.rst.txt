Tutorial
========

A minimal example
-----------------

The simplest example has two nodes: an :func:`~pywr.core.Input` which adds flow to the network and an :func:`~pywr.core.Output` which removes flow from the network.

.. code-block:: python

    from pywr.core import Model, Input, Output

    # create a model (including an empty network)
    model = Model()

    # create two nodes: a supply, and a demand
    supply = Input(model, name='supply')
    demand = Output(model, name='demand')

    # create a connection from the supply to the demand
    supply.connect(demand)

While technically valid, this model isn't very interesting because we haven't set any constraints or costs on flows in the network.

Let's add some flow constraints to the problem:

.. code-block:: python

    # set maximum flows
    supply.max_flow = 10.0
    demand.max_flow = 6.0

The default minimum flow for a :func:`~pywr.core.BaseNode` is zero, so we don't need to set it explicitly.

The model still doesn't do anything as it's missing costs for flow through the nodes. If the cost of supply is less than the benefit received from satisfying demand, flow in the network will occur (within the models constraints).

.. code-block:: python
   
    # set cost (+ve) or benefit (-ve)
    supply.cost = 3.0
    demand.cost = -100.0

Next we need to tell the model how long to run for. As an example, we'll use a daily timestep for all of 2015.

.. code-block:: python

    import datetime
    from pywr.core import Timestepper

    model.timestepper = Timestepper(
        pandas.to_datetime('2015-01-01'),  # first day
        pandas.to_datetime('2015-12-31'),  # last day
        datetime.timedelta(1)  # interval
    )

In order to capture the output from the model we need to use a recorder, such as the :func:`pywr.recorders.NumpyArrayNodeRecorder`.

.. code-block:: python

    from pywr.recorders import NumpyArrayNodeRecorder

    recorder = NumpyArrayNodeRecorder(model, supply)

Finally we are ready to run our model:

.. code-block:: python

    # lets get this party started!
    model.run()

We can check the result for the first timestep by accessing the recorder's data property:

.. code-block:: python

    scenario = 0
    timestep = 0
    print(recorder.data[scenario][timestep])  # prints 6.0

The result of this example model is trivial: the supply exceeds the demand, so the maximum flow at the demand is the limiting factor.

The model defined about could equally have been defined in a JSON document:

.. literalinclude:: tutorial.json

The model can be loaded using the ``Model.load`` method:

.. code-block:: python

    # load the model
    model = Model.load("tutorial.json")
    # run, forest, run!
    model.run()

See further information in on JSON see the :ref:`json-model-format` section.
