.. _extending-pywr-nodes:

Extending Pywr with custom Nodes
--------------------------------

Nodes and subclasses thereof provide the basic network structure in Pywr. There are several different types
of node available. Two major categories exist: flow nodes and storage nodes. Flow nodes are the typical nodes
that represent rivers, pipes and other features from, through or to which a resource can flow. These nodes are
typically characterised by minimum and maximum flow rates. Storage nodes provide the ability to store resource
from one time-step to another, and are characterised by minimum, maximum and initial volumes.

There are three fundamental types of node in Pywr:

* ``Input`` nodes add water to the system
* ``Output`` nodes remove water from the system
* ``Link`` nodes do not add or remove water from the system

There is a fourth node type, ``Storage``, which can be considered fundamental because there are special rules for
it's behaviour in the linear programme used to solve the water balance:

* ``Storage`` nodes can carry water from one timestep to the next

All other node types in Pywr are subclasses of these base types. For example, the ``pywr.nodes.Catchment`` node type
is a special case of ``Input`` where the ``min_flow`` and ``max_flow`` properties are equal.

The most common way to create a new node type is using a compound node. A compound node contains one or more existing
nodes and is used to manage common or more complex arrangements of the basic node types. An example of a compound node
is the ``PiecewiseLink``. It is composed of a link (``OUT_1``) which receives water from upstream and an link
(``IN_1``) which conveys water downstream, connected by a set of links in parallel (``LNK_1`` ... ``LNK_N``) each with
a different ``max_flow`` and ``cost``, illustrated below::

                               /-->-- LNK_1 -->--\
    UPSTREAM -->-- OUT_1 -->--|--->--  ...  -->---|-->-- IN_1 -->-- DOWNSTREAM
                               \-->-- LNK_N -->--/

Let's look at an example to create a new node type that represents a leaky pipe. To remove water from the system we
need to use an output node (``LEAK``), with two links representing the boundaries of the compound node (``INFLOW`` and
``OUTFLOW``)::

    UPSTREAM -->-- INFLOW -->-- OUTFLOW -->-- DOWNSTREAM
                     |
                     \------>--  LEAK

This is a simple structure which represents leakage as a demand with a maximum value and a benefit to be supplied. It
is slightly flawed as the leakage volume does not vary proportionally to flow through the link, but is sufficient as
an example:

.. code-block:: python

    from pywr.nodes import Node, Link, Output

    class LeakyPipe(Node):
        def __init__(self, leakage, leakage_cost=-99999, *args, **kwargs):
            self.allow_isolated = True  # Required for compound nodes

            super(LeakyPipe, self).__init__(*args, **kwargs)

            # Define the internal nodes. The parent of the nodes is defined to identify them as sub-nodes.
            self.inflow = Link(self.model, name='{} In'.format(self.name), parent=self)
            self.outflow = Link(self.model, name='{} Out'.format(self.name), parent=self)
            self.leak = Output(self.model, name='{} Leak'.format(self.name), parent=self)

            # Connect the internal nodes
            self.inflow.connect(self.outflow)
            self.inflow.connect(self.leak)

            # Define the properties of the leak (demand and benefit)
            self.leak.max_flow = leakage
            self.leak.cost = leakage_cost

        def iter_slots(self, slot_name=None, is_connector=True):
            # This is required so that connecting to this node actually connects to the outflow sub-node, and
            # connecting from this node actually connects to the input sub-node
            if is_connector:
                yield self.outflow
            else:
                yield self.inflow

        def after(self, timestep):
            # Make the flow on the compound node appear as the flow _after_ the leak
            self.commit_all(self.outflow.flow)
            # Make sure save is done after setting aggregated flow
            super(LeakyPipe, self).after(timestep)

        @classmethod
        def load(cls, data, model):
            del(data["type"])
            leakage = data.pop("leakage")
            leakage_cost = data.pop("leakage_cost", None)
            return cls(model, leakage, leakage_cost, **data)


The custom node does not need to be "registered", unlike ``Parameters``, as this is done automatically using
metaclasses. The new node type can be referenced from a JSON model provided that the class has been imported before
the JSON is loaded:

.. code-block:: python

    from pywr.model import Model
    import leakypipe
    
    model = Model.load("leaky_pipe_model.sjon")


.. code-block:: yaml

    {
        "type": "leakypipe",
        "leakage": "1.0"
    }

The ``allow_isolated`` attribute identifies nodes of this type as compound nodes. Without this the model would raise
an error that the node is not connected to the rest of the network, as the connections are actually to its sub-nodes.

The ``after`` method is not required but is useful so that recorders can be attached to the compound node. Without
this the flow would appear to be zero as the flow doesn't *actually* pass through the compound node.

The ``iter_slots`` method is required so that connecting to/from the node (e.g. ``upstream.connect(leaky)``) creates
connections to the sub-nodes.

A more advanced representation of the leaky pipe could use an additional ``AggregatedNode`` to constrain the ratio
of flow through the ``OUTFLOW`` and ``LEAK`` nodes. [*]_

.. [*] ``AggregatedNode`` is actually another fundamental node type, as this behaviour requires special treatment
       in the linear programme.
