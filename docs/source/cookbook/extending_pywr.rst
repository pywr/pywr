Extending Pywr
--------------

Pywr is a Python library and as such is designed in an object oriented fashion. These objects (or classes) form
the basic structure of Pywr, but much of the functionality in Pywr is provided via specific sub-classes designed
to do specific instances. For example, there exist many specialised `Parameter` classes to provide specific types
of input data. One of the benefits of being a Python library is that users can extend the base functionality in their
own projects to provide custom functionality. In this section of the documentation there are some examples and guidance
for extending Pywr in your projects.


Extending Nodes
===============

Nodes and subclasses thereof provide the basic network structure in Pywr. There are several different types
of node available. Two major categories exist: flow nodes and storage nodes. Flow nodes are the typical nodes
that represent rivers, pipes and other features from, through or to which a resource can flow. These nodes are
typically characterised by minimum and maximum flow rates. Storage nodes provide the ability to store resource
from one time-step to another, and are characterised by minimum, maximum and initial volumes.

The most common way to make new node types in Pywr is to create a subclass which creates a compound of one or more
existing nodes. In this way the subclass acts as a helper for managing more complex or common arrangements of the
basic node types.


Let's look at an example to create a new node type that represents a leaky pipe.




.. code-block:: python

    from pywr.nodes import Node, Link, Output

    class LeakyPipe(Node):
        def __init__(self, *args, **kwargs):
            self.allow_isolated = True
            super(LeakyPipe, self).__init__(*args, **kwargs)

            # Define the internal nodes
            self.inflow = Link(self.model, name='{} In'.format(self.name), parent=self)
            self.outflow = Link(self.model, name='{} Out'.format(self.name), parent=self)
            self.inflow.connect(self.outflow)

            # Self output for the link
            self.leak = Output(self.model, name='{} Leak'.format(self.name), parent=self)
            self.inflow.connect(self.leak)

        def iter_slots(self, slot_name=None, is_connector=True):
            if is_connector:
                yield self.outflow
            else:
                yield self.inflow

        def after(self, timestep):
            # Make the flow on this node appear as the flow after the leak
            self.commit_all(self.outflow.flow)
            # Make sure save is done after setting aggregated flow
            super(LeakyPipe, self).after(timestep)



    baseline = ConstantParameter(5.0)
    profile = MonthlyParameter(
        [0.8, 0.8, 0.8, 0.8, 1.1, 1.1, 1.1, 1.1, 0.8, 0.8, 0.8, 0.8]
    )
    agg = AggregatedParameter(
        parameters=[baseline, profile],
        agg_func="product"
    )



Extending Parameters
====================




Extending Recorders
===================

