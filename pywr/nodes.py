import numpy as np

from pywr import _core
from pywr._core import Node as BaseNode
from pywr._core import (BaseInput, BaseLink, BaseOutput, StorageInput,
    StorageOutput, Timestep, ScenarioIndex)

from pywr.parameters import pop_kwarg_parameter, load_parameter, load_parameter_values

from pywr.domains import Domain

class Drawable(object):
    """Mixin class for objects that are drawable on a diagram of the network.
    """
    def __init__(self, *args, **kwargs):
        self.position = kwargs.pop('position', None)
        self.color = kwargs.pop('color', 'black')
        self.visible = kwargs.pop('visible', True)
        super(Drawable, self).__init__(*args, **kwargs)


class Connectable(object):
    """A mixin class providing methods for connecting nodes in the model graph"""
    def iter_slots(self, slot_name=None, is_connector=True):
        """ Returns the object(s) wich should be connected to given slot_name

        Overload this method when implementing compound nodes which have
        multiple slots and may return something other than self.

        is_connector is True when self's connect method has been used. I.e. self
        is connecting to another object. This is useful for providing an
        appropriate response object in circumstances where a subnode should make
        the actual connection rather than self.
        """
        if slot_name is not None:
            raise ValueError('{} does not have slot: {}'.format(self, slot_name))
        yield self

    def connect(self, node, from_slot=None, to_slot=None):
        """Create an edge from this Node to another Node

        Parameters
        ----------
        node : Node
            The node to connect to
        from_slot : object (optional)
            The outgoing slot on this node to connect to
        to_slot : object (optional)
            The incoming slot on the target node to connect to
        """
        if self.model is not node.model:
            raise RuntimeError("Can't connect Nodes in different Models")
        if not isinstance(node, Connectable):
            raise TypeError("Other node ({}) is not connectable.".format(node))

        # Get slot from this node
        for node1 in self.iter_slots(slot_name=from_slot, is_connector=True):
            # And slot to connect from other node
            for node2 in node.iter_slots(slot_name=to_slot, is_connector=False):
                self.model.graph.add_edge(node1, node2)
        self.model.dirty = True

    def disconnect(self, node=None, slot_name=None, all_slots=True):
        """Remove a connection from this Node to another Node

        Parameters
        ----------
        node : Node (optional)
            The node to remove the connection to. If another node is not
            specified, all connections from this node will be removed.
        slot_name : integer (optional)
            If specified, only remove the connection to a specific slot name.
            Otherwise connections from all slots are removed.
        """
        if node is not None:
            self._disconnect(node, slot_name=slot_name, all_slots=all_slots)
        else:
            neighbors = self.model.graph.neighbors(self)
            for neighbor in [neighbor for neighbor in neighbors]:
                self._disconnect(neighbor, slot_name=slot_name, all_slots=all_slots)

    def _disconnect(self, node, slot_name=None, all_slots=True):
        """As disconnect, except node argument is required"""
        disconnected = False
        try:
            self.model.graph.remove_edge(self, node)
        except:
            for node_slot in node.iter_slots(slot_name=slot_name, is_connector=False, all_slots=all_slots):
                try:
                    self.model.graph.remove_edge(self, node_slot)
                except nx.exception.NetworkXError:
                    pass
                else:
                    disconnected = True
        else:
            disconnected = True
        if not disconnected:
            raise nx.exception.NetworkXError('{} is not connected to {}'.format(self, node))
        self.model.dirty = True


class NodeMeta(type):
    """Node metaclass used to keep a registry of Node classes"""
    # node subclasses are stored in a dict for convenience
    node_registry = {}
    def __new__(meta, name, bases, dct):
        return super(NodeMeta, meta).__new__(meta, name, bases, dct)
    def __init__(cls, name, bases, dct):
        super(NodeMeta, cls).__init__(name, bases, dct)
        cls.node_registry[name.lower()] = cls
    def __call__(cls, *args, **kwargs):
        # Create new instance of Node (or subclass thereof)
        node = type.__call__(cls, *args, **kwargs)
        # Add node to Model graph. This needs to be done here, so that if the
        # __init__ method of Node raises an exception it is not added.
        node.model.graph.add_node(node)
        node.model.dirty = True
        return node


class Node(Drawable, Connectable, BaseNode, metaclass=NodeMeta):
    """Base object from which all other nodes inherit

    This BaseNode is not connectable by default, and the Node class should
    be used for actual Nodes in the model. The BaseNode provides an abstract
    class for other Node types (e.g. StorageInput) that are not directly
    Connectable.
    """
    def __init__(self, model, name, **kwargs):
        """Initialise a new Node object

        Parameters
        ----------
        model : Model
            The model the node belongs to
        name : string
            A unique name for the node
        """

        position = kwargs.pop("position", {})

        color = kwargs.pop('color', 'black')
        min_flow = pop_kwarg_parameter(kwargs, 'min_flow', 0.0)
        if min_flow is None:
            min_flow = 0.0
        max_flow = pop_kwarg_parameter(kwargs, 'max_flow', float('inf'))
        cost = pop_kwarg_parameter(kwargs, 'cost', 0.0)
        conversion_factor = pop_kwarg_parameter(kwargs, 'conversion_factor', 1.0)

        super(Node, self).__init__(model, name, **kwargs)

        self.slots = {}
        self.color = color
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.cost = cost
        self.conversion_factor = conversion_factor
        self.position = position

    def check(self):
        """Check the node is valid

        Raises an exception if the node is invalid
        """
        pass

    @classmethod
    def load(cls, data, model):
        name = data.pop('name')

        cost = data.pop('cost', 0.0)
        min_flow = data.pop('min_flow', None)
        max_flow = data.pop('max_flow', None)

        data.pop('type')
        node = cls(model=model, name=name,
                   **data)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        max_flow = load_parameter(model, max_flow)
        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0
        if max_flow is None:
            max_flow = np.inf
        node.cost = cost
        node.min_flow = min_flow
        node.max_flow = max_flow

        return node


class Input(Node, BaseInput):
    """A general input at any point in the network

    """
    def __init__(self, *args, **kwargs):
        """Initialise a new Input node

        Parameters
        ----------
        min_flow : float (optional)
            A simple minimum flow constraint for the input. Defaults to None
        max_flow : float (optional)
            A simple maximum flow constraint for the input. Defaults to 0.0
        """
        super(Input, self).__init__(*args, **kwargs)
        self.color = '#F26C4F' # light red


class Output(Node, BaseOutput):
    """A general output at any point from the network

    """
    def __init__(self, *args, **kwargs):
        """Initialise a new Output node

        Parameters
        ----------
        min_flow : float (optional)
            A simple minimum flow constraint for the output. Defaults to 0.0
        max_flow : float (optional)
            A simple maximum flow constraint for the output. Defaults to None
        """
        kwargs['color'] = kwargs.pop('color', '#FFF467')  # light yellow
        super(Output, self).__init__(*args, **kwargs)


class Link(Node, BaseLink):
    """A link in the supply network, such as a pipe

    Connections between Nodes in the network are created using edges (see the
    Node.connect and Node.disconnect methods). However, these edges cannot
    hold constraints (e.g. a maximum flow constraint). In this instance a Link
    node should be used.
    """
    def __init__(self, *args, **kwargs):
        """Initialise a new Link node

        Parameters
        ----------
        max_flow : float or function (optional)
            A maximum flow constraint on the link, e.g. 5.0
        """
        kwargs['color'] = kwargs.pop('color', '#A0A0A0')  # 45% grey
        super(Link, self).__init__(*args, **kwargs)


class Storage(Drawable, Connectable, _core.Storage, metaclass=NodeMeta):
    """A generic storage Node

    In terms of connections in the network the Storage node behaves like any
    other node, provided there is only 1 input and 1 output. If there are
    multiple sub-nodes the connections need to be explicit about which they
    are connecting to. For example:

    >>> storage(model, 'reservoir', num_outputs=1, num_inputs=2)
    >>> supply.connect(storage)
    >>> storage.connect(demand1, from_slot=0)
    >>> storage.connect(demand2, from_slot=1)

    The attribtues of the sub-nodes can be modified directly (and
    independently). For example:

    >>> storage.outputs[0].max_flow = 15.0

    If a recorder is set on the storage node, instead of recording flow it
    records changes in storage. Any recorders set on the output or input
    sub-nodes record flow as normal.
    """
    def __init__(self, model, name, num_outputs=1, num_inputs=1, *args, **kwargs):
        # cast number of inputs/outputs to integer
        # this is needed if values come in as strings sometimes
        num_outputs = int(num_outputs)
        num_inputs = int(num_inputs)

        min_volume = pop_kwarg_parameter(kwargs, 'min_volume', 0.0)
        if min_volume is None:
            min_volume = 0.0
        max_volume = pop_kwarg_parameter(kwargs, 'max_volume', 0.0)
        initial_volume = kwargs.pop('initial_volume', 0.0)
        initial_volume_pc = kwargs.pop('initial_volume_pc', None)
        cost = pop_kwarg_parameter(kwargs, 'cost', 0.0)
        level = pop_kwarg_parameter(kwargs, 'level', None)
        area = pop_kwarg_parameter(kwargs, 'area', None)

        position = kwargs.pop("position", {})

        super(Storage, self).__init__(model, name, **kwargs)

        self.outputs = []
        for n in range(0, num_outputs):
            self.outputs.append(StorageOutput(model, name="[output{}]".format(n), parent=self))

        self.inputs = []
        for n in range(0, num_inputs):
            self.inputs.append(StorageInput(model, name="[input{}]".format(n), parent=self))

        self.min_volume = min_volume
        self.max_volume = max_volume
        self.initial_volume = initial_volume
        self.initial_volume_pc = initial_volume_pc
        self.cost = cost
        self.position = position
        self.level = level
        self.area = area

        # TODO FIXME!
        # StorageOutput and StorageInput are Cython classes, which do not have
        # NodeMeta as their metaclass, therefore they don't get added to the
        # model graph automatically.
        for node in self.outputs:
            self.model.graph.add_node(node)
        for node in self.inputs:
            self.model.graph.add_node(node)

        # TODO: keyword arguments for input and output nodes specified with prefix
        '''
        input_kwargs, output_kwargs = {}, {}
        keys = list(kwargs.keys())
        for key in keys:
            if key.startswith('input_'):
                input_kwargs[key.replace('input_', '')] = kwargs.pop(key)
            elif key.startswith('output_'):
                output_kwargs[key.replace('output_', '')] = kwargs.pop(key)
        '''

    def iter_slots(self, slot_name=None, is_connector=True, all_slots=False):
        if is_connector:
            if not self.inputs:
                raise StopIteration
            if slot_name is None:
                if all_slots or len(self.inputs) == 1:
                    for node in self.inputs:
                        yield node
                else:
                    raise ValueError("Must specify slot identifier.")
            else:
                try:
                    yield self.inputs[slot_name]
                except IndexError:
                    raise IndexError('{} does not have slot: {}'.format(self, slot_name))
        else:
            if not self.outputs:
                raise StopIteration
            if slot_name is None:
                if all_slots or len(self.outputs) == 1:
                    for node in self.outputs:
                        yield node
                else:
                    raise ValueError("Must specify slot identifier.")
            else:
                yield self.outputs[slot_name]

    def check(self):
        pass  # TODO

    @classmethod
    def load(cls, data, model):
        name = data.pop('name')
        num_inputs = int(data.pop('inputs', 1))
        num_outputs = int(data.pop('outputs', 1))

        if 'initial_volume' not in data and 'initial_volume_pc' not in data:
            raise ValueError('Initial volume must be specified in absolute or relative terms.')

        initial_volume = data.pop('initial_volume', 0.0)
        initial_volume_pc = data.pop('initial_volume_pc', None)
        max_volume = data.pop('max_volume')
        min_volume = data.pop('min_volume', 0.0)
        level = data.pop('level', None)
        area = data.pop('area', None)
        cost = data.pop('cost', 0.0)

        data.pop('type', None)
        # Create the instance
        node = cls(model=model, name=name, num_inputs=num_inputs, num_outputs=num_outputs, **data)

        # Load the parameters after the instance has been created to prevent circular
        # loading errors

        # Try to coerce initial volume to float.
        try:
            initial_volume = float(initial_volume)
        except TypeError:
            initial_volume = load_parameter_values(model, initial_volume)
        node.initial_volume = initial_volume
        node.initial_volume_pc = initial_volume_pc

        max_volume = load_parameter(model, max_volume)
        if max_volume is not None:
            node.max_volume = max_volume

        min_volume = load_parameter(model, min_volume)
        if min_volume is not None:
            node.min_volume = min_volume

        cost = load_parameter(model, cost)
        if cost is None:
            cost = 0.0
        node.cost = cost

        if level is not None:
            level = load_parameter(model, level)
        node.level = level

        if area is not None:
            area = load_parameter(model, area)
        node.area = area

        return node

    def __repr__(self):
        return '<{} "{}">'.format(self.__class__.__name__, self.name)


class VirtualStorage(Drawable, _core.VirtualStorage, metaclass=NodeMeta):
    """A virtual storage unit

    Parameters
    ----------
    model: pywr.core.Model
    name: str
        The name of the virtual node
    nodes: list of nodes
        List of inflow/outflow nodes that affect the storage volume
    factors: list of floats
        List of factors to multiply node flow by. Positive factors remove
        water from the storage, negative factors remove it.
    min_volume: float or parameter
        The minimum volume the storage is allowed to reach.
    max_volume: float or parameter
        The maximum volume of the storage.
    initial_volume: float
        The initial storage volume.
    cost: float or parameter
        The cost of flow into/outfrom the storage.

    Notes
    -----
    TODO: The cost property is not currently respected. See issue #242.
    """
    def __init__(self, model, name, nodes, **kwargs):
        min_volume = pop_kwarg_parameter(kwargs, 'min_volume', 0.0)
        if min_volume is None:
            min_volume = 0.0
        max_volume = pop_kwarg_parameter(kwargs, 'max_volume', 0.0)
        if 'volume' in kwargs:
            # support older API where volume kwarg was the initial volume
            initial_volume = kwargs.pop('volume')
        else:
            initial_volume = kwargs.pop('initial_volume', 0.0)
        cost = pop_kwarg_parameter(kwargs, 'cost', 0.0)

        position = kwargs.pop("position", {})
        factors = kwargs.pop('factors', None)

        super(VirtualStorage, self).__init__(model, name, **kwargs)

        self.min_volume = min_volume
        self.max_volume = max_volume
        self.initial_volume = initial_volume
        self.cost = cost
        self.position = position
        self.nodes = nodes

        if factors is None:
            self.factors = [1.0 for i in range(len(nodes))]
        else:
            self.factors = factors

    def check(self):
        super(VirtualStorage, self).check()
        if self.cost not in (0.0, None):
            raise NotImplementedError("VirtualStorage does not currently support a non-zero cost.")

    @classmethod
    def load(cls, data, model):
        del(data["type"])
        nodes = []
        for node_name in data.pop("nodes"):
            nodes.append(model._get_node_from_ref(model, node_name))
        node = cls(model, nodes=nodes, **data)
        return node

class AnnualVirtualStorage(VirtualStorage):
    """A virtual storage which resets annually, useful for licences

    See documentation for `pywr.core.VirtualStorage`.

    Parameters
    ----------
    reset_day: int
        The day of the month (0-31) to reset the volume to the initial value.
    reset_month: int
        The month of the year (0-12) to reset the volume to the initial value.
    reset_to_initial_volume: bool
        Reset the volume to the initial volume instead of maximum volume each year (default is False).

    """
    def __init__(self, *args, **kwargs):
        self.reset_day = kwargs.pop('reset_day', 1)
        self.reset_month = kwargs.pop('reset_month', 1)
        self.reset_to_initial_volume = kwargs.pop('reset_to_initial_volume', False)
        self._last_reset_year = None

        super(AnnualVirtualStorage, self).__init__(*args, **kwargs)

    def reset(self):
        super(AnnualVirtualStorage, self).reset()
        self._last_reset_year = None

    def before(self, ts):
        super(AnnualVirtualStorage, self).before(ts)

        # Reset the storage volume if necessary
        if ts.year != self._last_reset_year:
            # I.e. we're in a new year and ...
            # ... we're at or past the reset month/day
            if ts.month > self.reset_month or \
                    (ts.month == self.reset_month and ts.day >= self.reset_day):
                # Reset to maximum volume (i.e. full capacity. )
                self._reset_storage_only(use_initial_volume=self.reset_to_initial_volume)
                self._last_reset_year = ts.year


class PiecewiseLink(Node):
    """ An extension of Node that represents a non-linear Link with a piece wise cost function.

    This object is intended to model situations where there is a benefit of supplying certain flow rates
    but beyond a fixed limit there is a change in (or zero) cost.

    Parameters
    ----------
    max_flow : iterable
        A monotonic increasing list of maximum flows for the piece wise function
    cost : iterable
        A list of costs corresponding to the max_flow steps

    Notes
    -----

    This Node is implemented using a compound node structure like so:

    ::

                | Separate Domain         |
        Output -> Sublink 0 -> Sub Output -> Input
               -> Sublink 1 ---^
               ...             |
               -> Sublink n ---|

    This means routes do not directly traverse this node due to the separate
    domain in the middle. Instead several new routes are made for each of
    the sublinks and connections to the Output/Input node. The reason for this
    breaking of the route is to avoid an geometric increase in the number
    of routes when multiple PiecewiseLinks are present in the same route.
    """
    def __init__(self, *args, **kwargs):
        self.allow_isolated = True
        costs = kwargs.pop('cost')
        max_flows = kwargs.pop('max_flow')
        super(PiecewiseLink, self).__init__(*args, **kwargs)

        if len(costs) != len(max_flows):
            raise ValueError("Piecewise max_flow and cost keywords must be the same length.")

        # TODO look at the application of Domains here. Having to use
        # Input/Output instead of BaseInput/BaseOutput because of a different
        # domain is required on the sub-nodes and they need to be connected
        self.sub_domain = Domain()
        self.input = Input(self.model, name='{} Input'.format(self.name), parent=self)
        self.output = Output(self.model, name='{} Output'.format(self.name), parent=self)

        self.sub_output = Output(self.model, name='{} Sub Output'.format(self.name), parent=self,
                             domain=self.sub_domain)
        self.sub_output.connect(self.input)
        self.sublinks = []
        for max_flow, cost in zip(max_flows, costs):
            self.sublinks.append(Input(self.model, name='{} Sublink {}'.format(self.name, len(self.sublinks)),
                                      cost=cost, max_flow=max_flow, parent=self, domain=self.sub_domain))
            self.sublinks[-1].connect(self.sub_output)
            self.output.connect(self.sublinks[-1])

    def iter_slots(self, slot_name=None, is_connector=True):
        if is_connector:
            yield self.input
        else:
            yield self.output
            # All sublinks are connected upstream and downstream
            #for link in self.sublinks:
            #    yield link

    def after(self, timestep):
        """
        Set total flow on this link as sum of sublinks
        """
        for lnk in self.sublinks:
            self.commit_all(lnk.flow)
        # Make sure save is done after setting aggregated flow
        super(PiecewiseLink, self).after(timestep)

    @classmethod
    def load(cls, data, model):
        # max_flow and cost should be lists of parameter definitions
        max_flow = [load_parameter(model, p) for p in data.pop('max_flow')]
        cost = [load_parameter(model, p) for p in data.pop('cost')]

        del(data["type"])
        return cls(model, max_flow=max_flow, cost=cost, **data)


class MultiSplitLink(PiecewiseLink):
    """ An extension of PiecewiseLink that includes additional slots to connect from.

    Conceptually this node looks like the following internally,

    ::

                 / -->-- X0 -->-- \\
        A -->-- Xo -->-- X1 -->-- Xi -->-- C
                 \\ -->-- X2 -->-- /
                         |
                         Bo -->-- Bi --> D

    An additional sublink in the PiecewiseLink (i.e. X2 above) and nodes
    (i.e. Bo and Bi) in this class are added for each extra slot.

    Finally a mechanism is provided to (optionally) fix the ratio between the
    last non-split sublink (i.e. X1) and each of the extra sublinks (i.e. X2).
    This mechanism uses `AggregatedNode` internally.

    Parameters
    ----------
    max_flow : iterable
        A monotonic increasing list of maximum flows for the piece wise function
    cost : iterable
        A list of costs corresponding to the max_flow steps
    extra_slots : int, optional (default 1)
        Number of additional slots (and sublinks) to provide. Must be greater
        than zero.
    slot_names : iterable, optional (default range of ints)
        The names by which to refer to the slots during connection to other
        nodes. Length must be one more than the number of extra_slots. The first
        item refers to the PiecewiseLink connection with the following items for
        each extra slot.
    factors : iterable, optional (default None)
        If given, the length must be equal to one more than the number of
        extra_slots. Each item is the proportion of total flow to pass through
        the additional sublinks. If no factor is required for a particular
        sublink then use `None` for its items. Factors are normalised prior to
        use in the solver.

    Notes
    -----
    Users must be careful when using the factor mechanism. Factors use the last
    non-split sublink (i.e. X1 but not X0). If this link is constrained with a
    maximum or minimum flow, or if it there is another unconstrained link
    (i.e. if X0 is unconstrained) then ratios across this whole node may not be
    enforced as expected.

    """
    def __init__(self, *args, **kwargs):
        self.allow_isolated = True
        costs = list(kwargs.pop('cost'))
        max_flows = list(kwargs.pop('max_flow'))

        extra_slots = kwargs.pop('extra_slots', 1)
        if extra_slots < 1:
            raise ValueError("extra_slots must be at least 1.")

        # No cost or maximum flow on the additional links
        # The max_flows could be problematic with the aggregated node.
        costs.extend([0.0]*extra_slots)
        max_flows.extend([None]*extra_slots)
        # Edit the kwargs to get the PiecewiseLink to setup as we want.
        kwargs['cost'] = costs
        kwargs['max_flow'] = max_flows

        # Default to integer names
        self.slot_names = list(kwargs.pop('slot_names', range(extra_slots+1)))
        if extra_slots+1 != len(self.slot_names):
            raise ValueError("slot_names must be one more than the number of extra_slots.")

        factors = kwargs.pop('factors', None)
        # Finally initialise the parent.
        super(MultiSplitLink, self).__init__(*args, **kwargs)

        self._extra_inputs = []
        self._extra_outputs = []
        n = len(self.sublinks) - extra_slots
        for i in range(extra_slots):
            # create a new input inside the piecewise link which only has access
            # to flow travelling via the last sublink (X2)
            otpt = Output(self.model, '{} Extra Output {}'.format(self.name, i),
                          domain=self.sub_domain, parent=self)
            inpt = Input(self.model, '{} Extra Input {}'.format(self.name, i), parent=self)

            otpt.connect(inpt)
            self.sublinks[n+i].connect(otpt)

            self._extra_inputs.append(inpt)
            self._extra_outputs.append(otpt)

        # Now create an aggregated node for addition constaints if required.
        if factors is not None:
            if extra_slots+1 != len(factors):
                raise ValueError("factors must have a length equal to extra_slots.")

            nodes = []
            valid_factors = []
            for r, nd in zip(factors, self.sublinks[n-1:]):
                if r is not None:
                    nodes.append(nd)
                    valid_factors.append(r)

            agg = AggregatedNode(self.model, "{} Agg".format(self.name), nodes)
            agg.factors = valid_factors


    def iter_slots(self, slot_name=None, is_connector=True):
        if is_connector:
            i = self.slot_names.index(slot_name)
            if i == 0:
                yield self.input
            else:
                yield self._extra_inputs[i-1]
        else:
            yield self.output


class AggregatedStorage( Drawable, _core.AggregatedStorage, metaclass=NodeMeta):
    """ An aggregated sum of other `Storage` nodes

    This object should behave like `Storage` by returning current `flow`, `volume` and `current_pc`.
    However this object can not be connected to others within the network.

    Parameters
    ----------
    model - `Model` instance
    name - str
    storage_nodes - list or iterable of `Storage` objects
        The `Storage` objects which to return the sum total of

    Notes
    -----
    This node can not be connected to other nodes in the network.

    """
    def __init__(self, model, name, storage_nodes, **kwargs):
        super(AggregatedStorage, self).__init__(model, name, **kwargs)
        self.storage_nodes = storage_nodes


class AggregatedNode(Drawable, _core.AggregatedNode, metaclass=NodeMeta):
    """ An aggregated sum of other `Node` nodes

    This object should behave like `Node` by returning current `flow`.
    However this object can not be connected to others within the network.

    Parameters
    ----------
    model - `Model` instance
    name - str
    nodes - list or iterable of `Node` objects
        The `Node` objects which to return the sum total of

    Notes
    -----
    This node can not be connected to other nodes in the network.

    """
    def __init__(self, model, name, nodes, **kwargs):
        super(AggregatedNode, self).__init__(model, name, **kwargs)
        self.nodes = nodes


class BreakLink(Node):
    """Compound node used to reduce the number of routes in a model

    Parameters
    ----------
    model : `pywr.model.Model`
    name : string
    min_flow : float or `pywr.parameters.Parameter`
    max_flow : float or `pywr.parameters.Parameter`
    cost : float or `pywr.parameters.Parameter`

    Notes
    -----

    In a model with form (3, 1, 3), i.e. 3 (A,B,C) inputs connected to 3
    outputs (D,E,F) via a bottleneck (X), there are 3*3 routes = 9 routes.

    ::

        A -->\\ /--> D
        B --> X --> E
        C -->/ \\--> F

    If X is a storage, there are only 6 routes: A->X_o, B->X_o, C->X_o and
    X_i->D_o, X_i->E_o, X_i->F_o.

    The `BreakLink` node is a compound node composed of a `Storage` with zero
    volume and a `Link`. It can be used in place of a normal `Link`, but
    with the benefit that it reduces the number of routes in the model (in
    the situation described above). The resulting LP is easier to solve.
    """
    allow_isolated = True

    def __init__(self, model, name, **kwargs):
        storage_name = "{} (storage)".format(name)
        link_name = "{} (link)".format(name)
        assert(storage_name not in model.nodes)
        assert(link_name not in model.nodes)
        self.storage = Storage(
            model,
            name=storage_name,
            min_volume=0,
            max_volume=0,
            initial_volume=0,
            cost=0,
        )
        self.link = Link(
            model,
            name=link_name
        )

        self.storage.connect(self.link)

        super(BreakLink, self).__init__(model, name, **kwargs)

    def min_flow():
        def fget(self):
            return self.link.min_flow
        def fset(self, value):
            self.link.min_flow = value
        return locals()
    min_flow = property(**min_flow())

    def max_flow():
        def fget(self):
            return self.link.max_flow
        def fset(self, value):
            self.link.max_flow = value
        return locals()
    max_flow = property(**max_flow())

    def cost():
        def fget(self):
            return self.link.cost
        def fset(self, value):
            self.link.cost = value
        return locals()
    cost = property(**cost())

    def iter_slots(self, slot_name=None, is_connector=True):
        if is_connector:
            # connecting FROM the transfer TO something else
            yield self.link
        else:
            # connecting FROM something else TO the transfer
            yield self.storage.outputs[0]

    def after(self, timestep):
        super(BreakLink, self).after(timestep)
        # update flow on transfer node to flow via link node
        self.commit_all(self.link.flow)

from pywr.domains.river import *
