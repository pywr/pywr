import numpy as np
from pywr import _core
from pywr._core import Node as BaseNode
from pywr._core import BaseInput, BaseLink, BaseOutput, StorageInput, StorageOutput
from pywr.parameters import (
    pop_kwarg_parameter,
    load_parameter,
    load_parameter_values,
    FlowDelayParameter,
)
from pywr.domains import Domain
import networkx as nx


class Drawable(object):
    """Mixin class for objects that are drawable on a diagram of the network."""

    def __init__(self, *args, **kwargs):
        self.position = kwargs.pop("position", {})
        self.color = kwargs.pop("color", "black")
        self.visible = kwargs.pop("visible", True)
        super(Drawable, self).__init__(*args, **kwargs)


class Loadable:
    """Mixin class that laods nodes from JSON data.

    Loading is performed in two stages. First a `pre_load` classmethod creates a node instance passing
    any non-parameter arguments via keyword to the respective Node's init method. After parameters are
    loaded (see `Model.load`) the `finalise_load` method is called on all node instances to assign
    concrete parameter instances where needed.
    """

    __parameter_attributes__ = ()
    __node_attributes__ = ()
    __parameter_value_attributes__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__parameters_to_load = None

    @classmethod
    def _pre_load_parameter(cls, key, value, param_data, non_param_data):
        # By default parameter references are loaded later, and no keyword is given to the `__init__`
        param_data[key] = value

    @classmethod
    def pre_load(cls, model, data):
        """Create a node instance from data.

        Parameter data and references are stored until later, and consumed by a call to `finalise_load`.

        Parameters
        ----------
        model : Model
            The model instance.
        data : dict
            The node's data.
        """
        # Filter non-parameter data and initialise with it only
        param_data = {}
        non_param_data = {}
        for key, value in data.items():
            if key in cls.__parameter_attributes__:
                cls._pre_load_parameter(key, value, param_data, non_param_data)
            elif key in cls.__node_attributes__:
                # Node references are converted to nodes immediately.
                if isinstance(value, list):
                    node = [model.pre_load_node(n) for n in value]
                else:
                    node = model.pre_load_node(value)
                non_param_data[key] = node
            elif key in cls.__parameter_value_attributes__:
                if isinstance(value, (float, int)):
                    non_param_data[key] = value
                else:
                    non_param_data[key] = load_parameter_values(model, value)
            else:
                non_param_data[key] = value

        obj = cls(model=model, **non_param_data)
        obj.__parameters_to_load = param_data
        return obj

    def finalise_load(self):
        """Finish loading a node by converting parameter name references to instance references."""
        for key, parameter_data in self.__parameters_to_load.items():
            if isinstance(parameter_data, list):
                # List of parameter references
                parameter = [load_parameter(self.model, d) for d in parameter_data]
            else:
                parameter = load_parameter(self.model, parameter_data)

            setattr(self, key, parameter)
        del self.__parameters_to_load


class Connectable(object):
    """A mixin class providing methods for connecting nodes in the model graph"""

    def iter_slots(self, slot_name=None, is_connector=True):
        """Returns the object(s) which should be connected to the given `slot_name`.

        Overload this method when implementing compound nodes which have
        multiple slots and may return something other than self.

        Parameters
        ----------
        slot_name : Optional[str], default=None
            The name of the slot.
        is_connector : bool, default=True
            This is True when self's connect method has been used. I.e. self
            is connecting to another object. This is useful for providing an
            appropriate response object in circumstances where a sub-node should make
            the actual connection rather than self.

        Returns
        -------
        Iterable[Connectable]
            The object(s) which should be connected to the given `slot_name`.
        """
        if slot_name is not None:
            raise ValueError("{} does not have slot: {}".format(self, slot_name))
        yield self

    def connect(self, node, from_slot=None, to_slot=None):
        """Create an edge from this `Node` to another `Node`.

        Parameters
        ----------
        node : Node
            The node to connect to.
        from_slot : Option[int], default=None
            The outgoing slot on this node to connect to.
        to_slot : Option[int], default=None
            The incoming slot on the target node to connect to.
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
        """Remove a connection from this `Node` to another `Node`.

        Parameters
        ----------
        node : Optional[Node], default=None.
            The node to remove the connection to. If another node is not
            specified, all connections from this node will be removed.
        slot_name : Optional[int], default=None
            If specified, only remove the connection to a specific slot name.
            Otherwise, connections from all slots are removed.
        all_slots : bool, default=True
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
        except nx.exception.NetworkXError:
            for node_slot in node.iter_slots(
                slot_name=slot_name, is_connector=False, all_slots=all_slots
            ):
                try:
                    self.model.graph.remove_edge(self, node_slot)
                except nx.exception.NetworkXError:
                    pass
                else:
                    disconnected = True
        else:
            disconnected = True
        if not disconnected:
            raise nx.exception.NetworkXError(
                "{} is not connected to {}".format(self, node)
            )
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


class Node(Loadable, Drawable, Connectable, BaseNode, metaclass=NodeMeta):
    """Base object from which all other nodes inherit

    This `BaseNode` is not connectable by default, and the `Node` class should
    be used for actual Nodes in the model. The `BaseNode` provides an abstract
    class for other `Node` types (e.g. `StorageInput`) that are not directly
    `Connectable`.
    """

    __parameter_attributes__ = ("cost", "min_flow", "max_flow")

    def __init__(self, model, name, **kwargs):
        """Initialise a new Node object.

        Parameters
        ----------
        model : Model
            The model the node belongs to.
        name : string
            A unique name for the node.
        """
        color = kwargs.pop("color", "black")
        min_flow = pop_kwarg_parameter(kwargs, "min_flow", 0.0)
        if min_flow is None:
            min_flow = 0.0
        max_flow = pop_kwarg_parameter(kwargs, "max_flow", float("inf"))
        cost = pop_kwarg_parameter(kwargs, "cost", 0.0)
        conversion_factor = pop_kwarg_parameter(kwargs, "conversion_factor", 1.0)

        super(Node, self).__init__(model, name, **kwargs)

        self.slots = {}
        self.color = color
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.cost = cost
        self.conversion_factor = conversion_factor

    def check(self):
        """Check the node is valid.

        Raises
        ------
        Exception
            If the node is invalid
        """
        pass


class Input(Node, BaseInput):
    """A general input at any point in the network.

    Examples
    --------
    Python
    ======
    ```python
    model = Model()
    Input(model=model, min_flow=1.0,max_flow=10.3, name="In")
    ```

    JSON
    ======
    ```json
    {
        "name": "In",
        "type": "Input",
        "min_flow": 1.0,
        "max_flow": 10.3
    }
    ```
    """

    def __init__(self, *args, **kwargs):
        """Initialise a new Input node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        min_flow : Optional[float | Parameter], default=0
            A simple minimum flow constraint for the node. Defaults to 0.
        max_flow : Optional[float | Parameter], default=Inf
            A simple maximum flow constraint for the node. Defaults to infinite.
        cost : Optional[float | Parameter], default=0
            The cost of supply.
        """
        super(Input, self).__init__(*args, **kwargs)
        self.color = "#F26C4F"  # light red


class Output(Node, BaseOutput):
    """A general output."""

    def __init__(self, *args, **kwargs):
        """Initialise a new Output node

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        min_flow : Optional[float | Parameter], default=0
            A simple minimum flow constraint for the node. Defaults to 0.
        max_flow : Optional[float | Parameter], default=Inf
            A simple maximum flow constraint for the node. Defaults to infinite.
        cost : Optional[float | Parameter], default=0
            The cost of supply.
        """
        kwargs["color"] = kwargs.pop("color", "#FFF467")  # light yellow
        super(Output, self).__init__(*args, **kwargs)


class Link(Node, BaseLink):
    """A link in the supply network, such as a pipe.

    Connections between `Nodes` in the network are created using edges (see the
    Node.connect and Node.disconnect methods). However, these edges cannot
    hold constraints (e.g. a maximum flow constraint). In this instance a Link
    node should be used.

    Parameters
    ----------
    model : Model
        The model instance.
    name : str
        The unique name of the node.
    min_flow : Optional[float | Parameter], default=0
        A simple minimum flow constraint for the node. Defaults to 0.
    max_flow : Optional[float | Parameter], default=Inf
        A simple maximum flow constraint for the node. Defaults to infinite.
    cost : Optional[float | Parameter], default=0
        The cost of supply.

    Examples
    --------
    Python
    ======
    ```python
    model = Model()
    Link(model=model, min_flow=1.0, name="Pipe")
    ```

    JSON
    ======
    ```json
    {
        "name": "Pipe",
        "type": "Link",
        "min_flow": 1.0,
    }
    ```
    """

    def __init__(self, *args, **kwargs):
        """Initialise a new Link node

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        min_flow : Optional[float | Parameter], default=0
            A simple minimum flow constraint for the node. Defaults to 0.
        max_flow : Optional[float | Parameter], default=Inf
            A simple maximum flow constraint for the node. Defaults to infinite.
        cost : Optional[float | Parameter], default=0
            The cost of supply.
        """
        kwargs["color"] = kwargs.pop("color", "#A0A0A0")  # 45% grey
        super(Link, self).__init__(*args, **kwargs)


class Storage(Loadable, Drawable, Connectable, _core.Storage, metaclass=NodeMeta):
    """A generic storage node holdin some water.

    Examples
    --------
    Python
    ==================
    ```python
    model = Model()
    node = Storage(model=model, cost=, name="Reservoir")
    curve = ConstantParameter(model=model, value=0.5)
    node.cost = ControlCurveParameter(
        model=model,
        name="Cost",
        storage_node=node,
        control_curves=[curve],
        values=[-0.1, -100]
    )
    ```

    JSON
    ==================
    ```json
    {
        "name": "Reservoir",
        "type": "Storage",
        "cost": {
            "type": "ControlCurve",
            "storage_node": "Reservir",
            "control_curves" [
                {
                    "type": "Constant",
                    "value": 0.5
                }
            ],
            "values": [-0.1, -100]
        }
    }
    ```

    Initial volume
    ------
    1. The `initial_volume` and initial_volume_pc` are both required if `max_volume is a [pywr.parameters.Parameter][]
    because the parameter will not be evaluated at the first time-step.
    2. If both `initial_volume` and initial_volume_pc` are given and `max_volume` is not a Parameter, then the absolute
    value is ignored.

    Connection
    ------
    In terms of connections in the network the Storage node behaves like any
    other node, provided there is only 1 input and 1 output. If there are
    multiple sub-nodes the connections need to be explicit about which they
    are connecting to. For example:

    ```python
    Storage(model, 'reservoir', num_outputs=1, num_inputs=2)
    supply.connect(storage)
    storage.connect(demand1, from_slot=0)
    storage.connect(demand2, from_slot=1)
    ```
    The attribtues of the sub-nodes can be modified directly (and
    independently). For example:

    ```python
    storage.outputs[0].max_flow = 15.0
    ```

    If a recorder is set on the storage node, instead of recording flow it
    records changes in storage. Any recorders set on the output or input
    sub-nodes record flow as normal.
    """

    __parameter_attributes__ = ("cost", "min_volume", "max_volume", "level", "area")
    __parameter_value_attributes__ = ("initial_volume",)

    def __init__(self, model, name, outputs=1, inputs=1, *args, **kwargs):
        """Initialise the node.
        Parameters
        ----------
        model : Model
            The model instance to which this storage node is attached.
        name : str
            The name of the storage node.
        inputs : Optional[integer], default=1
            The number of input nodes to create internally.
        outputs : Optional[integer], default=1
            The number of output nodes to create internally.

        Other Parameters
        ----------------
        min_volume : Optional[float], default=0
            The minimum volume of the storage.
        max_volume : Optional[float | Parameter], default=0
            The maximum volume of the storage.
        initial_volume : Optional[float], default=0
            Specify the initial volume in absolute terms.
        initial_volume_pc : Optional[float], default=0
            Specify initial volume in proportional terms. `initial_volume` and initial_volume_pc` are required
            if `max_volume is a parameter because the parameter will not be evaluated at the first time-step.
            If both are given and `max_volume` is not a Parameter, then the absolute value is ignored.
        cost : Optional[float | Parameter], default=0
            The cost of net flow in to the storage node. I.e. a positive cost penalises increasing volume by
            giving a benefit to negative net flow (release), and a negative cost penalises decreasing volume
            by giving a benefit to positive net flow (inflow).
        area : Optional[float | Parameter], default=None
            Optional float or Parameter defining the area of the storage node. These values are
            accessible through the `get_area` method.
        level : Optional[float | Parameter], default=None
            Optional float or Parameter defining the level of the storage node. These values are
            accessible through the `get_level` method.
        """
        min_volume = pop_kwarg_parameter(kwargs, "min_volume", 0.0)
        if min_volume is None:
            min_volume = 0.0
        max_volume = pop_kwarg_parameter(kwargs, "max_volume", 0.0)
        initial_volume = kwargs.pop("initial_volume", None)
        initial_volume_pc = kwargs.pop("initial_volume_pc", None)
        cost = pop_kwarg_parameter(kwargs, "cost", 0.0)
        level = pop_kwarg_parameter(kwargs, "level", None)
        area = pop_kwarg_parameter(kwargs, "area", None)

        super(Storage, self).__init__(model, name, **kwargs)

        self.outputs = []
        for n in range(0, outputs):
            self.outputs.append(
                StorageOutput(model, name="[output{}]".format(n), parent=self)
            )

        self.inputs = []
        for n in range(0, inputs):
            self.inputs.append(
                StorageInput(model, name="[input{}]".format(n), parent=self)
            )

        self.min_volume = min_volume
        self.max_volume = max_volume
        self.initial_volume = initial_volume
        self.initial_volume_pc = initial_volume_pc
        self.cost = cost
        self.level = level
        self.area = area

        # StorageOutput and StorageInput are Cython classes, which do not have
        # NodeMeta as their metaclass, therefore they don't get added to the
        # model graph automatically.
        for node in self.outputs:
            self.model.graph.add_node(node)
        for node in self.inputs:
            self.model.graph.add_node(node)

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
                    raise IndexError(
                        "{} does not have slot: {}".format(self, slot_name)
                    )
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

    def __repr__(self):
        return '<{} "{}">'.format(self.__class__.__name__, self.name)


class VirtualStorage(Loadable, Drawable, _core.VirtualStorage, metaclass=NodeMeta):
    """A virtual storage unit.

    Attributes
    ----------
    model : Model
        The model instance.
    name : str
        The unique name of the node.
    nodes : list[Node]
        List of inflow/outflow nodes that affect the storage volume.
    factors : Optional[list[float]], default=None
        List of factors to multiply each node's flow in `nodes` by. Positive factors remove
        water from the storage, negative factors remove it. When `None`, the flow is not scaled.
    """

    # TODO: The cost property is not currently respected. See issue #242.
    __parameter_attributes__ = ("min_volume", "max_volume")
    __node_attributes__ = ("nodes",)

    def __init__(self, model, name, nodes, **kwargs):
        """Instantiate the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        nodes : list[Node]
            List of inflow/outflow nodes that affect the storage volume.

        Other Parameters
        ----------------
        factors : Optional[list[float]], default=None
            List of factors to multiply each node's flow in `nodes` by. Positive factors remove
            water from the storage, negative factors remove it. When `None`, the flow is not scaled.
        min_volume : Optional[float | Parameter], default=0.0
            The minimum volume the storage is allowed to reach.
        max_volume : Optional[float | Parameter], default=0.0
            The maximum volume of the storage.
        initial_volume : Optional[float], default=None
            Specify initial volume in absolute terms.
        initial_volume_pc : Optional[float], default=None
            Specify initial volume in proportional terms. Both `initial_volume_pc` and `initial_volume` are required if `max_volume`
            is a parameter because the parameter will not be evaluated at the first time-step. If both are given
            and `max_volume` is not a Parameter, then the absolute value is ignored.
        cost : Optional[float | Parameter], default=None
            The cost of flow into/outfrom the storage. The cost property is not currently respected (see issue #242),
            therefore, leave this to `None`.

        Notes
        -----
        This is an abstract class used to build other virtual storages. Do not instantiate it unless you are inheriting
        for this class.
        """
        min_volume = pop_kwarg_parameter(kwargs, "min_volume", 0.0)
        if min_volume is None:
            min_volume = 0.0
        max_volume = pop_kwarg_parameter(kwargs, "max_volume", 0.0)
        initial_volume = kwargs.pop("initial_volume", None)
        initial_volume_pc = kwargs.pop("initial_volume_pc", None)
        cost = pop_kwarg_parameter(kwargs, "cost", 0.0)

        factors = kwargs.pop("factors", None)

        super(VirtualStorage, self).__init__(model, name, **kwargs)

        self.min_volume = min_volume
        self.max_volume = max_volume
        self.initial_volume = initial_volume
        self.initial_volume_pc = initial_volume_pc
        self.cost = cost
        self.nodes = nodes

        if factors is None:
            self.factors = [1.0 for i in range(len(nodes))]
        else:
            self.factors = factors

    def check(self):
        super(VirtualStorage, self).check()
        if self.cost not in (0.0, None):
            raise NotImplementedError(
                "VirtualStorage does not currently support a non-zero cost."
            )


class RollingVirtualStorage(
    Loadable, Drawable, _core.RollingVirtualStorage, metaclass=NodeMeta
):
    """A rolling virtual storage node useful for implementing rolling licences.

    The volume of the node is updated each timestep with the volume of water utilised in the timestep immediately
    prior to the rolling period.

    If the initial volume of the storage is less than the maximum volume then the parameter will calculate
    an initial utilisation value. This is set equal to

        (max volume - initial volume) / (timesteps - 1).

    This utilisation is assumed to occurred equally across each timestep of the rolling period. The storage is replenished
    by this value for each timestep until a full rolling period is completed. At this point, replenishment will
    be based on the previous utilisation of the storage during the model run. Note that this changes the previous
    default behaviour of the node up to Pywr version `1.17.1`, where no initial utilisation was calculated. This meant
    that it was impossible for the storage volume to be higher than the initial volume even if this was lower than the
    max volume.

    Attributes
    ----------
    model : Model
        The model instance.
    name : str
        The unique name of the node.
    nodes : list[Node]
        List of inflow/outflow nodes that affect the storage volume.
    factors : Optional[list[float]], default=None
        List of factors to multiply each node's flow in `nodes` by. Positive factors remove
        water from the storage, negative factors remove it. When `None`, the flow is not scaled.
    timesteps : int
        The number of timesteps to apply to the rolling storage over.
    days : int
        The number of days to apply the rolling storage over.

    Notes
    -----
    The cost property is not currently respected (see issue #242), therefore leave this to `None`.
    """

    __parameter_attributes__ = ("min_volume", "max_volume")
    __node_attributes__ = ("nodes",)

    def __init__(self, model, name, nodes, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        nodes : list[Node]
            List of inflow/outflow nodes that affect the storage volume.

        Other Parameters
        ----------------
        factors : Optional[list[float]], default=None
            List of factors to multiply each node's flow in `nodes` by. Positive factors remove
            water from the storage, negative factors remove it. When `None`, the flow is not scaled.
        min_volume : Optional[float | Parameter], default=0.0
            The minimum volume the storage is allowed to reach.
        max_volume : Optional[float | Parameter], default=0.0
            The maximum volume of the storage.
        initial_volume : Optional[float], default=None
            Specify initial volume in absolute terms.
        initial_volume_pc : Optional[float], default=None
            Specify initial volume in proportional terms. Both `initial_volume_pc` and `initial_volume` are required if `max_volume`
            is a parameter because the parameter will not be evaluated at the first time-step. If both are given
            and `max_volume` is not a Parameter, then the absolute value is ignored.
        cost : Optional[float | Parameter], default=None
            The cost of flow into/outfrom the storage. The cost property is not currently respected (see issue #242),
            therefore leave this to `None`.
        timesteps : Optional[int], default=0
            The number of timesteps to apply to the rolling storage over.
        days : Optional[int], default=None
            The number of days to apply the rolling storage over. Specifying a number of days (instead of a number
            of timesteps) is only valid with models running a timestep of daily frequency.

        Raises
        ------
        ValueError
            When `timesteps` is 0 and `days` is `None`.
        """
        min_volume = pop_kwarg_parameter(kwargs, "min_volume", 0.0)
        if min_volume is None:
            min_volume = 0.0
        max_volume = pop_kwarg_parameter(kwargs, "max_volume", 0.0)
        initial_volume = kwargs.pop("initial_volume", 0.0)
        initial_volume_pc = kwargs.pop("initial_volume_pc", None)
        cost = pop_kwarg_parameter(kwargs, "cost", 0.0)
        factors = kwargs.pop("factors", None)
        days = kwargs.pop("days", None)
        timesteps = kwargs.pop("timesteps", 0)

        if not timesteps and not days:
            raise ValueError("Either `timesteps` or `days` must be specified.")

        super().__init__(model, name, **kwargs)

        self.min_volume = min_volume
        self.max_volume = max_volume
        self.initial_volume = initial_volume
        self.initial_volume_pc = initial_volume_pc
        self.cost = cost
        self.nodes = nodes
        self.days = days
        self.timesteps = timesteps

        if factors is None:
            self.factors = [1.0 for i in range(len(nodes))]
        else:
            self.factors = factors

    def check(self):
        super().check()
        if self.cost not in (0.0, None):
            raise NotImplementedError(
                "RollingVirtualStorage does not currently support a non-zero cost."
            )

    def setup(self, model):
        if self.days is not None and self.days > 0:
            try:
                self.timesteps = self.days // self.model.timestepper.delta
            except TypeError:
                raise TypeError(
                    "A rolling period defined as a number of days is only valid with daily time-steps."
                )
        if self.timesteps < 1:
            raise ValueError(
                "The number of time-steps for a RollingVirtualStorage node must be greater than one."
            )
        super().setup(model)


class AnnualVirtualStorage(VirtualStorage):
    """A virtual storage which resets annually, useful for annual licences.

    Examples
    --------
    Reset the license using the financial year.

    Python
    ======
    ```python
    model = Model()
    node = Link(model=model, name="Track flow")
    AnnualVirtualStorage(model=model, reset_month=4, name="Annual license", max_volume=10, nodes=[node])
    ```

    JSON
    ======
    ```json
    {
        "name": "Annual license",
        "type": "AnnualVirtualStorage",
        "reset_month": 4,
        "max_volume": 10,
        "nodes": ["Track flow"]
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    name : str
        The unique name of the node.
    nodes : list[Node]
        List of inflow/outflow nodes that affect the storage volume.
    factors : Optional[list[float]], default=None
        List of factors to multiply each node's flow in `nodes` by. Positive factors remove
        water from the storage, negative factors remove it. When `None`, the flow is not scaled.
    min_volume : Optional[float | Parameter], default=0.0
            The minimum volume the storage is allowed to reach.
    max_volume : Optional[float | Parameter], default=0.0
        The maximum volume of the storage.
    initial_volume : Optional[float], default=None
        Specify initial volume in absolute terms.
    initial_volume_pc : Optional[float], default=None
        Specify initial volume in proportional terms. Both `initial_volume_pc` and `initial_volume` are required if `max_volume`
        is a parameter because the parameter will not be evaluated at the first time-step. If both are given
        and `max_volume` is not a Parameter, then the absolute value is ignored.
    cost : Optional[float | Parameter], default=None
        The cost of flow into/outfrom the storage. The cost property is not currently respected (see issue #242),
        therefore leave this to `None`.
    reset_day: int
        The day of the month (0-31) to reset the volume to the initial value.
    reset_month: int
        The month of the year (0-12) to reset the volume to the initial value.
    reset_to_initial_volume: bool
        Reset the volume to the initial volume instead of maximum volume each year.

    Notes
    -----
    The cost property is not currently respected (see issue #242), therefore leave this to `None`.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        nodes : list[Node]
            List of inflow/outflow nodes that affect the storage volume.
        factors : Optional[list[float]], default=None
            List of factors to multiply each node's flow in `nodes` by. Positive factors remove
            water from the storage, negative factors remove it. When `None`, the flow is not scaled.
        min_volume : Optional[float | Parameter], default=0.0
            The minimum volume the storage is allowed to reach.
        max_volume : Optional[float | Parameter], default=0.0
            The maximum volume of the storage.
        initial_volume : Optional[float], default=None
            Specify initial volume in absolute terms.
        initial_volume_pc : Optional[float], default=None
            Specify initial volume in proportional terms. Both `initial_volume_pc` and `initial_volume` are required if `max_volume`
            is a parameter because the parameter will not be evaluated at the first time-step. If both are given
            and `max_volume` is not a Parameter, then the absolute value is ignored.
        cost : Optional[float | Parameter], default=None
            The cost of flow into/outfrom the storage. The cost property is not currently respected (see issue #242),
            therefore leave this to `None`.
        reset_day: Optional[int], default=1
            The day of the month (1-31) to reset the volume to the initial value.
        reset_month: Optional[int], default=1
            The month of the year (1-12) to reset the volume to the initial value.
        reset_to_initial_volume: Optional[bool], default=False
            Reset the volume to the initial volume instead of maximum volume each year (default is False).
        """
        self.reset_day = kwargs.pop("reset_day", 1)
        self.reset_month = kwargs.pop("reset_month", 1)
        self.reset_to_initial_volume = kwargs.pop("reset_to_initial_volume", False)
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
            if ts.month > self.reset_month or (
                ts.month == self.reset_month and ts.day >= self.reset_day
            ):
                # Reset maximum volume depending on user preference ...
                use_initial_volume = self.reset_to_initial_volume
                if ts.index == 0 and isinstance(self.max_volume, Parameter):
                    # ... if it is the first timestep and max volume is a parameter, then we can only reset to
                    # initial volume because the max volume would not be evaluated
                    use_initial_volume = True

                self._reset_storage_only(use_initial_volume=use_initial_volume)
                self._last_reset_year = ts.year
                self.active = True


class SeasonalVirtualStorage(AnnualVirtualStorage):
    """A virtual storage node that operates only for a specified period within a year.

    This node is most useful for representing licences that are only enforced during specified periods. The
    `reset_day` and `reset_month` parameters indicate when the node starts operating and the `end_day` and `end_month`
    when it stops operating. For the period when the node is not operating, the volume of the node remains unchanged
    and the node does not apply any constraints to the model.

    The `end_day` and `end_month` represents a date when the node stops operating. This
    situation represents a licence that operates across a year boundary. For example, one that is active between
    October and March and not active between April and September.

    Attributes
    ----------
    reset_day : int
        The day of the month (0-31) when the node starts operating and its volume is reset to the initial value or
        maximum volume.
    reset_month : int
        The month of the year (0-12) when the node starts operating and its volume is reset to the initial value or
        maximum volume.
    reset_to_initial_volume : bool
        Reset the volume to the initial volume instead of maximum volume each year (default is False).
    end_day : int
        The day of the month (0-31) when the node stops operating.
    end_month : int
        The month of the year (0-12) when the node stops operating.

    Notes
    -----
    The cost property is not currently respected (see issue #242), therefore leave this to `None`.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        nodes : list[Node]
            List of inflow/outflow nodes that affect the storage volume.
        factors : Optional[list[float]], default=None
            List of factors to multiply each node's flow in `nodes` by. Positive factors remove
            water from the storage, negative factors remove it. When `None`, the flow is not scaled.
        min_volume : Optional[float | Parameter], default=0.0
            The minimum volume the storage is allowed to reach.
        max_volume : Optional[float | Parameter], default=0.0
            The maximum volume of the storage.
        initial_volume : Optional[float], default=None
            Specify initial volume in absolute terms.
        initial_volume_pc : Optional[float], default=None
            Specify initial volume in proportional terms. Both `initial_volume_pc` and `initial_volume` are required if `max_volume`
            is a parameter because the parameter will not be evaluated at the first time-step. If both are given
            and `max_volume` is not a Parameter, then the absolute value is ignored.
        cost : Optional[float | Parameter], default=None
            The cost of flow into/outfrom the storage. The cost property is not currently respected (see issue #242),
            therefore leave this to `None`.
        reset_day : int
            The day of the month (1-31) when the node starts operating and its volume is reset to the initial value or
            maximum volume.
        reset_month : int
            The month of the year (1-12) when the node starts operating and its volume is reset to the initial value or
            maximum volume.
        reset_to_initial_volume : Optional[bool], default=False
            Reset the volume to the initial volume instead of maximum volume each year
        end_day : Optional[int], default=31
            The day of the month (1-31) when the node stops operating.
        end_month : Optional[int], default=12
            The month of the year (1-12) when the node stops operating.
        """
        self.end_day = kwargs.pop("end_day", 31)
        self.end_month = kwargs.pop("end_month", 12)
        self._last_active_year = None

        super().__init__(*args, **kwargs)

    def before(self, ts):
        super().before(ts)

        if ts.year != self._last_active_year:
            if ts.index == 0:
                if self._last_reset_year == ts.year:
                    # First timestep is later in year than reset date
                    if self.end_month < self.reset_month or (
                        self.end_month == self.reset_month
                        and self.end_day <= self.reset_day
                    ):
                        # end date is earlier in year than reset date so do not deactivate node in first year
                        self._last_active_year = ts.year
                else:
                    # First timestep is earlier in year than reset date
                    if self.end_month > self.reset_month or (
                        self.end_month == self.reset_month
                        and self.end_day >= self.reset_day
                    ):
                        # end date is later in year than reset date so node needs to be deactivated
                        self.active = False
            elif ts.month > self.end_month or (
                ts.month == self.end_month and ts.day >= self.end_day
            ):
                self._last_active_year = ts.year
                self.active = False


class MonthlyVirtualStorage(VirtualStorage):
    """A virtual storage that resets after a given number of months.

    Examples
    --------
    Reset the license after 3 months.

    Python
    ======
    ```python
    model = Model()
    node = Link(model=model, name="Track flow")
    MonthlyVirtualStorage(model=model, month=3, name="Monthly license", max_volume=10, nodes=[node])
    ```

    JSON
    ======
    ```json
    {
        "name": "Monthly license",
        "type": "MonthlyVirtualStorage",
        "month": 3,
        "max_volume": 10,
        "nodes": ["Track flow"]
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    name : str
        The unique name of the node.
    nodes : list[Node]
        List of inflow/outflow nodes that affect the storage volume.
    factors : Optional[list[float]], default=None
        List of factors to multiply each node's flow in `nodes` by. Positive factors remove
        water from the storage, negative factors remove it. When `None`, the flow is not scaled.
    months : int
        The number of months after which the storage volume resets.
    initial_months : int
        The number of months into the reset period the storages is at when the model run starts.
    reset_to_initial_volume : bool
        Reset the volume to the initial volume instead of maximum volume each year.

    Notes
    -----
    The cost property is not currently respected (see issue #242), therefore leave this to `None`.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        nodes : list[Node]
            List of inflow/outflow nodes that affect the storage volume.
        factors : Optional[list[float]], default=None
            List of factors to multiply each node's flow in `nodes` by. Positive factors remove
            water from the storage, negative factors remove it. When `None`, the flow is not scaled.
        min_volume : Optional[float | Parameter], default=0.0
            The minimum volume the storage is allowed to reach.
        max_volume : Optional[float | Parameter], default=0.0
            The maximum volume of the storage.
        initial_volume : Optional[float], default=None
            Specify initial volume in absolute terms.
        initial_volume_pc : Optional[float], default=None
            Specify initial volume in proportional terms. Both `initial_volume_pc` and `initial_volume` are required if `max_volume`
            is a parameter because the parameter will not be evaluated at the first time-step. If both are given
            and `max_volume` is not a Parameter, then the absolute value is ignored.
        cost : Optional[float | Parameter], default=None
            The cost of flow into/outfrom the storage. The cost property is not currently respected (see issue #242),
            therefore leave this to `None`.
        months : Optional[int], default 1
            The number of months after which the storage volume resets.
        initial_months : Optional[int], default 0
            The number of months into the reset period the storages is at when the model run starts.
        reset_to_initial_volume : Optional[bool], default=False
            Reset the volume to the initial volume instead of maximum volume each year (default is False).
        """
        self.months = kwargs.pop("months", 1)
        self.initial_months = kwargs.pop("initial_months", 0)
        self.reset_to_initial_volume = kwargs.pop("reset_to_initial_volume", False)
        self._count = self.initial_months - 1
        self._last_month = None
        super(MonthlyVirtualStorage, self).__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self._count = self.initial_months - 1
        self._last_month = None

    def before(self, ts):
        super().before(ts)
        if ts.month != self._last_month:
            self._last_month = ts.month
            self._count += 1
            if self._count == self.months:
                self._count = 0
                self._reset_storage_only(
                    use_initial_volume=self.reset_to_initial_volume
                )


class PiecewiseLink(Node):
    """An extension of Node that represents a non-linear Link with a piece-wise cost function.

    This object is intended to model situations where there is a benefit of supplying certain flow rates,
    but beyond a fixed limit, there is a change in (or zero) cost.

    Attributes
    ---------
    max_flows : Iterable[float]
        A monotonic increasing list of maximum flows for the piece wise function
    costs : Iterable[float]
        A list of costs corresponding to the max_flow steps

    Notes
    -----
    This node is implemented using a compound node structure like so:

    ```mermaid
    graph LR
        Output --> Sublink_0
        Output --> Sublink_1
        Output --> Sublink_n
        Sublink_0 --> Sub_Output
        Sublink_1 --> Sub_Output
        Sublink_n --> Sub_Output
        Sub_Output --> Input
    ```

    This means routes do not directly traverse this node due to the separate
    domain in the middle. Instead, several new routes are made for each of
    the sub-links and connections to the Output/Input node. The reason for this
    breaking of the route is to avoid a geometric increase in the number
    of routes when multiple `PiecewiseLinks` are present in the same route.
    """

    __parameter_attributes__ = ("costs", "max_flows")

    def __init__(self, model, nsteps, *args, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        nsteps : int
            The number of internal piecewise links to add.

        Other Parameters
        ----------------
        max_flows : Optional[Iterable[float]] = None
            A monotonic increasing list of maximum flows for the piece wise function
        costs : Optional[Iterable[float]] = None
            A list of costs corresponding to the max_flow steps

        Other parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        """
        self.allow_isolated = True
        name = kwargs.pop("name")
        costs = kwargs.pop("costs", None)
        max_flows = kwargs.pop("max_flows", None)

        # TODO look at the application of Domains here. Having to use
        # Input/Output instead of BaseInput/BaseOutput because of a different
        # domain is required on the sub-nodes and they need to be connected
        self.sub_domain = Domain()
        self.input = Input(model, name="{} Input".format(name), parent=self)
        self.output = Output(model, name="{} Output".format(name), parent=self)

        self.sub_output = Output(
            model,
            name="{} Sub Output".format(name),
            parent=self,
            domain=self.sub_domain,
        )
        self.sub_output.connect(self.input)
        self.sublinks = []
        for i in range(nsteps):
            sublink = Input(
                model,
                name="{} Sublink {}".format(name, i),
                parent=self,
                domain=self.sub_domain,
            )
            self.sublinks.append(sublink)
            sublink.connect(self.sub_output)
            self.output.connect(self.sublinks[-1])

        super().__init__(model, *args, name=name, **kwargs)

        if costs is not None:
            self.costs = costs
        if max_flows is not None:
            self.max_flows = max_flows

    def get_min_flow(self, si: "ScenarioIndex"):
        """Get the total minimum flow through the sub link.

        Parameters
        ----------
        si: ScenarioIndex
            The scenario index.

        Returns
        -------
        float
            The total min flow.
        """
        return sum([sl.get_min_flow(si) for sl in self.sublinks])

    def get_max_flow(self, si: "ScenarioIndex"):
        """Get the total maximum flow through the sub link.

        Parameters
        ----------
        si: ScenarioIndex
            The scenario index.

        Returns
        -------
        float
            The total max flow.
        """
        return sum([sl.get_max_flow(si) for sl in self.sublinks])

    def costs():
        def fget(self):
            return [sl.cost for sl in self.sublinks]

        def fset(self, values):
            if len(self.sublinks) != len(values):
                raise ValueError(
                    f"Piecewise costs must be the same length as the number of "
                    f"sub-links ({len(self.sublinks)})."
                )
            for i, sl in enumerate(self.sublinks):
                sl.cost = values[i]

        return locals()

    costs = property(**costs())

    def max_flows():
        def fget(self):
            return [sl.max_flow for sl in self.sublinks]

        def fset(self, values):
            if len(self.sublinks) != len(values):
                raise ValueError(
                    f"Piecewise max_flows must be the same length as the number of "
                    f"sub-links ({len(self.sublinks)})."
                )
            for i, sl in enumerate(self.sublinks):
                sl.max_flow = values[i]

        return locals()

    max_flows = property(**max_flows())

    def iter_slots(self, slot_name=None, is_connector=True):
        if is_connector:
            yield self.input
        else:
            yield self.output

    def after(self, timestep):
        """
        Set total flow on this link as sum of sublinks
        """
        for lnk in self.sublinks:
            self.commit_all(lnk.flow)
        # Make sure save is done after setting aggregated flow
        super(PiecewiseLink, self).after(timestep)


class MultiSplitLink(PiecewiseLink):
    """An extension of PiecewiseLink that includes additional slots to connect from.

    Conceptually, this node looks like the following internally:

    ```mermaid
    graph LR
        A --> X0
        X0 --> Xi
        A --> Xo
        Xo --> X1
        X1 --> Xi
        Xo --> X2
        X2 --> Xi
        Xi --> C
        X2 --> Bo
        Bo --> Bi
        Bi --> D
    ```

    An additional sub-link in the PiecewiseLink (i.e., X2 above) and nodes
    (i.e., Bo and Bi) in this class are added for each extra slot.

    Finally, a mechanism is provided to (optionally) fix the ratio between the
    last non-split sub-link (i.e., X1) and each of the extra sub-links (i.e., X2).
    This mechanism uses `AggregatedNode` internally.

    Notes
    -----
    Users must be careful when using the factor mechanism. Factors use the last
    non-split sub-link (i.e., X1 but not X0). If this link is constrained with a
    maximum or minimum flow, or if it there is another unconstrained link
    (i.e., if X0 is unconstrained) then ratios across this whole node may not be
    enforced as expected.
    """

    def __init__(self, model, nsteps, *args, **kwargs):
        """Initialise the node.
        Parameters
        ----------
        model : Model
            The model instance.
        nsteps : int
            The number of sub-links.

        Other Parameters
        ----------------
        name : str
            The unique name of the node.
        max_flows : Iterable[float]
            A monotonic increasing list of maximum flows for the piece wise function
        costs : Iterable[float]
            A list of costs corresponding to the `max_flow` steps
        extra_slots : Optional[int], default=1
            Number of additional slots (and sub-links) to provide. Must be greater
            than zero.
        slot_names : Optional[Iterable[str | int]], default=range of ints
            The names by which to refer to the slots during connection to other
            nodes. Length must be one more than the number of extra_slots. The first
            item refers to the `PiecewiseLink` connection with the following items for
            each extra slot.
        factors : Optional[Iterable[float]], default=None
            If given, the length must be equal to one more than the number of
            extra_slots. Each item is the proportion of total flow to pass through
            the additional sub-links. If no factor is required for a particular
            sub-link then use `None` for its items. Factors are normalised prior to
            use in the solver.
        """
        self.allow_isolated = True
        costs = kwargs.pop("costs", None)
        max_flows = kwargs.pop("max_flows", None)

        extra_slots = kwargs.pop("extra_slots", 1)
        if extra_slots < 1:
            raise ValueError("extra_slots must be at least 1.")

        # Default to integer names
        self.slot_names = list(kwargs.pop("slot_names", range(extra_slots + 1)))
        if extra_slots + 1 != len(self.slot_names):
            raise ValueError(
                "slot_names must be one more than the number of extra_slots."
            )

        factors = kwargs.pop("factors", None)
        # Finally initialise the parent.
        super(MultiSplitLink, self).__init__(
            model, nsteps + extra_slots, *args, **kwargs
        )

        self._extra_inputs = []
        self._extra_outputs = []
        n = len(self.sublinks) - extra_slots
        for i in range(extra_slots):
            # create a new input inside the piecewise link which only has access
            # to flow travelling via the last sublink (X2)
            otpt = Output(
                self.model,
                "{} Extra Output {}".format(self.name, i),
                domain=self.sub_domain,
                parent=self,
            )
            inpt = Input(
                self.model, "{} Extra Input {}".format(self.name, i), parent=self
            )

            otpt.connect(inpt)
            self.sublinks[n + i].connect(otpt)

            self._extra_inputs.append(inpt)
            self._extra_outputs.append(otpt)

        if costs is not None:
            # No cost or maximum flow on the additional links
            self.costs = costs + [0.0] * extra_slots
        if max_flows is not None:
            # The max_flows could be problematic with the aggregated node.
            self.max_flows = max_flows + [None] * extra_slots

        # Now create an aggregated node for addition constraints if required.
        if factors is not None:
            if extra_slots + 1 != len(factors):
                raise ValueError("factors must have a length equal to extra_slots.")

            nodes = []
            valid_factors = []
            for r, nd in zip(factors, self.sublinks[n - 1 :]):
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
                yield self._extra_inputs[i - 1]
        else:
            yield self.output


class AggregatedStorage(
    Loadable, Drawable, _core.AggregatedStorage, metaclass=NodeMeta
):
    """An aggregated sum of other [pywr.nodes.Storage][] nodes.

    This object should behave like [pywr.nodes.Storage][] by returning the current `flow`,
    `volume` and `current_pc` as sum of the flows and volumes of the nodes provided in
    `storage_nodes`. However, this object can not be connected to others within the network.

    Examples
    --------
    Python
    ======
    ```python
    model = Model()
    st1 = Storage(model=model, name="Storage 1", max_volume=10)
    st2 = Storage(model=model, name="Storage 2", max_volume=2)
    AggregatedStorage(model=model, storage_nodes=[st1, st2], name="Combined")
    ```

    JSON
    ======
    ```json
    {
        "name": "Combined",
        "type": "AggregatedStorage",
        "storage_nodes": ["Storage 1", "Storage 2"]
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    name : str
        The unique name of the node.
    storage_nodes : list[Storage]
        The `Storage` objects to return the sum of

    Notes
    -----
    This node cannot be connected to other nodes in the network.
    """

    __node_attributes__ = ("storage_nodes",)

    def __init__(self, model, name, storage_nodes, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        storage_nodes : list[Storage]
            The `Storage` objects to return the sum of

        """
        super(AggregatedStorage, self).__init__(model, name, **kwargs)
        self.storage_nodes = storage_nodes


class AggregatedNode(Loadable, Drawable, _core.AggregatedNode, metaclass=NodeMeta):
    """An aggregated sum of other [pywr.nodes.Node][] nodes.

    This object should behave like [pywr.nodes.Node][] by returning the current `flow`
    as a sum of the flow through the nodes provided in `nodes`. However, it cannot be
    connected to others within the network.

    Examples
    --------
    Python
    ======
    ```python
    model = Model()
    link_node = Link(model=model, name="Works", max_flow=2)
    input_node = Input(model=model, name="Input", min_flow=0.5)
    AggregatedNode(model=model, nodes=[link_node, input_node], name="Combined flow")
    ```

    JSON
    ======
    ```json
    {
        "name": "Combined",
        "type": "AggregatedNode",
        "nodes": ["Works", "Input"]
    }
    ```

    Attributes
    ----------
    nodes : list[Node]
        The `Node` objects which to return the sum total of
    flow_weights : Optional[list[float]]
        Scale the flow of each node by the given weights.

    Notes
    -----
    This node cannot be connected to other nodes in the network.
    """

    __parameter_attributes__ = ("factors", "min_flow", "max_flow")
    __node_attributes__ = ("nodes",)

    def __init__(self, model, name, nodes, flow_weights=None, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.
        nodes : list[Node]
            The `Node` objects which to return the sum total of
        flow_weights : Optional[list[float]], default=None
            Scale the flow of each node by the given weights. When `None` weights are ignored.
        """
        super(AggregatedNode, self).__init__(model, name, **kwargs)
        self.nodes = nodes
        self.flow_weights = flow_weights


class BreakLink(Node):
    """Compound node used to reduce the number of routes in a model

    Attributes
    ----------
    link : Link
        The link node connect to the storage node.
    storage : Storage
        The storage node.

    Notes
    -----

    In a model with form (3, 1, 3), i.e. 3 (A,B,C) inputs connected to 3
    outputs (D,E,F) via a bottleneck (X), there are 3*3 routes = 9 routes.

    ```mermaid
    graph LR
        A --> X
        B --> X
        C --> X
        X --> D
        X --> E
        X --> F
    ```
    If X is a storage, there are only 6 routes: A -> X<sub>_o</sub>, B -> X<sub>_o</sub>, C -> X<sub>_o</sub> and
    X<sub>_i</sub> -> D, X<sub>_i</sub> -> E, X<sub>_i</sub> -> F_o, where `_o` indicates
    the output node of the storage and `_i` its input.

    The `BreakLink` node is a compound node composed of a [pywr.nodes.Storage]() with zero
    volume and a [pywr.nodes.Link][]. It can be used in place of a normal [pywr.nodes.Link][], but
    with the benefit that it reduces the number of routes in the model (in
    the situation described above). The resulting Linear Porgramming problem is easier to solve.
    """

    allow_isolated = True

    def __init__(self, model, name, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.

        Other Parameters
        ----------------
        min_flow : Optional[float | Parameter], default=0
            A simple minimum flow constraint for the node. Default to 0.
        max_flow : Optional[float | Parameter], default=Inf
            A simple maximum flow constraint for the node. Defaults to infinite.
        cost : Optional[float | Parameter], default=0
            The cost of supply.
        """
        storage_name = "{} (storage)".format(name)
        link_name = "{} (link)".format(name)
        assert storage_name not in model.nodes
        assert link_name not in model.nodes
        self.storage = Storage(
            model,
            name=storage_name,
            min_volume=0,
            max_volume=0,
            initial_volume=0,
            cost=0,
        )
        self.link = Link(model, name=link_name)

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


class DelayNode(Node):
    """A node that delays flow for a given number of time-steps or days.

    This is a composite node consisting internally of a [pywr.nodes.Input][] and an [pywr.nodes.Output][] node. A
    [pywr.parameters.FlowDelayParameter][] is used to delay the flow of the output node for a given period prior
    to this delayed flow being set as the flow of the input node. Connections to the node are connected
    to the internal output node and connections from the node are connected to the internal input
    node.

    Attributes
    ----------
    input : Input
        The internal input node.
    output : Output
        The internal output node.
    delay_param : FlowDelayParameter
        The parameter applied to the min_flow and max_flow attributes of
        the input node.

    Examples
    --------
    Python
    ======
    ```python
    model = Model()
    DelayNode(model=model, days=2, name="Release lag", cost=3)
    ```

    JSON
    ======
    ```json
    {
        "name": "Release lag",
        "type": "DelayNode",
        "days": 2,
        "cost": 3
    }
    ```
    """

    def __init__(self, model, name, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.

        Other Parameters
        ----------------
        timesteps: [int], default=0
            Number of time steps to delay the flow.
        days: Optional[int], default=0
            Number of days to delay the flow. Specifying a number of days (instead of a number
            of time steps) is only valid if the number of days is exactly divisible by the model
            timestep delta.
        initial_flow: [float], default=0
            Flow provided by node for initial time steps prior to any delayed flow being available.
            This is constant across all delayed time steps and any model scenarios. Default is 0.0.
        min_flow : Optional[float | Parameter], default=0
            A simple minimum flow constraint for the node. Defaults to 0.
        max_flow : Optional[float | Parameter], default=Inf
            A simple maximum flow constraint for the node. Defaults to infinite.
        cost : Optional[float | Parameter], default=0
            The cost of supply.
        """
        self.allow_isolated = True
        output_name = "{} Output".format(name)
        input_name = "{} Input".format(name)
        param_name = "{} - delay parameter".format(name)
        assert output_name not in model.nodes
        assert input_name not in model.nodes
        assert param_name not in model.parameters

        days = kwargs.pop("days", 0)
        timesteps = kwargs.pop("timesteps", 0)
        initial_flow = kwargs.pop("initial_flow", 0.0)

        self.output = Output(model, name=output_name, parent=self)
        self.delay_param = FlowDelayParameter(
            model,
            self.output,
            timesteps=timesteps,
            days=days,
            initial_flow=initial_flow,
            name=param_name,
        )
        self.input = Input(
            model,
            name=input_name,
            min_flow=self.delay_param,
            max_flow=self.delay_param,
            parent=self,
        )
        super().__init__(model, name, **kwargs)

    def iter_slots(self, slot_name=None, is_connector=True):
        if is_connector:
            yield self.input
        else:
            yield self.output

    def after(self, timestep):
        super().after(timestep)
        # delayed flow is saved to the DelayNode
        self.commit_all(self.input.flow)


class LossLink(Node):
    """A node that has a proportional loss.

    A fixed proportional loss of all flows through this node is sent to an internal output node. Max and min flows
    applied to this node are enforced on the net output after losses. The node itself records the net output
    in its flow attribute (which would be used by any attached recorders).

    Attributes
    ----------
    output : Output
        The output node where the lost flow is routed.
    gross : Link
        The link node for the gross flow..
    net : Link
        The link node for the net flow.
    agg : AggregatedNode
        The aggregated node used to enforce the flow constraints.
    loss_factor_type : Literal["gross", "net"]
        The type of loss.
    loss_factor : float
        The loss factor.

    Examples
    --------
    Python
    ======
    ```python
    model = Model()
    LossLink(model=model, loss_factor=0.02, name="Works")
    ```

    JSON
    ======
    ```json
    {
        "name": "Works",
        "type": "LossLink",
        "loss_factor": 0.02
    }
    ```

    Notes
    -----
    1. There is currently a limitation that the loss factor must be a literal constant (i.e. not a parameter) when
    `loss_factor_type` is set to "gross".
    2. Any recorder attached to this node records the net output.
    """

    __parameter_attributes__ = ("loss_factor", "max_flow", "min_flow", "cost")

    def __init__(self, model, name, **kwargs):
        """Initialise the node.

        Parameters
        ----------
        model : Model
            The model instance.
        name : str
            The unique name of the node.

        Other Parameters
        ----------------
        min_flow : Optional[float | Parameter], default=0
            A simple minimum flow constraint for the node. Default to 0.
        max_flow : Optional[float | Parameter], default=Inf
            A simple maximum flow constraint for the node. Defaults to infinite.
        cost : Optional[float | Parameter], default=0
            The cost of supply.
        loss_factor : float | Parameter
            The proportion of flow that is lost through this node. Must be greater than or equal to zero. If zero
            then no-losses are calculated. This value is either a proportion of gross or net flow depending on the
            value of `loss_factor_type`.
        loss_factor_type : Optional[Literal["gross", "net"]], default="net"
            Either "gross" or "net" to specify whether the loss factor is applied as a proportion of gross or
             net flow respectively.
        """
        self.allow_isolated = True

        output_name = "{} Output".format(name)
        gross_name = "{} Gross".format(name)
        net_name = "{} Net".format(name)
        agg_name = "{} Aggregated".format(name)

        assert output_name not in model.nodes
        assert gross_name not in model.nodes
        assert net_name not in model.nodes
        assert agg_name not in model.nodes

        self.output = Output(model, name=output_name, parent=self)
        self.gross = Link(model, name=gross_name, parent=self)
        self.net = Link(model, name=net_name, parent=self)
        self.gross.connect(self.output)
        self.gross.connect(self.net)

        self.agg = AggregatedNode(model, name=agg_name, nodes=[self.net, self.output])
        self.loss_factor_type = kwargs.pop("loss_factor_type", "net")
        self.loss_factor = kwargs.pop("loss_factor", 0.0)

        super().__init__(model, name, **kwargs)

    def setup(self, model):
        super().setup(model)
        value = self.loss_factor

        if value == 0.0:
            # 0% loss; no flow to the output loss node.
            self.agg.factors = None
            self.output.max_flow = 0.0
        elif value == 1.0 and self.loss_factor_type == "gross":
            # 100% loss; all flow to the output loss node
            self.agg.factors = None
            self.output.max_flow = float("inf")
            self.net.max_flow = 0.0
        else:
            self.output.max_flow = float("inf")
            if self.loss_factor_type == "net":
                self.agg.factors = [1.0, value]

            elif self.loss_factor_type == "gross":
                # TODO this will error in the case of a `value` being a parameter
                self.agg.factors = [1.0 - float(value), float(value)]
            else:
                raise ValueError(
                    f'Unrecognised `loss_factor_type` "{self.loss_factor_type}".'
                    f'Please use either "gross" or "net".'
                )

    def min_flow():
        def fget(self):
            return self.net.min_flow

        def fset(self, value):
            self.net.min_flow = value

        return locals()

    min_flow = property(**min_flow())

    def max_flow():
        def fget(self):
            return self.net.max_flow

        def fset(self, value):
            self.net.max_flow = value

        return locals()

    max_flow = property(**max_flow())

    def cost():
        def fget(self):
            return self.net.cost

        def fset(self, value):
            self.net.cost = value

        return locals()

    cost = property(**cost())

    def iter_slots(self, slot_name=None, is_connector=True):
        if is_connector:
            yield self.net
        else:
            yield self.gross

    def after(self, timestep):
        super().after(timestep)
        # Net flow is saved to the node.
        self.commit_all(self.net.flow)


class ShadowStorage(Loadable, Drawable, _core.ShadowStorage, metaclass=NodeMeta):
    pass


class ShadowNode(Loadable, Drawable, _core.ShadowNode, metaclass=NodeMeta):
    pass


from pywr.domains.river import *  # noqa
