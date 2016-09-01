#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from pywr import _core
# Cython objects availble in the core namespace
from pywr.parameters import *
from pywr._core import BaseInput, BaseLink, BaseOutput, \
    StorageInput, StorageOutput, Timestep, ScenarioIndex
from pywr.parameters._parameters import Parameter as BaseParameter
from pywr._core import Node as BaseNode
from pywr.recorders import load_recorder
import os
import networkx as nx
import inspect
import pandas
import datetime
import json
from six import with_metaclass
import copy

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = UnicodeWarning)


class Timestepper(object):
    def __init__(self, start="2015-01-01", end="2015-12-31", delta=1):
        self.start = start
        self.end = end
        self.delta = delta
        self._last_length = None
        self.reset()

    def __iter__(self, ):
        return self

    def __len__(self, ):
        return int((self.end-self.start)/self.delta) + 1

    def reset(self, start=None):
        """ Reset the timestepper

        If start is None it resets to the original self.start, otherwise
        start is used as the new starting point.
        """
        self._current = None
        current_length = len(self)

        if start is None:
            self._next = _core.Timestep(self.start, 0, self.delta.days)
        else:
            # Calculate actual index from new position
            diff = start - self.start
            if diff.days % self.delta.days != 0:
                raise ValueError('New starting position is not compatible with the existing starting position and timestep.')
            index = diff.days / self.delta.days
            self._next = _core.Timestep(start, index, self.delta.days)

        length_changed = self._last_length != current_length
        self._last_length = current_length
        return length_changed

    def __next__(self, ):
        return self.next()

    def next(self, ):
        self._current = current = self._next
        if current.datetime > self.end:
            raise StopIteration()

        # Increment to next timestep
        self._next = _core.Timestep(current.datetime + self.delta, current.index + 1, self.delta.days)

        # Return this timestep
        return current

    def start():
        def fget(self):
            return self._start
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._start = value
            else:
                self._start = pandas.to_datetime(value)
        return locals()
    start = property(**start())

    def end():
        def fget(self):
            return self._end
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._end = value
            else:
                self._end = pandas.to_datetime(value)
        return locals()
    end = property(**end())

    def delta():
        def fget(self):
            return self._delta
        def fset(self, value):
            try:
                self._delta = pandas.Timedelta(days=value)
            except TypeError:
                self._delta = pandas.to_timedelta(value)
        return locals()
    delta = property(**delta())

    @property
    def current(self, ):
        """ Return the current Timestep.

        If iteration has not begun this will return the starting Timestep.
        """
        if self._current is None:
            return self._next
        return self._current

    @property
    def datetime_index(self):
        """ Return a `pandas.DatetimeIndex` using the start, end and delta of this object

        This is useful for creating `pandas.DataFrame` objects from Model results
        """
        freq = '{}D'.format(self.delta.days)
        return pandas.date_range(self.start, self.end, freq=freq)


class Scenario(_core.Scenario):
    pass



class ScenarioCollection(_core.ScenarioCollection):
    pass


class NodeIterator(object):
    """Iterator for Nodes in a Model which also supports indexing

    Notes
    -----
    Although it's not very efficient to have to read through all of the nodes
    in a model when accessing one by name (e.g. model.nodes['reservoir']), it's
    easier than having to keep a dictionary up to date. The solvers should
    avoid using this class, and use Model.graph.nodes() directly.
    """
    def __init__(self, model):
        self.model = model
        self.position = 0
        self.length = None

    def _nodes(self, hide_children=True):
        for node in self.model.graph.nodes():
            if hide_children is False or node.parent is None:  # don't return child nodes (e.g. StorageInput)
                yield node
        raise StopIteration()

    def __getitem__(self, key):
        """Get a node from the graph by it's name"""
        for node in self._nodes(hide_children=False):
            if node.name == key:
                return node
        raise KeyError("'{}'".format(key))

    def __delitem__(self, key):
        """Remove a node from the graph by it's name"""
        node = self[key]
        self.model.graph.remove_node(node)

    def keys(self):
        for node in self._nodes():
            yield node.name

    def values(self):
        for node in self._nodes():
            yield node

    def items(self):
        for node in self._nodes():
            yield (node.name, node)

    def __len__(self):
        """Returns the number of nodes in the model"""
        return len(list(self._nodes()))

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.position == 0:
            self.nodes = list(self._nodes())
            self.length = len(self.nodes)
        if self.position < self.length:
            node = self.nodes[self.position]
            self.position += 1
            return node
        raise StopIteration()

    def __call__(self):
        # support for old API
        return self


class NamedIterator(object):
    def __init__(self):
        self._objects = []

    def __getitem__(self, key):
        """Get a node from the graph by it's name"""
        for obj in self._objects:
            if obj.name == key:
                return obj
        raise KeyError("'{}'".format(key))

    def __delitem__(self, key):
        """Remove a node from the graph by it's name"""
        obj = self[key]
        self._objects.remove(rec)

    def __setitem__(self, key, obj):
        # TODO: check for name collisions / duplication
        self._objects.append(obj)

    def keys(self):
        for obj in self._objects:
            yield obj.name

    def values(self):
        for obj in self._objects:
            yield obj

    def items(self):
        for obj in self._objects:
            yield (obj.name, obj)

    def __len__(self):
        """Returns the number of nodes in the model"""
        return len(self._objects)

    def __iter__(self):
        return iter(self._objects)

    def append(self, obj):
        # TODO: check for name collisions / duplication
        self._objects.append(obj)


class Model(object):
    """Model of a water supply network"""
    def __init__(self, **kwargs):
        """Initialise a new Model instance

        Parameters
        ----------
        solver : string
            The name of the underlying solver to use. See the `pywr.solvers`
            package. If no value is given, the default GLPK solver is used.
        start : pandas.Timestamp
            The date of the first timestep in the model
        end : pandas.Timestamp
            The date of the last timestep in the model
        timestep : int or datetime.timedelta
            Number of days in each timestep
        """
        self.graph = nx.DiGraph()
        self.metadata = {}

        solver_name = kwargs.pop('solver', None)

        # time arguments
        start = kwargs.pop("start", "2015-01-01")
        end = kwargs.pop("end", "2015-12-31")
        timestep = kwargs.pop("timestep", 1)
        self.timestepper = Timestepper(start, end, timestep)

        self.data = {}
        self.failure = set()
        self.dirty = True

        self.path = kwargs.pop('path', None)
        if self.path is not None:
            if os.path.exists(self.path) and not os.path.isdir(self.path):
                self.path = os.path.dirname(self.path)

        # Import this here once everything else is defined.
        # This avoids circular references in the solver classes
        from .solvers import solver_registry
        if solver_name is not None:
            # use specific solver
            solver = None
            name1 = solver_name.lower()
            for cls in solver_registry:
                if name1 == cls.name.lower():
                    solver = cls
            if solver is None:
                raise KeyError('Unrecognised solver: {}'.format(solver_name))
        else:
            # use default solver
            solver = solver_registry[0]
        self.solver = solver()

        self.parameters = NamedIterator()
        self.recorders = NamedIterator()
        self.scenarios = ScenarioCollection()

        if kwargs:
            key = list(kwargs.keys())[0]
            raise TypeError("'{}' is an invalid keyword argument for this function".format(key))

        self.reset()

    def check(self):
        """Check the validity of the model

        Raises an Exception if the model is invalid.
        """
        nodes = self.graph.nodes()
        for node in nodes:
            node.check()
        self.check_graph()

    def check_graph(self):
        all_nodes = set(self.graph.nodes())
        routes = self.find_all_routes(BaseInput, BaseOutput, valid=(BaseLink, BaseInput, BaseOutput))
        # identify nodes that aren't in at least one route
        seen = set()
        for route in routes:
            for node in route:
                seen.add(node)
        isolated_nodes = all_nodes ^ seen
        for node in isolated_nodes:
            if node.allow_isolated is False:
                raise ModelStructureError("Node is not part of a valid route: {}".format(node.name))

    @property
    def nodes(self):
        """Returns a model node iterator"""
        return NodeIterator(self)
    # support for old API
    node = nodes

    def edges(self):
        """Returns a list of Edges in the model

        An edge is described as a 2-tuple of the source and dest Nodes.
        """
        return self.graph.edges()

    @classmethod
    def loads(cls, data, model=None, path=None, solver=None):
        """Read JSON data from a string and parse it as a model document"""
        data = json.loads(data)
        cls._load_includes(data, path)
        return cls.load(data, model, path, solver)

    @classmethod
    def _load_includes(cls, data, path=None):
        """Load included JSON references

        Parameters
        ----------
        data : dict
            The model dictionary.
        path : str
            Path to the model document (None if in-memory).

        This method is private and shouldn't need to be called by the user.
        Note that the data dictionary is modified in-place.
        """
        if "includes" in data:
            for filename in data["includes"]:
                if path is not None:
                    filename = os.path.join(os.path.dirname(path), filename)
                with open(filename, "r") as f:
                    include_data = json.loads(f.read())
                for key, value in include_data.items():
                    if isinstance(value, list):
                        try:
                            data[key].extend(value)
                        except KeyError:
                            data[key] = value
                    elif isinstance(value, dict):
                        try:
                            data[key].update(value)
                        except KeyError:
                            data[key] = value
                    else:
                        raise TypeError("Invalid type for key \"{}\" in include \"{}\".".format(key, path))
        return None  # data modified in-place

    @classmethod
    def load(cls, data, model=None, path=None, solver=None):
        """Load an existing model

        Parameters
        ----------
        data : file-like, string, or dict
            A file-like object to read JSON data from, a filename to read,
            or a parsed dict
        model : Model (optional)
            An existing model to append to
        path : str (optional)
            Path to the model document for relative pathnames
        solver : str (optional)
            Name of the solver to use for the model. This overrides the solver
            section of the model document.
        """
        if isinstance(data, basestring):
            # argument is a filename
            path = data
            with open(path, "r") as f:
                data = f.read()
            return cls.loads(data, model, path, solver)

        if hasattr(data, 'read'):
            # argument is a file-like object
            data = data.read()
            return cls.loads(data, model, path, solver)

        # data is a dictionary, make a copy to avoid modify the input
        data = copy.deepcopy(data)

        cls._load_includes(data, path)

        try:
            solver_data = data['solver']
        except KeyError:
            solver_name = solver
        else:
            solver_name = data['solver']['name']

        try:
            timestepper_data = data['timestepper']
        except KeyError:
            start = end = None
            timestep = 1
        else:
            start = pandas.to_datetime(timestepper_data['start'])
            end = pandas.to_datetime(timestepper_data['end'])
            timestep = int(timestepper_data['timestep'])

        if model is None:
            model = cls(
                solver=solver_name,
                start=start,
                end=end,
                timestep=timestep,
                path=path,
            )
        model.metadata = data["metadata"]

        # collect nodes to load
        nodes_to_load = {}
        for node_data in data["nodes"]:
            node_name = node_data["name"]
            nodes_to_load[node_name] = node_data
        model._nodes_to_load = nodes_to_load

        # collect parameters to load
        try:
            parameters_to_load = data["parameters"]
        except KeyError:
            parameters_to_load = {}
        else:
            for key, value in parameters_to_load.items():
                if isinstance(value, dict):
                    parameters_to_load[key]["name"] = key
        model._parameters_to_load = parameters_to_load

        # collect recorders to load
        try:
            recorders_to_load = data["recorders"]
        except KeyError:
            recorders_to_load = {}
        else:
            for key, value in recorders_to_load.items():
                if isinstance(value, dict):
                    recorders_to_load[key]["name"] = key
        model._recorders_to_load = recorders_to_load

        # load parameters and recorders
        for name, rdata in model._recorders_to_load.items():
            load_recorder(model, rdata)
        while True:
            try:
                name, pdata = model._parameters_to_load.popitem()
            except KeyError:
                break
            parameter = load_parameter(model, pdata, name)
            if not isinstance(parameter, BaseParameter):
                raise TypeError("Named parameters cannot be literal values. Use type \"constant\" instead.")

        # load the remaining nodes
        for node_name in list(nodes_to_load.keys()):
            node = cls._get_node_from_ref(model, node_name)

        del(model._recorders_to_load)
        del(model._parameters_to_load)
        del(model._nodes_to_load)

        # load edges
        for edge_data in data['edges']:
            node_from_name = edge_data[0]
            node_to_name = edge_data[1]
            if len(edge_data) > 2:
                slot_from, slot_to = edge_data[2:]
            else:
                slot_from = slot_to = None
            node_from = model.nodes[node_from_name]
            node_to = model.nodes[node_to_name]
            node_from.connect(node_to, from_slot=slot_from, to_slot=slot_to)

        # load recorders
        if 'recorders' in data:
            for recorder_data in data['recorders']:
                load_recorder(model, recorder_data)

        return model

    @classmethod
    def _get_node_from_ref(cls, model, node_name):
        try:
            # first check if node has already been loaded
            node = model.nodes[node_name]
        except KeyError:
            # if not, load it now
            node_data = model._nodes_to_load[node_name]
            node_type = node_data['type'].lower()
            cls = node_registry[node_type]
            node = cls.load(node_data, model)
            del(model._nodes_to_load[node_name])
        return node

    def find_all_routes(self, type1, type2, valid=None, max_length=None, domain_match='strict'):
        """Find all routes between two nodes or types of node

        Parameters
        ----------
        type1 : Node class or instance
            The source node instance (or class)
        type2 : Node class or instance
            The destination  node instance (or class)
        valid : tuple of Node classes
            A tuple of Node classes that the route can traverse. For example,
            a route between a Catchment and Terminator can generally only
            traverse River nodes.
        max_length : integer
            Maximum length of the route including start and end nodes.
        domain_match : string
            A string to control the behaviour of different domains on the route.
                'strict' : all nodes must have the same domain as the first node.
                'any' : any domain is permitted on any node (i.e. nodes can have different domains)
                'different' : at least two different domains must be present on the route

        Returns a list of all the routes between the two nodes. A route is
        specified as a list of all the nodes between the source and
        destination with the same domain has the source.
        """

        nodes = self.graph.nodes()

        if inspect.isclass(type1):
            # find all nodes of type1
            type1_nodes = []
            for node in nodes:
                if isinstance(node, type1):
                    type1_nodes.append(node)
        else:
            type1_nodes = [type1]

        if inspect.isclass(type2):
            # find all nodes of type2
            type2_nodes = []
            for node in nodes:
                if isinstance(node, type2):
                    type2_nodes.append(node)
        else:
            type2_nodes = [type2]

        # find all routes between type1_nodes and type2_nodes
        all_routes = []
        for node1 in type1_nodes:
            for node2 in type2_nodes:
                for route in nx.all_simple_paths(self.graph, node1, node2):
                    is_valid = True
                    # Check valid intermediate nodes
                    if valid is not None and len(route) > 2:
                        for node in route[1:-1]:
                            if not isinstance(node, valid):
                                is_valid = False
                    # Check domains
                    if domain_match == 'strict':
                        # Domains must match the first node
                        for node in route[1:]:
                            if node.domain != route[0].domain:
                                is_valid = False
                    elif domain_match == 'different':
                        # Ensure at least two different domains are present
                        domains_found = set()
                        for node in route:
                            domains_found.add(node.domain)
                        if len(domains_found) < 2:
                            is_valid = False
                    elif domain_match == 'any':
                        # No filtering required
                        pass
                    else:
                        raise ValueError("domain_match '{}' not understood.".format(domain_match))

                    # Check length
                    if max_length is not None:
                        if len(route) > max_length:
                            is_valid = False

                    if is_valid:
                        all_routes.append(route)

        return all_routes

    def step(self):
        """Step the model forward by one day"""
        if self.dirty:
            self.setup()
        self.timestep = next(self.timestepper)
        return self._step()

    def _step(self):
        self.before()
        # reset any failures
        self.failure = set()
        # solve the current timestep
        ret = self.solve()
        self.after()
        return ret

    def solve(self):
        """Call solver to solve the current timestep"""
        return self.solver.solve(self)

    def run(self, until_date=None, until_failure=False, reset=True):
        """Run model until exit condition is reached

        Parameters
        ----------
        until_date : datetime (optional)
            Stop model when date is reached
        until_failure: bool (optional)
            Stop model run when failure condition occurs
        reset : bool (optional)
            If true, start the run from the beginning. Otherwise continue
            from the current state.

        Returns the number of last Timestep that was run.
        """
        if self.dirty:
            self.setup()
            self.timestepper.reset()
        elif reset:
            self.reset()
        for timestep in self.timestepper:
            self.timestep = timestep
            ret = self._step()
            if until_failure is True and self.failure:
                return timestep
            elif until_date and timestep.datetime > until_date:
                return timestep
            elif timestep.datetime > self.timestepper.end:
                return timestep
        self.finish()
        try:
            # Can only return timestep object if the iterator went
            # through at least one iteration
            return timestep
        except UnboundLocalError:
            return None

    def setup(self, ):
        """Setup the model for the first time or if it has changed since
        last run."""
        self.scenarios.setup()
        length_changed = self.timestepper.reset()
        for node in self.graph.nodes():
            node.setup(self)
        for recorder in self.recorders:
            recorder.setup()
        self.solver.setup(self)
        self.reset()
        self.dirty = False

    def reset(self, start=None):
        """Reset model to it's initial conditions"""
        length_changed = self.timestepper.reset(start=start)
        for node in self.nodes:
            if length_changed:
                node.setup(self)
            node.reset()
        for recorder in self.recorders:
            if length_changed:
                recorder.setup()
            recorder.reset()

    def before(self):
        for node in self.graph.nodes():
            node.before(self.timestep)

    def after(self):
        for node in self.graph.nodes():
            node.after(self.timestep)
        for recorder in self.recorders:
            recorder.save()

    def finish(self):
        for recorder in self.recorders:
            recorder.finish()

    def to_dataframe(self):
        """ Return a DataFrame from any Recorders with a `to_dataframe` attribute

        """
        dfs = {r.name: r.to_dataframe() for r in self.recorders if hasattr(r, 'to_dataframe')}
        df = pandas.concat(dfs, axis=1)
        df.columns.set_names('Recorder', level=0, inplace=True)
        return df


class Domain(_core.Domain):
    def __init__(self, name='default', **kwargs):
        super(Domain, self).__init__(name)
        self.color = kwargs.pop('color', '#FF6600')

class Drawable(object):
    """Mixin class for objects that are drawable on a diagram of the network.
    """
    def __init__(self, *args, **kwargs):
        self.position = kwargs.pop('position', None)
        self.color = kwargs.pop('color', 'black')
        self.visible = kwargs.pop('visible', True)
        super(Drawable, self).__init__(*args, **kwargs)


class Connectable(object):
    """
    Connectable is a mixin class that provides methods for connect the object to
    others view a NetworkX graph store in self.model.graph
    """
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
            for neighbor in neighbors:
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


# node subclasses are stored in a dict for convenience
node_registry = {}
class NodeMeta(type):
    """Node metaclass used to keep a registry of Node classes"""
    def __new__(meta, name, bases, dct):
        return super(NodeMeta, meta).__new__(meta, name, bases, dct)
    def __init__(cls, name, bases, dct):
        super(NodeMeta, cls).__init__(name, bases, dct)
        node_registry[name.lower()] = cls
    def __call__(cls, *args, **kwargs):
        # Create new instance of Node (or subclass thereof)
        node = type.__call__(cls, *args, **kwargs)
        # Add node to Model graph. This needs to be done here, so that if the
        # __init__ method of Node raises an exception it is not added.
        node.model.graph.add_node(node)
        node.model.dirty = True
        return node


class Node(with_metaclass(NodeMeta, Drawable, Connectable, BaseNode)):
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


class InputFromOtherDomain(Input):
    """A input in to the network that is connected to an output from another domain

    Parameters
    ----------
    conversion_factor : float (optional)
        A factor that is multiplied by the upstream output to calculate the input flow rate.
        This is typically used for losses and/or unit conversion.
    """
    def __init__(self, *args, **kwargs):
        Input.__init__(self, *args, **kwargs)

        import warnings
        warnings.warn("InputFromOtherDomain class is deprecated as the functionality is provided by Input.")


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


class Blender(Link):
    """Blender node to maintain a constant ratio between two supply routes"""
    def __init__(self, *args, **kwargs):
        """Initialise a new Blender node

        Parameters
        ----------
        ratio : float (optional)
            The ratio to constraint the two routes by (0.0-0.1). If no value is
            given a default value of 0.5 is used.
        """
        Link.__init__(self, *args, **kwargs)
        self.slots = {1: None, 2: None}

        self.properties['ratio'] = pop_kwarg_parameter(kwargs, 'ratio', 0.5)


class Storage(with_metaclass(NodeMeta, Drawable, Connectable, _core.Storage)):
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
        if 'volume' in kwargs:
            # support older API where volume kwarg was the initial volume
            initial_volume = kwargs.pop('volume')
        else:
            initial_volume = kwargs.pop('initial_volume', 0.0)
        cost = pop_kwarg_parameter(kwargs, 'cost', 0.0)

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
        self.cost = cost
        self.position = position

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
        initial_volume = float(data.pop('initial_volume'))
        max_volume = float(data.pop('max_volume'))
        min_volume = data.pop('min_volume', None)
        if min_volume is not None:
            min_volume = float(min_volume)
        cost = data.pop('cost', 0.0)
        data.pop('type', None)
        node = cls(
            model=model, name=name, num_inputs=num_inputs,
            num_outputs=num_outputs, initial_volume=initial_volume,
            max_volume=max_volume, min_volume=min_volume,
            **data
        )

        cost = load_parameter(model, cost)
        if cost is None:
            cost = 0.0
        node.cost = cost

        return node

    def __repr__(self):
        return '<{} "{}">'.format(self.__class__.__name__, self.name)


class VirtualStorage(with_metaclass(NodeMeta, Drawable, _core.VirtualStorage)):
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


class AnnualVirtualStorage(VirtualStorage):
    def __init__(self, *args, **kwargs):
        self.reset_day = kwargs.pop('reset_day', 1)
        self.reset_month = kwargs.pop('reset_month', 1)
        self._last_reset_year = None

        super(AnnualVirtualStorage, self).__init__(*args, **kwargs)

    def before(self, ts):
        super(AnnualVirtualStorage, self).before(ts)

        # Reset the storage volume if necessary
        if ts.datetime.year != self._last_reset_year:
            # I.e. we're in a new year and ...
            # ... we're at or past the reset month/day
            if ts.datetime.month > self.reset_month or \
                    (ts.datetime.month == self.reset_month and ts.datetime.day >= self.reset_day):
                self._reset_storage_only()
                self._last_reset_year = ts.datetime.year


class PiecewiseLink(Node):
    """ An extension of Node that represents a non-linear Link with a piece wise cost function.

    This object is intended to model situations where there is a benefit of supplying certain flow rates
    but beyond a fixed limit there is a change in (or zero) cost.

    This Node is implemented using a compound node structure like so:
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

    Parameters
    ----------
    max_flow : iterable
        A monotonic increasing list of maximum flows for the piece wise function
    cost : iterable
        A list of costs corresponding to the max_flow steps

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
        # accept plurals of max_flow and cost
        try:
            data["max_flow"] = data.pop("max_flows")
        except KeyError:
            pass
        try:
            data["cost"] = data.pop("costs")
        except KeyError:
            pass
        del(data["type"])
        return cls(model, **data)


class MultiSplitLink(PiecewiseLink):
    """ An extension of PiecewiseLink that includes additional additional slots to connect from.

    Conceptually this node looks like the following internally,

             / -->-- X0 -->-- \
    A -->-- Xo -->-- X1 -->-- Xi -->-- C
             \ -->-- X2 -->-- /
                     |
                     Bo -->-- Bi --> D

    An additional sublink in the PiecewiseLink (i.e. X2 above) and nodes (i.e. Bo and Bi) in this
     class are added for each extra slot.

    Finally a mechanism is provided to (optionally) fix the ratio between the last non-split
     sublink (i.e. X1) and each of extra sublinks (i.e. X2). This mechanism uses `AggregatedNode`
     internally.

    Parameters
    ----------
    max_flow : iterable
        A monotonic increasing list of maximum flows for the piece wise function
    cost : iterable
        A list of costs corresponding to the max_flow steps
    extra_slots : int, optional (default 1)
        Number of additional slots (and sublinks) to provide. Must be greater than zero.
    slot_names : iterable, optional (default range of ints)
        The names to by which to refer to the slots during connection to other nodes. Length
        must be one more than the number of extra_slots. The first item refers to the PiecewiseLink
        connection with the following items for each extra slot.
    factors : iterable, optional (default None)
        If given the length must be equal one more than the number of extra_slots. Each item is the proportion
        of total flow to pass through the additional sublinks. If no factor is required for a particular
        sublink then use `None` for its items. Factors are normalised prior to use in the solver.

    Notes
    -----
    Users must be careful when using the factormechanism. Factors use the last non-split
     sublink (i.e. X1 but not X0). If this link is constrained with a maximum or minimum flow,
     or if it there is another unconstrained link (i.e. if X0 is unconstrained) then ratios
     across this whole node may not be enforced as expected.

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


class AggregatedStorage(with_metaclass(NodeMeta, Drawable, _core.AggregatedStorage)):
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


class AggregatedNode(with_metaclass(NodeMeta, Drawable, _core.AggregatedNode)):
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

class ModelStructureError(Exception):
    pass

from .domains.river import *
