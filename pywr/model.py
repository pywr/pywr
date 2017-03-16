import os
import pandas
import json
import networkx as nx
from past.builtins import basestring
import copy
from packaging.version import parse as parse_version
import warnings
import inspect
import time

import pywr
from pywr.timestepper import Timestepper

from pywr.nodes import NodeMeta
from pywr.parameters import load_parameter
from pywr.recorders import load_recorder

from pywr._core import (BaseInput, BaseLink, BaseOutput, StorageInput,
    StorageOutput, Timestep, ScenarioIndex)
from pywr._component import ROOT_NODE
from pywr.nodes import Storage, AggregatedStorage, AggregatedNode, VirtualStorage
from pywr._core import ScenarioCollection, Scenario
from pywr._component import Component
from pywr.parameters._parameters import load_dataframe
from pywr.parameters._parameters import Parameter as BaseParameter
from pywr.recorders import ParameterRecorder, IndexParameterRecorder, Recorder

class ModelDocumentWarning(Warning): # TODO
    pass

class ModelStructureError(Exception): # TODO
    pass


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
        self.dirty = True

        self.path = kwargs.pop('path', None)
        if self.path is not None:
            if os.path.exists(self.path) and not os.path.isdir(self.path):
                self.path = os.path.dirname(self.path)

        # Import this here once everything else is defined.
        # This avoids circular references in the solver classes
        from pywr.solvers import solver_registry
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
        self.component_graph = nx.DiGraph()
        self.component_graph.add_node(ROOT_NODE)
        self.component_tree_flat = None

        self.tables = {}
        self.scenarios = ScenarioCollection(self)

        if kwargs:
            key = list(kwargs.keys())[0]
            raise TypeError("'{}' is an invalid keyword argument for this function".format(key))

        self.reset()

    @property
    def components(self):
        return NamedIterator(n for n in self.component_graph.nodes() if n != ROOT_NODE)

    @property
    def recorders(self):
        return NamedIterator(n for n in self.components if isinstance(n, Recorder))

    @property
    def parameters(self):
        return NamedIterator(n for n in self.components if isinstance(n, BaseParameter))

    @property
    def variables(self):
        return NamedIterator(n for n in self.parameters if n.is_variable)

    @property
    def constraints(self):
        return NamedIterator(n for n in self.recorders if n.is_constraint)

    @property
    def objectives(self):
        return NamedIterator(n for n in self.recorders if n.is_objective)

    def check(self):
        """Check the validity of the model

        Raises an Exception if the model is invalid.
        """
        for node in self.nodes:
            node.check()
        self.check_graph()

    def check_graph(self):
        """Check the connectivity of the graph is valid"""
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

    def edges(self):
        """Returns a list of Edges in the model

        An edge is described as a 2-tuple of the source and dest Nodes.
        """
        return self.graph.edges()

    @classmethod
    def loads(cls, data, model=None, path=None, solver=None):
        """Read JSON data from a string and parse it as a model document"""
        try:
            data = json.loads(data)
        except ValueError as e:
            message = e.args[0]
            if path:
                e.args = ("{} [{}]".format(e.args[0], os.path.basename(path)),)
            raise(e)
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
                    try:
                        include_data = json.loads(f.read())
                    except ValueError as e:
                        message = e.args[0]
                        if path:
                            e.args = ("{} [{}]".format(e.args[0], os.path.basename(filename)),)
                        raise(e)
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

        # check minimum version
        try:
            minimum_version = data["metadata"]["minimum_version"]
        except KeyError:
            warnings.warn("Missing \"minimum_version\" item in metadata.", ModelDocumentWarning)
        else:
            minimum_version = parse_version(minimum_version)
            pywr_version = parse_version(pywr.__version__)
            if pywr_version < minimum_version:
                warnings.warn("Document requires version {} or newer, but only have {}.".format(minimum_version, pywr_version), RuntimeWarning)

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

        # load scenarios
        try:
            scenarios_data = data["scenarios"]
        except KeyError:
            # Leave to default of no scenarios
            pass
        else:
            for scen_data in scenarios_data:
                scen_name = scen_data["name"]
                size = scen_data["size"]
                ensemble_names = scen_data.pop("ensemble_names", None)
                s_slice = scen_data.pop("slice", None)
                if s_slice:
                    s_slice = slice(*s_slice)
                Scenario(model, scen_name, size=size, slice=s_slice, ensemble_names=ensemble_names)

        try:
            scenario_combinations = data["scenario_combinations"]
        except KeyError:
            pass
        else:
            model.scenarios.user_combinations = scenario_combinations

        # load table references
        try:
            tables_data = data["tables"]
        except KeyError:
            # Default to no table entries
            pass
        else:
            for table_name, table_data in tables_data.items():
                model.tables[table_name] = load_dataframe(model, table_data)

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
            cls = NodeMeta.node_registry[node_type]
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

        nodes = sorted(self.graph.nodes(), key=lambda n:n.name)

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

        # Now sort the routes to ensure determinism
        all_routes = sorted(all_routes, key=lambda r: tuple(n.name for n in r))
        return all_routes

    def step(self):
        """ Step the model forward by one day

        This method progresses the model by one time-step. The anatomy
        of a time-step is as follows:
          1. Call `Model.setup` if the `Model.dirty` is True.
          2. Progress the `Model.timestepper` by one step.
          3. Call `Model.before` to ensure all nodes and components are ready for solve.
            a. Call `Node.before` on all nodes
            b. Refresh the component dependency tree
            c. Call `Component.before` on all components, respecting dependency order
            d. Call `Parameter.calc_values` on all Parameters, respecting dependency order
          4. Call `Model.solve` to solve the linear programme
          5. Call `Model.after` to ensure all nodes and components
            complete any work in the timestep.

        It is important to note that the current timestep object is the
        same during phases (3), (4) and (5) above. However the internal state of
        nodes changes during phase (4) and (5). During stages (3) and (5) the
        nodes are updated before the components. Therefore during the component
        update in phase (5) the internal state of nodes is already updated (e.g.
        current storage volumes). This has consequences for any component
        algorithms in phase (5) that rely on the state being as it was before
        this update. In general component `after` methods should not recompute
        any component state or rely on the internal node state.

        A dependency tree is used during the `before` and `after` updates of
        components. This ensures that components that rely on the state of
        another component are updated first.


        See also
        --------
        `Component`


        """
        if self.dirty:
            self.setup()
        self.timestep = next(self.timestepper)
        return self._step()

    def _step(self):
        self.before()
        # solve the current timestep
        ret = self.solve()
        self.after()
        return ret

    def solve(self):
        """Call solver to solve the current timestep"""
        return self.solver.solve(self)

    def run(self):
        """Run the model
        """
        t0 = time.time()
        timestep = None
        try:
            if self.dirty:
                self.setup()
                self.timestepper.reset()
            else:
                self.reset()
            t1 = time.time()
            for timestep in self.timestepper:
                self.timestep = timestep
                ret = self._step()
            t2 = time.time()
        finally:
            self.finish()
        t3 = time.time()

        if timestep is None:
            raise RuntimeError("Nothing to run! Timestepper length is {}".format(len(self.timestepper)))

        # return ModelResult instance
        time_taken = t2 - t1
        time_taken_with_overhead = t3 - t0
        num_scenarios = len(self.scenarios.combinations)
        try:
            speed = (timestep.index * num_scenarios) / time_taken
        except ZeroDivisionError:
            speed = float('nan')
        result = ModelResult(
            num_scenarios=num_scenarios,
            timestep=timestep,
            time_taken=time_taken,
            time_taken_with_overhead=time_taken_with_overhead,
            speed=speed,
            solver_name=self.solver.name,
            solver_stats=self.solver.stats,
            version=pywr.__version__,
            git_hash=pywr.__git_hash__,
        )
        return result

    def setup(self, ):
        """Setup the model for the first time or if it has changed since
        last run."""
        self.scenarios.setup()
        length_changed = self.timestepper.reset()
        for node in self.graph.nodes():
            node.setup(self)

        components = self.flatten_component_tree(rebuild=True)
        for component in components:
            component.setup()

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

        components = self.flatten_component_tree(rebuild=False)
        for component in components:
            if length_changed:
                component.setup()
            component.reset()

    def before(self):
        """ Perform initialisation work before solve on each timestep.

        This method calls the `before()` method on all nodes and components
        in the model. Nodes are updated first, components second.

        See also
        --------
        `Model.step`
        """
        for node in self.graph.nodes():
            node.before(self.timestep)
        components = self.flatten_component_tree(rebuild=False)
        for component in components:
            component.before()
        for component in components:
            if isinstance(component, BaseParameter):
                component.calc_values(self.timestep)

    def after(self):
        for node in self.graph.nodes():
            node.after(self.timestep)
        components = self.flatten_component_tree(rebuild=False)
        for component in components:
            component.after()

    def finish(self):
        for node in self.graph.nodes():
            node.finish()
        components = self.flatten_component_tree(rebuild=False)
        for component in components:
            component.finish()

    def to_dataframe(self):
        """ Return a DataFrame from any Recorders with a `to_dataframe` attribute

        """
        dfs = {r.name: r.to_dataframe() for r in self.recorders if hasattr(r, 'to_dataframe')}
        df = pandas.concat(dfs, axis=1)
        df.columns.set_names('Recorder', level=0, inplace=True)
        return df

    def flatten_component_tree(self, rebuild=False):
        if self.component_tree_flat is None or rebuild is True:
            self.component_tree_flat = []
            G = self.component_graph

            # Test some properties of the dependency tree
            # Do not permit cycles in the dependencies
            ncycles = len(list(nx.simple_cycles(G)))
            if ncycles != 0:
                raise ModelStructureError("Cyclical ({}) dependencies found in the model's components.".format(ncycles))
            # Do not permit self-loops
            for n in G.nodes_with_selfloops():
                raise ModelStructureError('Component "{}" contains a self-loop.'.format(n))

            for node in nx.dfs_postorder_nodes(G, ROOT_NODE):
                if node == ROOT_NODE:
                    continue
                self.component_tree_flat.append(node)
            # order components so that they can be iterated over easily in an
            # sequence which respects dependencies
            self.component_tree_flat = self.component_tree_flat[::-1]
        return self.component_tree_flat

    def find_orphaned_parameters(self):
        """Helper function to find orphaned parameters

        Returns a set of parameters which:
            1) Have no parent components
            2) Are not referenced directly by a node
        """
        all_parameters = set(self.parameters)
        visited = set()
        # add all parameters referenced by another component
        for parameter in self.components:
            if parameter.parents:
                visited.add(parameter)
        # find all parameters referenced by a node
        for node in self.graph.nodes():
            for component in node.components:
                visited.add(component)
        # identify unseen parameters
        orphans = all_parameters - visited
        return orphans

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
        return

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

class NamedIterator(object):
    def __init__(self, objects=None):
        if objects:
            self._objects = list(objects)
        else:
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
        self._objects.remove(obj)

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

class ModelResult(object):
    def __init__(self, num_scenarios, timestep, time_taken, time_taken_with_overhead, speed,
                 solver_name, solver_stats, version, git_hash):
        self.timestep = timestep
        self.timesteps = timestep.index + 1
        self.time_taken = time_taken
        self.time_taken_with_overhead = time_taken_with_overhead
        self.speed = speed
        self.num_scenarios = num_scenarios
        self.solver_name = solver_name
        self.solver_stats = solver_stats
        self.version = version
        self.git_hash = git_hash

    def to_dict(self):
        return {attr: value for attr, value in self.__dict__.items()}

    def to_dataframe(self):
        d = self.to_dict()
        # Update timestep to use the underlying pandas Timestamp
        d['timestep'] = d['timestep'].datetime
        # Must flatten the solver stats dict before passing to pandas
        solver_stats = d.pop('solver_stats')
        for k, v in solver_stats.items():
            d['solver_stats.{}'.format(k)] = v
        df = pandas.DataFrame(pandas.Series(d), columns=["Value"])
        return df

    def __repr__(self):
        return "Model executed {:d} scenarios in {:.1f} seconds, running at {:.1f} timesteps per second.".format(self.num_scenarios, self.time_taken_with_overhead, self.speed)

    def _repr_html_(self):
        return self.to_dataframe()._repr_html_()
