#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
from IPython.core.magic_arguments import kwds
import networkx as nx
import numpy as np
import inspect
import pandas
import datetime
import xml.etree.ElementTree as ET
from six import with_metaclass

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = UnicodeWarning)

from .licenses import LicenseCollection

class Model(object):
    """Model of a water supply network"""
    def __init__(self, solver=None, parameters=None):
        """Initialise a new Model instance

        Parameters
        ----------
        solver : string
            The name of the underlying solver to use. See the `pywr.solvers`
            package. If no value is given, the default GLPK solver is used.
        parameters : dict of Parameters
            A dictionary of parameters to initialise the model with. Parameters
            can also be added, modified or removed after the Model has been
            initialised.
        """
        self.graph = nx.DiGraph()
        self.xml_path = None  # keep track of XML location, for relative paths
        self.metadata = {}
        self.parameters = {
            # default parameter values
            'timestamp_start': pandas.to_datetime('2015-01-01'),
            'timestamp_finish': pandas.to_datetime('2015-12-31'),
            'timestep': datetime.timedelta(1),
        }
        if parameters is not None:
            self.parameters.update(parameters)
        self.data = {}
        self.failure = set()
        self.dirty = True

        if solver is not None:
            # use specific solver
            try:
                self.solver = SolverMeta.solvers[solver.lower()]
            except KeyError:
                raise KeyError('Unrecognised solver: {}'.format(solver))
        else:
            # use default solver
            self.solver = solvers.SolverGLPK()

        self.node = {}
        self.group = {}

        self.reset()

    def check(self):
        """Check the validity of the model

        Raises an Exception if the model is invalid.
        """
        nodes = self.graph.nodes()
        for node in nodes:
            node.check()

    def nodes(self):
        """Returns a list of Nodes in the model"""
        return self.graph.nodes()

    def edges(self):
        """Returns a list of Edges in the model

        An edge is described as a 2-tuple of the source and dest Nodes.
        """
        return self.graph.edges()

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
        self.before()
        # reset any failures
        self.failure = set()
        # solve the current timestep
        ret = self.solve()
        self.timestamp += self.parameters['timestep']
        self.after()
        return ret

    def solve(self):
        """Call solver to solve the current timestep"""
        return self.solver.solve(self)

    def run(self, until_date=None, until_failure=False):
        """Run model until exit condition is reached

        Parameters
        ----------
        until_date : datetime (optional)
            Stop model when date is reached
        until_failure: bool (optional)
            Stop model run when failure condition occurs

        Returns the number of timesteps that were run.
        """
        if self.timestamp > self.parameters['timestamp_finish']:
            return
        timesteps = 0
        while True:
            ret = self.step()
            timesteps += 1
            if until_failure is True and self.failure:
                return timesteps
            elif until_date and self.timestamp > until_date:
                return timesteps
            elif self.timestamp > self.parameters['timestamp_finish']:
                return timesteps

    def reset(self):
        """Reset model to it's initial conditions"""
        self.timestamp = self.parameters['timestamp_start']
        for node in self.nodes():
            node.reset()

    def before(self):
        for node in self.nodes():
            node.before()

    def after(self):
        for node in self.nodes():
            node.after()

    def xml(self):
        """Serialize the Model to XML"""
        xml_model = ET.Element('pywr')

        xml_metadata = ET.SubElement(xml_model, 'metadata')
        for key, value in self.metadata.items():
            xml_metadata_item = ET.SubElement(xml_metadata, key)
            xml_metadata_item.text = value

        xml_parameters = ET.SubElement(xml_model, 'parameters')
        for key, value in self.parameters.items():
            pass # TODO

        xml_data = ET.SubElement(xml_model, 'data')
        for name, ts in self.data.items():
            xml_ts = ts.xml(name)
            xml_data.append(xml_ts)

        xml_nodes = ET.SubElement(xml_model, 'nodes')
        for node in self.nodes():
            xml_node = node.xml()
            xml_nodes.append(xml_node)

        xml_edges = ET.SubElement(xml_model, 'edges')
        for edge in self.edges():
            node_from, node_to = edge
            xml_edge = ET.SubElement(xml_edges, 'edge')
            xml_edge.set('from', node_from.name)
            xml_edge.set('to', node_to.name)

        xml_groups = ET.SubElement(xml_model, 'groups')
        for name, group in self.group.items():
            xml_group = group.xml()
            xml_groups.append(xml_group)

        return xml_model

    @classmethod
    def from_xml(cls, xml, path=None):
        """Deserialize a Model from XML"""
        xml_solver = xml.find('solver')
        if xml_solver is not None:
            solver = xml_solver.get('name')
        else:
            solver = None

        model = Model(solver=solver)

        # parse metadata
        xml_metadata = xml.find('metadata')
        if xml_metadata is not None:
            for xml_metadata_item in xml_metadata.getchildren():
                key = xml_metadata_item.tag.lower()
                value = xml_metadata_item.text.strip()
                model.metadata[key] = value

        if path:
            model.xml_path = os.path.abspath(path)
        else:
            model.xml_path = None

        # parse model parameters
        for xml_parameters in xml.findall('parameters'):
            for xml_parameter in xml_parameters.getchildren():
                key, parameter = Parameter.from_xml(model, xml_parameter)
                model.parameters[key] = parameter

        # parse data
        xml_datas = xml.find('data')
        if xml_datas:
            for xml_data in xml_datas.getchildren():
                ts = Timeseries.from_xml(model, xml_data)

        # parse nodes
        for node_xml in xml.find('nodes'):
            tag = node_xml.tag.lower()
            node_cls = node_registry[tag]
            node = node_cls.from_xml(model, node_xml)

        # parse edges
        xml_edges = xml.find('edges')
        for xml_edge in xml_edges.getchildren():
            tag = xml_edge.tag.lower()
            if tag != 'edge':
                raise ValueError()
            from_name = xml_edge.get('from')
            to_name = xml_edge.get('to')
            from_node = model.node[from_name]
            to_node = model.node[to_name]
            from_slot = xml_edge.get('from_slot')
            if from_slot is not None:
                from_slot = int(from_slot)
            to_slot = xml_edge.get('to_slot')
            if to_slot is not None:
                to_slot = int(to_slot)
            from_node.connect(to_node, from_slot=from_slot, to_slot=to_slot)

        # parse groups
        xml_groups = xml.find('groups')
        if xml_groups:
            for xml_group in xml_groups.getchildren():
                group = Group.from_xml(model, xml_group)

        return model

    def path_rel_to_xml(self, path):
        if self.xml_path is None:
            return os.path.abspath(path)
        else:
            return os.path.abspath(os.path.join(os.path.dirname(self.xml_path), path))

class SolverMeta(type):
    """Solver metaclass used to keep a registry of Solver classes"""
    solvers = {}
    def __new__(cls, clsname, bases, attrs):
        newclass = super(SolverMeta, cls).__new__(cls, clsname, bases, attrs)
        cls.solvers[newclass.name.lower()] = newclass
        return newclass

class Solver(with_metaclass(SolverMeta)):
    """Solver base class from which all solvers should inherit"""
    name = 'default'
    def solve(self, model):
        raise NotImplementedError('Solver should be subclassed to provide solve()')

class Property(object):
    pass

class Parameter(Property):
    def value(self, index=None):
        raise NotImplementedError()

    def xml(*args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_xml(cls, model, xml):
        # TODO: this doesn't look nice - need to rethink xml specification?
        parameter_types = {
            'const': ParameterConstant,
            'constant': ParameterConstant,
            'timestamp': ParameterConstant,
            'timedelta': ParameterConstant,
            'datetime': ParameterConstant,
            'timeseries': ParameterConstant,
            'python': ParameterFunction,
        }
        parameter_type = xml.get('type')
        return parameter_types[parameter_type].from_xml(model, xml)

class ParameterConstant(Parameter):
    def __init__(self, value=None):
        self._value = value

    def value(self, index=None):
        return self._value

    def xml(self, key):
        parameter_xml = ET.Element('parameter')
        parameter_xml.set('key', key)
        if isinstance(self._value, float):
            parameter_type = 'const'
            parameter_xml.text = str(self._value)
        elif isinstance(self._value, pandas.tslib.Timestamp):
            parameter_type = 'datetime'
            parameter_xml.text = str(self._value)
        elif isinstance(self._value, datetime.timedelta):
            parameter_type = 'timedelta'
            # try to represent the timedelta in sensible units
            total_seconds = self._value.total_seconds()
            if total_seconds % (60*60*24) == 0:
                units = 'days'
                parameter_xml.text = str(int(total_seconds / (60*60*24)))
            elif total_seconds % (60*60) == 0:
                units = 'hours'
                parameter_xml.text = str(int(total_seconds / (60*60)))
            elif total_seconds % (60) == 0:
                units = 'minutes'
                parameter_xml.text = str(int(total_seconds / 60))
            else:
                units = 'seconds'
                parameter_xml.text = str(int(total_seconds))
            parameter_xml.set('units', units)
        else:
            raise TypeError()
        parameter_xml.set('type', parameter_type)
        return parameter_xml

    @classmethod
    def from_xml(cls, model, xml):
        parameter_type = xml.get('type')
        key = xml.get('key')
        if parameter_type == 'const' or parameter_type == 'constant':
            try:
                value = float(xml.text)
            except:
                value = xml.text
            return key, ParameterConstant(value=value)
        elif parameter_type == 'timeseries':
            name = xml.text
            return key, model.data[name]
        elif parameter_type == 'datetime':
            return key, pandas.to_datetime(xml.text)
        elif parameter_type == 'timedelta':
            units = xml.get('units')
            value = float(xml.text)
            if units is None:
                units = 'seconds'
            units = units.lower()
            if units[-1] != 's':
                units = units + 's'
            td = datetime.timedelta(**{units: value})
            return key, td
        else:
            raise NotImplementedError('Unknown parameter type: {}'.format(parameter_type))

class ParameterFunction(Parameter):
    def __init__(self, parent, func):
        self._parent = parent
        self._func = func

    def value(self, index=None):
        return self._func(self._parent, index)

    @classmethod
    def from_xml(cls, xml):
        raise NotImplementedError('TODO')

class ParameterMonthlyProfile(Parameter):
    def __init__(self, values):
        if len(values) != 12:
            raise ValueError("12 values must be given for a monthly profile.")
        self._values = values

    def value(self, index=None):
        return self._values[index.month-1]

    @classmethod
    def from_xml(cls, xml):
        raise NotImplementedError('TODO')

class Timeseries(Parameter):
    def __init__(self, name, df, metadata=None):
        self.name = name
        self.df = df
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def value(self, index):
        return self.df[index]

    def xml(self, name):
        xml_ts = ET.Element('timeseries')
        xml_ts.set('name', self.name)
        for key, value in self.metadata.items():
            xml_meta = ET.SubElement(xml_ts, key)
            xml_meta.text = value
        return xml_ts

    @classmethod
    def read(self, model, **kwargs):
        name = kwargs['name']
        if name in model.data:
            raise ValueError('Timeseries with name "{}" already exists.'.format(name))

        filetype = None
        if 'type' in kwargs:
            filetype = kwargs['type']
        elif 'path' in kwargs:
            ext = kwargs['path'].split('.')[-1].lower()
            if ext == 'csv':
                filetype = 'csv'
            elif ext in ('xls', 'xlsx', 'xlsm'):
                filetype = 'excel'
            else:
                raise ValueError('Unrecognised timeseries type: {}'.format(ext))
        # TODO: other filetypes (SQLite? HDF5?)
        if filetype is None:
            raise ValueError('Unknown timeseries type.')
        if filetype == 'csv':
            path = model.path_rel_to_xml(kwargs['path'])
            df = pandas.read_csv(
                path,
                index_col=0,
                parse_dates=True,
                dayfirst=True,
            )
        elif filetype == 'excel':
            path = model.path_rel_to_xml(kwargs['path'])
            sheet = kwargs['sheet']
            df = pandas.read_excel(
                path,
                sheet,
                index_col=0,
                parse_dates=True,
                dayfirst=True,
            )

        df = df[kwargs['column']]
        # create a new timeseries object
        ts = Timeseries(name, df, metadata=kwargs)
        # register the timeseries in the model
        model.data[name] = ts
        return ts

    @classmethod
    def from_xml(self, model, xml):
        name = xml.get('name')
        properties = {}
        for child in xml.getchildren():
            properties[child.tag.lower()] = child.text
        properties['name'] = name

        if 'dayfirst' not in properties:
            # default to british dates
            properties['dayfirst'] = True

        ts = self.read(model, **properties)

        return ts

class Variable(Property):
    """This property is used for values that are calculated by the model, i.e.
    variables. It provides a method of saving the results at each timestep.
    """
    def __init__(self, initial=0.0):
        self._initial = initial
        self.reset()

    def value(self, index=None):
        return self._value

    def reset(self):
        self._value = self._initial
        self._saved_values = []

    def save(self):
        self._saved_values.append(self._value)

    @classmethod
    def from_xml(cls, model, xml):
        key = xml.get('key')
        try:
            value = float(xml.text)
        except:
            value = xml.text
        return key, ParameterConstant(value=value)

class PropertiesDict(dict):
    def __setitem__(self, key, value):
        if not isinstance(value, Property):
            value = ParameterConstant(value)
        dict.__setitem__(self, key, value)

class Domain(object):
    def __init__(self, name='default', **kwargs):
        self.name = name
        self.color = kwargs.pop('color', '#FF6600')

# node subclasses are stored in a dict for convenience
node_registry = {}
class NodeMeta(type):
    """Node metaclass used to keep a registry of Node classes"""
    def __new__(meta, name, bases, dct):
        return super(NodeMeta, meta).__new__(meta, name, bases, dct)
    def __init__(cls, name, bases, dct):
        super(NodeMeta, cls).__init__(name, bases, dct)
        node_registry[name.lower()] = cls

class Node(with_metaclass(NodeMeta)):
    """Base object from which all other nodes inherit"""
    def __init__(self, model, name, domain=Domain(), position=None, **kwargs):
        """Initialise a new Node object

        Parameters
        ----------
        model : Model
            The model the node belongs to
        position : 2-tuple of floats
            The location of the node in the schematic, e.g. (3.0, 4.5)
        name : string
            A unique name for the node
        """
        self.model = model
        model.graph.add_node(self)
        model.dirty = True

        self.color = 'black'
        self.position = position
        self.domain = domain
        self.visible = kwargs.pop('visible', True)
        self.parent = kwargs.pop('parent', None)

        if not hasattr(self, 'name'):
            # set name, avoiding issues with multiple inheritance
            self.__name = None
            self.name = name

        self.slots = {}

        self.properties = PropertiesDict({
            'cost': self.pop_kwarg_parameter(kwargs, 'cost', 0.0)
        })

    def __repr__(self):
        if self.name:
            # e.g. <Node "oxford">
            return '<{} "{}">'.format(self.__class__.__name__, self.name)
        else:
            return '<{} "{}">'.format(self.__class__.__name__, hex(id(self)))

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        # check for name collision
        if name in self.model.node:
            raise ValueError('A node with the name "{}" already exists.'.format(name))
        # remove old name
        try:
            del(self.model.node[self.__name])
        except KeyError:
            pass
        # apply new name
        self.__name = name
        self.model.node[name] = self

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

        node1 = self
        # check slots are valid
        if from_slot is not None:
            if from_slot not in self.slots:
                raise ValueError('{} does not have slot: {}'.format(self.__class__.__name__, from_slot))
            node1 = self.slots[from_slot]

        node2 = node
        if to_slot is not None:
            if to_slot not in node.slots:
                raise ValueError('{} does not have slot: {}'.format(node.__class__.__name__, to_slot))
            node2 = node.slots[to_slot]
        else:
            # Add default to_slot for Storage and its subclases
            if isinstance(node2, Storage):
                node2 = node.slots['output']
            if isinstance(node2, PiecewiseLink):
                # must recursively add all sublinks
                for sublink in node2.sublinks:
                    print('Connecting sublink...')
                    node1.connect(sublink)
                return

        self.model.graph.add_edge(node1, node2)
        self.model.dirty = True

    def disconnect(self, node=None):
        """Remove a connection from this Node to another Node

        Parameters
        ----------
        node : Node (optional)
            The node to remove the connection to. If another node is not
            specified, all connections from this node will be removed.
        """
        if node is not None:
            self._disconnect(node)
        else:
            neighbors = self.model.graph.neighbors(self)
            for neighbor in neighbors:
                self._disconnect(neighbor)

    def _disconnect(self, node):
        """As disconnect, except node argument is required"""
        self.model.graph.remove_edge(self, node)
        for slot, slot_node in node.slots.items():
            if slot_node is self:
                node.slots[slot] = None
        for slot, slot_node in self.slots.items():
            if slot_node is node:
                self.slots[slot] = None
        self.model.dirty = True

    def check(self):
        """Check the node is valid

        Raises an exception if the node is invalid
        """
        if not isinstance(self.position, (tuple, list,)):
            raise TypeError('{} position has invalid type ({})'.format(self, type(self.position)))
        if not len(self.position) == 2:
            raise ValueError('{} position has invalid length ({})'.format(self, len(self.position)))

    def reset(self):
        # reset variables
        for key, parameter in self.properties.items():
            try:
                parameter.reset()
            except AttributeError:
                pass

    def before(self):
        """Called before the current timestep begins
        """
        pass

    def commit(self, volume):
        """Commit a volume of water actually supplied/transferred/received

        Parameter
        ---------
        volume : float
            The volume to commit

        This should be implemented by the various node subclasses.
        """
        pass

    def after(self):
        """Called after the current timestep has finished
        """
        # save variables
        for key, parameter in self.properties.items():
            try:
                parameter.save()
            except AttributeError:
                pass

    def pop_kwarg_parameter(self, kwargs, key, default):
        """Pop a parameter from the keyword arguments dictionary

        Parameters
        ----------
        kwargs : dict
            A keyword arguments dictionary
        key : string
            The argument name, e.g. 'flow'
        default : object
            The default value to use if the dictionary does not have that key

        Returns a Parameter
        """
        value = kwargs.pop(key, default)
        if isinstance(value, Parameter):
            return value
        elif callable(value):
            return ParameterFunction(self, value)
        else:
            return ParameterConstant(value=value)

    def xml(self):
        """Serialize the node to an XML object

        The tag of the XML node returned is the same as the class name. For the
        base Node object a <node /> is returned, but this will differ for
        subclasses, e.g. Supply.xml returns a <supply /> element.

        Returns an xml.etree.ElementTree.Element object
        """
        xml = ET.fromstring('<{} />'.format(self.__class__.__name__.lower()))
        xml.set('name', self.name)
        xml.set('x', str(self.position[0]))
        xml.set('y', str(self.position[1]))
        for key, prop in self.properties.items():
            prop_xml = prop.xml(key)
            xml.append(prop_xml)
        return xml

    @classmethod
    def from_xml(cls, model, xml):
        """Deserialize a node from an XML object

        Parameters
        ----------
        model : Model
            The model to add the node to
        xml : xml.etree.ElementTree.Element
            The XML element representing the node

        Returns a Node instance, or an instance of the appropriate subclass.
        """
        tag = xml.tag.lower()
        node_cls = node_registry[tag]
        name = xml.get('name')
        x = float(xml.get('x'))
        y = float(xml.get('y'))
        node = node_cls(model, name=name, position=(x, y,))
        for prop_xml in xml.findall('parameter'):
            key, prop = Parameter.from_xml(model, prop_xml)
            node.properties[key] = prop
        for var_xml in xml.findall('variable'):
            key, prop = Variable.from_xml(model, var_xml)
            node.properties[key] = prop
        return node

class Input(Node):
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

        self.properties['min_flow'] = self.pop_kwarg_parameter(kwargs, 'min_flow', 0.0)
        self.properties['max_flow'] = self.pop_kwarg_parameter(kwargs, 'max_flow', None)

        self.licenses = None
        self.properties['flow'] = Variable(initial=kwargs.pop('flow', 0.0))

    def before(self, ):
        super(Input, self).before()
        self.properties['flow']._value = 0.0

    def commit(self, volume, ):
        super(Input, self).commit(volume)
        self.properties['flow']._value += volume

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

        self.properties['conversion_factor'] = self.pop_kwarg_parameter(kwargs, 'conversion_factor', 1.0)


class Output(Node):
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
        Node.__init__(self, *args, **kwargs)
        self.color = '#FFF467' # light yellow

        self.properties['min_flow'] = self.pop_kwarg_parameter(kwargs, 'min_flow', 0.0)
        self.properties['max_flow'] = self.pop_kwarg_parameter(kwargs, 'max_flow', None)
        self.properties['benefit'] = self.pop_kwarg_parameter(kwargs, 'benefit', 0.0)
        self.properties['flow'] = Variable(initial=kwargs.pop('flow', 0.0))

    def before(self, ):
        super(Output, self).before()
        self.properties['flow']._value = 0.0

    def commit(self, volume, ):
        super(Output, self).commit(volume)
        self.properties['flow']._value += volume

class Supply(Input):
    """A supply in the network

    The base supply node should be sufficient to represent simply supplies
    which do not interact with other components (e.g. a groundwater source
    or a bulk supply from another zone/company). For more complex supplies
    use the appropriate subclass (e.g. RiverAbstraction or Reservoir).
    """
    def __init__(self, *args, **kwargs):
        """Initialise a new Supply node

        Parameters
        ----------
        max_flow : float (optional)
            A simple maximum flow constraint for the supply. For more complex
            constraints a License instance should be used.
        """
        Input.__init__(self, *args, **kwargs)
        self.color = '#F26C4F' # light red

        self.properties['max_flow'] = self.pop_kwarg_parameter(kwargs, 'max_flow', 0.0)

        self.licenses = None

    def commit(self, volume):
        super(Supply, self).commit(volume)
        if self.licenses is not None:
            self.licenses.commit(volume)

    def reset(self):
        Node.reset(self)
        if self.licenses:
            self.licenses.refresh()

    def xml(self):
        xml = super(Supply, self).xml()
        if self.licenses is not None:
            xml.append(self.licenses.xml())
        return xml

    @classmethod
    def from_xml(cls, model, xml):
        node = Node.from_xml(model, xml)
        licensecollection_xml = xml.find('licensecollection')
        if licensecollection_xml is not None:
            node.licenses = LicenseCollection.from_xml(licensecollection_xml)
        return node

class Demand(Output):
    """A demand in the network"""
    def __init__(self, *args, **kwargs):
        """Initialise a new Demand node

        Parameters
        ----------
        demand : float
            The amount of water to demand each timestep
        consumption : float
            The proportion of water received that is consumed. The remaining
            water can be discharged back into the river system using a
            DemandDischarge node. e.g. a value of 0.7 means 70% of the water is
            consumed and the remaining 30% can be discharged. The default is
            that 100% of the water is consumed.
        """
        Output.__init__(self, *args, **kwargs)
        self.color = '#FFF467' # light yellow

        self.properties['demand'] = self.pop_kwarg_parameter(kwargs, 'demand', 0.0)

        self.properties['benefit'] = self.pop_kwarg_parameter(kwargs, 'benefit', 1000.0)

        self.properties['consumption'] = self.pop_kwarg_parameter(kwargs, 'consumption', 1.0)

    def before(self):
        self._supplied = 0.0

    def commit(self, volume):
        super(Demand, self).commit(volume)

        self._supplied += volume

    def after(self):
        """Check if the demand has been satisfied this timestep
        """
        demanded = self.properties['demand'].value(self.model.timestamp)
        if self._supplied < demanded:
            self.model.failure.add((self, 'demand',))

class Link(Node):
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
        Node.__init__(self, *args, **kwargs)
        self.color = '#A0A0A0' # 45% grey

        self.properties['max_flow'] = self.pop_kwarg_parameter(kwargs, 'max_flow', None)

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

        self.properties['ratio'] = self.pop_kwarg_parameter(kwargs, 'ratio', 0.5)

class StorageInput(Input):
    def commit(self, volume):
        super(StorageInput, self).commit(volume)
        self.parent.commit(-volume)

class StorageOutput(Output):
    def commit(self, volume):
        super(StorageOutput, self).commit(volume)
        self.parent.commit(volume)

class Storage(Node):
    """A generic storage Node"""
    def __init__(self, model, *args, **kwargs):
        Node.__init__(self,  model, *args, **kwargs)
        self.color = 'green' # light yellow
        # keyword arguments for input and output nodes specified with prefix
        input_kwargs, output_kwargs = {}, {}
        for key in kwargs.keys():
            if key.startswith('input_'):
                input_kwargs[key.replace('input_', '')] = kwargs.pop(key)
            elif key.startswith('output_'):
                output_kwargs[key.replace('output_', '')] = kwargs.pop(key)

        # subnodes require position of this node
        input_kwargs['position'] = output_kwargs['position'] = self.position
        # subnodes have the same domain as this node
        input_kwargs['domain'] = output_kwargs['domain'] = self.domain

        # output node should have the same benefit as Storage
        output_kwargs['benefit'] = kwargs.pop('benefit', 0.0)
        def func(parent, index):
            return self.output.properties['benefit'].value(index)
        self.properties['benefit'] = ParameterFunction(self, func)

        self.input = StorageInput(model, name="{} Input".format(self.name),
                           visible=False, parent=self, **input_kwargs)
        self.slots['input'] = self.input
        self.output = StorageOutput(model, name="{} Output".format(self.name),
                             visible=False, parent=self, **output_kwargs)
        self.slots['output'] = self.output

        self.properties['current_volume'] = Variable(initial=kwargs.pop('current_volume', 0.0))
        self.properties['max_volume'] = self.pop_kwarg_parameter(kwargs, 'max_volume', 0.0)
        def func(parent, index):
            return parent.properties['current_volume'].value(index)/parent.properties['max_volume'].value(index)
        self.properties['current_pc_full'] = ParameterFunction(self, func)

    def commit(self, volume, ):
        super(Storage, self).commit(volume)
        timestep = self.model.parameters['timestep']
        if isinstance(timestep, datetime.timedelta):
            timestep = timestep.days
        self.properties['current_volume']._value += volume*timestep

    def check(self):
        Node.check(self)
        self.input.check()
        self.output.check()
        index = self.model.timestamp
        # check volume doesn't exceed maximum volume
        assert(self.properties['max_volume'].value(index) >= self.properties['current_volume'].value(index))

    def connect(self, node, from_slot='input', to_slot=None):
        super(Storage, self).connect(node, from_slot=from_slot, to_slot=to_slot)


class PiecewiseLink(Node):
    """ An extension of Nodes that represents a non-linear Link with a piece wise cost function.

    This object is intended to model situations where there is a benefit of supplying certain flow rates
    but beyond a fixed limit there is a change in (or zero) cost.


    """
    def __init__(self, model, *args, **kwargs):
        """
        Parameters
        ----------
        max_flow : iterable
            A monotonic increasing list of maximum flows for the piece wise function
        cost : iterable
            A list of costs corresponding to the max_flow steps
        """
        # Grab cost and max_flow keywords before given to Node.__init__
        costs = kwargs.pop('cost')
        max_flows = kwargs.pop('max_flow')
        Node.__init__(self,  model, *args, **kwargs)

        if len(costs) != len(max_flows):
            raise ValueError("Piecewise max_flow and cost keywords must be the same length.")

        self.sublinks = []
        for max_flow, cost in zip(max_flows, costs):
            self.sublinks.append(Link(model, name='{} Sublink {}'.format(self.name, len(self.sublinks)),
                                      cost=cost, max_flow=max_flow, visible=False, parent=self,
                                      position=self.position, domain=self.domain))

    def connect(self, node, from_slot=None, to_slot=None):
        """
        Overload Node.connect to connect node to all sublinks rather than directly to this PiecewiseLink.
        """
        for sublink in self.sublinks:
            sublink.connect(node, from_slot=from_slot, to_slot=to_slot)

    def disconnect(self, node=None):
        """
        Overload Node.disconnect to disconnect node from all sublinks.
        """
        for sublink in self.sublinks:
            sublink.disconnect(node=node)


class Group(object):
    """A group of nodes

    This class is useful for applying a license constraint (or set of
    constraints) to a group of Supply nodes.
    """
    def __init__(self, model, name, nodes=None):
        self.model = model
        if nodes is None:
            self.nodes = set()
        else:
            self.nodes = set(nodes)
        self.__name = name
        self.name = name
        self.licenses = None

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        try:
            del(self.model.group[self.__name])
        except KeyError:
            pass
        self.__name = name
        self.model.group[name] = self

    def xml(self):
        xml = ET.Element('group')
        xml.set('name', self.name)
        # members
        xml_members = ET.SubElement(xml, 'members')
        for node in self.nodes:
            member = ET.SubElement(xml_members, 'member')
            member.set('name', node.name)
        # licenses
        if self.licenses:
            xml_licensecollection = self.licenses.xml()
            xml.append(xml_licensecollection)
        return xml

    @classmethod
    def from_xml(cls, model, xml):
        name = xml.get('name')
        group = Group(model, name)
        # members
        xml_members = xml.find('members')
        for xml_member in xml_members:
            name = xml_member.get('name')
            node = model.node[name]
            group.nodes.add(node)
        # licenses
        xml_licensecollection = xml.find('licensecollection')
        if xml_licensecollection:
            group.licenses = LicenseCollection.from_xml(xml_licensecollection)
        return group

from . import solvers
