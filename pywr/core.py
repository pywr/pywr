#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

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
    
    def find_all_routes(self, type1, type2, valid=None):
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
        
        Returns a list of all the routes between the two nodes. A route is
        specified as a list of all the nodes between the source and
        destination.
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
                    if valid is not None and len(route) > 2:
                        for node in route[1:-1]:
                            if not isinstance(node, valid):
                                is_valid = False
                    if is_valid:
                        all_routes.append(route)
        
        return all_routes
    
    def step(self):
        """Step the model forward by one day"""
        ret = self.solve()
        self.timestamp += self.parameters['timestep']
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
            # TODO: more complex assessment of "failure"
            if until_failure is True and ret[1] != ret[2]:
                return timesteps
            elif until_date and self.timestamp > until_date:
                return timesteps
            elif self.timestamp > self.parameters['timestamp_finish']:
                return timesteps
    
    def reset(self):
        """Reset model to it's initial conditions"""
        # TODO: this will need more, e.g. reservoir states, license states
        self.timestamp = self.parameters['timestamp_start']
    
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
    def from_xml(cls, xml):
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

class Parameter(object):
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

class Timeseries(object):
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
    def from_xml(self, model, xml):
        name = xml.get('name')
        properties = {}
        for child in xml.getchildren():
            properties[child.tag.lower()] = child.text
        if properties['type'] == 'pandas':
            # TODO: additional data formats (e.g. XLS/XLSX and SQLite)
            # TODO: better handling of british/american dates (currently assumes british)
            df = pandas.read_csv(properties['path'], index_col=0, parse_dates=True, dayfirst=True)
            df = df[properties['column']]
            ts = Timeseries(name, df, metadata=properties)
            model.data[name] = ts
        else:
            raise NotImplementedError()
        return ts

class Variable(object):
    def __init__(self, initial=0.0):
        self._initial = initial
        self._value = initial

    def value(self, index=None):
        return self._value

    @classmethod
    def from_xml(cls, model, xml):
        key = xml.get('key')
        try:
            value = float(xml.text)
        except:
            value = xml.text
        return key, ParameterConstant(value=value)

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
    
    def __init__(self, model, position=None, name=None, **kwargs):
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
        self.color = 'black'
        self.position = position
        self.__name = None
        self.name = name
        
        self.properties = {
            'cost': ParameterConstant(value=0.0)
        }
    
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
        try:
            del(self.model.node[self.__name])
        except KeyError:
            pass
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
        self.model.graph.add_edge(self, node)
        if from_slot is not None:
            self.slots[from_slot] = node
        if to_slot is not None:
            node.slots[to_slot] = self
    
    def disconnect(self, node=None):
        """Remove a connection from this Node to another Node
        
        Parameters
        ----------
        node : Node (optional)
            The node to remove the connection to. If another node is not
            specified, all connections from this node will be removed.
        """
        if node is not None:
            self.model.graph.remove_edge(self, node)
        else:
            neighbors = self.model.graph.neighbors(self)
            for neighbor in neighbors:
                self.model.graph.remove_edge(self, neighbor)
    
    def check(self):
        """Check the node is valid
        
        Raises an exception if the node is invalid
        """
        if not isinstance(self.position, (tuple, list,)):
            raise TypeError('{} position has invalid type ({})'.format(self, type(self.position)))
        if not len(self.position) == 2:
            raise ValueError('{} position has invalid length ({})'.format(self, len(self.position)))

    def commit(self, volume, chain):
        """Commit a volume of water actually supplied/transferred/received
        
        Parameter
        ---------
        volume : float
            The volume to commit
        chain : string
            The position in the route of the node for this commit. This must be
            one of: 'first' (the node supplied water), 'middle' (the node
            transferred water) or 'last' (the node received water).
        
        This should be implemented by the various node subclasses.
        """
        pass
    
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

class Supply(Node):
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
        Node.__init__(self, *args, **kwargs)
        self.color = '#F26C4F' # light red
        
        max_flow = kwargs.pop('max_flow', 0.0)
        if callable(max_flow):
            self.properties['max_flow'] = ParameterFunction(self, max_flow)
        else:
            self.properties['max_flow'] = ParameterConstant(value=max_flow)
        
        self.licenses = None
    
    def commit(self, volume, chain):
        super(Supply, self).commit(volume, chain)
        if self.licenses is not None:
            self.licenses.commit(volume)

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

class Demand(Node):
    """A demand in the network"""
    def __init__(self, *args, **kwargs):
        """Initialise a new Demand node
        
        Parameters
        ----------
        demand : float
            The amount of water to demand each timestep
        """
        Node.__init__(self, *args, **kwargs)
        self.color = '#FFF467' # light yellow
        
        self.properties['demand'] = ParameterConstant(value=kwargs.pop('demand',10.0))

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
        
        if 'max_flow' in kwargs:
            max_flow = kwargs.pop('max_flow', 0.0)
            if callable(max_flow):
                self.properties['max_flow'] = ParameterFunction(self, max_flow)
            else:
                self.properties['max_flow'] = ParameterConstant(value=max_flow)

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

        if 'ratio' in kwargs:
            self.properties['ratio'] = ParameterConstant(value=kwargs['ratio'])
        else:
            self.properties['ratio'] = ParameterConstant(value=0.5)

class Catchment(Node):
    """A hydrological catchment, supplying water to the river network"""
    def __init__(self, *args, **kwargs):
        """Initialise a new Catchment node
        
        Parameters
        ----------
        flow : float or function
            The amount of water supplied by the catchment each timestep
        """
        Node.__init__(self, *args, **kwargs)
        self.color = '#82CA9D' # green
        
        flow = kwargs.pop('flow', 2.0)
        if callable(flow):
            self.properties['flow'] = ParameterFunction(self, flow)
        else:
            self.properties['flow'] = ParameterConstant(value=flow)        
    
    def check(self):
        Node.check(self)
        successors = self.model.graph.successors(self)
        if not len(successors) == 1:
            raise ValueError('{} has invalid number of successors ({})'.format(self, len(successors)))

class River(Node):
    """A node in the river network
    
    This node may have multiple upstream nodes (i.e. a confluence) but only
    one downstream node.
    """
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)
        self.color = '#6ECFF6' # blue

class RiverSplit(River):
    """A split in the river network"""
    def __init__(self, *args, **kwargs):
        """Initialise a new RiverSplit instance
        
        Parameters
        ----------
        split : float or function
            The ratio to apportion the flow between the two downstream nodes as
            a ratio (0.0-1.0). If no value is given a default value of 0.5 is
            used.
        """
        River.__init__(self, *args, **kwargs)
        self.slots = {1: None, 2: None}
        
        if 'split' in kwargs:
            self.properties['split'] = ParameterConstant(value=kwargs['split'])
        else:
            self.properties['split'] = ParameterConstant(value=0.5)

class Discharge(River):
    """An inline discharge to the river network
    
    This node is similar to a catchment, but sits inline to the river network,
    rather than at the head of the river.
    """
    def __init__(self, *args, **kwargs):
        River.__init__(self, *args, **kwargs)
        
        flow = kwargs.pop('flow', 0.0)
        if callable(flow) is not None:
            self.properties['flow'] = ParameterFunction(self, flow)
        else:
            self.properties['flow'] = ParameterConstant(value=flow)

class Terminator(Node):
    """A sink in the river network
    
    This node is required to close the network and is used by some of the
    routing algorithms. Every river must end in a Terminator.
    """
    pass

class RiverGauge(River):
    pass

class RiverAbstraction(Supply, River):
    """An abstraction from the river network"""
    pass

class Reservoir(Supply, Demand):
    """A reservoir"""
    def __init__(self, *args, **kwargs):
        super(Reservoir, self).__init__(*args, **kwargs)
        
        # reservoir cannot supply more than it's current volume
        def func(parent, index):
            return self.properties['current_volume'].value(index)
        self.properties['max_flow'] = ParameterFunction(self, func)
        
        def func(parent, index):
            current_volume = self.properties['current_volume'].value(index)
            max_volume = self.properties['max_volume'].value(index)
            return max_volume - current_volume
        self.properties['demand'] = ParameterFunction(self, func)

    def commit(self, volume, chain):
        super(Reservoir, self).commit(volume, chain)
        # update the volume remaining in the reservoir
        if chain == 'first':
            # reservoir supplied some water
            self.properties['current_volume']._value -= volume
        elif chain == 'last':
            # reservoir received some water
            self.properties['current_volume']._value += volume

    def check(self):
        super(Reservoir, self).check()
        index = self.model.timestamp
        # check volume doesn't exceed maximum volume
        assert(self.properties['max_volume'].value(index) >= self.properties['current_volume'].value(index))

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
