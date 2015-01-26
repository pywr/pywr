#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import networkx as nx
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as pyplot
import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import inspect
import pandas
import datetime

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
warnings.simplefilter(action = "ignore", category = UnicodeWarning)

inf = float('inf')
TIMESTEP = datetime.timedelta(1)

class Model(object):
    def __init__(self, solver=None):
        self.graph = nx.DiGraph()
        self.metadata = {}
        self.parameters = {}
        self.data = {}
        
        if solver is not None:
            # use specific solver
            try:
                self.solver = SolverMeta.solvers[solver.lower()]
            except KeyError:
                raise KeyError('Unrecognised solver: {}'.format(solver))
        else:
            # use default solver
            self.solver = SolverCyLP()
        
        self.timestamp = pandas.to_datetime('2015-01-5')
    
    def plot(self, volume_labels=None, node_labels=None):
        fig = pyplot.figure()
        ax = pyplot.subplot(111)
        pyplot.axis('equal')
        nodes = self.graph.nodes()
        edges = self.graph.edges()
        pos = dict([(node, node.position) for node in nodes])
        edge_colors = []
        river_types = (Catchment, River, Terminator,)
        for edge in edges:
            if isinstance(edge[0], river_types) and isinstance(edge[1], river_types):
                color = '#00AEEF' # cyan
            else:
                color = 'black'
            edge_colors.append(color)
        colors = [node.color for node in nodes]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes, node_color=colors)
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color=edge_colors)
        
        if volume_labels is not None:
            volume_labels = dict([(k,'{:.3g}'.format(v)) for k,v in volume_labels.items()])
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=volume_labels)
        
        if node_labels is not None:
            node_labels = dict([(k,'{:.3g}'.format(v)) for k,v in node_labels.items()])
            nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10)
        
        catchment_nodes = [node for node in nodes if isinstance(node, Catchment)]
        nx.draw_networkx_labels(self.graph, pos, nodelist=catchment_nodes, labels=dict([(n, n.properties['flow'].value(self.timestamp)) for n in catchment_nodes]), font_size=10)
        
        return fig
    
    def check(self):
        nodes = self.graph.nodes()
        for node in nodes:
            node.check()

    def nodes(self):
        return self.graph.nodes()
    
    def find_all_routes(self, type1, type2, valid=None):
        '''Find all routes between two nodes or types of node'''
        
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
        '''Step the model forward by one day'''
        ret = self.solve()
        self.timestamp += TIMESTEP
        return ret

    def solve(self):
        '''Call solver to solve the current timestep'''
        return self.solver.solve(self)

class SolverMeta(type):
    solvers = {}
    def __new__(cls, clsname, bases, attrs):
        newclass = super(SolverMeta, cls).__new__(cls, clsname, bases, attrs)
        cls.solvers[newclass.name.lower()] = newclass
        return newclass

class Solver(object):
    '''Solver base class from which all solvers should inherit'''
    __metaclass__ = SolverMeta
    name = 'default'
    def solve(self, model):
        raise NotImplementedError('Solver should be subclassed to provide solve()')

class SolverCyLP(Solver):
    name = 'CyLP'
    def solve(self, model):
        timestamp = model.timestamp
        
        routes = model.find_all_routes(Supply, Demand, valid=(Link,))
        count_routes = len(routes)
        assert(count_routes > 0)
        
        costs = []
        by_supply = {}
        by_demand = {}
        for n, route in enumerate(routes):
            supply_node = route[0]
            demand_node = route[-1]
            by_supply.setdefault(supply_node, [])
            by_supply[supply_node].append(n)
            by_demand.setdefault(demand_node, [])
            by_demand[demand_node].append(n)
            cost = 0.0
            for node in route:
                cost += node.properties['cost'].value(timestamp)
            costs.append(cost)
        
        s = CyClpSimplex()
        x = s.addVariable('x', count_routes)
        
        for n, route in enumerate(routes):
            col = x[n]
            s += 0.0 <= col <= inf
        
        for supply_node, idxs in by_supply.items():
            cols = x[idxs]
            # maximum supply from node is limited by max_flow parameter and licenses
            max_flow_parameter = supply_node.properties['max_flow'].value(timestamp)
            max_flow_license = inf
            if supply_node.licenses is not None:
                max_flow_license = supply_node.licenses.available(timestamp)
            max_flow = min(max_flow_parameter, max_flow_license)
            s += cols.sum() <= max_flow
        
        total_water_demanded = 0.0
        for demand_node, idxs in by_demand.items():
            cols = x[idxs]
            demand_value = demand_node.properties['demand'].value(timestamp)
            s += cols.sum() <= demand_value
            total_water_demanded += demand_value

        # river flow constraints
        for supply_node, idxs in by_supply.items():
            if isinstance(supply_node, RiverAbstraction):
                flow_constraint = 0.0
                # find all routes from a catchment to the abstraction
                river_routes = model.find_all_routes(Catchment, supply_node)
                upstream_abstractions = {}
                for route in river_routes:
                    route = route[::-1]
                    coefficient = 1.0
                    for n, node in enumerate(route):
                        if isinstance(node, Catchment):
                            # catchments add water
                            flow_constraint += (node.properties['flow'].value(timestamp) * coefficient)
                        elif isinstance(node, RiverSplit):
                            # splits
                            if node.slots[1] is route[n-1]:
                                coefficient *= node.properties['split'].value(timestamp)
                            elif node.slots[2] is route[n-1]:
                                coefficient *= (1 - node.properties['split'].value(timestamp))
                            else:
                                raise RuntimeError()
                        elif isinstance(node, RiverAbstraction):
                            # abstractions remove water
                            upstream_abstractions.setdefault(node, 1.0)
                            upstream_abstractions[node] *= coefficient
                abstraction_idxs = []
                abstraction_coefficients = []
                for upstream_node, coefficient in upstream_abstractions.items():
                    cols = by_supply[upstream_node]
                    abstraction_idxs.extend(cols)
                    abstraction_coefficients.extend([coefficient]*len(cols))
                s += CyLPArray(abstraction_coefficients) * x[abstraction_idxs] <= flow_constraint
        
        # workaround for bug in CyLP where problem fails with only 1 variable
        if count_routes == 1:
            y = s.addVariable('y', 1)
        
        # TODO: two-phase solve
        # if resource state < 1 (for any source), skip 1
        # 1) minimise cost
        # 2) maximise high resource state usage
        s.optimizationDirection = 'max'
        s.objective = (1+max(costs)-CyLPArray(costs)) * x
        
        s.logLevel = 0
        status = s.primal()
        result = [round(value, 3) for value in s.primalVariableSolution['x']]
        
        total_water_supplied = sum(result)

        volumes_links = {}
        volumes_nodes = {}
        for n, route in enumerate(routes):
            if result[n] > 0:
                for m in range(0, len(route)):
                    volumes_nodes.setdefault(route[m], 0.0)
                    volumes_nodes[route[m]] += result[n]

                    if m+1 == len(route):
                        break
                    
                    pair = (route[m], route[m+1])
                    volumes_links.setdefault(pair, 0.0)
                    volumes_links[pair] += result[n]
        
        # commit the volume of water actually supplied
        for n, route in enumerate(routes):
            route[0].commit(result[n], chain='first')
            for node in route[1:-1]:
                node.commit(result[n], chain='middle')
            route[-1].commit(result[n], chain='last')

        for k,v in volumes_links.items():
            v = round(v,3)
            if v:
                volumes_links[k] = v
            else:
                del(volumes_links[k])
        
        for k,v in volumes_nodes.items():
            v = round(v,3)
            if v:
                volumes_nodes[k] = v
            else:
                del(volumes_nodes[k])
        
        assert(status == 'optimal')
        
        return status, round(total_water_demanded, 3), round(total_water_supplied, 3)

class Parameter(object):
    def __init__(self, value=None):
        self._value = value
    
    def value(self, index=None):
        return self._value

class ParameterFunction(object):
    def __init__(self, parent, func):
        self._parent = parent
        self._func = func

    def value(self, index=None):
        return self._func(self._parent, index)

class Timeseries(object):
    def __init__(self, df):
        self.df = df
    
    def value(self, index):
        return self.df[index]

class Variable(object):
    def __init__(self, initial=0.0):
        self._initial = initial
        self._value = initial

    def value(self, index=None):
        return self._value

# node subclasses are stored in a dict for convenience
node_registry = {}
class NodeMeta(type):
    def __new__(meta, name, bases, dct):
        return super(NodeMeta, meta).__new__(meta, name, bases, dct)
    def __init__(cls, name, bases, dct):
        super(NodeMeta, cls).__init__(name, bases, dct)
        node_registry[name.lower()] = cls

class Node(object):
    '''Base object from which all other nodes inherit'''
    __metaclass__ = NodeMeta
    
    def __init__(self, model, position=None, name=None, **kwargs):
        self.model = model
        model.graph.add_node(self)
        self.color = 'black'
        self.position = position
        self.name = name
        
        self.properties = {
            'cost': Parameter(value=0.0)
        }
    
    def __repr__(self):
        if self.name:
            return '<{} "{}">'.format(self.__class__.__name__, self.name)
        else:
            return '<{} "{}">'.format(self.__class__.__name__, hex(id(self)))
    
    def connect(self, node, slot=None):
        '''Create a connection from this Node to another Node'''
        if self.model is not node.model:
            raise RuntimeError("Can't connect Nodes in different Models")
        self.model.graph.add_edge(self, node)
        if slot is not None:
            self.slots[slot] = node
    
    def disconnect(self, node=None):
        '''Remove a connection from this Node to another Node
        
        If another Node is not specified, all connections from this Node will
        be removed.
        '''
        if node is not None:
            self.model.graph.remove_edge(self, node)
        else:
            neighbors = self.model.graph.neighbors(self)
            for neighbor in neighbors:
                self.model.graph.remove_edge(self, neighbor)
    
    def check(self):
        if not isinstance(self.position, (tuple, list,)):
            raise TypeError('{} position has invalid type ({})'.format(self, type(self.position)))
        if not len(self.position) == 2:
            raise ValueError('{} position has invalid length ({})'.format(self, len(self.position)))

    def commit(self, volume, chain):
        '''Commit a volume of water actually supplied
        
        This should be implemented by the various node classes
        '''
        pass

class Supply(Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)
        self.color = '#F26C4F' # light red
        
        if 'max_flow' in kwargs:
            self.properties['max_flow'] = Parameter(value=kwargs['max_flow'])
        else:
            self.properties['max_flow'] = Parameter(value=0)
        
        self.licenses = None
    
    def commit(self, volume, chain):
        super(Supply, self).commit(volume, chain)
        if self.licenses is not None:
            self.licenses.commit(volume)

class Demand(Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)
        self.color = '#FFF467' # light yellow
        
        self.properties['demand'] = Parameter(value=10)

class Link(Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)
        self.color = '#A0A0A0' # 45% grey

class Catchment(Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)
        self.color = '#82CA9D' # green
        
        self.properties['flow'] = Parameter(value=2.0)
    
    def check(self):
        Node.check(self)
        successors = self.model.graph.successors(self)
        if not len(successors) == 1:
            raise ValueError('{} has invalid number of successors ({})'.format(self, len(successors)))

class River(Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)
        self.color = '#6ECFF6' # blue

class RiverSplit(River):
    def __init__(self, *args, **kwargs):
        River.__init__(self, *args, **kwargs)
        self.slots = {1: None, 2: None}
        
        if 'split' in kwargs:
            self.properties['split'] = Parameter(value=kwargs['split'])
        else:
            self.properties['split'] = Parameter(value=0.5)

class Terminator(Node):
    pass

class RiverAbstraction(Supply, River):
    pass

class Reservoir(Supply, Demand):
    def __init__(self, *args, **kwargs):
        super(Reservoir, self).__init__(*args, **kwargs)
        
        # reservoir cannot supply more than it's current volume
        def func(parent, index):
            return self.properties['current_volume'].value(index)
        self.properties['max_flow'] = ParameterFunction(self, func)
        
        def func(parent, index):
            current_volume = self.properties['current_volume'].value(index)
            max_volume = self.properties['max_volume'].value(index)
            print(max_volume - current_volume)
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
