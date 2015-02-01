#!/usr/bin/env python

from ..core import *

import glpk

inf = float('inf')

class SolverGLPK(Solver):
    name = 'GLPK'
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

        lp = glpk.LPX()
        
        # add a column for each route
        lp.cols.add(count_routes)
        for n, route in enumerate(routes):
            col = lp.cols[n]
            col.bounds = 0, inf
        
        for supply_node, idxs in by_supply.items():
            # maximum supply from node is limited by max_flow parameter and licenses
            max_flow_parameter = supply_node.properties['max_flow'].value(timestamp)
            max_flow_license = inf
            if supply_node.licenses is not None:
                max_flow_license = supply_node.licenses.available(timestamp)
            max_flow = min(max_flow_parameter, max_flow_license)
            lp.rows.add(1)
            row = lp.rows[-1]
            row.matrix = [(idx, 1.0) for idx in idxs]
            row.bounds = 0, max_flow

        total_water_demanded = 0.0
        for demand_node, idxs in by_demand.items():
            demand_value = demand_node.properties['demand'].value(timestamp)
            lp.rows.add(1)
            row = lp.rows[-1]
            row.matrix = [(idx, 1.0) for idx in idxs]
            row.bounds = 0, demand_value
            total_water_demanded += demand_value
        
        # intermediate node max flow constraints
        max_flow_constraints = {}
        for n, route in enumerate(routes):
            intermediate_nodes = route[1:-1]
            for node in intermediate_nodes:
                if 'max_flow' in node.properties:
                    max_flow_constraints.setdefault(node, [])
                    max_flow_constraints[node].append(n)
        for node, route_idxs in max_flow_constraints.items():
            lp.rows.add(1)
            row = lp.rows[-1]
            row.matrix = [(idx, 1.0) for idx in route_idxs]
            row.bounds = 0, node.properties['max_flow'].value(timestamp)

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
                lp.rows.add(1)
                row = lp.rows[-1]
                row.matrix = [(abstraction_idxs[n], abstraction_coefficients[n]) for n in range(0, len(abstraction_idxs))]
                row.bounds = 0, flow_constraint
        
        lp.obj.maximize = True
        for n in range(len(lp.cols)):
            lp.obj[n] = 1 + max(costs) - costs[n]
        
        lp.simplex()
        
        result = [round(value.primal, 3) for value in lp.cols]

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
        
        status = lp.status
        if status == 'opt':
            status = 'optimal'
        
        assert(status == 'optimal')
        
        return status, round(total_water_demanded, 3), round(total_water_supplied, 3), volumes_links, volumes_nodes
