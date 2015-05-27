#!/usr/bin/env python

from ..core import *

import glpk

inf = float('inf')

class SolverGLPK(Solver):
    name = 'GLPK'

    def solve(self, model):
        if model.dirty:
            '''
            This section should only need to be run when the model has changed
            structure (e.g. a new connection has been made).
            '''

            lp = self.lp = glpk.LPX()
            lp.obj.maximize = True

            # find all routes between supply and demand nodes
            routes = self.routes = model.find_all_routes(Supply, Demand, valid=(Link,))
            
            # find all routes between catchments and river gauges
            mrf_routes = self.mrf_routes = model.find_all_routes(Catchment, RiverGauge)
            river_gauge_nodes = self.river_gauge_nodes = dict([(node, {}) for node in set([route[-1] for route in mrf_routes])])
            for river_gauge_node, info in river_gauge_nodes.items():
                info['river_routes'] = []
            for route in mrf_routes:
                river_gauge_nodes[route[-1]]['river_routes'].append(route)

            # each route between a supply and demand is represented as a column
            supply_nodes = self.supply_nodes = {}
            demand_nodes = self.demand_nodes = {}
            intermediate_max_flow_constraints = self.intermediate_max_flow_constraints = {}
            lp.cols.add(len(routes))
            for col_idx, route in enumerate(routes):
                col = lp.cols[col_idx]
                col.bounds = 0, None  # supply must be >= 0
                supply_node = route[0]
                demand_node = route[-1]

                supply_nodes.setdefault(supply_node, {'cols': [], 'col_idxs': []})
                supply_nodes[supply_node]['cols'].append(col)
                supply_nodes[supply_node]['col_idxs'].append(col_idx)

                demand_nodes.setdefault(demand_node, {'cols': [], 'col_idxs': []})
                demand_nodes[demand_node]['cols'].append(col)
                demand_nodes[demand_node]['col_idxs'].append(col_idx)

                intermediate_nodes = route[1:-1]
                for node in intermediate_nodes:
                    if 'max_flow' in node.properties and node not in intermediate_max_flow_constraints:
                        row_idx = lp.rows.add(1)
                        row = lp.rows[row_idx]
                        intermediate_max_flow_constraints[node] = row
                        col_idxs = []
                        for col_idx, route in enumerate(routes):
                            if node in route:
                                col_idxs.append(col_idx)
                        row.matrix = [(idx, 1.0) for idx in col_idxs]

            for supply_node, info in supply_nodes.items():
                row_idx = lp.rows.add(1)
                row = lp.rows[row_idx]
                info['supply_constraint'] = row
                info['matrix'] = [(idx, 1.0) for idx in info['col_idxs']]

                if isinstance(supply_node, RiverAbstraction):
                    row_idx = lp.rows.add(1)
                    row = lp.rows[row_idx]
                    info['river_constraint'] = row
                    info['river_routes'] = river_routes = model.find_all_routes(Catchment, supply_node)

            for demand_node, info in demand_nodes.items():
                row_idx = lp.rows.add(1)
                row = lp.rows[row_idx]
                info['demand_constraint'] = row
                info['matrix'] = [(idx, 1.0) for idx in info['col_idxs']]
            
            # add mrf constraint rows
            for river_gauge_node, info in river_gauge_nodes.items():
                row_idx = lp.rows.add(1)
                row = lp.rows[row_idx]
                info['mrf_constraint'] = row

            model.dirty = False
        else:
            lp = self.lp
            supply_nodes = self.supply_nodes
            demand_nodes = self.demand_nodes
            routes = self.routes
            intermediate_max_flow_constraints = self.intermediate_max_flow_constraints
            river_gauge_nodes = self.river_gauge_nodes

        timestamp = self.timestamp = model.timestamp

        # apply route costs as objective function
        costs = []
        for col_idx, route in enumerate(routes):
            cost = 0.0
            for node in route:
                cost += node.properties['cost'].value(timestamp)
            costs.append(cost)
        max_cost = max(costs)
        for col_idx, cost in enumerate(costs):
            lp.obj[col_idx] = 1 + max_cost - cost

        # supply is limited by a maximum flow, and any licenses
        for supply_node, info in supply_nodes.items():
            row = info['supply_constraint']
            max_flow_parameter = supply_node.properties['max_flow'].value(timestamp)
            max_flow_license = inf
            if supply_node.licenses is not None:
                max_flow_license = supply_node.licenses.available(timestamp)
            max_flow = min(max_flow_parameter, max_flow_license)
            row.matrix = info['matrix']
            row.bounds = 0, max_flow

        # demands require water, but can only accept a certain amount
        total_water_demanded = 0.0
        for demand_node, info in demand_nodes.items():
            row = info['demand_constraint']
            demand_value = demand_node.properties['demand'].value(timestamp)
            row.matrix = info['matrix']
            row.bounds = 0, demand_value
            total_water_demanded += demand_value

        # intermediate node max flow constraints
        for node, row in intermediate_max_flow_constraints.items():
            row.bounds = 0, node.properties['max_flow'].value(timestamp)

        # river flow constraints
        for supply_node, info in supply_nodes.items():
            if isinstance(supply_node, RiverAbstraction):
                river_routes = info['river_routes']
                flow_constraint, abstraction_idxs, abstraction_coefficients = self.upstream_constraint(river_routes)
                row = info['river_constraint']
                row.matrix = [(abstraction_idxs[n], abstraction_coefficients[n]) for n in range(0, len(abstraction_idxs))]
                row.bounds = 0, flow_constraint
        
        # mrf constraints
        for river_gauge_node, info in river_gauge_nodes.items():
            mrf_value = river_gauge_node.properties['mrf'].value(timestamp)
            row = info['mrf_constraint']
            if mrf_value is None:
                row.matrix = []
                row.bounds = 0, inf
                continue
            river_routes = info['river_routes']
            flow_constraint, abstraction_idxs, abstraction_coefficients = self.upstream_constraint(river_routes)
            flow_constraint = max(0, flow_constraint - mrf_value)
            row.matrix = [(abstraction_idxs[n], abstraction_coefficients[n]) for n in range(0, len(abstraction_idxs))]
            row.bounds = 0, flow_constraint

        # solve the linear programme
        lp.simplex()
        assert(lp.status == 'opt')
        status = 'optimal'

        # retrieve the results
        result = [round(value.primal, 3) for value in lp.cols]
        total_water_supplied = sum(result)

        # commit the volume of water actually supplied
        for n, route in enumerate(routes):
            route[0].commit(result[n], chain='first')
            for node in route[1:-1]:
                node.commit(result[n], chain='middle')
            route[-1].commit(result[n], chain='last')

        # calculate the total amount of water transferred via each node/link
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
        for k, v in volumes_links.items():
            v = round(v, 3)
            if v:
                volumes_links[k] = v
            else:
                del(volumes_links[k])
        for k, v in volumes_nodes.items():
            v = round(v, 3)
            if v:
                volumes_nodes[k] = v
            else:
                del(volumes_nodes[k])

        return status, round(total_water_demanded, 3), round(total_water_supplied, 3), volumes_links, volumes_nodes

    def upstream_constraint(self, river_routes):
        '''Calculate parameters for river flow constraint'''
        flow_constraint = 0.0
        upstream_abstractions = {}
        for route in river_routes:
            # traverse the route from abstraction back up to catchments
            route = route[::-1]
            coefficient = 1.0
            for n, node in enumerate(route):
                if isinstance(node, Catchment):
                    # catchments add water
                    flow_constraint += (node.properties['flow'].value(self.timestamp) * coefficient)
                elif isinstance(node, RiverSplit):
                    # splits
                    if node.slots[1] is route[n-1]:
                        coefficient *= node.properties['split'].value(self.timestamp)
                    elif node.slots[2] is route[n-1]:
                        coefficient *= (1 - node.properties['split'].value(self.timestamp))
                    else:
                        raise RuntimeError()
                elif isinstance(node, RiverAbstraction):
                    # abstractions remove water
                    upstream_abstractions.setdefault(node, 1.0)
                    upstream_abstractions[node] *= coefficient
        abstraction_idxs = []
        abstraction_coefficients = []
        for upstream_node, coefficient in upstream_abstractions.items():
            cols = self.supply_nodes[upstream_node]['col_idxs']
            abstraction_idxs.extend(cols)
            abstraction_coefficients.extend([coefficient]*len(cols))
        
        return flow_constraint, abstraction_idxs, abstraction_coefficients
