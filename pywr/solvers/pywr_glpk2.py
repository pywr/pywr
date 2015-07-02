

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

            routes = model.find_all_routes(Input, Output, valid=(Link,))
            first_index = lp.cols.add(len(routes))
            routes = self.routes = list(zip([lp.cols[index] for index in range(first_index, first_index+len(routes))], routes))

            input_nodes = self.input_nodes = {}
            output_nodes = self.output_nodes = {}
            intermediate_max_flow_constraints = self.intermediate_max_flow_constraints = {}

            for col, route in routes:
                col.bounds = 0, None  # input must be >= 0
                input_node = route[0]
                output_node = route[-1]

                input_nodes.setdefault(input_node, {'cols': [], 'col_idxs': []})
                input_nodes[input_node]['cols'].append(col)
                input_nodes[input_node]['col_idxs'].append(col.index)

                output_nodes.setdefault(output_node, {'cols': [], 'col_idxs': []})
                output_nodes[output_node]['cols'].append(col)
                output_nodes[output_node]['col_idxs'].append(col.index)

                # find constraints on intermediate nodes
                intermediate_nodes = route[1:-1]
                for node in intermediate_nodes:
                    if 'max_flow' in node.properties and node not in intermediate_max_flow_constraints:
                        row_idx = lp.rows.add(1)
                        row = lp.rows[row_idx]
                        intermediate_max_flow_constraints[node] = row
                        col_idxs = []
                        for col, route in routes:
                            if node in route:
                                col_idxs.append(col.index)
                        row.matrix = [(idx, 1.0) for idx in col_idxs]
                        
            # initialise the structure (only) for the input constraint
            for input_node, info in input_nodes.items():
                row_idx = lp.rows.add(1)
                row = lp.rows[row_idx]
                info['input_constraint'] = row
                info['matrix'] = [(idx, 1.0) for idx in info['col_idxs']]

            for output_node, info in output_nodes.items():
                # add a column for each output
                col_idx = lp.cols.add(1)
                col = lp.cols[col_idx]
                info['output_col'] = col
                # mass balance between input and output
                row_idx = lp.rows.add(1)
                row = lp.rows[row_idx]
                row.bounds = 0, 0
                input_matrix = [(idx, 1.0) for idx in info['col_idxs']]
                output_matrix = [(col_idx, -1.0)]
                row.matrix = input_matrix + output_matrix
                info['output_row'] = row

            # TODO add min flow requirement
            """
            # add mrf constraint rows
            for river_gauge_node, info in river_gauge_nodes.items():
                row_idx = lp.rows.add(1)
                row = lp.rows[row_idx]
                info['mrf_constraint'] = row
            """

        else:
            lp = self.lp
            input_nodes = self.input_nodes
            output_nodes = self.output_nodes
            routes = self.routes
            intermediate_max_flow_constraints = self.intermediate_max_flow_constraints
            #blenders = self.blenders
            #groups = self.groups

        timestamp = self.timestamp = model.timestamp

        for node in model.nodes():
            node.before()

        # the cost of a route is equal to the sum of the route's node's costs
        costs = []
        for col, route in routes:
            cost = 0.0
            for node in route[0:-1]:
                cost += node.properties['cost'].value(timestamp)
            lp.obj[col.index] = -cost

        # there is a benefit for inputting water to outputs
        for output_node, info in output_nodes.items():
            col = info['output_col']
            cost = output_node.properties['benefit'].value(timestamp)
            lp.obj[col.index] = cost

        # input is limited by a minimum and maximum flow, and any licenses
        for input_node, info in input_nodes.items():
            row = info['input_constraint']
            max_flow_parameter = input_node.properties['max_flow'].value(timestamp)
            max_flow_license = inf
            if input_node.licenses is not None:
                max_flow_license = input_node.licenses.available(timestamp)
            max_flow = min(max_flow_parameter, max_flow_license)
            min_flow = input_node.properties['min_flow'].value(timestamp)
            row.matrix = info['matrix']
            row.bounds = min_flow, max_flow

        # outputs require a water between a min and maximium flow
        total_water_outputed = 0.0
        for output_node, info in output_nodes.items():
            # update output for the current timestep
            col = info['output_col']
            max_flow = output_node.properties['max_flow'].value(timestamp)
            min_flow = output_node.properties['min_flow'].value(timestamp)
            col.bounds = min_flow, max_flow
            total_water_outputed += min_flow

        # intermediate node max flow constraints
        for node, row in intermediate_max_flow_constraints.items():
            row.bounds = 0, node.properties['max_flow'].value(timestamp)

        # TODO add min flow requirement
        """
        # mrf constraints
        for river_gauge_node, info in river_gauge_nodes.items():
            mrf_value = river_gauge_node.properties['mrf'].value(timestamp)
            row = info['mrf_constraint']
            if mrf_value is None:
                row.bounds = None, None
                continue
            river_routes = info['river_routes']
            flow_constraint, abstraction_idxs, abstraction_coefficients = self.upstream_constraint(river_routes)
            flow_constraint = max(0, flow_constraint - mrf_value)
            row.matrix = [(abstraction_idxs[n], abstraction_coefficients[n]) for n in range(0, len(abstraction_idxs))]
            row.bounds = 0, flow_constraint


        # blender constraints
        for blender, info in blenders.items():
            matrix = []
            row = info['blender_constraint']
            ratio = blender.properties['ratio'].value(self.timestamp)
            for col_idx, sign in info['routes']:
                if sign == 1:
                    matrix.append((col_idx, sign*(1-ratio)))
                else:
                    matrix.append((col_idx, sign*ratio))
            row.matrix = matrix

        # groups
        for group, info in groups.items():
            if group.licenses is None:
                continue
            row = info['group_constraint']
            if group.licenses is None:
                row.bounds = None, None
            else:
                row.bounds = 0, group.licenses.available(timestamp)
        """

        # solve the linear programme
        lp.simplex()
        assert(lp.status == 'opt')
        status = 'optimal'

        # retrieve the results
        result = [round(col.primal, 3) for col, route in routes]
        total_water_supplied = sum(result)

        # commit the volume of water actually supplied
        for n, (col, route) in enumerate(routes):
            route[0].commit(result[n], chain='first')
            for node in route[1:-1]:
                node.commit(result[n], chain='middle')
            route[-1].commit(result[n], chain='last')

        for node in model.nodes():
            node.after()

        # calculate the total amount of water transferred via each node/link
        volumes_links = {}
        volumes_nodes = {}
        for n, (col, route) in enumerate(routes):
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

        return status, round(total_water_outputed, 3), round(total_water_supplied, 3), volumes_links, volumes_nodes