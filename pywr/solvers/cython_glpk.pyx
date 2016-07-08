from libc.stdlib cimport malloc, free
from cython.view cimport array as cvarray

cimport cython

from pywr._core import BaseInput, BaseOutput, BaseLink
from pywr._core cimport *
from pywr.core import ModelStructureError
import time
include "glpk.pxi"

inf = float('inf')

cdef class AbstractNodeData:
    cdef public int id
    cdef public bint is_link

cdef class CythonGLPKSolver:
    cdef glp_prob* prob
    cdef glp_smcp smcp
    cdef int idx_col_routes
    cdef int idx_row_non_storages
    cdef int idx_row_cross_domain
    cdef int idx_row_storages
    cdef int idx_row_aggregated
    cdef int idx_row_aggregated_min_max

    cdef list routes
    cdef list non_storages
    cdef list storages
    cdef list routes_cost
    cdef list all_nodes
    cdef list nodes_with_cost
    cdef int num_nodes
    cdef int num_routes
    cdef int num_storages
    cdef cvarray node_costs_arr
    cdef cvarray node_flows_arr
    cdef cvarray route_flows_arr
    cdef cvarray change_in_storage_arr
    cdef list aggregated
    cdef public object stats

    def __cinit__(self):
        # create a new problem
        self.prob = glp_create_prob()
        # disable console messages
        glp_init_smcp(&self.smcp)
        self.smcp.msg_lev = GLP_MSG_ERR

    def __init__(self):
        self.stats = None

    def __dealloc__(self):
        # free the problem
        glp_delete_prob(self.prob)

    cpdef object setup(self, model):
        cdef Node supply
        cdef Node demand
        cdef Node node
        cdef AggregatedNode agg_node
        cdef double min_flow
        cdef double max_flow
        cdef double cost
        cdef double avail_volume
        cdef int col, row
        cdef int* ind
        cdef double* val
        cdef double lb
        cdef double ub
        cdef Timestep timestep
        cdef int status
        cdef cross_domain_row
        cdef int n

        self.all_nodes = list(model.graph.nodes())
        self.nodes_with_cost = []

        n = 0
        for _node in self.all_nodes:
            _node.__data = AbstractNodeData()
            _node.__data.id = n
            if hasattr(_node, "get_cost"):
                self.nodes_with_cost.append(_node)
            if isinstance(_node, BaseLink):
                _node.__data.is_link = True
            n += 1
        self.num_nodes = n

        self.node_costs_arr = cvarray(shape=(self.num_nodes,), itemsize=sizeof(double), format="d")
        self.node_flows_arr = cvarray(shape=(self.num_nodes,), itemsize=sizeof(double), format="d")

        routes = model.find_all_routes(BaseInput, BaseOutput, valid=(BaseLink, BaseInput, BaseOutput))
        # Find cross-domain routes
        cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')

        non_storages = []
        storages = []
        aggregated = []
        aggregated_min_max = []
        for some_node in model.graph.nodes():
            if isinstance(some_node, (BaseInput, BaseLink, BaseOutput)):
                non_storages.append(some_node)
            elif isinstance(some_node, Storage):
                storages.append(some_node)
            elif isinstance(some_node, AggregatedNode):
                if some_node.factors is not None:
                    aggregated.append(some_node)
                if some_node.min_flow > -inf or \
                   some_node.max_flow < inf:
                    aggregated_min_max.append(some_node)

        if len(routes) == 0:
            raise ModelStructureError("Model has no valid routes")
        if len(non_storages) == 0:
            raise ModelStructureError("Model has no non-storage nodes")

        self.num_routes = len(routes)
        self.route_flows_arr = cvarray(shape=(self.num_routes,), itemsize=sizeof(double), format="d")
        self.num_storages = len(storages)
        if self.num_storages > 0:
            self.change_in_storage_arr = cvarray(shape=(self.num_storages,), itemsize=sizeof(double), format="d")

        # clear the previous problem
        glp_erase_prob(self.prob)
        glp_set_obj_dir(self.prob, GLP_MIN)
        # add a column for each route
        self.idx_col_routes = glp_add_cols(self.prob, <int>(len(routes)))

        # create a lookup for the cross-domain routes.
        cross_domain_cols = {}
        for cross_domain_route in cross_domain_routes:
            # These routes are only 2 nodes. From demand to supply
            demand, supply = cross_domain_route
            # TODO make this time varying.
            conv_factor = supply.get_conversion_factor()
            supply_cols = [(n, conv_factor) for n, route in enumerate(routes) if route[0] is supply]
            # create easy lookup for the route columns this demand might
            # provide cross-domain connection to
            if demand in cross_domain_cols:
                cross_domain_cols[demand].extend(supply_cols)
            else:
                cross_domain_cols[demand] = supply_cols

        # explicitly set bounds on route and demand columns
        for col, route in enumerate(routes):
            glp_set_col_bnds(self.prob, self.idx_col_routes+col, GLP_LO, 0.0, DBL_MAX)

        # constrain supply minimum and maximum flow
        self.idx_row_non_storages = glp_add_rows(self.prob, len(non_storages))
        # Add rows for the cross-domain routes.
        if len(cross_domain_cols) > 0:
            self.idx_row_cross_domain = glp_add_rows(self.prob, len(cross_domain_cols))

        cross_domain_row = 0
        for row, some_node in enumerate(non_storages):
            # Differentiate betwen the node type.
            # Input & Output only apply their flow constraints when they
            # are the first and last node on the route respectively.
            if isinstance(some_node, BaseInput):
                cols = [n for n, route in enumerate(routes) if route[0] is some_node]
            elif isinstance(some_node, BaseOutput):
                cols = [n for n, route in enumerate(routes) if route[-1] is some_node]
            else:
                # Other nodes apply their flow constraints to all routes passing through them
                cols = [n for n, route in enumerate(routes) if some_node in route]
            ind = <int*>malloc((1+len(cols)) * sizeof(int))
            val = <double*>malloc((1+len(cols)) * sizeof(double))
            for n, c in enumerate(cols):
                ind[1+n] = 1+c
                val[1+n] = 1
            glp_set_mat_row(self.prob, self.idx_row_non_storages+row, len(cols), ind, val)
            glp_set_row_bnds(self.prob, self.idx_row_non_storages+row, GLP_FX, 0.0, 0.0)
            free(ind)
            free(val)

            # Add constraint for cross-domain routes
            # i.e. those from a demand to a supply
            if some_node in cross_domain_cols:
                col_vals = cross_domain_cols[some_node]
                ind = <int*>malloc((1+len(col_vals)+len(cols)) * sizeof(int))
                val = <double*>malloc((1+len(col_vals)+len(cols)) * sizeof(double))
                for n, c in enumerate(cols):
                    ind[1+n] = 1+c
                    val[1+n] = -1
                for n, (c, v) in enumerate(col_vals):
                    ind[1+n+len(cols)] = 1+c
                    val[1+n+len(cols)] = 1./v
                glp_set_mat_row(self.prob, self.idx_row_cross_domain+cross_domain_row, len(col_vals)+len(cols), ind, val)
                glp_set_row_bnds(self.prob, self.idx_row_cross_domain+cross_domain_row, GLP_FX, 0.0, 0.0)
                free(ind)
                free(val)
                cross_domain_row += 1

        # storage
        if len(storages):
            self.idx_row_storages = glp_add_rows(self.prob, len(storages))
        for col, storage in enumerate(storages):
            cols_output = [n for n, route in enumerate(routes) if route[-1] in storage.outputs and route[0] not in storage.inputs]
            cols_input = [n for n, route in enumerate(routes) if route[0] in storage.inputs and route[-1] not in storage.outputs]
            ind = <int*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(int))
            val = <double*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(double))
            for n, c in enumerate(cols_output):
                ind[1+n] = self.idx_col_routes+c
                val[1+n] = 1
            for n, c in enumerate(cols_input):
                ind[1+len(cols_output)+n] = self.idx_col_routes+c
                val[1+len(cols_output)+n] = -1
            glp_set_mat_row(self.prob, self.idx_row_storages+col, len(cols_output)+len(cols_input), ind, val)
            free(ind)
            free(val)

        # aggregated node flow ratio constraints
        self.idx_row_aggregated = self.idx_row_storages + len(storages)
        for agg_node in aggregated:
            nodes = agg_node.nodes
            factors = agg_node.factors
            assert(len(nodes) == len(factors))

            row = glp_add_rows(self.prob, len(agg_node.nodes)-1)

            cols = []
            for node in nodes:
                cols.append([n for n, route in enumerate(routes) if node in route])

            # normalise factors
            f0 = factors[0]
            factors_norm = [f0/f for f in factors]

            # update matrix
            for n in range(len(nodes)-1):
                length = len(cols[0])+len(cols[n+1])
                ind = <int*>malloc(1+length * sizeof(int))
                val = <double*>malloc(1+length * sizeof(double))
                for i, c in enumerate(cols[0]):
                    ind[1+i] = 1+c
                    val[1+i] = 1.0
                for i, c in enumerate(cols[n+1]):
                    ind[1+len(cols[0])+i] = 1+c
                    val[1+len(cols[0])+i] = -factors_norm[n+1]
                glp_set_mat_row(self.prob, row+n, length, ind, val)
                free(ind)
                free(val)

                glp_set_row_bnds(self.prob, row+n, GLP_FX, 0.0, 0.0)

        # aggregated node min/max flow constraints
        if aggregated_min_max:
            self.idx_row_aggregated_min_max = glp_add_rows(self.prob, len(aggregated_min_max))
        for row, agg_node in enumerate(aggregated_min_max):
            row = self.idx_row_aggregated_min_max + row
            nodes = agg_node.nodes
            min_flow = agg_node.min_flow
            max_flow = agg_node.max_flow
            if min_flow is None:
                min_flow = -inf
            if max_flow is None:
                max_flow = inf
            min_flow = inf_to_dbl_max(min_flow)
            max_flow = inf_to_dbl_max(max_flow)
            matrix = set()
            for node in nodes:
                for n, route in enumerate(routes):
                    if node in route:
                        matrix.add(n)
            length = len(matrix)
            ind = <int*>malloc(1+length * sizeof(int))
            val = <double*>malloc(1+length * sizeof(double))
            for i, col in enumerate(matrix):
                ind[1+i] = 1+col
                val[1+i] = 1.0
            glp_set_mat_row(self.prob, row, length, ind, val)
            glp_set_row_bnds(self.prob, row, constraint_type(min_flow, max_flow), min_flow, max_flow)
            free(ind)
            free(val)

        # update route properties
        routes_cost = []
        for col, route in enumerate(routes):
            route_cost = []
            route_cost.append(route[0].__data.id)
            for node in route[1:-1]:
                if isinstance(node, BaseLink):
                    route_cost.append(node.__data.id)
            route_cost.append(route[-1].__data.id)
            routes_cost.append(route_cost)

        self.routes = routes
        self.non_storages = non_storages
        self.storages = storages
        self.routes_cost = routes_cost

        # reset stats
        self.stats = {
            'total': 0.0,
            'lp_solve': 0.0,
            'result_update': 0.0,
            'bounds_update_routes': 0.0,
            'bounds_update_nonstorage': 0.0,
            'bounds_update_storage': 0.0
        }

        self.stats['bounds_update_routes'] = 0.0
        self.stats['bounds_update_nonstorage'] = 0.0
        self.stats['bounds_update_storage'] = 0.0

    cpdef object solve(self, model):
        t0 = time.clock()
        cdef int[:] scenario_combination
        cdef int scenario_id
        cdef ScenarioIndex scenario_index
        for scenario_index in model.scenarios.combinations:
            self._solve_scenario(model, scenario_index)
        self.stats['total'] += time.clock() - t0

    @cython.boundscheck(False)
    cdef object _solve_scenario(self, model, ScenarioIndex scenario_index):
        cdef Node node
        cdef Storage storage
        cdef AbstractNode _node
        cdef AbstractNodeData data
        cdef double min_flow
        cdef double max_flow
        cdef double cost
        cdef double max_volume
        cdef double avail_volume
        cdef int col
        cdef int* ind
        cdef double* val
        cdef double lb
        cdef double ub
        cdef Timestep timestep
        cdef int status
        cdef cross_domain_col
        cdef list route
        cdef int node_id
        cdef double flow

        timestep = model.timestep
        routes = self.routes
        non_storages = self.non_storages
        storages = self.storages

        # update route cost

        t0 = time.clock()

        # update the cost of each node in the model
        cdef double[:] node_costs = self.node_costs_arr
        for _node in self.nodes_with_cost:
            data = _node.__data
            node_costs[data.id] = _node.get_cost(timestep, scenario_index)

        # calculate the total cost of each route
        for col, route in enumerate(routes):
            cost = 0.0
            for node_id in self.routes_cost[col]:
                cost += node_costs[node_id]
            glp_set_obj_coef(self.prob, self.idx_col_routes+col, cost)

        self.stats['bounds_update_routes'] += time.clock() - t0
        t0 = time.clock()

        # update non-storage properties
        for col, node in enumerate(non_storages):
            min_flow = inf_to_dbl_max(node.get_min_flow(timestep, scenario_index))
            max_flow = inf_to_dbl_max(node.get_max_flow(timestep, scenario_index))
            glp_set_row_bnds(self.prob, self.idx_row_non_storages+col, constraint_type(min_flow, max_flow), min_flow, max_flow)

        self.stats['bounds_update_nonstorage'] += time.clock() - t0
        t0 = time.clock()

        # update storage node constraint
        for col, storage in enumerate(storages):
            max_volume = storage.get_max_volume(timestep, scenario_index)
            avail_volume = max(storage._volume[scenario_index._global_id] - storage.get_min_volume(timestep, scenario_index), 0.0)
            # change in storage cannot be more than the current volume or
            # result in maximum volume being exceeded
            lb = -avail_volume/timestep.days
            ub = (max_volume-storage._volume[scenario_index._global_id])/timestep.days
            glp_set_row_bnds(self.prob, self.idx_row_storages+col, constraint_type(lb, ub), lb, ub)

        self.stats['bounds_update_storage'] += time.clock() - t0

        # attempt to solve the linear programme
        t0 = time.clock()
        glp_simplex(self.prob, &self.smcp)

        status = glp_get_status(self.prob)
        if status != GLP_OPT:
            raise RuntimeError(status_string[status])

        self.stats['lp_solve'] += time.clock() - t0
        t0 = time.clock()

        cdef double[:] route_flows = self.route_flows_arr
        for col in range(0, self.num_routes):
            route_flows[col] = glp_get_col_prim(self.prob, col+1)

        cdef double[:] change_in_storage = self.change_in_storage_arr
        for col in range(0, self.num_storages):
            change_in_storage[col] = glp_get_row_prim(self.prob, self.idx_row_storages+col)

        # collect the total flow via each node
        cdef double[:] node_flows = self.node_flows_arr
        node_flows[:] = 0.0
        for n in range(0, self.num_routes):
            route = routes[n]
            flow = route_flows[n]
            # first and last node
            _node = route[0]
            data = _node.__data
            node_flows[data.id] += flow
            _node = route[-1]
            data = _node.__data
            node_flows[data.id] += flow
            # intermediate nodes
            for _node in route[1:-1]:
                data = _node.__data
                if data.is_link:
                    node_flows[data.id] += flow

        # commit the total flows
        for n in range(0, self.num_nodes):
            self.all_nodes[n].commit(scenario_index._global_id, node_flows[n])

        self.stats['result_update'] += time.clock() - t0

        return route_flows, change_in_storage
