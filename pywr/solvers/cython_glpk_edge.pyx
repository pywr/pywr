from libc.stdlib cimport malloc, free
from libc.math cimport abs
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np

cimport cython

from .glpk cimport GLPKSolver, BasisManager, simplex, set_obj_coef, set_row_bnds, set_col_bnds, set_mat_row

from pywr._core import BaseInput, BaseOutput, BaseLink
from pywr._core cimport *
from pywr.core import ModelStructureError
import time

include "glpk.pxi"

cdef class AbstractNodeData:
    cdef public int id
    cdef public bint is_link
    cdef public list in_edges, out_edges

cdef class CythonGLPKEdgeSolver(GLPKSolver):
    cdef int idx_col_edges
    cdef int idx_row_non_storages
    cdef int idx_row_link_mass_bal
    cdef int idx_row_cross_domain
    cdef int idx_row_storages
    cdef int idx_row_virtual_storages
    cdef int idx_row_aggregated
    cdef int idx_row_aggregated_min_max

    cdef list non_storages
    cdef list storages
    cdef list virtual_storages
    cdef list aggregated

    cdef list all_nodes
    cdef list all_edges

    cdef int num_nodes
    cdef int num_edges
    cdef int num_storages
    cdef int num_scenarios
    cdef cvarray edge_cost_arr
    cdef cvarray edge_flows_arr
    cdef cvarray node_flows_arr
    cdef public cvarray route_flows_arr
    cdef public object stats

    # Internal representation of the basis for each scenario
    cdef BasisManager basis_manager
    cdef bint is_first_solve
    cdef public bint save_routes_flows
    cdef public bint retry_solve

    def __init__(self, use_presolve=False, time_limit=None, iteration_limit=None, message_level='error',
                 save_routes_flows=False, retry_solve=False):
        super().__init__(use_presolve, time_limit, iteration_limit, message_level)
        self.stats = None
        self.is_first_solve = True
        self.save_routes_flows = save_routes_flows
        self.retry_solve = retry_solve
        self.basis_manager = BasisManager()

    def setup(self, model):
        super().setup(model)

        cdef Node input
        cdef Node output
        cdef AbstractNode some_node
        cdef AbstractNode _node
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
        cdef int n, num

        self.all_nodes = list(sorted(model.graph.nodes(), key=lambda n: n.fully_qualified_name))
        self.all_edges = edges = list(model.graph.edges())
        if not self.all_nodes or not self.all_edges:
            raise ModelStructureError("Model is empty")

        for n, _node in enumerate(self.all_nodes):
            _node.__data = AbstractNodeData()
            _node.__data.id = n
            _node.__data.in_edges = []
            _node.__data.out_edges = []
            if isinstance(_node, BaseLink):
                _node.__data.is_link = True

        self.num_nodes = len(self.all_nodes)

        self.edge_cost_arr = cvarray(shape=(len(self.all_edges),), itemsize=sizeof(double), format="d")
        self.edge_flows_arr = cvarray(shape=(len(self.all_edges),), itemsize=sizeof(double), format="d")
        self.node_flows_arr = cvarray(shape=(self.num_nodes,), itemsize=sizeof(double), format="d")

        # Find cross-domain routes
        cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')

        link_nodes = []
        non_storages = []
        storages = []
        virtual_storages = []
        aggregated_with_factors = []
        aggregated = []

        for some_node in self.all_nodes:
            if isinstance(some_node, (BaseInput, BaseLink, BaseOutput)):
                non_storages.append(some_node)
                if isinstance(some_node, BaseLink):
                    link_nodes.append(some_node)
            elif isinstance(some_node, VirtualStorage):
                virtual_storages.append(some_node)
            elif isinstance(some_node, Storage):
                storages.append(some_node)
            elif isinstance(some_node, AggregatedNode):
                if some_node.factors is not None:
                    aggregated_with_factors.append(some_node)
                aggregated.append(some_node)

        if len(non_storages) == 0:
            raise ModelStructureError("Model has no non-storage nodes")

        self.num_edges = len(edges)
        self.num_scenarios = len(model.scenarios.combinations)
        self.num_storages = len(storages)

        # clear the previous problem
        glp_erase_prob(self.prob)
        glp_set_obj_dir(self.prob, GLP_MIN)
        # add a column for each edge
        self.idx_col_edges = glp_add_cols(self.prob, self.num_edges)

        # create a lookup for edges associated with each node (ignoring cross domain edges)
        for row, (start_node, end_node) in enumerate(self.all_edges):
            if start_node.domain != end_node.domain:
                continue
            start_node.__data.out_edges.append(row)
            end_node.__data.in_edges.append(row)

        # create a lookup for the cross-domain routes.
        cross_domain_cols = {}
        for cross_domain_route in cross_domain_routes:
            # These routes are only 2 nodes. From output to input
            output, input = cross_domain_route
            # note that the conversion factor is not time varying
            conv_factor = input.get_conversion_factor()
            input_cols = [(n, conv_factor) for n in input.__data.out_edges]
            # create easy lookup for the route columns this output might
            # provide cross-domain connection to
            if output in cross_domain_cols:
                cross_domain_cols[output].extend(input_cols)
            else:
                cross_domain_cols[output] = input_cols

        # explicitly set bounds on route and demand columns
        for row, edge in enumerate(edges):
            set_col_bnds(self.prob, self.idx_col_edges+row, GLP_LO, 0.0, DBL_MAX)

        # Apply nodal flow constraints
        self.idx_row_non_storages = glp_add_rows(self.prob, len(non_storages))
        # # Add rows for the cross-domain routes.
        if len(cross_domain_cols) > 0:
            self.idx_row_cross_domain = glp_add_rows(self.prob, len(cross_domain_cols))

        cross_domain_row = 0
        for row, some_node in enumerate(non_storages):
            # Differentiate betwen the node type.
            # Input and other nodes use the outgoing edge flows to apply the flow constraint on
            # This requires the mass balance constraints to ensure the inflow and outflow are equal
            # The Output nodes, in contrast, apply the constraint to the incoming flow (because there is no out going flow)
            if isinstance(some_node, BaseInput):
                cols = some_node.__data.out_edges
                if len(some_node.__data.in_edges) != 0:
                    raise ModelStructureError(f'Input node "{some_node.name}" should not have any upstream '
                                              f'connections.')
            elif isinstance(some_node, BaseOutput):
                cols = some_node.__data.in_edges
                if len(some_node.__data.out_edges) != 0:
                    raise ModelStructureError(f'Output node "{some_node.name}" should not have any downstream '
                                              f'connections.')
            else:
                # Other nodes apply their flow constraints to all routes passing through them
                cols = some_node.__data.out_edges

            ind = <int*>malloc((1+len(cols)) * sizeof(int))
            val = <double*>malloc((1+len(cols)) * sizeof(double))
            for n, c in enumerate(cols):
                ind[1+n] = 1+c
                val[1+n] = 1
            set_mat_row(self.prob, self.idx_row_non_storages+row, len(cols), ind, val)
            set_row_bnds(self.prob, self.idx_row_non_storages+row, GLP_FX, 0.0, 0.0)

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
                set_mat_row(self.prob, self.idx_row_cross_domain+cross_domain_row, len(col_vals)+len(cols), ind, val)
                set_row_bnds(self.prob, self.idx_row_cross_domain+cross_domain_row, GLP_FX, 0.0, 0.0)
                # glp_set_row_name(self.prob, self.idx_row_cross_domain+cross_domain_row,
                #                  b'cd.'+some_node.fully_qualified_name.encode('utf-8'))
                free(ind)
                free(val)
                cross_domain_row += 1

        # Add mass balance constraints
        if len(link_nodes) > 0:
            self.idx_row_link_mass_bal = glp_add_rows(self.prob, len(link_nodes))
        for row, some_node in enumerate(link_nodes):

            in_cols = some_node.__data.in_edges
            out_cols = some_node.__data.out_edges
            ind = <int*>malloc((1+len(in_cols)+len(out_cols)) * sizeof(int))
            val = <double*>malloc((1+len(in_cols)+len(out_cols)) * sizeof(double))
            for n, c in enumerate(in_cols):
                ind[1+n] = 1+c
                val[1+n] = 1
            for n, c in enumerate(out_cols):
                ind[1+len(in_cols)+n] = 1+c
                val[1+len(in_cols)+n] = -1
            set_mat_row(self.prob, self.idx_row_link_mass_bal+row, len(in_cols)+len(out_cols), ind, val)
            set_row_bnds(self.prob, self.idx_row_link_mass_bal+row, GLP_FX, 0.0, 0.0)

            free(ind)
            free(val)

        # storage
        if len(storages):
            self.idx_row_storages = glp_add_rows(self.prob, len(storages))
        for row, storage in enumerate(storages):

            cols_output = []
            for output in storage.outputs:
                cols_output.extend(output.__data.in_edges)
            cols_input = []
            for input in storage.inputs:
                cols_input.extend(input.__data.out_edges)

            ind = <int*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(int))
            val = <double*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(double))
            for n, c in enumerate(cols_output):
                ind[1+n] = self.idx_col_edges+c
                val[1+n] = 1
            for n, c in enumerate(cols_input):
                ind[1+len(cols_output)+n] = self.idx_col_edges+c
                val[1+len(cols_output)+n] = -1

            set_mat_row(self.prob, self.idx_row_storages+row, len(cols_output)+len(cols_input), ind, val)
            # glp_set_row_name(self.prob, self.idx_row_storages+row,
            #                  b's.'+storage.fully_qualified_name.encode('utf-8'))
            free(ind)
            free(val)

        # virtual storage
        if len(virtual_storages):
            self.idx_row_virtual_storages = glp_add_rows(self.prob, len(virtual_storages))
        for row, storage in enumerate(virtual_storages):
            # We need to handle the same route appearing twice here.
            cols = {}

            for i, some_node in enumerate(storage.nodes):
                if isinstance(some_node, BaseOutput):
                    node_cols = some_node.__data.in_edges
                else:
                    node_cols = some_node.__data.out_edges

                for n in node_cols:
                    try:
                        cols[n] += storage.factors[i]
                    except KeyError:
                        cols[n] = storage.factors[i]

            ind = <int*>malloc((1+len(cols)) * sizeof(int))
            val = <double*>malloc((1+len(cols)) * sizeof(double))
            for n, (c, f) in enumerate(cols.items()):
                ind[1+n] = self.idx_col_edges+c
                val[1+n] = -f

            set_mat_row(self.prob, self.idx_row_virtual_storages+row, len(cols), ind, val)
            # glp_set_row_name(self.prob, self.idx_row_virtual_storages+row,
            #                  b'vs.'+storage.fully_qualified_name.encode('utf-8'))
            free(ind)
            free(val)

        # aggregated node flow ratio constraints
        if len(aggregated_with_factors):
            self.idx_row_aggregated = self.idx_row_virtual_storages + len(virtual_storages)
        for agg_node in aggregated_with_factors:
            nodes = agg_node.nodes
            factors = agg_node.factors
            assert(len(nodes) == len(factors))

            row = glp_add_rows(self.prob, len(agg_node.nodes)-1)

            cols = []
            for node in nodes:
                if isinstance(node, BaseOutput):
                    cols.append([c for c in node.__data.in_edges])
                else:
                    cols.append([c for c in node.__data.out_edges])

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
                set_mat_row(self.prob, row+n, length, ind, val)
                free(ind)
                free(val)

                set_row_bnds(self.prob, row+n, GLP_FX, 0.0, 0.0)
                # glp_set_row_name(self.prob, row+n,
                #                  'ag.f{}.{}'.format(n, agg_node.fully_qualified_name).encode('utf-8'))

        # aggregated node min/max flow constraints
        if aggregated:
            self.idx_row_aggregated_min_max = glp_add_rows(self.prob, len(aggregated))
        for row, agg_node in enumerate(aggregated):
            row = self.idx_row_aggregated_min_max + row
            nodes = agg_node.nodes

            weights = agg_node.flow_weights
            if weights is None:
                weights = [1.0]*len(nodes)

            matrix = {}
            for some_node, w in zip(nodes, weights):
                if isinstance(some_node, BaseOutput):
                    node_cols = some_node.__data.in_edges
                else:
                    node_cols = some_node.__data.out_edges

                for n in node_cols:
                    matrix[n] = w

            length = len(matrix)
            ind = <int*>malloc(1+length * sizeof(int))
            val = <double*>malloc(1+length * sizeof(double))
            for i, col in enumerate(sorted(matrix)):
                ind[1+i] = 1+col
                val[1+i] = matrix[col]
            set_mat_row(self.prob, row, length, ind, val)
            set_row_bnds(self.prob, row, GLP_FX, 0.0, 0.0)
            # glp_set_row_name(self.prob, row, b'ag.'+agg_node.fully_qualified_name.encode('utf-8'))
            free(ind)
            free(val)

        self.non_storages = non_storages
        self.storages = storages
        self.virtual_storages = virtual_storages
        self.aggregated = aggregated

        self.basis_manager.init_basis(self.prob, len(model.scenarios.combinations))
        self.is_first_solve = True

        # reset stats
        self.stats = {
            'total': 0.0,
            'lp_solve': 0.0,
            'result_update': 0.0,
            'bounds_update_nonstorage': 0.0,
            'bounds_update_storage': 0.0,
            'objective_update': 0.0,
            'number_of_rows': glp_get_num_rows(self.prob),
            'number_of_cols': glp_get_num_cols(self.prob),
            'number_of_nonzero': glp_get_num_nz(self.prob),
            'number_of_edges': len(self.all_edges),
            'number_of_nodes': len(self.all_nodes)
        }

    def reset(self):
        # Resetting this triggers a crashing of a new basis in each scenario
        self.is_first_solve = True

    cpdef object solve(self, model):
        GLPKSolver.solve(self, model)
        t0 = time.perf_counter()
        cdef int[:] scenario_combination
        cdef int scenario_id
        cdef ScenarioIndex scenario_index
        for scenario_index in model.scenarios.combinations:
            self._solve_scenario(model, scenario_index)
        self.stats['total'] += time.perf_counter() - t0
        # After solving this is always false
        self.is_first_solve = False

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef object _solve_scenario(self, model, ScenarioIndex scenario_index):
        cdef Node node
        cdef Storage storage
        cdef AbstractNode _node
        cdef AbstractNodeData data
        cdef AggregatedNode agg_node
        cdef double min_flow
        cdef double max_flow
        cdef double cost
        cdef double max_volume
        cdef double min_volume
        cdef double avail_volume
        cdef double t0
        cdef int col, row
        cdef int* ind
        cdef double* val
        cdef double lb
        cdef double ub
        cdef Timestep timestep
        cdef int status, simplex_ret
        cdef cross_domain_col
        cdef list route
        cdef int node_id, indptr, nedges
        cdef double flow
        cdef int n, m
        cdef Py_ssize_t length

        timestep = model.timestep
        cdef list edges = self.all_edges
        nedges = self.num_edges
        cdef list non_storages = self.non_storages
        cdef list storages = self.storages
        cdef list virtual_storages = self.virtual_storages
        cdef list aggregated = self.aggregated

        # update route cost

        t0 = time.perf_counter()

        # Initialise the cost on each edge to zero
        cdef double[:] edge_costs = self.edge_cost_arr
        for col in range(nedges):
            edge_costs[col] = 0.0

        # update the cost of each node in the model
        for _node in self.all_nodes:
            cost = _node.get_cost(scenario_index)
            data = _node.__data

            # Link nodes have edges connected upstream & downstream. We apply
            # half the cost assigned to the node to all the connected edges.
            # The edge costs are then the mean of the node costs at either end.
            if data.is_link:
                cost /= 2

            for col in data.in_edges:
                edge_costs[col] += cost
            for col in data.out_edges:
                edge_costs[col] += cost

        # calculate the total cost of each route
        for col in range(nedges):
            set_obj_coef(self.prob, self.idx_col_edges+col, edge_costs[col])

        self.stats['objective_update'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        # update non-storage properties
        for row, node in enumerate(non_storages):
            min_flow = inf_to_dbl_max(node.get_min_flow(scenario_index))
            if abs(min_flow) < 1e-8:
                min_flow = 0.0
            max_flow = inf_to_dbl_max(node.get_max_flow(scenario_index))
            if abs(max_flow) < 1e-8:
                max_flow = 0.0

            set_row_bnds(self.prob, self.idx_row_non_storages+row, constraint_type(min_flow, max_flow),
                         min_flow, max_flow)

        for row, agg_node in enumerate(aggregated):
            min_flow = inf_to_dbl_max(agg_node.get_min_flow(scenario_index))
            if abs(min_flow) < 1e-8:
                min_flow = 0.0
            max_flow = inf_to_dbl_max(agg_node.get_max_flow(scenario_index))
            if abs(max_flow) < 1e-8:
                max_flow = 0.0
            set_row_bnds(self.prob, self.idx_row_aggregated_min_max + row, constraint_type(min_flow, max_flow),
                         min_flow, max_flow)

        self.stats['bounds_update_nonstorage'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        # update storage node constraint
        for row, storage in enumerate(storages):
            max_volume = storage.get_max_volume(scenario_index)
            min_volume = storage.get_min_volume(scenario_index)

            if max_volume == min_volume:
                set_row_bnds(self.prob, self.idx_row_storages+row, GLP_FX, 0.0, 0.0)
            else:
                avail_volume = max(storage._volume[scenario_index.global_id] - min_volume, 0.0)
                # change in storage cannot be more than the current volume or
                # result in maximum volume being exceeded
                lb = -avail_volume/timestep.days
                ub = max(max_volume - storage._volume[scenario_index.global_id], 0.0) / timestep.days

                if abs(lb) < 1e-8:
                    lb = 0.0
                if abs(ub) < 1e-8:
                    ub = 0.0
                set_row_bnds(self.prob, self.idx_row_storages+row, constraint_type(lb, ub), lb, ub)

        # update virtual storage node constraint
        for row, storage in enumerate(virtual_storages):
            max_volume = storage.get_max_volume(scenario_index)
            min_volume = storage.get_min_volume(scenario_index)

            if max_volume == min_volume:
                set_row_bnds(self.prob, self.idx_row_virtual_storages+row, GLP_FX, 0.0, 0.0)
            else:
                avail_volume = max(storage._volume[scenario_index.global_id] - min_volume, 0.0)
                # change in storage cannot be more than the current volume or
                # result in maximum volume being exceeded
                lb = -avail_volume/timestep.days
                ub = max(max_volume - storage._volume[scenario_index.global_id], 0.0) / timestep.days

                if abs(lb) < 1e-8:
                    lb = 0.0
                if abs(ub) < 1e-8:
                    ub = 0.0
                set_row_bnds(self.prob, self.idx_row_virtual_storages+row, constraint_type(lb, ub), lb, ub)

        self.stats['bounds_update_storage'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        # Set the basis for this scenario
        self.basis_manager.set_basis(self.prob, self.is_first_solve, scenario_index.global_id)
        # attempt to solve the linear programme
        simplex_ret = simplex(self.prob, self.smcp)
        status = glp_get_status(self.prob)
        if (status != GLP_OPT or simplex_ret != 0) and self.retry_solve:
            # try creating a new basis and resolving
            print('Retrying solve with new basis.')
            glp_std_basis(self.prob)
            simplex_ret = simplex(self.prob, self.smcp)
            status = glp_get_status(self.prob)

        if status != GLP_OPT or simplex_ret != 0:
            # If problem is not solved. Print some debugging information and error.
            print("Simplex solve returned: {} ({})".format(simplex_status_string[simplex_ret], simplex_ret))
            print("Simplex status: {} ({})".format(status_string[status], status))
            print("Scenario ID: {}".format(scenario_index.global_id))
            print("Timestep index: {}".format(timestep.index))
            self.dump_mps(b'pywr_glpk_debug.mps')
            self.dump_lp(b'pywr_glpk_debug.lp')

            self.smcp.msg_lev = GLP_MSG_DBG
            # Retry solve with debug messages
            simplex_ret = simplex(self.prob, self.smcp)
            status = glp_get_status(self.prob)
            raise RuntimeError('Simplex solver failed with message: "{}", status: "{}".'.format(
                simplex_status_string[simplex_ret], status_string[status]))
        # Now save the basis
        self.basis_manager.save_basis(self.prob, scenario_index.global_id)

        self.stats['lp_solve'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        cdef double[:] edge_flows = self.edge_flows_arr

        for col in range(self.num_edges):
            edge_flows[col] = glp_get_col_prim(self.prob, col+1)

        # collect the total flow via each node
        cdef double[:] node_flows = self.node_flows_arr
        node_flows[:] = 0.0
        for n, edge in enumerate(edges):
            flow = edge_flows[n]
            if flow == 0:
                continue

            for _node in edge:
                data = _node.__data
                if data.is_link:
                    # Link nodes are connected upstream & downstream so
                    # we take half of flow from each edge.
                    node_flows[data.id] += flow / 2
                else:
                    node_flows[data.id] += flow

        # commit the total flows
        for n in range(0, self.num_nodes):
            _node = self.all_nodes[n]
            _node.commit(scenario_index.global_id, node_flows[n])

        self.stats['result_update'] += time.perf_counter() - t0
