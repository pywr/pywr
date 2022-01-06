from cpython cimport array
import array
from libc.stdlib cimport malloc, free
from libc.math cimport abs, isnan
from libc.setjmp cimport setjmp, longjmp, jmp_buf
from libc.float cimport DBL_MAX
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
cimport cython

from pywr._core import BaseInput, BaseOutput, BaseLink
from pywr._core cimport *
from pywr.core import ModelStructureError
import time

from .libglpk cimport *
import logging
logger = logging.getLogger(__name__)

# Constants and message strings
# =============================
status_string = [
    None,
    'solution is undefined',
    'solution is feasible',
    'solution is infeasible',
    'no feasible solution exists',
    'solution is optimal',
    'solution is unbounded',
]

simplex_status_string = [
    None,
    'invalid basis',
    'singular matrix',
    'ill-conditioned matrix',
    'invalid bounds',
    'solver failed',
    'objective lower limit reached',
    'objective upper limit reached',
    'iteration limit exceeded',
    'time limit exceeded',
    'no primal feasible solution',
    'no dual feasible solution',
    'root LP optimum not provided',
    'search terminated by application',
    'relative mip gap tolerance reached',
    'no primal/dual feasible solution',
    'no convergence',
    'numerical instability',
    'invalid data',
    'result out of range',
]

message_levels = {
    'off': GLP_MSG_OFF,
    'error': GLP_MSG_ERR,
    'normal': GLP_MSG_ON,
    'all': GLP_MSG_ALL,
    'debug': GLP_MSG_DBG,
}

# Inline helper functions
# =======================
cdef inline int constraint_type(double a, double b):
    if abs(a - b) < 1e-8:
        return GLP_FX
    elif b == DBL_MAX:
        if a == -DBL_MAX:
            return GLP_FR
        else:
            return GLP_LO
    elif a == -DBL_MAX:
        return GLP_UP
    else:
        return GLP_DB


cdef double inf = float('inf')


cdef inline double dbl_max_to_inf(double a):
    if a == DBL_MAX:
        return inf
    elif a == -DBL_MAX:
        return -inf
    return a

cdef inline double inf_to_dbl_max(double a):
    if a == inf:
        return DBL_MAX
    elif a == -inf:
        return -DBL_MAX
    return a

# Error handling
# ==============
class GLPKError(Exception):
    pass
class GLPKInternalError(Exception):
    pass


cdef jmp_buf error_ctx
cdef bint has_glpk_errored = 0

cdef void error_hook(void *info):
    # Free GLPK memory; this will destroy the entire GLPK environment for this process
    # Potentially invalidating other models.
    global has_glpk_errored
    has_glpk_errored = True;
    glp_free_env();
    longjmp((<jmp_buf*>info)[0], <int>1)


cdef class AbstractNodeData:
    """Helper class for caching node data for the solver."""
    cdef public int id
    cdef public bint is_link
    cdef public list in_edges, out_edges
    cdef public int row


cdef class AggNodeFactorData:
    """Helper class for caching data for Aggregated Nodes that have factors defined by parameters."""
    cdef public int row
    cdef public int node_ind
    cdef public cvarray ind_ptr
    cdef public cvarray inds
    cdef public cvarray vals


cdef class GLPKSolver:
    cdef glp_prob* prob
    cdef glp_smcp smcp

    cdef public bint use_presolve
    cdef bint has_presolved
    cdef public bint use_unsafe_api
    cdef public bint set_fixed_flows_once
    cdef public bint set_fixed_costs_once
    cdef public bint set_fixed_factors_once

    def __cinit__(self):
        self.prob = glp_create_prob()

    def __dealloc__(self):
        # If there's been an error the GLPK environment is destroyed and the pointer is invalid.
        if not has_glpk_errored:
            glp_delete_prob(self.prob)

    def __init__(self, use_presolve=False, time_limit=None, iteration_limit=None, message_level="error",
                 set_fixed_flows_once=False, set_fixed_costs_once=False, set_fixed_factors_once=False, use_unsafe_api=False):
        self.use_presolve = use_presolve
        self.use_unsafe_api = use_unsafe_api
        self.set_solver_options(time_limit, iteration_limit, message_level)
        glp_term_hook(term_hook, NULL)
        self.has_presolved = False
        self.set_fixed_flows_once = set_fixed_flows_once
        self.set_fixed_costs_once = set_fixed_costs_once
        self.set_fixed_factors_once = set_fixed_factors_once
        # Register the error hook
        glp_error_hook(error_hook, &error_ctx)

    def set_solver_options(self, time_limit, iteration_limit, message_level):
        glp_init_smcp(&self.smcp)
        self.smcp.msg_lev = message_levels[message_level]
        if time_limit is not None:
            self.smcp.tm_lim = time_limit
        if iteration_limit is not None:
            self.smcp.it_lim = iteration_limit

    def setup(self, model):
        self.has_presolved = False

    def reset(self):
        pass

    cpdef object solve(self, model):
        if self.use_presolve and not self.has_presolved:
            self.smcp.presolve = GLP_ON
            self.has_presolved = True
        else:
            self.smcp.presolve = GLP_OFF

    cpdef dump_mps(self, filename):
        glp_write_mps(self.prob, GLP_MPS_FILE, NULL, filename)

    cpdef dump_lp(self, filename):
        glp_write_lp(self.prob, NULL, filename)

    cpdef dump_glpk(self, filename):
        glp_write_prob(self.prob, 0, filename)


cdef class BasisManager:
    cdef int[:, :] row_stat
    cdef int[:, :] col_stat

    cdef init_basis(self, glp_prob* prob, int nscenarios):
        """Initialise the arrays used for storing the LP basis by scenario"""
        cdef int nrows = glp_get_num_rows(prob)
        cdef int ncols = glp_get_num_cols(prob)
        self.row_stat = np.empty((nscenarios, nrows), dtype=np.int32)
        self.col_stat = np.empty((nscenarios, ncols), dtype=np.int32)

    cdef save_basis(self, glp_prob* prob, int global_id):
        """Save the current basis for scenario associated with global_id"""
        cdef int i
        cdef int nrows = glp_get_num_rows(prob)
        cdef int ncols = glp_get_num_cols(prob)
        for i in range(nrows):
            self.row_stat[global_id, i] = glp_get_row_stat(prob, i + 1)
        for i in range(ncols):
            self.col_stat[global_id, i] = glp_get_col_stat(prob, i + 1)

    cdef set_basis(self, glp_prob* prob, bint is_first_solve, int global_id):
        """Set the current basis for scenario associated with global_id"""
        cdef int i, nrows, ncols

        if is_first_solve:
            # First time solving we use the default advanced basis
            glp_adv_basis(prob, 0)
        else:
            # Otherwise we restore basis from previous solve of this scenario
            nrows = glp_get_num_rows(prob)
            ncols = glp_get_num_cols(prob)
            for i in range(nrows):
                glp_set_row_stat(prob, i + 1, self.row_stat[global_id, i])
            for i in range(ncols):
                glp_set_col_stat(prob, i + 1, self.col_stat[global_id, i])


cdef int term_hook(void *info, const char *s):
    """ Callback function to print GLPK messages through Python's print function """
    # TODO make this use logging.
    message = s.strip().decode('UTF-8')
    if message.startswith("Constructing initial basis"):
        pass
    elif message.startswith("Size of triangular part is"):
        pass
    else:
        print(message)
    return 1



# Unsafe interface to GLPK
# This interface performs no checks or error handling
cdef int simplex_unsafe(glp_prob *P, glp_smcp parm):
    """Unsafe wrapped call to `glp_simplex`"""
    return glp_simplex(P, &parm)

cdef set_obj_coef_unsafe(glp_prob *P, int j, double coef):
    """Unsafe wrapped call to `glp_set_obj_coef`"""
    glp_set_obj_coef(P, j, coef)

cdef set_row_bnds_unsafe(glp_prob *P, int i, int type, double lb, double ub):
    """Wrapped call to `glp_set_row_bnds`"""
    glp_set_row_bnds(P, i, type, lb, ub)

cdef set_col_bnds_unsafe(glp_prob *P, int i, int type, double lb, double ub):
    """Wrapped call to `glp_set_col_bnds`"""
    glp_set_col_bnds(P, i, type, lb, ub)

cdef set_mat_row_unsafe(glp_prob *P, int i, int len, int * ind, double * val):
    """Wrapped call to `glp_set_mat_row`"""
    glp_set_mat_row(P, i, len, ind, val)

# Safe interface to GLPK
# This interface checks for NaNs and uses setjmp to catch GLPK errors

glpk_error_msg = "This is an internal GLPK error and will have invalidated the entire GLPK environment. " \
                 "All models will need to be recreated to continue. " \
                 "This error should be resolved and the model re-run."

cdef int simplex_safe(glp_prob *P, glp_smcp parm) except? -1:
    """Wrapped call to `glp_simplex`"""
    if not setjmp(error_ctx):
        return glp_simplex(P, &parm)
    else:
        raise GLPKInternalError("An error occurred in `glp_simplex`." + glpk_error_msg)


cdef int set_obj_coef_safe(glp_prob *P, int j, double coef) except -1:
    """Wrapped call to `glp_set_obj_coef`"""
    if isnan(coef):
        raise GLPKError(f"NaN detected in objective coefficient of column {j}")

    if not setjmp(error_ctx):
        glp_set_obj_coef(P, j, coef)
    else:
        raise GLPKInternalError("An error occurred in `glp_set_obj_coef`." + glpk_error_msg)


cdef int set_row_bnds_safe(glp_prob *P, int i, int type, double lb, double ub) except -1:
    """Wrapped call to `glp_set_row_bnds`"""
    if isnan(lb):
        raise GLPKError(f"NaN detected in lower bounds of row {i}")
    if isnan(ub):
        raise GLPKError(f"NaN detected in upper bounds of row {i}")
    if not setjmp(error_ctx):
        glp_set_row_bnds(P, i, type, lb, ub)
    else:
        raise GLPKInternalError("An error occurred in `glp_set_row_bnds`." + glpk_error_msg)


cdef int set_col_bnds_safe(glp_prob *P, int i, int type, double lb, double ub) except -1:
    """Wrapped call to `glp_set_col_bnds`"""
    if isnan(lb):
        raise GLPKError(f"NaN detected in lower bounds of column {i}")
    if isnan(ub):
        raise GLPKError(f"NaN detected in upper bounds of column {i}")
    if not setjmp(error_ctx):
        glp_set_col_bnds(P, i, type, lb, ub)
    else:
        raise GLPKInternalError("An error occurred in `glp_set_col_bnds`." + glpk_error_msg)


cdef int set_mat_row_safe(glp_prob *P, int i, int len, int* ind, double* val) except -1:
    """Wrapped call to `glp_set_mat_row`"""
    if len == 0:
        raise GLPKError("Attempting to set a constraint row with zero entries. This should not happen. It is likely caused "
                        "by invalid or unsupported network configuration, but should generally be caught earlier by Pywr. "
                        "If you experience this error please report it to the Pywr developers.")

    for j in range(len):
        if isnan(val[j+1]):
            raise GLPKError(f"NaN detected in matrix coefficient of row {i} and column {ind[j+1]}")

    if not setjmp(error_ctx):
        glp_set_mat_row(P, i, len, ind, val)
    else:
        raise GLPKInternalError(f"An error occurred in `glp_set_mat_row` for row {i}." + glpk_error_msg)


cdef class CythonGLPKSolver(GLPKSolver):
    cdef int idx_col_routes
    cdef int idx_row_non_storages
    cdef int idx_row_cross_domain
    cdef int idx_row_storages
    cdef int idx_row_virtual_storages
    cdef int idx_row_aggregated
    cdef int idx_row_aggregated_with_factors
    cdef int idx_row_aggregated_min_max

    cdef public list routes
    cdef list non_storages
    cdef list non_storages_with_dynamic_bounds
    cdef list non_storages_with_constant_bounds
    cdef list nodes_with_dynamic_cost
    cdef list storages
    cdef list virtual_storages
    cdef list aggregated
    cdef list aggregated_with_factors
    cdef list aggregated_with_dynamic_factors
    cdef list aggregated_with_constant_factors

    cdef int[:] routes_cost
    cdef int[:] routes_cost_indptr

    cdef list all_nodes
    cdef int num_nodes
    cdef int num_routes
    cdef int num_storages
    cdef int num_scenarios
    cdef cvarray node_costs_arr
    cdef cvarray node_flows_arr
    cdef public cvarray route_flows_arr
    cdef public object stats

    # Internal representation of the basis for each scenario
    cdef BasisManager basis_manager
    cdef bint is_first_solve
    cdef public bint save_routes_flows
    cdef public bint retry_solve

    def __init__(self, use_presolve=False, time_limit=None, iteration_limit=None, message_level='error',
                 save_routes_flows=False, retry_solve=False, **kwargs):
        super().__init__(use_presolve, time_limit, iteration_limit, message_level, **kwargs)
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
        if not self.all_nodes:
            raise ModelStructureError("Model is empty")

        for n, _node in enumerate(self.all_nodes):
            _node.__data = AbstractNodeData()
            _node.__data.id = n
            if isinstance(_node, BaseLink):
                _node.__data.is_link = True

        self.num_nodes = len(self.all_nodes)

        self.node_costs_arr = cvarray(shape=(self.num_nodes,), itemsize=sizeof(double), format="d")
        self.node_flows_arr = cvarray(shape=(self.num_nodes,), itemsize=sizeof(double), format="d")

        routes = model.find_all_routes(BaseInput, BaseOutput, valid=(BaseLink, BaseInput, BaseOutput))
        # Find cross-domain routes
        cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')

        nodes_with_dynamic_cost = []  # Only these nodes' costs are updated each timestep
        non_storages = []
        non_storages_with_dynamic_bounds = []  # These nodes' constraints are updated each timestep
        non_storages_with_constant_bounds = []  # These nodes' constraints are updated only at reset
        storages = []
        virtual_storages = []
        aggregated_with_factors = []
        aggregated_with_dynamic_factors = []
        aggregated_with_constant_factors = []
        aggregated = []

        for some_node in self.all_nodes:
            if isinstance(some_node, (BaseInput, BaseLink, BaseOutput)):
                non_storages.append(some_node)
            elif isinstance(some_node, VirtualStorage):
                virtual_storages.append(some_node)
            elif isinstance(some_node, Storage):
                storages.append(some_node)
            elif isinstance(some_node, AggregatedNode):
                if some_node.factors is not None:
                    aggregated_with_factors.append(some_node)
                    some_node.__agg_factor_data = AggNodeFactorData()
                aggregated.append(some_node)

        if len(routes) == 0:
            raise ModelStructureError("Model has no valid routes")
        if len(non_storages) == 0:
            raise ModelStructureError("Model has no non-storage nodes")

        self.num_routes = len(routes)
        self.num_scenarios = len(model.scenarios.combinations)

        if self.save_routes_flows:
            # If saving flows this array needs to be 2D (one for each scenario)
            self.route_flows_arr = cvarray(shape=(self.num_scenarios, self.num_routes),
                                           itemsize=sizeof(double), format="d")
        else:
            # Otherwise the array can just be used to store a single solve to save some memory
            self.route_flows_arr = cvarray(shape=(self.num_routes, ), itemsize=sizeof(double), format="d")

        # clear the previous problem
        glp_erase_prob(self.prob)
        glp_set_obj_dir(self.prob, GLP_MIN)
        # add a column for each route
        self.idx_col_routes = glp_add_cols(self.prob, <int>(len(routes)))

        # create a lookup for the cross-domain routes.
        cross_domain_cols = {}
        for cross_domain_route in cross_domain_routes:
            # These routes are only 2 nodes. From output to input
            output, input = cross_domain_route
            # note that the conversion factor is not time varying
            conv_factor = input.get_conversion_factor()
            input_cols = [(n, conv_factor) for n, route in enumerate(routes) if route[0] is input]
            # create easy lookup for the route columns this output might
            # provide cross-domain connection to
            if output in cross_domain_cols:
                cross_domain_cols[output].extend(input_cols)
            else:
                cross_domain_cols[output] = input_cols

        # explicitly set bounds on route and demand columns
        for col, route in enumerate(routes):
            set_col_bnds_safe(self.prob, self.idx_col_routes+col, GLP_LO, 0.0, DBL_MAX)

        # constrain supply minimum and maximum flow
        self.idx_row_non_storages = glp_add_rows(self.prob, len(non_storages))
        # Add rows for the cross-domain routes.
        if len(cross_domain_cols) > 0:
            self.idx_row_cross_domain = glp_add_rows(self.prob, len(cross_domain_cols))

        # Calculate the fixed costs
        for some_node in self.all_nodes:
            try:
                fixed_cost = some_node.has_fixed_cost
            except AttributeError:
                fixed_cost = False

            if self.set_fixed_costs_once and fixed_cost:
                # With a fixed cost this should work with no scenario index
                cost = some_node.get_cost(None)
                data = some_node.__data
                self.node_costs_arr[data.id] = cost
            else:
                nodes_with_dynamic_cost.append(some_node)

        cross_domain_row = 0
        for row, node in enumerate(non_storages):
            node.__data.row = row
            # Differentiate betwen the node type.
            # Input & Output only apply their flow constraints when they
            # are the first and last node on the route respectively.
            if isinstance(node, BaseInput):
                cols = [n for n, route in enumerate(routes) if route[0] is node]
            elif isinstance(node, BaseOutput):
                cols = [n for n, route in enumerate(routes) if route[-1] is node]
            else:
                # Other nodes apply their flow constraints to all routes passing through them
                cols = [n for n, route in enumerate(routes) if node in route]

            if len(cols) == 0:
                raise ModelStructureError(f'{node.__class__.__name__} node "{node.name}" is not part of any routes.')

            ind = <int*>malloc((1+len(cols)) * sizeof(int))
            val = <double*>malloc((1+len(cols)) * sizeof(double))
            for n, c in enumerate(cols):
                ind[1+n] = 1+c
                val[1+n] = 1
            try:
                # Always use the safe API in setup
                set_mat_row_safe(self.prob, self.idx_row_non_storages+row, len(cols), ind, val)
                set_row_bnds_safe(self.prob, self.idx_row_non_storages + row, GLP_FX, 0.0, 0.0)
            except GLPKError, GLPKInternalError:
                logger.error(f"A GLPK error occurred during creation of flow constraints for node: {node}")
                raise
            finally:
                free(ind)
                free(val)
            # Now test whether this node has fixed flow constraints
            if self.set_fixed_flows_once and node.has_constant_flows:
                non_storages_with_constant_bounds.append(node)
            else:
                non_storages_with_dynamic_bounds.append(node)

            # Add constraint for cross-domain routes
            # i.e. those from a demand to a supply
            if node in cross_domain_cols:
                col_vals = cross_domain_cols[node]
                ind = <int*>malloc((1+len(col_vals)+len(cols)) * sizeof(int))
                val = <double*>malloc((1+len(col_vals)+len(cols)) * sizeof(double))
                for n, c in enumerate(cols):
                    ind[1+n] = 1+c
                    val[1+n] = -1
                for n, (c, v) in enumerate(col_vals):
                    ind[1+n+len(cols)] = 1+c
                    val[1+n+len(cols)] = 1./v

                try:
                    # Always use the safe API in setup
                    set_mat_row_safe(self.prob, self.idx_row_cross_domain+cross_domain_row, len(col_vals)+len(cols), ind, val)
                    set_row_bnds_safe(self.prob, self.idx_row_cross_domain+cross_domain_row, GLP_FX, 0.0, 0.0)
                except GLPKError, GLPKInternalError:
                    logger.error(f"A GLPK error occurred during creation of cross-domain constraints for node: {node}")
                    raise
                finally:
                    free(ind)
                    free(val)
                cross_domain_row += 1

        # storage
        if len(storages):
            self.idx_row_storages = glp_add_rows(self.prob, len(storages))
        for row, storage in enumerate(storages):
            cols_output = [n for n, route in enumerate(routes)
                           if route[-1] in storage.outputs and route[0] not in storage.inputs]
            cols_input = [n for n, route in enumerate(routes)
                          if route[0] in storage.inputs and route[-1] not in storage.outputs]
            ind = <int*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(int))
            val = <double*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(double))
            for n, c in enumerate(cols_output):
                ind[1+n] = self.idx_col_routes+c
                val[1+n] = 1
            for n, c in enumerate(cols_input):
                ind[1+len(cols_output)+n] = self.idx_col_routes+c
                val[1+len(cols_output)+n] = -1
            try:
                # Always use the safe API in setup
                set_mat_row_safe(self.prob, self.idx_row_storages+row, len(cols_output)+len(cols_input), ind, val)
            except GLPKError, GLPKInternalError:
                logger.error(f"A GLPK error occurred during creation of storage constraints for node: {storage}")
                raise
            finally:
                free(ind)
                free(val)

        # virtual storage
        if len(virtual_storages):
            self.idx_row_virtual_storages = glp_add_rows(self.prob, len(virtual_storages))
        for row, storage in enumerate(virtual_storages):
            # We need to handle the same route appearing twice here.
            cols = {}
            for n, route in enumerate(routes):
                for some_node in route:
                    try:
                        i = storage.nodes.index(some_node)
                    except ValueError:
                        pass
                    else:
                        try:
                            cols[n] += storage.factors[i]
                        except KeyError:
                            cols[n] = storage.factors[i]

            ind = <int*>malloc((1+len(cols)) * sizeof(int))
            val = <double*>malloc((1+len(cols)) * sizeof(double))
            for n, (c, f) in enumerate(cols.items()):
                ind[1+n] = self.idx_col_routes+c
                val[1+n] = -f

            try:
                # Always use the safe API in setup
                set_mat_row_safe(self.prob, self.idx_row_virtual_storages+row, len(cols), ind, val)
            except GLPKError, GLPKInternalError:
                logger.error(f"A GLPK error occurred during creation of virtual storage constraints for node: {storage}")
                raise
            finally:
                free(ind)
                free(val)

        # Add constraint rows for aggregated nodes with dynamics factors and
        # cache data so row values can be updated in solve
        if len(aggregated_with_factors):
            self.idx_row_aggregated = self.idx_row_virtual_storages + len(virtual_storages)
        for agg_node in aggregated_with_factors:
            nodes = agg_node.nodes

            row = glp_add_rows(self.prob, len(agg_node.nodes)-1)
            agg_node.__agg_factor_data.row = row

            cols = []
            ind_ptr = [0,]
            first_node_cols = [0] + [n+1 for n, route in enumerate(routes) if nodes[0] in route]
            agg_node.__agg_factor_data.node_ind = len(first_node_cols)
            for i, node in enumerate(nodes[1:]):
                cols.extend(first_node_cols + [n+1 for n, route in enumerate(routes) if node in route])
                ind_ptr.append(len(cols))

            agg_node.__agg_factor_data.ind_ptr = cvarray(shape=(len(ind_ptr),), itemsize=sizeof(int), format="i")
            for i, v in enumerate(ind_ptr):
                agg_node.__agg_factor_data.ind_ptr[i] = v

            agg_node.__agg_factor_data.inds = cvarray(shape=(len(cols),), itemsize=sizeof(int), format="i")
            agg_node.__agg_factor_data.vals = cvarray(shape=(len(cols),), itemsize=sizeof(double), format="d")
            for i, v in enumerate(cols):
                agg_node.__agg_factor_data.inds[i] = v
                agg_node.__agg_factor_data.vals[i] = 1.0

            for n in range(len(nodes)-1):
                # Always use the safe API in setup
                set_row_bnds_safe(self.prob, row+n, GLP_FX, 0.0, 0.0)

            # Determine whether the factors should be updated in reset or at each time-step
            if self.set_fixed_factors_once and agg_node.has_constant_factors:
                aggregated_with_constant_factors.append(agg_node)
            else:
                aggregated_with_dynamic_factors.append(agg_node)

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
                for n, route in enumerate(routes):
                    if some_node in route:
                        matrix[n] = w
            length = len(matrix)
            ind = <int*>malloc(1+length * sizeof(int))
            val = <double*>malloc(1+length * sizeof(double))
            for i, col in enumerate(sorted(matrix)):
                ind[1+i] = 1+col
                val[1+i] = matrix[col]
            try:
                # Always use the safe API in setup
                set_mat_row_safe(self.prob, row, length, ind, val)
                set_row_bnds_safe(self.prob, row, GLP_FX, 0.0, 0.0)
            except GLPKError, GLPKInternalError:
                logger.error(f"A GLPK error occurred during creation of aggregated node constraints for node: {agg_node}")
                raise
            finally:
                free(ind)
                free(val)

        # update route properties
        routes_cost = []
        routes_cost_indptr = [0, ]
        for col, route in enumerate(routes):
            route_cost = []
            route_cost.append(route[0].__data.id)
            for some_node in route[1:-1]:
                if isinstance(some_node, BaseLink):
                    route_cost.append(some_node.__data.id)
            route_cost.append(route[-1].__data.id)
            routes_cost.extend(route_cost)
            routes_cost_indptr.append(len(routes_cost))

        assert(len(routes_cost_indptr) == len(routes) + 1)

        self.routes_cost_indptr = np.array(routes_cost_indptr, dtype=np.int32)
        self.routes_cost = np.array(routes_cost, dtype=np.int32)

        self.routes = routes
        self.non_storages = non_storages
        self.non_storages_with_constant_bounds = non_storages_with_constant_bounds
        self.non_storages_with_dynamic_bounds = non_storages_with_dynamic_bounds
        self.nodes_with_dynamic_cost = nodes_with_dynamic_cost
        self.storages = storages
        self.virtual_storages = virtual_storages
        self.aggregated = aggregated
        self.aggregated_with_factors = aggregated_with_factors
        self.aggregated_with_dynamic_factors = aggregated_with_dynamic_factors
        self.aggregated_with_constant_factors = aggregated_with_constant_factors

        self.basis_manager.init_basis(self.prob, len(model.scenarios.combinations))
        self.is_first_solve = True

        # reset stats
        self.stats = {
            'total': 0.0,
            'lp_solve': 0.0,
            'result_update': 0.0,
            'constraint_update_factors': 0.0,
            'bounds_update_nonstorage': 0.0,
            'bounds_update_storage': 0.0,
            'objective_update': 0.0,
            'number_of_rows': glp_get_num_rows(self.prob),
            'number_of_cols': glp_get_num_cols(self.prob),
            'number_of_nonzero': glp_get_num_nz(self.prob),
            'number_of_routes': len(routes),
            'number_of_nodes': len(self.all_nodes)
        }

    def reset(self):
        # Resetting this triggers a crashing of a new basis in each scenario
        self.is_first_solve = True
        self._update_nonstorage_constant_bounds()
        self._update_constant_agg_factors()

    cdef int _update_nonstorage_constant_bounds(self) except -1:
        """Update the bounds of non-storage where they are constants.

        These bounds do not need updating every time-step.
        """
        cdef Node node
        cdef double min_flow
        cdef double max_flow
        cdef int row
        if self.non_storages_with_constant_bounds is None:
            return 0
        # update non-storage constraints with constant bounds
        for node in self.non_storages_with_constant_bounds:
            row = node.__data.row
            min_flow = inf_to_dbl_max(node.get_constant_min_flow())
            if abs(min_flow) < 1e-8:
                min_flow = 0.0
            max_flow = inf_to_dbl_max(node.get_constant_max_flow())
            if abs(max_flow) < 1e-8:
                max_flow = 0.0
            # Always use the safe API in reset
            set_row_bnds_safe(self.prob, self.idx_row_non_storages + row, constraint_type(min_flow, max_flow),
                         min_flow, max_flow)

    cdef int _update_constant_agg_factors(self) except -1:
        cdef AggregatedNode agg_node
        cdef AggNodeFactorData agg_data
        cdef int n, i, ptr
        cdef Py_ssize_t length
        cdef int[::1] inds
        cdef double[::1] vals
        cdef int[:] indptr_array
        if self.aggregated_with_constant_factors is None:
            return 0

        # Update constraint matrix values for aggregated nodes that have factors defined as parameters
        for agg_node in self.aggregated_with_constant_factors:
            factors_norm = agg_node.get_factors_norm(None)

            agg_data = agg_node.__agg_factor_data
            inds = agg_data.inds
            vals = agg_data.vals
            indptr_array = agg_data.ind_ptr

            for n in range(len(agg_node.nodes)-1):

                ptr = indptr_array[n]
                length = indptr_array[n+1] - ptr

                # only update factors for second node of each row, factor values for first node are already 1.0
                for i in range(ptr + agg_data.node_ind, ptr + length):
                    vals[i] = -factors_norm[n+1]

                # 'length - 1' is used here because the ind and val slices start with a padded zero value.
                # This is required by 'set_mat_row'.
                # Always use the safe API in reset
                set_mat_row_safe(self.prob, agg_data.row+n, length-1, &inds[ptr], &vals[ptr])

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
        cdef AggNodeFactorData agg_data
        cdef AggregatedNode agg_node
        cdef double min_flow
        cdef double max_flow
        cdef double cost
        cdef double max_volume
        cdef double min_volume
        cdef double avail_volume
        cdef double t0
        cdef int col, row
        cdef double lb
        cdef double ub
        cdef Timestep timestep
        cdef int status, simplex_ret
        cdef cross_domain_col
        cdef list route
        cdef int node_id, indptr, nroutes
        cdef int[::1] inds
        cdef double[::1] vals
        cdef double flow
        cdef int n, m, i, ptr
        cdef Py_ssize_t length
        cdef int[:] indptr_array
        cdef double[:] factors_norm

        timestep = model.timestep
        cdef list routes = self.routes
        nroutes = len(routes)
        cdef list non_storages = self.non_storages
        cdef list non_storages_with_dynamic_bounds = self.non_storages_with_dynamic_bounds
        cdef list storages = self.storages
        cdef list virtual_storages = self.virtual_storages
        cdef list aggregated = self.aggregated

        # update route cost

        t0 = time.perf_counter()

        # update the cost of each node in the model
        cdef double[:] node_costs = self.node_costs_arr
        for _node in self.nodes_with_dynamic_cost:
            data = _node.__data
            node_costs[data.id] = _node.get_cost(scenario_index)

        # calculate the total cost of each route
        for col in range(nroutes):
            cost = 0.0
            for indptr in range(self.routes_cost_indptr[col], self.routes_cost_indptr[col+1]):
                node_id = self.routes_cost[indptr]
                cost += node_costs[node_id]

            if abs(cost) < 1e-8:
                cost = 0.0
            if self.use_unsafe_api:
                set_obj_coef_unsafe(self.prob, self.idx_col_routes+col, cost)
            else:
                set_obj_coef_safe(self.prob, self.idx_col_routes+col, cost)

        self.stats['objective_update'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        # Update constraint matrix values for aggregated nodes that have factors defined as parameters
        for agg_node in self.aggregated_with_dynamic_factors:

            factors_norm = agg_node.get_factors_norm(scenario_index)

            agg_data = agg_node.__agg_factor_data
            inds = agg_data.inds
            vals = agg_data.vals
            indptr_array = agg_data.ind_ptr

            for n in range(len(agg_node.nodes)-1):

                ptr = indptr_array[n]
                length = indptr_array[n+1] - ptr

                # only update factors for second node of each row, factor values for first node are already 1.0
                for i in range(ptr + agg_data.node_ind, ptr + length):
                    vals[i] = -factors_norm[n+1]

                # 'length - 1' is used here because the ind and val slices start with a padded zero value.
                # This is required by 'set_mat_row'.
                if self.use_unsafe_api:
                    set_mat_row_unsafe(self.prob, agg_data.row+n, length-1, &inds[ptr], &vals[ptr])
                else:
                    set_mat_row_safe(self.prob, agg_data.row+n, length-1, &inds[ptr], &vals[ptr])

        self.stats['constraint_update_factors'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        # update non-storage properties
        for node in non_storages_with_dynamic_bounds:
            row = node.__data.row
            min_flow = inf_to_dbl_max(node.get_min_flow(scenario_index))
            if abs(min_flow) < 1e-8:
                min_flow = 0.0
            max_flow = inf_to_dbl_max(node.get_max_flow(scenario_index))
            if abs(max_flow) < 1e-8:
                max_flow = 0.0
            if self.use_unsafe_api:
                set_row_bnds_unsafe(self.prob, self.idx_row_non_storages+row, constraint_type(min_flow, max_flow),
                                    min_flow, max_flow)
            else:
                set_row_bnds_safe(self.prob, self.idx_row_non_storages+row, constraint_type(min_flow, max_flow),
                                  min_flow, max_flow)

        for row, agg_node in enumerate(aggregated):
            min_flow = inf_to_dbl_max(agg_node.get_min_flow(scenario_index))
            if abs(min_flow) < 1e-8:
                min_flow = 0.0
            max_flow = inf_to_dbl_max(agg_node.get_max_flow(scenario_index))
            if abs(max_flow) < 1e-8:
                max_flow = 0.0
            if self.use_unsafe_api:
                set_row_bnds_unsafe(self.prob, self.idx_row_aggregated_min_max + row, constraint_type(min_flow, max_flow),
                                    min_flow, max_flow)
            else:
                set_row_bnds_safe(self.prob, self.idx_row_aggregated_min_max + row, constraint_type(min_flow, max_flow),
                                  min_flow, max_flow)

        self.stats['bounds_update_nonstorage'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        # update storage node constraint
        for row, storage in enumerate(storages):
            max_volume = storage.get_max_volume(scenario_index)
            min_volume = storage.get_min_volume(scenario_index)

            if max_volume == min_volume:
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_storages+row, GLP_FX, 0.0, 0.0)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_storages + row, GLP_FX, 0.0, 0.0)
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
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_storages+row, constraint_type(lb, ub), lb, ub)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_storages + row, constraint_type(lb, ub), lb, ub)

        # update virtual storage node constraint
        for row, storage in enumerate(virtual_storages):
            max_volume = storage.get_max_volume(scenario_index)
            min_volume = storage.get_min_volume(scenario_index)

            if max_volume == min_volume:
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_virtual_storages+row, GLP_FX, 0.0, 0.0)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_virtual_storages+row, GLP_FX, 0.0, 0.0)
            elif not storage.active:
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_virtual_storages+row, GLP_FR, -DBL_MAX, DBL_MAX)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_virtual_storages+row, GLP_FR, -DBL_MAX, DBL_MAX)
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
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_virtual_storages+row, constraint_type(lb, ub), lb, ub)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_virtual_storages+row, constraint_type(lb, ub), lb, ub)

        self.stats['bounds_update_storage'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        # Set the basis for this scenario
        self.basis_manager.set_basis(self.prob, self.is_first_solve, scenario_index.global_id)
        # attempt to solve the linear programme
        if self.use_unsafe_api:
            simplex_ret = simplex_unsafe(self.prob, self.smcp)
        else:
            simplex_ret = simplex_safe(self.prob, self.smcp)

        status = glp_get_status(self.prob)
        if (status != GLP_OPT or simplex_ret != 0) and self.retry_solve:
            # try creating a new basis and resolving
            print('Retrying solve with new basis.')
            glp_std_basis(self.prob)
            if self.use_unsafe_api:
                simplex_ret = simplex_unsafe(self.prob, self.smcp)
            else:
                simplex_ret = simplex_safe(self.prob, self.smcp)
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
            if self.use_unsafe_api:
                simplex_ret = simplex_unsafe(self.prob, self.smcp)
            else:
                simplex_ret = simplex_safe(self.prob, self.smcp)
            status = glp_get_status(self.prob)
            raise RuntimeError('Simplex solver failed with message: "{}", status: "{}".'.format(
                simplex_status_string[simplex_ret], status_string[status]))
        # Now save the basis
        self.basis_manager.save_basis(self.prob, scenario_index.global_id)

        self.stats['lp_solve'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        cdef double[:] route_flows
        if self.save_routes_flows:
            route_flows = self.route_flows_arr[scenario_index.global_id, :]
        else:
            route_flows = self.route_flows_arr

        for col in range(0, self.num_routes):
            route_flows[col] = glp_get_col_prim(self.prob, col+1)

        # collect the total flow via each node
        cdef double[:] node_flows = self.node_flows_arr
        node_flows[:] = 0.0
        for n, route in enumerate(routes):
            flow = route_flows[n]
            if flow == 0:
                continue
            length = len(route)
            for m, _node in enumerate(route):
                data = _node.__data
                if (m == 0) or (m == length-1) or data.is_link:
                    node_flows[data.id] += flow

        # commit the total flows
        for n in range(0, self.num_nodes):
            _node = self.all_nodes[n]
            _node.commit(scenario_index.global_id, node_flows[n])

        self.stats['result_update'] += time.perf_counter() - t0


cdef class CythonGLPKEdgeSolver(GLPKSolver):
    cdef int idx_col_edges
    cdef int idx_row_non_storages
    cdef int idx_row_link_mass_bal
    cdef int idx_row_cross_domain
    cdef int idx_row_storages
    cdef int idx_row_virtual_storages
    cdef int idx_row_aggregated
    cdef int idx_row_aggregated_with_factors
    cdef int idx_row_aggregated_min_max

    cdef list non_storages
    cdef list non_storages_with_dynamic_bounds
    cdef list non_storages_with_constant_bounds
    cdef list nodes_with_dynamic_cost
    cdef list storages
    cdef list virtual_storages
    cdef list aggregated
    cdef list aggregated_with_factors
    cdef list aggregated_with_dynamic_factors
    cdef list aggregated_with_constant_factors

    cdef list all_nodes
    cdef list all_edges

    cdef int num_nodes
    cdef int num_edges
    cdef int num_storages
    cdef int num_scenarios
    cdef cvarray edge_cost_arr
    cdef cvarray edge_fixed_cost_arr
    cdef cvarray edge_flows_arr
    cdef cvarray node_flows_arr
    cdef public cvarray route_flows_arr
    cdef public object stats

    # Internal representation of the basis for each scenario
    cdef BasisManager basis_manager
    cdef bint is_first_solve
    cdef public bint retry_solve


    def __init__(self, use_presolve=False, time_limit=None, iteration_limit=None, message_level='error',
                 save_routes_flows=False, retry_solve=False, **kwargs):
        super().__init__(use_presolve, time_limit, iteration_limit, message_level, **kwargs)
        self.stats = None
        self.is_first_solve = True
        self.retry_solve = retry_solve
        self.basis_manager = BasisManager()

    def setup(self, model):
        super().setup(model)

        cdef Node input
        cdef Node output
        cdef Node node
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
        self.edge_fixed_cost_arr = cvarray(shape=(len(self.all_edges),), itemsize=sizeof(double), format="d")
        self.edge_flows_arr = cvarray(shape=(len(self.all_edges),), itemsize=sizeof(double), format="d")
        self.node_flows_arr = cvarray(shape=(self.num_nodes,), itemsize=sizeof(double), format="d")

        # Find cross-domain routes
        cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')

        nodes_with_dynamic_cost = []
        link_nodes = []
        non_storages = []
        non_storages_with_dynamic_bounds = []  # These nodes' constraints are updated each timestep
        non_storages_with_constant_bounds = []  # These nodes' constraints are updated only at reset
        storages = []
        virtual_storages = []
        aggregated_with_factors = []
        aggregated_with_dynamic_factors = []
        aggregated_with_constant_factors = []
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
                    some_node.__agg_factor_data = AggNodeFactorData()
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
            # Always use safe API in setup
            set_col_bnds_safe(self.prob, self.idx_col_edges+row, GLP_LO, 0.0, DBL_MAX)

        # Apply nodal flow constraints
        self.idx_row_non_storages = glp_add_rows(self.prob, len(non_storages))
        # # Add rows for the cross-domain routes.
        if len(cross_domain_cols) > 0:
            self.idx_row_cross_domain = glp_add_rows(self.prob, len(cross_domain_cols))

        self.edge_fixed_cost_arr[...] = 0.0
        # Calculate fixed costs
        for some_node in self.all_nodes:
            try:
                fixed_cost = some_node.has_fixed_cost
            except AttributeError:
                fixed_cost = False

            if self.set_fixed_costs_once and fixed_cost:
                # With a fixed cost this should work with no scenario index
                cost = some_node.get_cost(None)
                data = some_node.__data

                # Link nodes have edges connected upstream & downstream. We apply
                # half the cost assigned to the node to all the connected edges.
                # The edge costs are then the mean of the node costs at either end.
                if data.is_link:
                    cost /= 2

                for col in data.in_edges:
                    self.edge_fixed_cost_arr[col] += cost
                for col in data.out_edges:
                    self.edge_fixed_cost_arr[col] += cost
            else:
                nodes_with_dynamic_cost.append(some_node)

        cross_domain_row = 0
        for row, node in enumerate(non_storages):
            node.__data.row = row
            # Differentiate between the node type.
            # Input and other nodes use the outgoing edge flows to apply the flow constraint on
            # This requires the mass balance constraints to ensure the inflow and outflow are equal
            # The Output nodes, in contrast, apply the constraint to the incoming flow (because there is no outgoing flow)
            if isinstance(node, BaseInput):
                cols = node.__data.out_edges
                if len(node.__data.in_edges) != 0:
                    raise ModelStructureError(f'Input node "{node.name}" should not have any upstream '
                                              f'connections.')
            elif isinstance(node, BaseOutput):
                cols = node.__data.in_edges
                if len(node.__data.out_edges) != 0:
                    raise ModelStructureError(f'Output node "{node.name}" should not have any downstream '
                                              f'connections.')
            else:
                # Other nodes apply their flow constraints to all routes passing through them
                cols = node.__data.out_edges

            if len(cols) == 0:
                raise ModelStructureError(f'{node.__class__.__name__} node "{node.name}" does not have any valid connections.')

            ind = <int*>malloc((1+len(cols)) * sizeof(int))
            val = <double*>malloc((1+len(cols)) * sizeof(double))
            for n, c in enumerate(cols):
                ind[1+n] = 1+c
                val[1+n] = 1
            try:
                # Always use safe API in setup
                set_mat_row_safe(self.prob, self.idx_row_non_storages+row, len(cols), ind, val)
                set_row_bnds_safe(self.prob, self.idx_row_non_storages + row, GLP_FX, 0.0, 0.0)
            except GLPKError, GLPKInternalError:
                logger.error(f"A GLPK error occurred during creation of flow constraints for node: {node}")
                raise
            finally:
                free(ind)
                free(val)

            # Now test whether this node has constant flow constraints
            if self.set_fixed_flows_once and node.has_constant_flows:
                non_storages_with_constant_bounds.append(node)
            else:
                non_storages_with_dynamic_bounds.append(node)

            # Add constraint for cross-domain routes
            # i.e. those from a demand to a supply
            if node in cross_domain_cols:
                col_vals = cross_domain_cols[node]
                ind = <int*>malloc((1+len(col_vals)+len(cols)) * sizeof(int))
                val = <double*>malloc((1+len(col_vals)+len(cols)) * sizeof(double))
                for n, c in enumerate(cols):
                    ind[1+n] = 1+c
                    val[1+n] = -1
                for n, (c, v) in enumerate(col_vals):
                    ind[1+n+len(cols)] = 1+c
                    val[1+n+len(cols)] = 1./v
                try:
                    # Always use safe API in setup
                    set_mat_row_safe(self.prob, self.idx_row_cross_domain+cross_domain_row, len(col_vals)+len(cols), ind, val)
                    set_row_bnds_safe(self.prob, self.idx_row_cross_domain+cross_domain_row, GLP_FX, 0.0, 0.0)
                except GLPKError, GLPKInternalError:
                    logger.error(f"A GLPK error occurred during creation of cross-domain constraints for node: {node}")
                    raise
                finally:
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

            try:
                # Always use safe API in setup
                set_mat_row_safe(self.prob, self.idx_row_link_mass_bal+row, len(in_cols)+len(out_cols), ind, val)
                set_row_bnds_safe(self.prob, self.idx_row_link_mass_bal+row, GLP_FX, 0.0, 0.0)
            except GLPKError, GLPKInternalError:
                logger.error(f"A GLPK error occurred during creation of mass-balance constraints for node: {some_node}")
                raise
            finally:
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

            try:
                # Always use safe API in setup
                set_mat_row_safe(self.prob, self.idx_row_storages+row, len(cols_output)+len(cols_input), ind, val)
            except GLPKError, GLPKInternalError:
                logger.error(f"A GLPK error occurred during creation of storage constraints for node: {storage}")
                raise
            finally:
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

            try:
                # Always use safe API in setup
                set_mat_row_safe(self.prob, self.idx_row_virtual_storages+row, len(cols), ind, val)
            except GLPKError, GLPKInternalError:
                logger.error(f"A GLPK error occurred during creation of virtual-storage constraints for node: {storage}")
                raise
            finally:
                free(ind)
                free(val)

        # Add constraint rows for aggregated nodes with dynamics factors and
        # cache data so row values can be updated in solve
        if len(aggregated_with_factors):
            self.idx_row_aggregated_with_factors = self.idx_row_virtual_storages + len(virtual_storages)
        for agg_node in aggregated_with_factors:
            nodes = agg_node.nodes

            row = glp_add_rows(self.prob, len(agg_node.nodes)-1)
            agg_node.__agg_factor_data.row = row

            cols = []
            ind_ptr = [0,]
            if isinstance(nodes[0], BaseOutput):
                first_node_cols = [0] + [c+1 for c in nodes[0].__data.in_edges]
            else:
                first_node_cols = [0] + [c+1 for c in nodes[0].__data.out_edges]

            agg_node.__agg_factor_data.node_ind = len(first_node_cols)
            for i, some_node in enumerate(nodes[1:]):
                if isinstance(some_node, BaseOutput):
                    cols.extend(first_node_cols + [c+1 for c in some_node.__data.in_edges])
                else:
                    cols.extend(first_node_cols + [c+1 for c in some_node.__data.out_edges])
                ind_ptr.append(len(cols))

            agg_node.__agg_factor_data.ind_ptr = cvarray(shape=(len(ind_ptr),), itemsize=sizeof(int), format="i")
            for i, v in enumerate(ind_ptr):
                agg_node.__agg_factor_data.ind_ptr[i] = v

            agg_node.__agg_factor_data.inds = cvarray(shape=(len(cols),), itemsize=sizeof(int), format="i")
            agg_node.__agg_factor_data.vals = cvarray(shape=(len(cols),), itemsize=sizeof(double), format="d")
            for i, v in enumerate(cols):
                agg_node.__agg_factor_data.inds[i] = v
                agg_node.__agg_factor_data.vals[i] = 1.0

            for n in range(len(nodes)-1):
                # Always use safe API in setup
                set_row_bnds_safe(self.prob, row+n, GLP_FX, 0.0, 0.0)

            # Determine whether the factors should be updated in reset or at each time-step
            if self.set_fixed_factors_once and agg_node.has_constant_factors:
                aggregated_with_constant_factors.append(agg_node)
            else:
                aggregated_with_dynamic_factors.append(agg_node)

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
            try:
                # Always use safe API in setup
                set_mat_row_safe(self.prob, row, length, ind, val)
                set_row_bnds_safe(self.prob, row, GLP_FX, 0.0, 0.0)
            except GLPKError, GLPKInternalError:
                logger.error(f"A GLPK error occurred during creation of aggregated node constraints for node: {storage}")
                raise
            finally:
                free(ind)
                free(val)

        self.non_storages = non_storages
        self.non_storages_with_dynamic_bounds = non_storages_with_dynamic_bounds
        self.non_storages_with_constant_bounds = non_storages_with_constant_bounds
        self.nodes_with_dynamic_cost = nodes_with_dynamic_cost
        self.storages = storages
        self.virtual_storages = virtual_storages
        self.aggregated = aggregated
        self.aggregated_with_factors = aggregated_with_factors
        self.aggregated_with_dynamic_factors = aggregated_with_dynamic_factors
        self.aggregated_with_constant_factors = aggregated_with_constant_factors

        self.basis_manager.init_basis(self.prob, len(model.scenarios.combinations))
        self.is_first_solve = True

        # reset stats
        self.stats = {
            'total': 0.0,
            'lp_solve': 0.0,
            'result_update': 0.0,
            'constraint_update_factors': 0.0,
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
        self._update_nonstorage_constant_bounds()
        self._update_constant_agg_factors()

    cdef int _update_nonstorage_constant_bounds(self) except -1:
        """Update the bounds of non-storage where they are constants.
        
        These bounds do not need updating every time-step.
        """
        cdef Node node
        cdef double min_flow
        cdef double max_flow
        cdef int row
        if self.non_storages_with_constant_bounds is None:
            return 0
        # update non-storage constraints with constant bounds
        for node in self.non_storages_with_constant_bounds:
            row = node.__data.row
            min_flow = inf_to_dbl_max(node.get_constant_min_flow())
            if abs(min_flow) < 1e-8:
                min_flow = 0.0
            max_flow = inf_to_dbl_max(node.get_constant_max_flow())
            if abs(max_flow) < 1e-8:
                max_flow = 0.0

            # Always use safe API in setup
            set_row_bnds_safe(self.prob, self.idx_row_non_storages+row, constraint_type(min_flow, max_flow),
                         min_flow, max_flow)

    cdef int _update_constant_agg_factors(self) except -1:
        cdef AggregatedNode agg_node
        cdef AggNodeFactorData agg_data
        cdef int n, i, ptr
        cdef Py_ssize_t length
        cdef int[::1] inds
        cdef double[::1] vals
        cdef int[:] indptr_array
        if self.aggregated_with_constant_factors is None:
            return 0

        # Update constraint matrix values for aggregated nodes that have factors defined as parameters
        for agg_node in self.aggregated_with_constant_factors:
            factors_norm = agg_node.get_factors_norm(None)

            agg_data = agg_node.__agg_factor_data
            inds = agg_data.inds
            vals = agg_data.vals
            indptr_array = agg_data.ind_ptr

            for n in range(len(agg_node.nodes)-1):

                ptr = indptr_array[n]
                length = indptr_array[n+1] - ptr

                # only update factors for second node of each row, factor values for first node are already 1.0
                for i in range(ptr + agg_data.node_ind, ptr + length):
                    vals[i] = -factors_norm[n+1]

                # 'length - 1' is used here because the ind and val slices start with a padded zero value.
                # This is required by 'set_mat_row'.
                # Always use safe API in setup
                set_mat_row_safe(self.prob, agg_data.row+n, length-1, &inds[ptr], &vals[ptr])

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
        cdef AggNodeFactorData agg_data
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
        cdef int[::1] inds
        cdef double[::1] vals
        cdef double flow
        cdef int n, m, i, ptr
        cdef int[:] indptr_array
        cdef double[:] factors_norm
        cdef Py_ssize_t length

        timestep = model.timestep
        cdef list edges = self.all_edges
        nedges = self.num_edges
        cdef list non_storages = self.non_storages
        cdef list non_storages_with_dynamic_bounds = self.non_storages_with_dynamic_bounds
        cdef list storages = self.storages
        cdef list virtual_storages = self.virtual_storages
        cdef list aggregated = self.aggregated

        # update route cost

        t0 = time.perf_counter()

        # Initialise the cost on each edge to zero
        cdef double[:] fixed_edge_costs = self.edge_fixed_cost_arr
        cdef double[:] edge_costs = self.edge_cost_arr
        for col in range(nedges):
            edge_costs[col] = fixed_edge_costs[col]

        # update the cost of each node in the model
        for _node in self.nodes_with_dynamic_cost:
            cost = _node.get_cost(scenario_index)
            data = _node.__data
            # Link nodes have edges connected upstream & downstream. We apply
            # half the cost assigned to the node to all the connected edges.
            # The edge costs are then the mean of any links at either end plus
            # the cost of any input or output nodes.
            if data.is_link:
                cost /= 2

            for col in data.in_edges:
                edge_costs[col] += cost
            for col in data.out_edges:
                edge_costs[col] += cost

        # calculate the total cost of each route
        for col in range(nedges):
            if self.use_unsafe_api:
                set_obj_coef_unsafe(self.prob, self.idx_col_edges+col, edge_costs[col])
            else:
                set_obj_coef_safe(self.prob, self.idx_col_edges+col, edge_costs[col])

        self.stats['objective_update'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        # Update constraint matrix values for aggregated nodes that have factors defined as parameters
        for agg_node in self.aggregated_with_dynamic_factors:

            factors_norm = agg_node.get_factors_norm(scenario_index)

            agg_data = agg_node.__agg_factor_data
            inds = agg_data.inds
            vals = agg_data.vals
            indptr_array = agg_data.ind_ptr

            for n in range(len(agg_node.nodes)-1):

                ptr = indptr_array[n]
                length = indptr_array[n+1] - ptr

                # only update factors for second node of each row, factor values for first node are already 1.0
                for i in range(ptr + agg_data.node_ind, ptr + length):
                    vals[i] = -factors_norm[n+1]

                # 'length - 1' is used here because the ind and val slices start with a padded zero value.
                # This is required by 'set_mat_row'.
                if self.use_unsafe_api:
                    set_mat_row_unsafe(self.prob, agg_data.row+n, length-1, &inds[ptr], &vals[ptr])
                else:
                    set_mat_row_safe(self.prob, agg_data.row+n, length-1, &inds[ptr], &vals[ptr])

        self.stats['constraint_update_factors'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        # update non-storage properties
        for node in non_storages_with_dynamic_bounds:
            row = node.__data.row
            min_flow = inf_to_dbl_max(node.get_min_flow(scenario_index))
            if abs(min_flow) < 1e-8:
                min_flow = 0.0
            max_flow = inf_to_dbl_max(node.get_max_flow(scenario_index))
            if abs(max_flow) < 1e-8:
                max_flow = 0.0
            if self.use_unsafe_api:
                set_row_bnds_unsafe(self.prob, self.idx_row_non_storages+row, constraint_type(min_flow, max_flow),
                         min_flow, max_flow)
            else:
                set_row_bnds_safe(self.prob, self.idx_row_non_storages+row, constraint_type(min_flow, max_flow),
                         min_flow, max_flow)

        for row, agg_node in enumerate(aggregated):
            min_flow = inf_to_dbl_max(agg_node.get_min_flow(scenario_index))
            if abs(min_flow) < 1e-8:
                min_flow = 0.0
            max_flow = inf_to_dbl_max(agg_node.get_max_flow(scenario_index))
            if abs(max_flow) < 1e-8:
                max_flow = 0.0
            if self.use_unsafe_api:
                set_row_bnds_unsafe(self.prob, self.idx_row_aggregated_min_max + row, constraint_type(min_flow, max_flow),
                             min_flow, max_flow)
            else:
                set_row_bnds_safe(self.prob, self.idx_row_aggregated_min_max + row, constraint_type(min_flow, max_flow),
                             min_flow, max_flow)

        self.stats['bounds_update_nonstorage'] += time.perf_counter() - t0
        t0 = time.perf_counter()

        # update storage node constraint
        for row, storage in enumerate(storages):
            max_volume = storage.get_max_volume(scenario_index)
            min_volume = storage.get_min_volume(scenario_index)

            if max_volume == min_volume:
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_storages+row, GLP_FX, 0.0, 0.0)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_storages+row, GLP_FX, 0.0, 0.0)
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
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_storages+row, constraint_type(lb, ub), lb, ub)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_storages+row, constraint_type(lb, ub), lb, ub)

        # update virtual storage node constraint
        for row, storage in enumerate(virtual_storages):
            max_volume = storage.get_max_volume(scenario_index)
            min_volume = storage.get_min_volume(scenario_index)

            if max_volume == min_volume:
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_virtual_storages+row, GLP_FX, 0.0, 0.0)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_virtual_storages+row, GLP_FX, 0.0, 0.0)
            elif not storage.active:
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_virtual_storages+row, GLP_FR, -DBL_MAX, DBL_MAX)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_virtual_storages+row, GLP_FR, -DBL_MAX, DBL_MAX)
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
                if self.use_unsafe_api:
                    set_row_bnds_unsafe(self.prob, self.idx_row_virtual_storages+row, constraint_type(lb, ub), lb, ub)
                else:
                    set_row_bnds_safe(self.prob, self.idx_row_virtual_storages+row, constraint_type(lb, ub), lb, ub)

        self.stats['bounds_update_storage'] += time.perf_counter() - t0

        t0 = time.perf_counter()

        # Set the basis for this scenario
        self.basis_manager.set_basis(self.prob, self.is_first_solve, scenario_index.global_id)
        # attempt to solve the linear programme
        if self.use_unsafe_api:
            simplex_ret = simplex_unsafe(self.prob, self.smcp)
        else:
            simplex_ret = simplex_safe(self.prob, self.smcp)
        status = glp_get_status(self.prob)
        if (status != GLP_OPT or simplex_ret != 0) and self.retry_solve:
            # try creating a new basis and resolving
            print('Retrying solve with new basis.')
            glp_std_basis(self.prob)
            if self.use_unsafe_api:
                simplex_ret = simplex_unsafe(self.prob, self.smcp)
            else:
                simplex_ret = simplex_safe(self.prob, self.smcp)
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
            if self.use_unsafe_api:
                simplex_ret = simplex_unsafe(self.prob, self.smcp)
            else:
                simplex_ret = simplex_safe(self.prob, self.smcp)
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
