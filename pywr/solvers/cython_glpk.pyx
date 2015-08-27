from libc.stdlib cimport malloc, free

from pywr.core import BaseInput, BaseOutput, BaseLink
from pywr._core cimport *

include "glpk.pxi"

cdef class CythonGLPKSolver:
    cdef glp_prob* prob
    cdef glp_smcp smcp
    cdef int idx_col_routes
    cdef int idx_col_demands
    cdef int idx_row_supplys
    cdef int idx_row_demands
    cdef int idx_row_storages
    
    cdef object routes
    cdef object supplys
    cdef object demands
    cdef object storages
    
    def __cinit__(self):
        # create a new problem
        self.prob = glp_create_prob()
        # disable console messages
        glp_init_smcp(&self.smcp)
        self.smcp.msg_lev = GLP_MSG_ERR

    def __dealloc__(self):
        # free the problem
        glp_free(self.prob)
    
    cpdef object solve(self, model):
        cdef Node supply
        cdef Node demand
        cdef double min_flow
        cdef double max_flow
        cdef double cost
        cdef double current_volume
        cdef int col
        cdef int* ind
        cdef double* val
        cdef double lb
        cdef double ub
        cdef Timestep timestep
        cdef int status
        
        timestep = model.timestep

        if model.dirty:
            routes = model.find_all_routes(BaseInput, BaseOutput, valid=(BaseLink, BaseInput, BaseOutput))
            
            supplys = []
            demands = []
            storages = []
            for node in model.nodes():
                if isinstance(node, BaseInput):
                    supplys.append(node)
                if isinstance(node, BaseOutput):
                    demands.append(node)
                if isinstance(node, Storage):
                    storages.append(node)

            assert(routes)
            assert(supplys)
            assert(demands)
            
            # clear the previous problem
            glp_erase_prob(self.prob)
            glp_set_obj_dir(self.prob, GLP_MIN)
            # add a column for each route
            self.idx_col_routes = glp_add_cols(self.prob, <int>(len(routes)))
            # add a column for each demand
            self.idx_col_demands = glp_add_cols(self.prob, <int>(len(demands)))

            # explicitly set bounds on route and demand columns
            for col, route in enumerate(routes):
                glp_set_col_bnds(self.prob, self.idx_col_routes+col, GLP_LO, 0.0, DBL_MAX)
            for col, demand in enumerate(demands):
                glp_set_col_bnds(self.prob, self.idx_col_demands+col, GLP_LO, 0.0, DBL_MAX)
                
            # constrain supply minimum and maximum flow
            self.idx_row_supplys = glp_add_rows(self.prob, len(supplys))
            for col, supply in enumerate(supplys):
                cols = [n for n, route in enumerate(routes) if route[0] is supply]
                ind = <int*>malloc((1+len(cols)) * sizeof(int))
                val = <double*>malloc((1+len(cols)) * sizeof(double))
                for n, c in enumerate(cols):
                    ind[1+n] = 1+c
                    val[1+n] = 1
                glp_set_mat_row(self.prob, self.idx_row_supplys+col, len(cols), ind, val)
                glp_set_row_bnds(self.prob, self.idx_row_supplys+col, GLP_FX, 0.0, 0.0)

            # link supply and demand variables
            self.idx_row_demands = glp_add_rows(self.prob, len(demands))
            for col, demand in enumerate(demands):
                cols = [n for n, route in enumerate(routes) if route[-1] is demand]
                ind = <int*>malloc((1+len(cols)+1) * sizeof(int))
                val = <double*>malloc((1+len(cols)+1) * sizeof(double))
                for n, c in enumerate(cols):
                    ind[1+n] = 1+c
                    val[1+n] = 1
                ind[1+len(cols)] = self.idx_col_demands+col
                val[1+len(cols)] = -1
                glp_set_mat_row(self.prob, self.idx_row_demands+col, len(cols)+1, ind, val)
                glp_set_row_bnds(self.prob, self.idx_row_demands+col, GLP_FX, 0.0, 0.0)

            # storage
            if len(storages):
                self.idx_row_storages = glp_add_rows(self.prob, len(storages))
            for col, storage in enumerate(storages):
                cols_output = [n for n, demand in enumerate(demands) if demand is storage.output]
                cols_input = [n for n, route in enumerate(routes) if route[0] is storage.input]
                ind = <int*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(int))
                val = <double*>malloc((1+len(cols_output)+len(cols_input)) * sizeof(double))
                for n, c in enumerate(cols_output):
                    ind[1+n] = self.idx_col_demands+c
                    val[1+n] = 1
                for n, c in enumerate(cols_input):
                    ind[1+len(cols_output)+n] = self.idx_col_routes+c
                    val[1+len(cols_output)+n] = -1
                glp_set_mat_row(self.prob, self.idx_row_storages+col, len(cols_output)+len(cols_input), ind, val)
            
            self.routes = routes
            self.supplys = supplys
            self.demands = demands
            self.storages = storages
            
            model.dirty = False
        else:
            routes = self.routes
            supplys = self.supplys
            demands = self.demands
            storages = self.storages

        # update route properties
        for col, route in enumerate(routes):
            supply = route[0]
            demand = route[-1]
            # TODO: cost should be for every node in route, not just supply
            cost = supply.get_cost(timestep)
            glp_set_obj_coef(self.prob, self.idx_col_routes+col, cost)
        
        # update supply properties
        for col, supply in enumerate(supplys):
            min_flow = inf_to_dbl_max(supply.get_min_flow(timestep))
            max_flow = inf_to_dbl_max(supply.get_max_flow(timestep))
            glp_set_row_bnds(self.prob, self.idx_row_supplys+col, constraint_type(min_flow, max_flow), min_flow, max_flow)

        # update demand properties
        for col, demand in enumerate(demands):
            min_flow = inf_to_dbl_max(demand.get_min_flow(timestep))
            max_flow = inf_to_dbl_max(demand.get_max_flow(timestep))
            cost = demand.get_cost(timestep)
            glp_set_col_bnds(self.prob, self.idx_col_demands+col, constraint_type(min_flow, max_flow), min_flow, max_flow)
            glp_set_obj_coef(self.prob, self.idx_col_demands+col, cost)

        # update storage node constraint
        for col, storage in enumerate(storages):
            max_volume = storage.get_max_volume(timestep)
            current_volume = storage._volume
            # change in storage cannot be more than the current volume or
            # result in maximum volume being exceeded
            lb = -current_volume
            ub = max_volume-current_volume
            glp_set_row_bnds(self.prob, self.idx_row_storages+col, constraint_type(lb, ub), lb, ub)

        # attempt to solve the linear programme
        glp_simplex(self.prob, &self.smcp)
        
        status = glp_get_status(self.prob)
        if status != GLP_OPT:
            raise RuntimeError(status_string[status])
        
        route_flow = [glp_get_col_prim(self.prob, col+1) for col in range(0, len(routes))]
        change_in_storage = [glp_get_row_prim(self.prob, self.idx_row_storages+col) for col in range(0, len(storages))]
        
        result = {}
        
        for route, flow in zip(routes, route_flow):
            for node in route:
                node.commit(flow)

        return route_flow, change_in_storage
