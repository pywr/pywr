# cython: profile=False
from libc.float cimport DBL_MAX
from libc.stdlib cimport malloc, free

from pywr._core import BaseInput, BaseOutput, BaseLink
from pywr._core cimport *

cdef extern from "lpsolve/lp_lib.h":
    cdef struct _lprec:
        pass
    ctypedef _lprec lprec
    ctypedef unsigned char MYBOOL
    cdef unsigned char FALSE
    cdef unsigned char TRUE
    cdef unsigned char FR
    cdef unsigned char LE
    cdef unsigned char GE
    cdef unsigned char EQ
    cdef int OPTIMAL
    ctypedef double REAL
    cdef int CRITICAL
    cdef int FULL
    cdef int PRESOLVE_ROWS
    cdef int PRESOLVE_COLS
    cdef int PRESOLVE_LINDEP

    lprec* make_lp(int rows, int columns)
    MYBOOL resize_lp(lprec *lp, int rows, int columns)
    char* get_statustext(lprec *lp, int statuscode);
    MYBOOL get_variables(lprec *lp, REAL *var)
    MYBOOL get_ptr_variables(lprec *lp, REAL **var)

    MYBOOL add_constraint(lprec *lp, REAL *row, int constr_type, REAL rh);
    MYBOOL add_constraintex(lprec *lp, int count, REAL *row, int *colno, int constr_type, REAL rh);
    MYBOOL set_add_rowmode(lprec *lp, MYBOOL turnon)
    void delete_lp(lprec *lp)
    void free_lp(lprec **plp)
    void set_maxim(lprec *lp)
    void set_minim(lprec *lp)

    MYBOOL add_column(lprec *lp, REAL *column);
    MYBOOL add_columnex(lprec *lp, int count, REAL *column, int *rowno);

    MYBOOL set_obj(lprec *lp, int colnr, REAL value);
    MYBOOL set_bounds(lprec *lp, int colnr, REAL lower, REAL upper);
    MYBOOL set_constr_type(lprec *lp, int rownr, int con_type)
    MYBOOL set_bounds(lprec *lp, int colnr, REAL lower, REAL upper)
    MYBOOL set_lowbo(lprec *lp, int colnr, REAL value)
    MYBOOL set_row(lprec *lp, int rownr, REAL *row)
    MYBOOL set_rowex(lprec *lp, int rownr, int count, REAL *row, int *colno)
    MYBOOL set_rh(lprec *lp, int rownr, REAL value);
    MYBOOL set_rh_range(lprec *lp, int rownr, REAL deltavalue);
    int get_Norig_rows(lprec *lp)
    int get_Nrows(lprec *lp)
    int get_Lrows(lprec *lp)

    int get_Norig_columns(lprec *lp)
    int get_Ncolumns(lprec *lp)
    int solve(lprec *lp);
    void print_lp(lprec *lp);
    void set_verbose(lprec *lp, int verbose);
    void set_presolve(lprec *lp, int presolvemode, int maxloops);
    int get_presolve(lprec *lp);
    int get_presolveloops(lprec *lp);

cdef inline int set_row_bnds(lprec *prob, int row, double a, double b):
    if a == b:
        set_constr_type(prob, row, EQ)
        set_rh(prob, row, a)
        #set_rh_range(prob, row, 0.0)
    elif b == DBL_MAX:
        if a == -DBL_MAX:
            set_constr_type(prob, row, FR)
            set_rh(prob, row, 0.0)
            #set_rh_range(prob, row, 0.0)
        else:
            set_constr_type(prob, row, GE)
            set_rh(prob, row, a)
            #set_rh_range(prob, row, 0.0)
    elif a == -DBL_MAX:
        set_constr_type(prob, row, LE)
        set_rh(prob, row, b)
        #set_rh_range(prob, row, 0.0)
    else:
        set_constr_type(prob, row, LE)
        set_rh(prob, row, b)
        set_rh_range(prob, row, b-a)

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

cdef class CythonLPSolveSolver:
    cdef lprec *prob
    cdef int idx_col_routes
    cdef int idx_col_demands
    cdef int idx_row_supplys
    cdef int idx_row_demands
    cdef int idx_row_cross_domain
    cdef int idx_row_storages

    cdef object routes
    cdef object supplys
    cdef object demands
    cdef object storages

    def __cinit__(self):
        # create a new problem
        self.prob = make_lp(0, 0)
        if self.prob is NULL:
            raise MemoryError()
        set_verbose(self.prob, CRITICAL)

    def __dealloc__(self):
        if self.prob is not NULL:
            # free the problem
            delete_lp(self.prob)


    cpdef object setup(self, model):
        cdef Node supply
        cdef Node demand
        cdef Node node
        cdef double min_flow
        cdef double max_flow
        cdef double cost
        cdef double avail_volume
        cdef int col
        cdef int* ind
        cdef double* val
        cdef double lb
        cdef double ub
        cdef Timestep timestep
        cdef int status
        cdef cross_domain_col
        cdef MYBOOL ret
        cdef REAL *ptr_var

        routes = model.find_all_routes(BaseInput, BaseOutput, valid=(BaseLink, BaseInput, BaseOutput))
        # Find cross-domain routes
        cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')

        supplys = []
        demands = []
        storages = []
        for some_node in model.nodes():
            if isinstance(some_node, (BaseInput, BaseLink)):
                supplys.append(some_node)
            if isinstance(some_node, BaseOutput):
                demands.append(some_node)
            if isinstance(some_node, Storage):
                storages.append(some_node)

        assert(routes)
        assert(supplys)
        assert(demands)

        # clear the previous problem
        ret = resize_lp(self.prob, 0, 0)
        if ret == FALSE:
            return -1
        set_minim(self.prob)


        # add a column for each route and demand
        self.idx_col_routes = get_Norig_columns(self.prob)+1
        self.idx_col_demands = self.idx_col_routes + len(routes)
        ret = resize_lp(self.prob, 0, len(routes)+len(demands))
        ret = set_add_rowmode(self.prob, TRUE)

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
            ret = add_columnex(self.prob, 0, NULL, NULL)
            ret = set_lowbo(self.prob, self.idx_col_routes+col, 0.0)
        for col, demand in enumerate(demands):
            ret = add_columnex(self.prob, 0, NULL, NULL)
            ret = set_lowbo(self.prob, self.idx_col_demands+col, 0.0)

        # constrain supply minimum and maximum flow
        self.idx_row_supplys = get_Norig_rows(self.prob)+1
        ret = resize_lp(self.prob, get_Norig_rows(self.prob)+len(supplys), get_Norig_columns(self.prob))
        for col, supply in enumerate(supplys):
            # TODO is this a bit hackish??
            if isinstance(supply, BaseInput):
                cols = [n for n, route in enumerate(routes) if route[0] is supply]
            else:
                cols = [n for n, route in enumerate(routes) if supply in route]

            ind = <int*>malloc(len(cols) * sizeof(int))
            val = <double*>malloc(len(cols) * sizeof(double))
            for n, c in enumerate(cols):
                ind[n] = 1+c
                val[n] = 1
            ret = add_constraintex(self.prob, len(cols), val, ind, GE, 0.0)

            #set_rowex(self.prob, self.idx_row_supplys+col, len(cols)+1, val, ind)
            #set_row_bnds(self.prob, self.idx_row_supplys+col, 0.0, 0.0)

        # link supply and demand variables
        self.idx_row_demands = get_Norig_rows(self.prob)+1
        ret = resize_lp(self.prob, get_Norig_rows(self.prob)+len(demands), get_Norig_columns(self.prob))
        if len(cross_domain_cols) > 0:
            self.idx_row_cross_domain = get_Norig_rows(self.prob)+1
            ret = resize_lp(self.prob, get_Norig_rows(self.prob)+len(cross_domain_cols), get_Norig_columns(self.prob))
        cross_domain_col = 0
        for col, demand in enumerate(demands):
            cols = [n for n, route in enumerate(routes) if route[-1] is demand]
            ind = <int*>malloc((1+len(cols)) * sizeof(int))
            val = <double*>malloc((1+len(cols)) * sizeof(double))
            for n, c in enumerate(cols):
                ind[n] = 1+c
                val[n] = 1
            ind[len(cols)] = self.idx_col_demands+col
            val[len(cols)] = -1
            ret = add_constraintex(self.prob, len(cols)+1, val, ind, EQ, 0.0)

        for col, demand in enumerate(demands):
            # Add constraint for cross-domain routes
            # i.e. those from a demand to a supply
            if demand in cross_domain_cols:
                col_vals = cross_domain_cols[demand]
                ind = <int*>malloc((1+len(col_vals)) * sizeof(int))
                val = <double*>malloc((1+len(col_vals)) * sizeof(double))
                for n, (c, v) in enumerate(col_vals):
                    ind[n] = 1+c
                    val[n] = 1./v
                ind[len(col_vals)] = self.idx_col_demands+col
                val[len(col_vals)] = -1
                add_constraintex(self.prob, len(col_vals)+1, val, ind, EQ, 0.0)
                cross_domain_col += 1

        # storage
        if len(storages):
            self.idx_row_storages = get_Norig_rows(self.prob)+1
            ret = resize_lp(self.prob, get_Norig_rows(self.prob)+len(storages), get_Norig_columns(self.prob))
        for col, storage in enumerate(storages):
            cols_output = [n for n, demand in enumerate(demands) if demand in storage.outputs]
            cols_input = [n for n, route in enumerate(routes) if route[0] in storage.inputs]
            ind = <int*>malloc((len(cols_output)+len(cols_input)) * sizeof(int))
            val = <double*>malloc((len(cols_output)+len(cols_input)) * sizeof(double))
            for n, c in enumerate(cols_output):
                ind[n] = self.idx_col_demands+c
                val[n] = 1
            for n, c in enumerate(cols_input):
                ind[len(cols_output)+n] = self.idx_col_routes+c
                val[len(cols_output)+n] = -1
            add_constraintex(self.prob, len(cols_output)+len(cols_input), val, ind, EQ, 0.0)

        ret = set_add_rowmode(self.prob, FALSE)
        self.routes = routes
        self.supplys = supplys
        self.demands = demands
        self.storages = storages

    cpdef object solve(self, model):
        cdef int[:] scenario_combination
        cdef int scenario_id
        for scenario_id, scenario_combination in enumerate(model.scenarios.combinations):
            self._solve_scenario(model, scenario_id, scenario_combination)

    cdef object _solve_scenario(self, model, int scenario_id, int[:] scenario_indices):
        cdef Node supply
        cdef Node demand
        cdef Node node
        cdef double min_flow
        cdef double max_flow
        cdef double cost
        cdef double avail_volume
        cdef int col
        cdef int* ind
        cdef double* val
        cdef double lb
        cdef double ub
        cdef Timestep timestep
        cdef int status
        cdef cross_domain_col
        cdef MYBOOL ret
        cdef REAL *ptr_var

        timestep = model.timestep

        routes = self.routes
        supplys = self.supplys
        demands = self.demands
        storages = self.storages

        # update route properties
        for col, route in enumerate(routes):
            cost = route[0].get_cost(timestep, scenario_indices)
            for node in route[1:-1]:
                if isinstance(node, BaseLink):
                    cost += node.get_cost(timestep, scenario_indices)
            set_obj(self.prob, self.idx_col_routes+col, cost)

        # update supply properties
        for col, supply in enumerate(supplys):
            min_flow = inf_to_dbl_max(supply.get_min_flow(timestep, scenario_indices))
            max_flow = inf_to_dbl_max(supply.get_max_flow(timestep, scenario_indices))
            set_row_bnds(self.prob, self.idx_row_supplys+col, min_flow, max_flow)


        # update demand properties
        for col, demand in enumerate(demands):
            min_flow = inf_to_dbl_max(demand.get_min_flow(timestep, scenario_indices))
            max_flow = inf_to_dbl_max(demand.get_max_flow(timestep, scenario_indices))
            cost = demand.get_cost(timestep, scenario_indices)
            set_bounds(self.prob, self.idx_col_demands+col, min_flow, max_flow)
            set_obj(self.prob, self.idx_col_demands+col, cost)

        # update storage node constraint
        for col, storage in enumerate(storages):
            max_volume = storage.get_max_volume(timestep, scenario_indices)
            avail_volume = max(storage._volume[scenario_id] - storage.get_min_volume(timestep), 0.0)
            # change in storage cannot be more than the current volume or
            # result in maximum volume being exceeded
            lb = -avail_volume/timestep.days
            ub = (max_volume-storage._volume[scenario_id])/timestep.days
            set_row_bnds(self.prob, self.idx_row_storages+col, lb, ub)

        #print_lp(self.prob)
        #set_presolve(self.prob, PRESOLVE_ROWS | PRESOLVE_COLS, get_presolveloops(self.prob))
        # attempt to solve the linear programme
        status = solve(self.prob)

        if status != OPTIMAL:
            raise RuntimeError(get_statustext(self.prob, status))

        get_ptr_variables(self.prob, &ptr_var)

        route_flow = [ptr_var[col] for col in range(0, len(routes))]
        change_in_storage = []

        result = {}

        for route, flow in zip(routes, route_flow):
            # TODO make this cleaner.
            route[0].commit(scenario_id, flow)
            route[-1].commit(scenario_id, flow)
            for node in route[1:-1]:
                if isinstance(node, BaseLink):
                    node.commit(scenario_id, flow)

        return route_flow, change_in_storage
