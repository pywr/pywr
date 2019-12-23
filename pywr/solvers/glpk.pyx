include "glpk.pxi"

import numpy as np
cimport numpy as np

cdef class GLPKSolver:
    def __cinit__(self):
        self.prob = glp_create_prob()

    def __dealloc__(self):
        glp_delete_prob(self.prob)

    def __init__(self, use_presolve=False, time_limit=None, iteration_limit=None, message_level="error"):
        self.use_presolve = use_presolve
        self.set_solver_options(time_limit, iteration_limit, message_level)
        glp_term_hook(term_hook, NULL)

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


cdef int simplex(glp_prob *P, glp_smcp parm):
    return glp_simplex(P, &parm)


cdef set_obj_coef(glp_prob *P, int j, double coef):
    IF SOLVER_DEBUG:
        assert np.isfinite(coef)
        if abs(coef) < 1e-9:
            if abs(coef) != 0.0:
                print(j, coef)
                assert False
    glp_set_obj_coef(P, j, coef)


cdef set_row_bnds(glp_prob *P, int i, int type, double lb, double ub):
    IF SOLVER_DEBUG:
        assert np.isfinite(lb)
        assert np.isfinite(ub)
        assert lb <= ub
        if abs(lb) < 1e-9:
            if abs(lb) != 0.0:
                print(i, type, lb, ub)
                assert False
        if abs(ub) < 1e-9:
            if abs(ub) != 0.0:
                print(i, type, lb, ub)
                assert False
    glp_set_row_bnds(P, i, type, lb, ub)


cdef set_col_bnds(glp_prob *P, int i, int type, double lb, double ub):
    IF SOLVER_DEBUG:
        assert np.isfinite(lb)
        assert np.isfinite(ub)
        assert lb <= ub
    glp_set_col_bnds(P, i, type, lb, ub)


cdef set_mat_row(glp_prob *P, int i, int len, int* ind, double* val):
    IF SOLVER_DEBUG:
        cdef int j
        for j in range(len):
            assert np.isfinite(val[j+1])
            assert abs(val[j+1]) > 1e-6
            assert ind[j+1] > 0
    glp_set_mat_row(P, i, len, ind, val)


