cdef extern from "glpk.h":
    ctypedef struct glp_prob:
        pass
    ctypedef struct glp_smcp:
        int msg_lev
        int meth
        double tol_bnd
        double tol_dj
        double tol_piv
        double obj_ll
        double obj_ul
        int it_lim
        int tm_lim
        int presolve


cdef class GLPKSolver:
    cdef glp_prob* prob
    cdef glp_smcp smcp

    cpdef object solve(self, model)
    # cpdef object dump(self, filename, format)


cdef class BasisManager:
    cdef int[:, :] row_stat
    cdef int[:, :] col_stat
    cdef init_basis(self, glp_prob* prob, int nscenarios)
    cdef save_basis(self, glp_prob* prob, int global_id)
    cdef set_basis(self, glp_prob* prob, bint is_first_solve, int global_id)


cdef int simplex(glp_prob *P, glp_smcp parm)
cdef set_obj_coef(glp_prob *P, int j, double coef)
cdef set_row_bnds(glp_prob *P, int i, int type, double lb, double ub)
cdef set_col_bnds(glp_prob *P, int i, int type, double lb, double ub)
cdef set_mat_row(glp_prob *P, int i, int len, int* ind, double* val)
