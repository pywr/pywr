from libc.setjmp cimport jmp_buf

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
    ctypedef struct glp_mpscp:
        pass

    int GLP_MIN = 1  # minimization
    int GLP_MAX = 2  # maximization

    int GLP_FR = 1  # free (unbounded) variable
    int GLP_LO = 2  # variable with lower bound
    int GLP_UP = 3  # variable with upper bound
    int GLP_DB = 4  # double-bounded variable
    int GLP_FX = 5  # fixed variable

    int GLP_UNDEF = 1  # solution is undefined
    int GLP_FEAS = 2  # solution is feasible
    int GLP_INFEAS = 3  # solution is infeasible
    int GLP_NOFEAS = 4  # no feasible solution exists
    int GLP_OPT = 5  # solution is optimal
    int GLP_UNBND = 6  # solution is unbounded

    int GLP_MSG_OFF = 0  # no output
    int GLP_MSG_ERR = 1  # warning and error messages only
    int GLP_MSG_ON = 2  # normal output
    int GLP_MSG_ALL = 3  # full output
    int GLP_MSG_DBG = 4  # debug output

    int GLP_PRIMAL = 1  # use primal simplex
    int GLP_DUALP = 2  # use dual; if it fails, use primal
    int GLP_DUAL = 3  # use dual simplex

    int GLP_MPS_DECK = 1  # fixed (ancient)
    int GLP_MPS_FILE = 2  # free (modern)

    int GLP_ON = 1
    int GLP_OFF = 0

    glp_prob* glp_create_prob()
    void glp_init_smcp(glp_smcp *parm)
    void glp_erase_prob(glp_prob *P)
    void glp_delete_prob(glp_prob *P)
    void glp_free(void *ptr)
    int glp_free_env()

    int glp_add_rows(glp_prob *P, int nrs)
    int glp_add_cols(glp_prob *P, int ncs)

    void glp_set_mat_row(glp_prob *P, int i, int len, const int ind[], const double val[])
    void glp_set_mat_col(glp_prob *P, int j, int len, const int ind[], const double val[])

    void glp_set_row_bnds(glp_prob *P, int i, int type, double lb, double ub)
    void glp_set_row_name(glp_prob *P, int i, const char *name)

    void glp_set_col_bnds(glp_prob *P, int j, int type, double lb, double ub)
    void glp_set_col_name(glp_prob *P, int j, const char *name)

    void glp_set_obj_coef(glp_prob *P, int j, double coef)

    void glp_set_obj_dir(glp_prob *P, int dir)

    void glp_std_basis(glp_prob *P)
    void glp_adv_basis(glp_prob *P, int flags)
    int glp_simplex(glp_prob *P, const glp_smcp *parm)

    int glp_get_status(glp_prob *P)
    int glp_term_out(int flag)
    void glp_term_hook(int (*func)(void *info, const char *s), void *info)
    void glp_error_hook(void (*func)(void *info), void *info)

    double glp_get_row_prim(glp_prob *P, int i)
    double glp_get_col_prim(glp_prob *P, int j)

    int glp_get_num_rows(glp_prob *P)
    int glp_get_num_cols(glp_prob *P)
    int glp_get_num_nz(glp_prob *P)

    int glp_write_mps(glp_prob *P, int fmt, const glp_mpscp *parm, const char *fname)
    int glp_write_lp(glp_prob *lp, const void *parm, const char *fname)
    int glp_write_prob(glp_prob *P, int flags, const char *fname)

    int glp_get_row_stat(glp_prob *P, int i)
    int glp_get_col_stat(glp_prob *P, int i)
    void glp_set_row_stat(glp_prob *P, int i, int state)
    void glp_set_col_stat(glp_prob *P, int i, int state)
