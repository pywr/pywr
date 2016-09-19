cdef extern from "glpk.h":
    cdef struct glp_graph:
        void *pool
        char *name
        int nv_max
        int nv
        int na
        glp_vertex **v
        void *index
        int v_size
        int a_size
    cdef struct glp_vertex:
        int i
        char *name
        void *entry
        void *data
        void *temp
        # line below triggers a cython bug
        # glp_arc *in
        glp_arc *out
    cdef struct glp_arc:
        glp_vertex *tail
        glp_vertex *head
        void *data
        void *temp
        glp_arc *t_prev
        glp_arc *t_next
        glp_arc *h_prev
        glp_arc *h_next

    glp_graph *glp_create_graph(int v_size, int a_size)
    int glp_add_vertices(glp_graph *G, int nadd)
    glp_arc *glp_add_arc(glp_graph *G, int i, int j)
    void glp_delete_graph(glp_graph *G)
    
    int glp_mincost_relax4(glp_graph *G, int v_rhs, int a_low, int a_cap,
          int a_cost, int crash, double *sol, int a_x, int a_rc)
    
    cdef enum:
        GLP_ENOPFS
        GLP_EDATA
        GLP_ERANGE

ctypedef struct v_data:
    double rhs
ctypedef struct a_data:
    double low
    double cap
    double cost
    double x
    double rc

cdef inline v_data *node(glp_vertex *v) nogil:
    return <v_data *>(v.data)
cdef inline a_data *arc(glp_arc *a) nogil:
    return <a_data *>(a.data)
