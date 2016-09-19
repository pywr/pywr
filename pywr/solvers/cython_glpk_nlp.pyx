from libc.stdlib cimport malloc, free

from pywr._core import BaseInput, BaseOutput, BaseLink
from pywr._core cimport *

include "glpk_graph.pxi"

class NoFeasibleSolutionError(Exception):
    pass
class InvalidDataError(Exception):
    pass
class IntegerOverflowError(Exception):
    pass

infinity = 10**6
cdef int inf = <int>infinity

# offsets of node data
cdef int v_rhs = 0
# offsets for arc data
cdef int a_low = 0
cdef int a_cap = sizeof(double) * 1
cdef int a_cost = sizeof(double) * 2
cdef int a_x = sizeof(double) * 3
cdef int a_rc = sizeof(double) * 4

cdef class CythonGLPKNLPSolver:
    cdef glp_graph *G
    
    cpdef object setup(self, model):
        pass # TODO

    cpdef object solve(self, model):
        #print('solve')
        for scenario_index in model.scenarios.combinations:
            self._solve_scenario(model, scenario_index)

    cdef object _solve_scenario(self, model, ScenarioIndex scenario_index):
        cdef double max_flow
        cdef glp_arc *a
        cdef glp_vertex *balance_vertex
        cdef double balance = 0
        
        timestep = model.timestep
        
        if self.G != NULL:
            glp_delete_graph(self.G)
        self.G = glp_create_graph(sizeof(v_data), sizeof(a_data))
        
        nodes = list(model.graph.nodes())
        edges = list(model.graph.edges())
        
        num_edges = len(model.graph.edges())
        num_nodes = len(model.graph.nodes())
        
        glp_add_vertices(self.G, <int>(num_nodes*2 + 1))
        
        balance_vertex = self.G.v[num_nodes*2 + 1]
        balance_vertex.data = malloc(sizeof(v_data))
        
        for n, node in enumerate(nodes):
            self.G.v[n+1].data = malloc(sizeof(v_data))  # FIXME: memory leak
            
            # set default rhs for nodes (in and out)
            (<v_data*>self.G.v[n+1].data).rhs = 0
            (<v_data*>self.G.v[n*2+1].data).rhs = 0
            
            # connect node.in -> node.out
            a = glp_add_arc(self.G, n+1, num_nodes+n+1)
            a.data = malloc(sizeof(a_data))
            (<a_data*>a.data).cap = inf
            (<a_data*>a.data).cost = 0
            (<a_data*>a.data).low = 0
            
            # surplus arc (???)
            if isinstance(node, BaseInput):
                a = glp_add_arc(self.G, 1+n, 1+num_nodes*2)
                
                a.data = malloc(sizeof(a_data))
                (<a_data*>a.data).cap = inf
                (<a_data*>a.data).cost = 99999
                (<a_data*>a.data).low = 0
            
            # deficit arc
            if isinstance(node, BaseOutput):
                a = glp_add_arc(self.G, balance_vertex.i, 1+n)
                
                a.data = malloc(sizeof(a_data))
                (<a_data*>a.data).cap = inf
                (<a_data*>a.data).cost = 99998
                (<a_data*>a.data).low = 0
        
        for n, edge in enumerate(edges):
            node_a, node_b = edge
            index_a = nodes.index(node_a) # FIXME
            index_b = nodes.index(node_b) # FIXME
            a = glp_add_arc(self.G, index_a*2+1, index_b+1) # A.out -> B.in
            
            a.data = malloc(sizeof(a_data))
            (<a_data*>a.data).cap = inf
            (<a_data*>a.data).cost = 0
            (<a_data*>a.data).low = 0
        
        print('nv', self.G.nv)
        print('na', self.G.na)
        
        for n, node in enumerate(nodes):
            if isinstance(node, BaseInput):
                max_flow = node.get_max_flow(timestep, scenario_index)
                (<v_data*>self.G.v[n+1].data).rhs = max_flow
                print('input', max_flow)
                balance += max_flow
            elif isinstance(node, BaseOutput):
                max_flow = node.get_max_flow(timestep, scenario_index)
                (<v_data*>self.G.v[n*2+1].data).rhs = -max_flow
                print('output', -max_flow)
                balance -= max_flow
        
        print('balance', -balance)
        (<v_data*>balance_vertex.data).rhs = -balance

        cdef double test
        for n in range(0, self.G.nv):
            test = (<v_data*>self.G.v[1+n].data).rhs
            print('rhs[{}] {}'.format(n+1, test))
        
        cdef int crash = 0
        cdef double sol
        cdef int ret

        # solve the minimum cost flow problem
        ret = glp_mincost_relax4(self.G, v_rhs, a_low, a_cap, a_cost, crash, &sol, a_x, a_rc)
        
        print('ret',ret)
        
        if ret != 0:
            if ret == GLP_ENOPFS:
                raise NoFeasibleSolutionError("No (primal) feasible solution exists.")
            elif ret == GLP_EDATA:
                raise InvalidDataError("Unable to start the search, because some problem data are either not integer-valued or out of range.")
            elif ret == GLP_ERANGE:
                raise IntegerOverflowError("Unable to start the search because of integer overflow.")
