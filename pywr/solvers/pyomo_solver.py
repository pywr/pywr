from . import Solver
from pywr._core import BaseInput, BaseOutput, BaseLink, Storage, VirtualStorage, AggregatedNode
from pywr.core import ModelStructureError
from pyomo.environ import (ConcreteModel, Set, Var, NonNegativeReals, RangeSet,
                           Constraint, Objective, minimize)
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import numpy as np


class PyomoSolver(Solver):
    name = 'pyomo'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_routes_flows = kwargs.pop('save_routes_flows', False)
        self.all_nodes = None
        self.routes = None
        self.cross_domain_routes = None
        self._stats = None

    def setup(self, model):

        self.all_nodes = list(sorted(model.graph.nodes(), key=lambda n: n.fully_qualified_name))
        if not self.all_nodes:
            raise ModelStructureError("Model is empty")

        self.routes = model.find_all_routes(BaseInput, BaseOutput, valid=(BaseLink, BaseInput, BaseOutput))
        # Find cross-domain routes
        self.cross_domain_routes = model.find_all_routes(BaseOutput, BaseInput, max_length=2, domain_match='different')
        if len(self.routes) == 0:
            raise ModelStructureError("Model has no valid routes")

        # create a lookup for the cross-domain routes.
        cross_domain_cols = {}
        for cross_domain_route in self.cross_domain_routes:
            # These routes are only 2 nodes. From output to input
            output, input = cross_domain_route
            # note that the conversion factor is not time varying
            conv_factor = input.get_conversion_factor()
            input_cols = [(n, conv_factor) for n, route in enumerate(self.routes)
                          if route[0] is input]
            # create easy lookup for the route columns this output might
            # provide cross-domain connection to
            if output in cross_domain_cols:
                cross_domain_cols[output].extend(input_cols)
            else:
                cross_domain_cols[output] = input_cols
        self.cross_domain_cols = cross_domain_cols

        self.num_routes = len(self.routes)
        self.num_scenarios = len(model.scenarios.combinations)
        if self.save_routes_flows:
            # If saving flows this array needs to be 2D (one for each scenario)
            self.routes_flows_array = np.empty(shape=(self.num_scenarios, self.num_routes))
        else:
            # Otherwise the array can just be used to store a single solve to save some memory
            self.routes_flows_array = np.empty(shape=(self.num_routes))


    def reset(self):
        pass

    def make_model(self, model, scenario_index, pyomo_model=None):

        # Make a new pyomo_model if not given one already.
        if pyomo_model is None:
            pyomo_model = ConcreteModel()

        self._make_sets(pyomo_model)
        self._make_variables(pyomo_model)
        self._make_constraints(pyomo_model, scenario_index)
        self._make_objective(pyomo_model, scenario_index)

        return pyomo_model

    def _make_sets(self, pyomo_model):
        """ Create the pyomo sets. """

        non_storages = []
        storages = []
        virtual_storages = []
        aggregated_with_factors = []
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
                aggregated.append(some_node)

        if len(non_storages) == 0:
            raise ModelStructureError("Model has no non-storage nodes")

        self.non_storages = non_storages
        self.storages = storages
        self.virtual_storages = virtual_storages
        self.aggregated = aggregated
        self.aggregated_with_factors = aggregated_with_factors

        # Nodal sets
        # We make these zero based arrays
        pyomo_model.non_storages = range(len(non_storages))
        pyomo_model.storages = range(len(storages))
        pyomo_model.virtual_storages = range(len(virtual_storages))
        pyomo_model.aggregated_with_factors = range(len(aggregated_with_factors))
        pyomo_model.aggregated = range(len(aggregated))
        # Route sets
        pyomo_model.routes = range(len(self.routes))
        pyomo_model.cross_domain_routes = range(len(self.cross_domain_routes))
        pyomo_model.cross_domain_cols = range(len(self.cross_domain_cols))

    def _make_variables(self, pyomo_model):

        pyomo_model.route_vars = Var(pyomo_model.routes, within=NonNegativeReals)

    def _make_objective(self, pyomo_model, scenario_index):

        def obj(m):
            # update route properties
            costs = []
            for col, route in enumerate(self.routes):
                cost = route[0].get_cost(scenario_index)
                for node in route[1:-1]:
                    if isinstance(node, BaseLink):
                        cost += node.get_cost(scenario_index)
                cost += route[-1].get_cost(scenario_index)
                costs.append(m.route_vars[col]*cost)
            return sum(costs)

        pyomo_model.obj = Objective(rule=obj, sense=minimize)

    def _make_constraints(self, pyomo_model, scenario_index):

        self._make_non_storage_constraints(pyomo_model, scenario_index)
        self._make_storage_constraints(pyomo_model, scenario_index)
        self._make_virtual_storage_constraints(pyomo_model, scenario_index)
        self._make_aggregated_with_factors_constraints(pyomo_model, scenario_index)
        self._make_aggregated_constraints(pyomo_model, scenario_index)

    def _make_non_storage_constraints(self, pyomo_model, scenario_index):
        routes = self.routes

        def constraint_cols(node):
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

            return cols

        def ns_constraint(m, i):
            node = self.non_storages[i]
            cols = constraint_cols(node)
            min_flow = node.get_min_flow(scenario_index)
            max_flow = node.get_max_flow(scenario_index)

            if len(cols) == 0:
                return Constraint.Feasible

            return min_flow <= sum([m.route_vars[c] for c in cols]) <= max_flow

        pyomo_model.non_storage_constraints = Constraint(pyomo_model.non_storages, rule=ns_constraint)

        # This is a bit of hack and only works because of new ordered dictionaries
        cross_domain_nodes = list(self.cross_domain_cols.keys())
        def cd_constraint(m, i):
            node = cross_domain_nodes[i]
            cols = constraint_cols(node)
            col_vals = self.cross_domain_cols[node]

            if len(cols) == 0 and len(col_vals) == 0:
                return Constraint.Feasible

            return 0.0 <= sum([m.route_vars[c]*-1 for c in cols]) + \
                   sum([m.route_vars[c]*1./v for c, v in col_vals]) <= 0.0

        pyomo_model.cross_domain_constraints = Constraint(pyomo_model.cross_domain_cols, rule=cd_constraint)

    def _make_storage_constraints(self, pyomo_model, scenario_index):
        routes = self.routes
        timestep = self.timestep

        def s_constraint(m, i):
            storage = self.storages[i]

            cols_output = [n for n, route in enumerate(routes)
                           if route[-1] in storage.outputs and route[0] not in storage.inputs]
            cols_input = [n for n, route in enumerate(routes)
                          if route[0] in storage.inputs and route[-1] not in storage.outputs]

            max_volume = storage.get_max_volume(scenario_index)
            min_volume = storage.get_min_volume(scenario_index)

            if max_volume == min_volume:
                lb = ub = 0.0
            else:
                avail_volume = max(storage.volume[scenario_index.global_id] - min_volume, 0.0)
                # change in storage cannot be more than the current volume or
                # result in maximum volume being exceeded
                lb = -avail_volume/timestep.days
                ub = max(max_volume - storage.volume[scenario_index.global_id], 0.0) / timestep.days

                if abs(lb) < 1e-8:
                    lb = 0.0
                if abs(ub) < 1e-8:
                    ub = 0.0

            if len(cols_output) == 0 and len(cols_input) == 0:
                return Constraint.Feasible

            return lb <= sum([m.route_vars[c] for c in cols_output]) - \
                             sum([m.route_vars[c] for c in cols_input]) <= ub

        pyomo_model.storage_constaints = Constraint(pyomo_model.storages, rule=s_constraint)

    def _make_virtual_storage_constraints(self, pyomo_model, scenario_index):
        routes = self.routes
        timestep = self.timestep

        def vs_constraint(m, i):

            storage = self.virtual_storages[i]

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

            max_volume = storage.get_max_volume(scenario_index)
            min_volume = storage.get_min_volume(scenario_index)

            if max_volume == min_volume:
                lb = ub = 0.0
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

            if len(cols) == 0:
                return Constraint.Feasible

            return lb <= sum(m.route_vars[c]*-f for c, f in cols.items()) <= ub

        pyomo_model.virtual_storage_constraints = Constraint(pyomo_model.virtual_storages, rule=vs_constraint)

    def _make_aggregated_with_factors_constraints(self, pyomo_model, scenario_index):
        routes = self.routes

        for n, agg_node in enumerate(self.aggregated_with_factors):

            nodes = agg_node.nodes
            factors = agg_node.factors
            assert(len(nodes) == len(factors))

            cols = []
            for node in nodes:
                cols.append([n for n, route in enumerate(routes) if node in route])

            # normalise factors
            f0 = factors[0]
            factors_norm = [f0/f for f in factors]

            # update matrix
            def agg_f_constraint(m, i):

                if len(cols[0]) == 0 and len(cols[i+1]) == 0:
                    return Constraint.Feasible

                return 0.0 <= sum([m.route_vars[c] for c in cols[0]]) \
                       - sum([m.route_vars[c]*factors_norm[i+1] for c in cols[i+1]]) \
                       <= 0
            # This is a bit weird
            setattr(pyomo_model, 'agg_factor_constraint_{}'.format(n),
                Constraint(range(len(agg_node.nodes)-1), rule=agg_f_constraint))

    def _make_aggregated_constraints(self, pyomo_model, scenario_index):
        routes = self.routes

        def ag_constraint(m, i):
            agg_node = self.aggregated[i]

            nodes = agg_node.nodes

            weights = agg_node.flow_weights
            if weights is None:
                weights = [1.0]*len(nodes)

            matrix = {}
            for some_node, w in zip(nodes, weights):
                for n, route in enumerate(routes):
                    if some_node in route:
                        matrix[n] = w

            min_flow = agg_node.get_min_flow(scenario_index)
            if abs(min_flow) < 1e-8:
                min_flow = 0.0
            max_flow = agg_node.get_max_flow(scenario_index)
            if abs(max_flow) < 1e-8:
                max_flow = 0.0

            if len(matrix) == 0:
                return Constraint.Feasible

            return min_flow <= sum(m.route_vars[c]*matrix[c] for c in sorted(matrix)) <= max_flow

        pyomo_model.aggregated_constraints = Constraint(pyomo_model.aggregated,
                                                        rule=ag_constraint)

    def solve(self, model):

        # Optimise
        # TODO make this configurable
        opt = SolverFactory('glpk')
        self.timestep = model.timestep

        # reset stats
        stats = {
            'total': 0.0,
            'lp_solve': 0.0,
            'result_update': 0.0,
            'bounds_update_nonstorage': 0.0,
            'bounds_update_storage': 0.0,
            'objective_update': 0.0,
            'number_of_rows': 'not implemented',
            'number_of_cols': 'not implemented',
            'number_of_nonzero': 'not implemented',
            'number_of_routes': len(self.routes),
            'number_of_nodes': len(self.all_nodes)
        }

        for scenario_index in model.scenarios.combinations:
            pyomo_model = self.make_model(model, scenario_index)
            results = opt.solve(pyomo_model)
            if results.solver.status == SolverStatus.ok and \
                    results.solver.termination_condition == TerminationCondition.optimal:
                self._apply_result(pyomo_model, scenario_index)
            elif results.solver.termination_condition == TerminationCondition.infeasible:
                raise RuntimeError('Pyomo solver failed due to infeasibility: "{}".'.format(
                                    results.solver.status))
            else:
                raise RuntimeError('Pyomo solver failed for reason other than infeasibility: "{}".'.format(
                                    results.solver.status))

        self._stats = stats

    @property
    def stats(self):
        return self._stats

    def _apply_result(self, pyomo_model, scenario_index):

        for col, route in enumerate(self.routes):
            flow = pyomo_model.route_vars[col].value
            # TODO make this cleaner.
            route[0].commit(scenario_index.global_id, flow)
            route[-1].commit(scenario_index.global_id, flow)
            for node in route[1:-1]:
                if isinstance(node, BaseLink):
                    node.commit(scenario_index.global_id, flow)

            if self.save_routes_flows:
                self.routes_flows_array[scenario_index.global_id, col] = pyomo_model.route_vars[col].value
            else:
                self.routes_flows_array[col] = pyomo_model.route_vars[col].value
