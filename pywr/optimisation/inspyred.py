import numpy as np
import inspyred
from . import BaseOptimisationWrapper


class InspyredOptimisationWrapper(BaseOptimisationWrapper):
    """ A wrapper to enable optimisation using inspyred.

    A generator, bounder and evaluator method are provided to use with the inspyred algorithms.
    """
    def generator(self, random, args):

        values = []
        for var in self.model_variables:
            lower = []
            upper = []
            if var.double_size > 0:
                lower.append(var.get_double_lower_bounds())
                upper.append(var.get_double_upper_bounds())

            if var.integer_size > 0:
                lower.append(var.get_integer_lower_bounds())
                upper.append(var.get_integer_upper_bounds())

            lower = np.concatenate(lower)
            upper = np.concatenate(upper)

            if len(lower) != len(upper):
                raise ValueError('Upper and lower bounds are different lengths. Malformed bound data from Parameter:'
                                 ' "{}"'.format(var.name))

            for i in range(len(lower)):
                values.append(random.uniform(lower[i], upper[i]))

        return values

    def evaluator(self, candidates, args):
        fitness = []
        for candidate in candidates:
            for ivar, var in enumerate(self.model_variables):
                l, u = self.model_variable_map[ivar], self.model_variable_map[ivar+1]
                if var.double_size > 0:
                    dj = slice(l, l + var.double_size)
                    var.set_double_variables(np.array(candidate[dj]))

                if var.integer_size > 0:
                    ij = slice(l + var.double_size, u)
                    # Convert integer variables to ints
                    ints = np.rint(np.array(candidate[ij])).astype(np.int)
                    var.set_integer_variables(ints)

            self.model.reset()
            run_stats = self.model.run()

            fitness.append(inspyred.ec.emo.Pareto([r.aggregated_value() for r in self.model_objectives]))
        return fitness

    def bounder(self, candidate, args):
        for ivar, var in enumerate(self.model_variables):
            lower = []
            upper = []
            if var.double_size > 0:
                lower.append(var.get_double_lower_bounds())
                upper.append(var.get_double_upper_bounds())

            if var.integer_size > 0:
                lower.append(var.get_integer_lower_bounds())
                upper.append(var.get_integer_upper_bounds())

            lower = np.concatenate(lower)
            upper = np.concatenate(upper)

            j = slice(self.model_variable_map[ivar], self.model_variable_map[ivar+1])
            candidate[j] = np.minimum(upper, np.maximum(lower, candidate[j]))
        return candidate
