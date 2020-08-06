from ..model import Model
import numpy as np


class BisectionSearchModel(Model):
    """A Pywr model that performs a bisection search.

    When a `BisectionSearchModel` is run it performs multiple simulations using a bisection
    algorithm to find the largest value of a parameter that satisfies all defined constraints.
    """
    def __init__(self, **kwargs):
        self.bisect_parameter = kwargs.pop('bisect_parameter', None)
        self.bisect_epsilon = kwargs.pop('bisect_epsilon', None)
        super().__init__(**kwargs)

    @classmethod
    def _load_from_dict(cls, data, model=None, path=None, solver=None, **kwargs):
        model = super()._load_from_dict(data, model=None, path=None, solver=None, **kwargs)

        try:
            bisect_data = data['bisection']
        except KeyError:
            pass
        else:
            model.bisect_parameter = bisect_data.get('parameter', None)
            model.bisect_epsilon = bisect_data.get('epsilon', None)
        return model

    def run(self):
        """Perform a bisection search using repeated simulation."""

        if self.bisect_epsilon is None:
            raise ValueError("Bisection epsilon is not defined.")
        if self.bisect_parameter is None:
            raise ValueError("Bisection parameter is not defined.")

        # Setup the model first
        self.setup()
        # Get the bisection parameter
        param = self.parameters[self.bisect_parameter]
        if param.double_size != 1 or param.integer_size != 0:
            raise ValueError("Bisection is only supported using a parameter with only a single double variable "
                             "(e.g ConstantParameter).")
        # Use the bounds of the parameter for the bisection search space
        min_value = param.get_double_lower_bounds()[0]
        max_value = param.get_double_upper_bounds()[0]
        if min_value >= max_value:
            raise ValueError("Minimum bounds of the bisection parameter must be strictly greater than its "
                             "maximum bounds.")

        if self.bisect_epsilon <= 0.0:
            raise ValueError("Bisection epsilon value must be greater than zero.")

        best_feasible = -np.inf
        while (max_value - min_value) > self.bisect_epsilon:
            # Compute the current value to try as the middle of the bounds
            current_value = (max_value + min_value) / 2

            print(min_value, current_value, max_value)
            # Update parameter & perform the simulation
            param.set_double_variables(np.array([current_value]))
            super().run()
            # Check for constraint failures
            if self.is_feasible():
                # No constraint failures; tighten lower bounds
                min_value = current_value
                best_feasible = max(best_feasible, current_value)
            else:
                # Failed; tighten upper bounds
                max_value = current_value

        # Finally, rerun at the best found feasible value
        # This is slightly inefficient, but ensures the model's recorders are at the best_feasible
        # value's results at the end of the simulation.
        param.set_double_variables(np.array([best_feasible]))
        return super().run()


