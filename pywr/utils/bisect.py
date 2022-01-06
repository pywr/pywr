from ..model import Model
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BisectionSearchModel(Model):
    """A Pywr model that performs a bisection search.

    When a `BisectionSearchModel` is run it performs multiple simulations using a bisection
    algorithm to find the largest value of a parameter that satisfies all defined constraints. The bisection
    proceeds until a minimum gap (epsilon) is reached. After the bisection the best feasible value is re-run
    to ensure the model ends with corresponding recorder values. If no feasible solutions are found then
    an error is raised.

    Parameters
    ==========
    bisect_parameter : Parameter
        The parameter to vary during the bisection process. This parameter must have a `double_size` of 1
        and have lower and upper bounds defined; the bisection is undertaken between these bounds.
    bisect_epsilon : float
        The termination criterion for the bisection process. When the bisection has narrowed to a gap
         of less than the `bisect_epsilon` the process is terminated.
    error_on_infeasible : bool (default True)
        If true a ValueError is raised if no feasible solution is found during the bisection process. If false
        no error is raised if there is no feasible solution, and a solution using the lower bounds of the
        bisection parameter is the final result.
    """

    def __init__(self, **kwargs):
        self.bisect_parameter = kwargs.pop("bisect_parameter", None)
        self.bisect_epsilon = kwargs.pop("bisect_epsilon", None)
        self.error_on_infeasible = kwargs.pop("error_on_infeasible", True)
        super().__init__(**kwargs)

    @classmethod
    def _load_from_dict(cls, data, model=None, path=None, solver=None, **kwargs):
        model = super()._load_from_dict(
            data, model=None, path=None, solver=None, **kwargs
        )

        try:
            bisect_data = data["bisection"]
        except KeyError:
            pass
        else:
            model.bisect_parameter = bisect_data.get("parameter", None)
            model.bisect_epsilon = bisect_data.get("epsilon", None)
            model.error_on_infeasible = bisect_data.get("error_on_infeasible", True)
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
            raise ValueError(
                "Bisection is only supported using a parameter with only a single double variable "
                "(e.g ConstantParameter)."
            )
        # Use the bounds of the parameter for the bisection search space
        min_value = param.get_double_lower_bounds()[0]
        max_value = param.get_double_upper_bounds()[0]
        if min_value >= max_value:
            raise ValueError(
                "Minimum bounds of the bisection parameter must be strictly greater than its "
                "maximum bounds."
            )

        if self.bisect_epsilon <= 0.0:
            raise ValueError("Bisection epsilon value must be greater than zero.")

        logger.info(
            f'Starting bisection using parameter "{self.bisect_parameter}" with epsilon of '
            f"{self.bisect_epsilon:.4f}."
        )

        best_feasible = -np.inf
        while (max_value - min_value) > self.bisect_epsilon:
            # Compute the current value to try as the middle of the bounds
            current_value = (max_value + min_value) / 2
            logger.debug(
                f"Performing bisection run with value: {current_value:.4f}; "
                f"bounds [{min_value:.4f}, {max_value:.4f}]"
            )
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

        # Ensure a feasible solution is found.
        if np.isneginf(best_feasible):
            if self.error_on_infeasible:
                raise ValueError(
                    "No feasible solutions found during bisection. Trying lowering the bounds on the "
                    "bisection parameter."
                )
            else:
                # If no feasible the "best" value is the lower bounds.
                best_feasible = min_value
        # Finally, rerun at the best found feasible value
        # This is slightly inefficient, but ensures the model's recorders are at the best_feasible
        # value's results at the end of the simulation.
        param.set_double_variables(np.array([best_feasible]))
        ret = super().run()
        logger.info(
            f'Bisection complete! Highest feasible value of "{self.bisect_parameter}" '
            f"found: {best_feasible:.4f}"
        )
        return ret
