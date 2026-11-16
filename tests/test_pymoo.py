import multiprocessing
import numpy as np
import pytest

pymoo = pytest.importorskip("pymoo")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.parallelization.starmap import StarmapParallelization
from pywr.optimisation.pymoo import PymooWrapper
from pywr.optimisation import clear_global_model_cache
import os

TEST_FOLDER = os.path.dirname(__file__)


@pytest.fixture()
def two_reservoir_wrapper():
    filename = os.path.join(TEST_FOLDER, "models", "two_reservoir.json")
    yield PymooWrapper(filename)
    clear_global_model_cache()


@pytest.fixture()
def two_reservoir_constrained_wrapper():
    filename = os.path.join(TEST_FOLDER, "models", "two_reservoir_constrained.json")
    yield PymooWrapper(filename)
    clear_global_model_cache()


def test_pymoo_init(two_reservoir_wrapper):
    """Test the initialisation of the pymoo problem."""
    p = two_reservoir_wrapper.problem

    assert p.n_var == 12
    assert p.n_obj == 2
    assert p.n_ieq_constr == 0
    assert p.n_eq_constr == 0


def test_pymoo_constrained_init(two_reservoir_constrained_wrapper):
    """Test the initialisation of a constrained pymoo problem."""
    p = two_reservoir_constrained_wrapper.problem

    assert p.n_var == 12
    assert p.n_obj == 2
    assert p.n_ieq_constr == 2  # double-bounded constraint expands to 2 inequalities
    assert p.n_eq_constr == 0


def test_pymoo_nsga2_step(two_reservoir_wrapper):
    """Undertake a single generation of NSGA2 with a small population."""
    algorithm = NSGA2(pop_size=10)
    minimize(two_reservoir_wrapper.problem, algorithm, ("n_gen", 1), verbose=False)


def test_pymoo_nsga2_step_constrained(two_reservoir_constrained_wrapper):
    """Undertake a single generation of NSGA2 on a constrained problem."""
    algorithm = NSGA2(pop_size=10)
    minimize(
        two_reservoir_constrained_wrapper.problem,
        algorithm,
        ("n_gen", 1),
        verbose=False,
    )


def test_pymoo_equality_constraint():
    """Test the equality constraint is evaluated as the difference from its bound."""

    class EqualityWrapper(PymooWrapper):
        def customise_model(self, model):
            for r in model.constraints:
                r.constraint_lower_bounds = 4.0
                r.constraint_upper_bounds = 4.0

    filename = os.path.join(TEST_FOLDER, "models", "two_reservoir_constrained.json")
    wrapper = EqualityWrapper(filename)
    try:
        p = wrapper.problem
        assert p.n_ieq_constr == 0
        assert p.n_eq_constr == 1

        x = 0.5 * (p.xl + p.xu)
        out = {}
        wrapper._pymoo_evaluate(x, out)
        value = wrapper.model_constraints[0].aggregated_value()
        np.testing.assert_allclose(out["H"], [value - 4.0])
    finally:
        clear_global_model_cache()


def test_pymoo_nsga2_step_parallel(two_reservoir_constrained_wrapper):
    """Undertake a single generation of NSGA2 with the population evaluated in parallel."""
    with multiprocessing.Pool(2) as pool:
        wrapper = PymooWrapper(
            two_reservoir_constrained_wrapper.pywr_model_json,
            elementwise_runner=StarmapParallelization(pool.starmap),
        )
        algorithm = NSGA2(pop_size=10)
        minimize(wrapper.problem, algorithm, ("n_gen", 1), verbose=False)
