import pytest

pymoo = pytest.importorskip("pymoo")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
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
