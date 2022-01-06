import os
import numpy as np
import pytest
from platypus import NSGAII
from pywr.utils.bisect import BisectionSearchModel
from pywr.optimisation.platypus import PlatypusWrapper
from pywr.optimisation import clear_global_model_cache
from helpers import load_model

TEST_FOLDER = os.path.dirname(__file__)


def test_simple_bisection():
    """Test basic bisection search"""

    model = load_model("simple1_bisect.json", model_klass=BisectionSearchModel)
    assert isinstance(model, BisectionSearchModel)

    # Perform bisection search
    model.run()

    np.testing.assert_allclose(
        model.parameters["demand"].get_double_variables(), [17.5]
    )


def test_simple_infeasible_bisection():
    """Test infeasible bisection search"""

    model = load_model(
        "simple1_infeasible_bisect.json", model_klass=BisectionSearchModel
    )
    assert isinstance(model, BisectionSearchModel)

    # Perform bisection search
    with pytest.raises(ValueError):
        model.run()


@pytest.fixture()
def simple_bisection_problem():
    filename = os.path.join(TEST_FOLDER, "models", "simple1_bisect.json")
    yield PlatypusWrapper(filename, model_klass=BisectionSearchModel)
    # Clean up the
    clear_global_model_cache()
    # We force deallocation the cache here to prevent problems using process pools
    # with pytest.
    import gc

    gc.collect()


def test_platypus_init(simple_bisection_problem):
    """Test the initialisation of the platypus problem."""
    p = simple_bisection_problem

    assert p.problem.nvars == 1
    assert p.problem.nobjs == 1
    assert p.problem.nconstrs == 1
    # Check the correct model class is created by the wrapper.
    assert isinstance(p.model, BisectionSearchModel)


def test_platypus_nsgaii_step(simple_bisection_problem):
    """Undertake a single step of the NSGAII algorithm with a small population."""
    algorithm = NSGAII(simple_bisection_problem.problem, population_size=10)
    algorithm.step()
