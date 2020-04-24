import pytest
platypus = pytest.importorskip("platypus")

from pywr.optimisation.platypus import PlatypusWrapper
from pywr.optimisation import clear_global_model_cache
from platypus import NSGAII, ProcessPoolEvaluator
import os


TEST_FOLDER = os.path.dirname(__file__)


class TwoReservoirWrapper(PlatypusWrapper):
    def customise_model(self, model):
        self.the_model_has_been_customised = True


@pytest.fixture()
def two_reservoir_problem():
    filename = os.path.join(TEST_FOLDER, 'models', 'two_reservoir.json')
    yield TwoReservoirWrapper(filename)
    # Clean up the
    clear_global_model_cache()
    # We force deallocation the cache here to prevent problems using process pools
    # with pytest.
    import gc
    gc.collect()


@pytest.fixture()
def two_reservoir_constrained_problem():
    filename = os.path.join(TEST_FOLDER, 'models', 'two_reservoir_constrained.json')
    yield TwoReservoirWrapper(filename)
    # Clean up the
    clear_global_model_cache()
    # We force deallocation the cache here to prevent problems using process pools
    # with pytest.
    import gc
    gc.collect()


def test_platypus_init(two_reservoir_problem):
    """ Test the initialisation of the platypus problem. """
    p = two_reservoir_problem

    assert p.problem.nvars == 12
    assert p.problem.nobjs == 2
    assert p.problem.nconstrs == 0
    # Check the `customise_model` method has been called.
    assert p.the_model_has_been_customised


def test_platypus_constrained_init(two_reservoir_constrained_problem):
    """ Test the initialisation of a constrained platypus problem. """
    p = two_reservoir_constrained_problem

    assert p.problem.nvars == 12
    assert p.problem.nobjs == 2
    assert p.problem.nconstrs == 2
    # Check the `customise_model` method has been called.
    assert p.the_model_has_been_customised


def test_platypus_nsgaii_step(two_reservoir_problem):
    """ Undertake a single step of the NSGAII algorithm with a small population. """
    algorithm = NSGAII(two_reservoir_problem.problem, population_size=10)
    algorithm.step()


def test_platypus_nsgaii_step(two_reservoir_constrained_problem):
    """ Undertake a single step of the NSGAII algorithm with a small population. """
    algorithm = NSGAII(two_reservoir_constrained_problem.problem, population_size=10)
    algorithm.step()


def test_platypus_nsgaii_process_pool(two_reservoir_problem):
    """ Undertake a single step of the NSGAII algorithm with a ProcessPool. """
    with ProcessPoolEvaluator(2) as evaluator:
        algorithm = NSGAII(two_reservoir_problem.problem, population_size=50, evaluator=evaluator)
        algorithm.run(10)


