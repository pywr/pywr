from pywr.recorders import TotalFlowNodeRecorder
from pywr.parameters import ConstantParameter
from pywr.optimisation.platypus import PlatypusWrapper, clear_global_model_cache
from platypus import NSGAII, ProcessPoolEvaluator
import pygmo as pg
from pywr.optimisation.pygmo import PygmoWrapper
import pytest
import os
from fixtures import simple_storage_model


TEST_FOLDER = os.path.dirname(__file__)


@pytest.fixture()
def optimisation_model(simple_storage_model):
    model = simple_storage_model

    # Modify the model to have two variables.
    inpt = model.nodes['Input']
    inpt.max_flow = ConstantParameter(model, inpt.max_flow, lower_bounds=0.0, upper_bounds=10.0, name='Input flow',
                                      is_variable=True)

    otpt = model.nodes['Output']
    otpt.max_flow = ConstantParameter(model, otpt.max_flow, lower_bounds=0.0, upper_bounds=10.0, name='Output flow',
                                      is_variable=True)

    # And two objectives.
    TotalFlowNodeRecorder(model, inpt, name='Total inflow', is_objective='maximise')
    TotalFlowNodeRecorder(model, otpt, name='Total outflow', is_objective='maximise')

    return model


def test_variables(optimisation_model):
    """ Test that the number of variables, objectives and constraints are correct. """

    model = optimisation_model

    assert len(model.variables) == 2
    assert len(model.objectives) == 2
    assert len(model.constraints) == 0


class TwoReservoirWrapper(PlatypusWrapper):
    def customise_model(self, model):
        self.the_model_has_been_customised = True


@pytest.fixture()
def two_reservoir_problem():
    filename = os.path.join(TEST_FOLDER, 'models', 'two_reservoir.json')
    yield TwoReservoirWrapper(filename)
    # Clean up the
    clear_global_model_cache()

def test_platypus_init(two_reservoir_problem):
    """ Test the initialisation of the platypus problem. """
    p = two_reservoir_problem

    assert p.problem.nvars == 12
    assert p.problem.nobjs == 2
    assert p.problem.nconstrs == 0
    # Check the `customise_model` method has been called.
    assert p.the_model_has_been_customised


def test_platypus_nsgaii_step(two_reservoir_problem):
    """ Undertake a single step of the NSGAII algorithm with a small population. """
    algorithm = NSGAII(two_reservoir_problem.problem, population_size=10)
    algorithm.step()


def test_platypus_nsgaii_process_pool(two_reservoir_problem):
    """ Undertake a single step of the NSGAII algorithm with a ProcessPool. """
    with ProcessPoolEvaluator(2) as evaluator:
        algorithm = NSGAII(two_reservoir_problem.problem, population_size=50, evaluator=evaluator)
        algorithm.run(10)


@pytest.fixture()
def two_reservoir_pygmo_wrapper():
    """ Two reservoir test optimisation problem in PygmoWrapper. """
    filename = os.path.join(TEST_FOLDER, 'models', 'two_reservoir.json')
    yield PygmoWrapper(filename)
    # Clean up the
    clear_global_model_cache()


def test_pygmo_single_generation(two_reservoir_pygmo_wrapper):
    """ Simple pygmo wrapper test. """
    wrapper = two_reservoir_pygmo_wrapper
    prob = pg.problem(wrapper)
    algo = pg.algorithm(pg.moead(gen=1))

    pg.mp_island.init_pool(2)
    isl = pg.island(algo=algo, prob=prob, size=50, udi=pg.mp_island())
    isl.evolve(1)
