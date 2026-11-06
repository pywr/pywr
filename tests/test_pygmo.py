import numpy as np
import pytest

pygmo = pytest.importorskip("pygmo")

import pygmo as pg
from pywr.optimisation.pygmo import PygmoWrapper
from pywr.optimisation import clear_global_model_cache
import os

TEST_FOLDER = os.path.dirname(__file__)


@pytest.fixture()
def two_reservoir_wrapper():
    """Two reservoir test optimisation problem in PygmoWrapper."""
    filename = os.path.join(TEST_FOLDER, "models", "two_reservoir.json")
    yield PygmoWrapper(filename)
    # Clean up the
    clear_global_model_cache()


@pytest.fixture()
def two_reservoir_constrained_wrapper():
    """Two reservoir test optimisation problem in PygmoWrapper."""
    filename = os.path.join(TEST_FOLDER, "models", "two_reservoir_constrained.json")
    yield PygmoWrapper(filename)
    # Clean up the
    clear_global_model_cache()


def test_pygmo_single_generation(two_reservoir_wrapper):
    """Simple pygmo wrapper test."""
    wrapper = two_reservoir_wrapper
    prob = pg.problem(wrapper)
    algo = pg.algorithm(pg.moead(gen=1))

    pg.mp_island.init_pool(2)
    isl = pg.island(algo=algo, prob=prob, size=50, udi=pg.mp_island())
    isl.evolve(1)


def test_pygmo_single_generation_constrained(two_reservoir_constrained_wrapper):
    """Simple pygmo wrapper test of constrained problem."""
    wrapper = two_reservoir_constrained_wrapper
    prob = pg.problem(wrapper)
    algo = pg.algorithm(pg.moead(gen=1))

    pg.mp_island.init_pool(2)
    isl = pg.island(algo=algo, prob=prob, size=50, udi=pg.mp_island())
    isl.evolve(1)


def test_pygmo_equality_constraint():
    """Test the equality constraint is evaluated as the difference from its bound."""

    class EqualityWrapper(PygmoWrapper):
        def customise_model(self, model):
            for r in model.constraints:
                r.constraint_lower_bounds = 4.0
                r.constraint_upper_bounds = 4.0

    filename = os.path.join(TEST_FOLDER, "models", "two_reservoir_constrained.json")
    wrapper = EqualityWrapper(filename)
    try:
        assert wrapper.get_nec() == 1
        assert wrapper.get_nic() == 0

        lower, upper = wrapper.get_bounds()
        x = 0.5 * (np.array(lower) + np.array(upper))
        f = wrapper.fitness(x)
        value = wrapper.model_constraints[0].aggregated_value()
        # Objectives are followed by the equality constraints
        np.testing.assert_allclose(f[wrapper.get_nobj()], value - 4.0)
    finally:
        clear_global_model_cache()
