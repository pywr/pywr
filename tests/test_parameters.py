"""
Test for individual Parameter classes
"""
from pywr.core import Model, Timestep, Scenario
from pywr.parameters import (ParameterArrayIndexed, ParameterConstantScenario, ParameterArrayIndexedScenarioMonthlyFactors,
                             ParameterMonthlyProfile, ParameterDailyProfile, ParameterMinimumCollection, ParameterMaximumCollection)

import datetime
import numpy as np
import pytest
import itertools

@pytest.fixture
def model(solver):
    return Model(solver=solver)


def test_parameter_array_indexed(model):
    """
    Test ParameterArrayIndexed

    """
    A = np.arange(len(model.timestepper), dtype=np.float64)
    p = ParameterArrayIndexed(A)
    p.setup(model)
    # scenario indices (not used for this test)
    si = np.array([0], dtype=np.int32)
    for v, ts in zip(A, model.timestepper):
        np.testing.assert_allclose(p.value(ts, si), v)

    # Now check that IndexError is raised if an out of bounds Timestep is given.
    ts = Timestep(datetime.datetime(2016, 1, 1), 366, 1.0)
    with pytest.raises(IndexError):
        p.value(ts, si)


def test_parameter_constant_scenario(model):
    """
    Test ParameterConstantScenario

    """
    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    p = ParameterConstantScenario(scB, np.arange(scB.size, dtype=np.float64))
    p.setup(model)
    ts = model.timestepper.current
    # Now ensure the appropriate value is returned for the Scenario B indices.
    for a, b in itertools.product(range(scA.size), range(scB.size)):
        np.testing.assert_allclose(p.value(ts, np.array([a, b], dtype=np.int32)), float(b))


def test_parameter_array_indexed_scenario_monthly_factors(model):
    """
    Test ParameterArrayIndexedScenarioMonthlyFactors

    """
    # Baseline timeseries data
    values = np.arange(len(model.timestepper), dtype=np.float64)

    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    # Random factors for each Scenario B value per month
    factors = np.random.rand(scB.size, 12)

    p = ParameterArrayIndexedScenarioMonthlyFactors(scB, values, factors)
    p.setup(model)

    # Iterate in time
    for v, ts in zip(values, model.timestepper):
        imth = ts.datetime.month - 1
        # Now ensure the appropriate value is returned for the Scenario B indices.
        for a, b in itertools.product(range(scA.size), range(scB.size)):
            f = factors[b, imth]
            np.testing.assert_allclose(p.value(ts, np.array([a, b], dtype=np.int32)), v*f)


def test_parameter_monthly_profile(model):
    """
    Test ParameterMonthlyProfile

    """
    values = np.arange(12, dtype=np.float64)
    p = ParameterMonthlyProfile(values)
    p.setup(model)

    # Iterate in time
    for ts in model.timestepper:
        imth = ts.datetime.month - 1
        np.testing.assert_allclose(p.value(ts, np.array([0], dtype=np.int32)), values[imth])


def test_parameter_daily_profile(model):
    """
    Test ParameterDailyProfile

    """
    values = np.arange(366, dtype=np.float64)
    p = ParameterDailyProfile(values)
    p.setup(model)

    # Iterate in time
    for ts in model.timestepper:
        iday = ts.datetime.dayofyear - 1
        np.testing.assert_allclose(p.value(ts, np.array([0], dtype=np.int32)), values[iday])


def test_parameter_min(model):
    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    values = np.arange(366, dtype=np.float64)
    p1 = ParameterDailyProfile(values)
    p2 = ParameterConstantScenario(scB, np.arange(scB.size, dtype=np.float64))

    p = ParameterMinimumCollection([p1, p2])
    p.setup(model)

    for ts in model.timestepper:
        iday = ts.datetime.dayofyear - 1
        for i in range(scB.size):
            np.testing.assert_allclose(p.value(ts, np.array([0, i], dtype=np.int32)), min(values[iday], i))


def test_parameter_max(model):
    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    values = np.arange(366, dtype=np.float64)
    p1 = ParameterDailyProfile(values)
    p2 = ParameterConstantScenario(scB, np.arange(scB.size, dtype=np.float64))

    p = ParameterMaximumCollection([p1, p2])
    p.setup(model)

    for ts in model.timestepper:
        iday = ts.datetime.dayofyear - 1
        for i in range(scB.size):
            np.testing.assert_allclose(p.value(ts, np.array([0, i], dtype=np.int32)), max(values[iday], i))