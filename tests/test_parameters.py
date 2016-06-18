"""
Test for individual Parameter classes
"""
from pywr.core import Model, Timestep, Scenario, ScenarioIndex
from pywr.parameters import BaseParameter, ArrayIndexedParameter, ConstantScenarioParameter, \
    ArrayIndexedScenarioMonthlyFactorsParameter, MonthlyProfileParameter, DailyProfileParameter, \
    AggregatedParameter, load_parameter

import datetime
import numpy as np
import pytest
import itertools

@pytest.fixture
def model(solver):
    return Model(solver=solver)


def test_parameter_array_indexed(model):
    """
    Test ArrayIndexedParameter

    """
    A = np.arange(len(model.timestepper), dtype=np.float64)
    p = ArrayIndexedParameter(A)
    p.setup(model)
    # scenario indices (not used for this test)
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    for v, ts in zip(A, model.timestepper):
        np.testing.assert_allclose(p.value(ts, si), v)

    # Now check that IndexError is raised if an out of bounds Timestep is given.
    ts = Timestep(datetime.datetime(2016, 1, 1), 366, 1.0)
    with pytest.raises(IndexError):
        p.value(ts, si)


def test_parameter_constant_scenario(model):
    """
    Test ConstantScenarioParameter

    """
    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    p = ConstantScenarioParameter(scB, np.arange(scB.size, dtype=np.float64))
    p.setup(model)
    ts = model.timestepper.current
    # Now ensure the appropriate value is returned for the Scenario B indices.
    for i, (a, b) in enumerate(itertools.product(range(scA.size), range(scB.size))):
        si = ScenarioIndex(i, np.array([a, b], dtype=np.int32))
        np.testing.assert_allclose(p.value(ts, si), float(b))


def test_parameter_array_indexed_scenario_monthly_factors(model):
    """
    Test ArrayIndexedParameterScenarioMonthlyFactors

    """
    # Baseline timeseries data
    values = np.arange(len(model.timestepper), dtype=np.float64)

    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    # Random factors for each Scenario B value per month
    factors = np.random.rand(scB.size, 12)

    p = ArrayIndexedScenarioMonthlyFactorsParameter(scB, values, factors)
    p.setup(model)

    # Iterate in time
    for v, ts in zip(values, model.timestepper):
        imth = ts.datetime.month - 1
        # Now ensure the appropriate value is returned for the Scenario B indices.
        for i, (a, b) in enumerate(itertools.product(range(scA.size), range(scB.size))):
            f = factors[b, imth]
            si = ScenarioIndex(i, np.array([a, b], dtype=np.int32))
            np.testing.assert_allclose(p.value(ts, si), v*f)


def test_parameter_monthly_profile(model):
    """
    Test MonthlyProfileParameter

    """
    values = np.arange(12, dtype=np.float64)
    p = MonthlyProfileParameter(values)
    p.setup(model)

    # Iterate in time
    for ts in model.timestepper:
        imth = ts.datetime.month - 1
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        np.testing.assert_allclose(p.value(ts, si), values[imth])


def test_parameter_daily_profile(model):
    """
    Test DailyProfileParameter

    """
    values = np.arange(366, dtype=np.float64)
    p = DailyProfileParameter(values)
    p.setup(model)

    # Iterate in time
    for ts in model.timestepper:
        iday = ts.datetime.dayofyear - 1
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        np.testing.assert_allclose(p.value(ts, si), values[iday])


class TestAggregatedParameter:
    """ Tests for AggregatedParameter"""

    def test_min(self, model):
        # Add two scenarios
        scA = Scenario(model, 'Scenario A', size=2)
        scB = Scenario(model, 'Scenario B', size=5)

        values = np.arange(366, dtype=np.float64)
        p1 = DailyProfileParameter(values)
        p2 = ConstantScenarioParameter(scB, np.arange(scB.size, dtype=np.float64))

        p = AggregatedParameter([p1, ], agg_func='min')
        p.add(p2)

        p.setup(model)
        for ts in model.timestepper:
            iday = ts.datetime.dayofyear - 1
            for i in range(scB.size):
                si = ScenarioIndex(i, np.array([0, i], dtype=np.int32))
                np.testing.assert_allclose(p.value(ts, si), min(values[iday], i))

    def test_max(self, model):
        # Add two scenarios
        scA = Scenario(model, 'Scenario A', size=2)
        scB = Scenario(model, 'Scenario B', size=5)

        values = np.arange(366, dtype=np.float64)
        p1 = DailyProfileParameter(values)
        p2 = ConstantScenarioParameter(scB, np.arange(scB.size, dtype=np.float64))

        p = AggregatedParameter([p1, p2], agg_func='max')
        p.setup(model)

        for ts in model.timestepper:
            iday = ts.datetime.dayofyear - 1
            for i in range(scB.size):
                si = ScenarioIndex(i, np.array([0, i], dtype=np.int32))
                np.testing.assert_allclose(p.value(ts, si), max(values[iday], i))

    def test_load(self, model):
        """ Test load from JSON dict"""
        data = {
            "type": "aggregated",
            "agg_func": "product",
            "parameters": [
                {
                    "type": "constant",
                    "values": 0.8
                },
                {
                    "type": "monthlyprofile",
                    "values": list(range(12))
                }
            ]
        }

        p = load_parameter(model, data)
        # Correct instance is loaded
        assert isinstance(p, AggregatedParameter)

        # Test correct aggregation is performed
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        for mth in range(1, 13):
            ts = Timestep(datetime.datetime(2016, mth, 1), 366, 1.0)
            np.testing.assert_allclose(p.value(ts, si), (mth-1)*0.8)



def test_parameter_child_variables():

    p1 = BaseParameter()
    # Default parameter
    assert p1 not in p1.variables
    assert len(p1.variables) == 0
    assert p1.parent is None
    assert len(p1.children) == 0

    c1 = BaseParameter()
    c1.parent = p1
    assert len(p1.children) == 1
    assert c1 in p1.children
    assert c1.parent == p1

    assert p1 not in p1.variables
    assert len(p1.variables) == 0

    c1.is_variable = True
    # Now parent should see find child as a variable
    assert p1 not in p1.variables
    assert c1 in p1.variables
    assert len(p1.variables) == 1

    # Test third level
    c2 = BaseParameter()
    c2.parent = c1
    c2.is_variable = True
    assert p1 not in p1.variables
    assert c1 in p1.variables
    assert c2 in p1.variables
    assert len(p1.variables) == 2

    # Disable middle parameter as variable
    c1.is_variable = False
    assert p1 not in p1.variables
    assert c1 not in p1.variables
    assert c2 in p1.variables
    assert len(p1.variables) == 1

    # Disable parent
    c1.parent = None

    assert len(p1.variables) == 0
    assert len(p1.children) == 0
    # Child variables still OK.
    assert c1 not in c1.variables
    assert c2 in c1.variables
    assert len(c1.variables) == 1


def test_with_a_better_name():

    data = {
        'type': 'scaledprofile',
        'scale': 50.0,
        'profile': {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                {
                    'type': 'monthlyprofile',
                    'values': [1.0]*12
                },
                {
                    'type': 'controlcurvemonthlyprofile',
                    'control_curve': [0.8, 0.6],
                    'values': [[1.05]*12,
                               [1.1]*12]
                }
            ]
        }
    }