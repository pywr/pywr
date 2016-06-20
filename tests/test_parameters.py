"""
Test for individual Parameter classes
"""
from pywr.core import Model, Timestep, Scenario, ScenarioIndex
from pywr.parameters import BaseParameter, ArrayIndexedParameter, ConstantScenarioParameter, \
    ArrayIndexedScenarioMonthlyFactorsParameter, MonthlyProfileParameter, DailyProfileParameter, \
    MinimumParameterCollection, MaximumParameterCollection, DataFrameParameter, load_parameter

import datetime
import numpy as np
import pandas as pd
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


def test_parameter_array_indexed_json_load(model, tmpdir):
    """Test ArrayIndexedParameter can be loaded from json dict"""
    # Daily time-step
    index = pd.date_range('2015-01-01', periods=365, freq='D', name='date')
    df = pd.DataFrame(np.arange(365), index=index, columns=['data'])
    df_path = tmpdir.join('df.csv')
    df.to_csv(str(df_path))

    data = {
        'type': 'arrayindexed',
        'url': str(df_path),
        'index_col': 'date',
        'parse_dates': True,
        'column': 'data',
    }

    p = load_parameter(model, data)
    p.setup(model)

    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    for v, ts in enumerate(model.timestepper):
        np.testing.assert_allclose(p.value(ts, si), v)

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


def test_parameter_min(model):
    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    values = np.arange(366, dtype=np.float64)
    p1 = DailyProfileParameter(values)
    p2 = ConstantScenarioParameter(scB, np.arange(scB.size, dtype=np.float64))

    p = MinimumParameterCollection([p1, ])
    p.add(p2)

    p.setup(model)
    for ts in model.timestepper:
        iday = ts.datetime.dayofyear - 1
        for i in range(scB.size):
            si = ScenarioIndex(i, np.array([0, i], dtype=np.int32))
            np.testing.assert_allclose(p.value(ts, si), min(values[iday], i))


def test_parameter_max(model):
    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    values = np.arange(366, dtype=np.float64)
    p1 = DailyProfileParameter(values)
    p2 = ConstantScenarioParameter(scB, np.arange(scB.size, dtype=np.float64))

    p = MaximumParameterCollection([p1, p2])
    p.setup(model)

    for ts in model.timestepper:
        iday = ts.datetime.dayofyear - 1
        for i in range(scB.size):
            si = ScenarioIndex(i, np.array([0, i], dtype=np.int32))
            np.testing.assert_allclose(p.value(ts, si), max(values[iday], i))


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



def test_parameter_df_upsampling(model):
    """ Test that the `DataFrameParameter` can upsample data from a `pandas.DataFrame` and return that correctly
    """
    # scenario indices (not used for this test)
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    # Use a 7 day timestep for this test and run 2015
    model.timestepper.delta = datetime.timedelta(7)
    model.timestepper.start = pd.to_datetime('2015-01-01')
    model.timestepper.end = pd.to_datetime('2015-12-31')

    # Daily time-step
    index = pd.date_range('2015-01-01', periods=365, freq='D')
    series = pd.Series(np.arange(365), index=index)

    p = DataFrameParameter(series)
    p.setup(model)

    A = series.resample('7D').mean()
    for v, ts in zip(A, model.timestepper):
        np.testing.assert_allclose(p.value(ts, si), v)

    model.reset()
    # Daily time-step that requires aligning
    index = pd.date_range('2014-12-31', periods=366, freq='D')
    series = pd.Series(np.arange(366), index=index)

    p = DataFrameParameter(series)
    p.setup(model)

    # offset the resample appropriately for the test
    A = series[1:].resample('7D').mean()
    for v, ts in zip(A, model.timestepper):
        np.testing.assert_allclose(p.value(ts, si), v)

    model.reset()
    # Daily time-step that is not covering the require range
    index = pd.date_range('2015-02-01', periods=365, freq='D')
    series = pd.Series(np.arange(365), index=index)

    p = DataFrameParameter(series)
    with pytest.raises(ValueError):
        p.setup(model)

    model.reset()
    # Daily time-step that is not covering the require range
    index = pd.date_range('2014-11-01', periods=365, freq='D')
    series = pd.Series(np.arange(365), index=index)

    p = DataFrameParameter(series)
    with pytest.raises(ValueError):
        p.setup(model)


def test_parameter_df_upsampling_multiple_columns(model):
    """ Test that the `DataFrameParameter` works with multiple columns that map to a `Scenario`
    """
    scA = Scenario(model, 'A', size=20)
    scB = Scenario(model, 'B', size=2)
    # scenario indices (not used for this test)

    # Use a 7 day timestep for this test and run 2015
    model.timestepper.delta = datetime.timedelta(7)
    model.timestepper.start = pd.to_datetime('2015-01-01')
    model.timestepper.end = pd.to_datetime('2015-12-31')

    # Daily time-step
    index = pd.date_range('2015-01-01', periods=365, freq='D')
    df = pd.DataFrame(np.random.rand(365, 20), index=index)

    p = DataFrameParameter(df, scenario=scA)
    p.setup(model)

    A = df.resample('7D', axis=0).mean()
    for v, ts in zip(A.values, model.timestepper):
        np.testing.assert_allclose([p.value(ts, ScenarioIndex(i, np.array([i], dtype=np.int32))) for i in range(20)], v)

    p = DataFrameParameter(df, scenario=scB)
    with pytest.raises(ValueError):
        p.setup(model)


def test_parameter_df_json_load(model, tmpdir):

    # Daily time-step
    index = pd.date_range('2015-01-01', periods=365, freq='D', name='date')
    df = pd.DataFrame(np.random.rand(365), index=index, columns=['data'])
    df_path = tmpdir.join('df.csv')
    df.to_csv(str(df_path))

    data = {
        'type': 'dataframe',
        'url': str(df_path),
        'index_col': 'date',
        'parse_dates': True,
    }

    p = load_parameter(model, data)
    p.setup(model)


