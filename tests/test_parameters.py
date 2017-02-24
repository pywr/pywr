"""
Test for individual Parameter classes
"""
from __future__ import division
from pywr.core import Model, Timestep, Scenario, ScenarioIndex, Storage, Link, Input, Output
from pywr.parameters import (Parameter, ArrayIndexedParameter, ConstantScenarioParameter,
    ArrayIndexedScenarioMonthlyFactorsParameter, MonthlyProfileParameter, DailyProfileParameter,
    DataFrameParameter, AggregatedParameter, ConstantParameter, CachedParameter,
    IndexParameter, AggregatedIndexParameter, RecorderThresholdParameter, ScenarioMonthlyProfileParameter,
    Polynomial1DParameter, Polynomial2DStorageParameter,
    FunctionParameter, AnnualHarmonicSeriesParameter, load_parameter)

from pywr.recorders import Recorder
from fixtures import simple_linear_model, simple_storage_model
from helpers import load_model

import os
import datetime
import numpy as np
import pandas as pd
import pytest
import itertools
from numpy.testing import assert_allclose

TEST_DIR = os.path.dirname(__file__)

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

def test_parameter_array_indexed_scenario_monthly_factors_json(model):
    model.path = os.path.join(TEST_DIR, "models")
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=3)
    
    p1 = ArrayIndexedScenarioMonthlyFactorsParameter.load(model, {
        "scenario": "Scenario A",
        "values": list(range(32)),
        "factors": [list(range(1, 13)),list(range(13, 25))],
    })

    p2 = ArrayIndexedScenarioMonthlyFactorsParameter.load(model, {
        "scenario": "Scenario B",
        "values": {
            "url": "timeseries1.csv",
            "index_col": "Timestamp",
            "column": "Data",
        },
        "factors": {
            "url": "monthly_profiles.csv",
            "index_col": "scenario",
        },
    })

    node1 = Input(model, "node1", max_flow=p1)
    node2 = Input(model, "node2", max_flow=p2)
    nodeN = Output(model, "nodeN", max_flow=None, cost=-1)
    node1.connect(nodeN)
    node2.connect(nodeN)

    model.timestepper.start = "2015-01-01"
    model.timestepper.end = "2015-01-31"
    model.run()


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


class TestScenarioMonthlyProfileParameter:

    def test_init(self, model):
        scenario = Scenario(model, 'A', 10)
        values = np.random.rand(10, 12)

        p = ScenarioMonthlyProfileParameter(scenario, values)

        p.setup(model)
        # Iterate in time
        for ts in model.timestepper:
            imth = ts.datetime.month - 1
            for i in range(scenario.size):
                si = ScenarioIndex(i, np.array([i], dtype=np.int32))
                np.testing.assert_allclose(p.value(ts, si), values[i, imth])

    def test_json(self, solver):
        model = load_model('scenario_monthly_profile.json', solver=solver)

        # check first day initalised
        assert (model.timestepper.start == datetime.datetime(2015, 1, 1))

        # check results
        supply1 = model.nodes['supply1']

        # Multiplication factors
        factors = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        ])

        for expected in (23.92, 22.14, 22.57, 24.97, 27.59):
            model.step()
            imth = model.timestepper.current.month - 1
            assert_allclose(supply1.flow, expected*factors[:, imth], atol=1e-7)

def test_parameter_daily_profile(model):
    """
    Test DailyProfileParameter

    """
    values = np.arange(366, dtype=np.float64)
    p = DailyProfileParameter(values)
    p.setup(model)

    # Iterate in time
    for ts in model.timestepper:
        month = ts.datetime.month
        day = ts.datetime.day
        iday = int((datetime.datetime(2016, month, day) - datetime.datetime(2016, 1, 1)).days)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        np.testing.assert_allclose(p.value(ts, si), values[iday])

def test_daily_profile_leap_day(model):
    """Test behaviour of daily profile parameter for leap years
    """
    inpt = Input(model, "input")
    otpt = Output(model, "otpt", max_flow=None, cost=-999)
    inpt.connect(otpt)
    inpt.max_flow = DailyProfileParameter(np.arange(0, 366, dtype=np.float64))

    # non-leap year
    model.timestepper.start = pd.to_datetime("2015-01-01")
    model.timestepper.end = pd.to_datetime("2015-12-31")
    model.run()
    assert_allclose(inpt.flow, 365) # NOT 364

    # leap year
    model.timestepper.start = pd.to_datetime("2016-01-01")
    model.timestepper.end = pd.to_datetime("2016-12-31")
    model.run()
    assert_allclose(inpt.flow, 365)

class TestAnnualHarmonicSeriesParameter:
    """ Tests for `AnnualHarmonicSeriesParameter` """
    def test_single_harmonic(self, model):

        p1 = AnnualHarmonicSeriesParameter(0.5, [0.25], [np.pi/4])
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))

        for ts in model.timestepper:
            doy = (ts.datetime.dayofyear - 1)/365
            np.testing.assert_allclose(p1.value(ts, si), 0.5 + 0.25*np.cos(doy*2*np.pi + np.pi/4))

    def test_double_harmonic(self, model):
        p1 = AnnualHarmonicSeriesParameter(0.5, [0.25, 0.3], [np.pi/4, np.pi/3])
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))

        for ts in model.timestepper:
            doy = (ts.datetime.dayofyear - 1) /365
            expected = 0.5 + 0.25*np.cos(doy*2*np.pi + np.pi / 4) + 0.3*np.cos(doy*4*np.pi + np.pi/3)
            np.testing.assert_allclose(p1.value(ts, si), expected)

    def test_load(self, model):

        data = {
            "type": "annualharmonicseries",
            "mean": 0.5,
            "amplitudes": [0.25],
            "phases": [np.pi/4]
        }

        p1 = load_parameter(model, data)

        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        for ts in model.timestepper:
            doy = (ts.datetime.dayofyear - 1) / 365
            np.testing.assert_allclose(p1.value(ts, si), 0.5 + 0.25 * np.cos(doy * 2 * np.pi + np.pi / 4))


class TestAggregatedParameter:
    """Tests for AggregatedParameter"""

    funcs = {"min": np.min, "max": np.max, "mean": np.mean, "median": np.median, "sum": np.sum}

    @pytest.mark.parametrize("agg_func", ["min", "max", "mean", "median", "sum"])
    def test_agg(self, model, agg_func):
        scenarioA = Scenario(model, "Scenario A", size=2)
        scenarioB = Scenario(model, "Scenario B", size=5)

        values = np.arange(366, dtype=np.float64)
        p1 = DailyProfileParameter(values)
        p2 = ConstantScenarioParameter(scenarioB, np.arange(scenarioB.size, dtype=np.float64))

        p = AggregatedParameter([p1, p2], agg_func=agg_func)

        func = TestAggregatedParameter.funcs[agg_func]

        p1.setup(model)
        p2.setup(model)
        p.setup(model)

        for ts in model.timestepper:
            for i in range(scenarioB.size):
                si = ScenarioIndex(i, np.array([0, i], dtype=np.int32))
                x = p1.value(ts, si)
                y = p2.value(ts, si)
                np.testing.assert_allclose(p.value(ts, si), func(np.array([x, y])))

    def test_load(self, model):
        """ Test load from JSON dict"""
        data = {
            "type": "aggregated",
            "agg_func": "product",
            "parameters": [
                0.8,
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

class DummyIndexParameter(IndexParameter):
    """A simple IndexParameter which returns a constant value"""
    def __init__(self, index, *args, **kwargs):
        super(DummyIndexParameter, self).__init__(*args, **kwargs)
        self._index = index
    def index(self, timestep, scenario_index):
        return self._index

def test_aggregated_index_parameter(model):
    """Basic tests of AggregatedIndexParameter"""

    parameters = []
    parameters.append(DummyIndexParameter(2))
    parameters.append(DummyIndexParameter(3))

    timestep = scenario_index = None  # lazy

    agg_index = AggregatedIndexParameter(parameters, "sum")
    assert(agg_index.index(timestep, scenario_index) == 5)

    agg_index = AggregatedIndexParameter(parameters, "max")
    assert(agg_index.index(timestep, scenario_index) == 3)

    agg_index = AggregatedIndexParameter(parameters, "min")
    assert(agg_index.index(timestep, scenario_index) == 2)

def test_aggregated_index_parameter_anyall(model):
    """Test `any` and `all` predicates"""
    timestep = scenario_index = None  # lazy
    data = [(0, 0), (1, 0), (0, 1), (1, 1), (1, 1, 1)]
    expected = [(False, False), (True, False), (True, False), (True, True), (True, True)]
    for item, (expected_any, expected_all) in zip(data, expected):
        parameters = [DummyIndexParameter(i) for i in item]
        agg_index_any = AggregatedIndexParameter(parameters, "any")
        agg_index_all = AggregatedIndexParameter(parameters, "all")
        assert(agg_index_any.index(timestep, scenario_index) == int(expected_any))
        assert(agg_index_all.index(timestep, scenario_index) == int(expected_all))

def test_parameter_child_variables():

    p1 = Parameter()
    # Default parameter
    assert p1 not in p1.variables
    assert len(p1.variables) == 0
    assert len(p1.parents) == 0
    assert len(p1.children) == 0

    c1 = Parameter()
    c1.parents.add(p1)
    assert len(p1.children) == 1
    assert c1 in p1.children
    assert p1 in c1.parents

    assert p1 not in p1.variables
    assert len(p1.variables) == 0

    c1.is_variable = True
    # Now parent should see find child as a variable
    assert p1 not in p1.variables
    assert c1 in p1.variables
    assert len(p1.variables) == 1

    # Test third level
    c2 = Parameter()
    c2.parents.add(c1)
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
    c1.parents.clear()

    assert len(p1.variables) == 0
    assert len(p1.children) == 0
    # Child variables still OK.
    assert c1 not in c1.variables
    assert c2 in c1.variables
    assert len(c1.variables) == 1


def test_scaled_profile_nested_load(model):
    """ Test `ScaledProfileParameter` loading with `AggregatedParameter` """

    s = Storage(model, 'Storage', max_volume=100.0)
    l = Link(model, 'Link')
    data = {
        'type': 'scaledprofile',
        'scale': 50.0,
        'profile': {
            'type': 'aggregated',
            'agg_func': 'product',
            'parameters': [
                {
                    'type': 'monthlyprofile',
                    'values': [0.5]*12
                },
                {
                    'type': 'monthlyprofilecontrolcurve',
                    'control_curves': [0.8, 0.6],
                    'values': [[1.0]*12, [0.7]*np.arange(12), [0.3]*12],
                    'storage_node': 'Storage'
                }
            ]
        }
    }

    l.max_flow = p = load_parameter(model, data)

    p.setup(model)

    # Test correct aggregation is performed
    model.scenarios.setup()
    s.setup(model)  # Init memory view on storage (bypasses usual `Model.setup`)

    s.initial_volume = 90.0
    model.reset()  # Set initial volume on storage
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    for mth in range(1, 13):
        ts = Timestep(datetime.datetime(2016, mth, 1), 366, 1.0)
        np.testing.assert_allclose(p.value(ts, si), 50.0*0.5*1.0)

    s.initial_volume = 70.0
    model.reset()  # Set initial volume on storage
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    for mth in range(1, 13):
        ts = Timestep(datetime.datetime(2016, mth, 1), 366, 1.0)
        np.testing.assert_allclose(p.value(ts, si), 50.0 * 0.5 * 0.7*(mth - 1))

    s.initial_volume = 30.0
    model.reset()  # Set initial volume on storage
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    for mth in range(1, 13):
        ts = Timestep(datetime.datetime(2016, mth, 1), 366, 1.0)
        np.testing.assert_allclose(p.value(ts, si), 50.0 * 0.5 * 0.3)


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

def test_simple_json_parameter_reference(solver):
    # note that parameters in the "parameters" section cannot be literals
    model = load_model("parameter_reference.json")
    max_flow = model.nodes["supply1"].max_flow
    assert(isinstance(max_flow, ConstantParameter))
    assert(max_flow.value(None, None) == 125.0)
    cost = model.nodes["demand1"].cost
    assert(isinstance(cost, ConstantParameter))
    assert(cost.value(None, None) == -10.0)

    assert(len(model.parameters) == 3) # only 3 parameters are named


def test_threshold_parameter(model):
    class DummyRecorder(Recorder):
        def __init__(self, *args, **kwargs):
            super(DummyRecorder, self).__init__(*args, **kwargs)
            self.data = np.array([[0.0]], dtype=np.float64)

    rec = DummyRecorder(model)

    timestep = Timestep(datetime.datetime(2016, 1, 2), 1, 1)
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    threshold = 10.0
    values = [50.0, 60.0]

    expected = [
        ("LT", (1, 0, 0)),
        ("GT", (0, 0, 1)),
        ("EQ", (0, 1, 0)),
        ("LE", (1, 1, 0)),
        ("GE", (0, 1, 1)),
    ]

    for predicate, (value_lt, value_eq, value_gt) in expected:
        param = RecorderThresholdParameter(rec, threshold, values, predicate)
        rec.data[...] = threshold - 5  # data is below threshold
        assert_allclose(param.value(timestep, si), values[value_lt])
        assert(param.index(timestep, si) == value_lt)
        rec.data[...] = threshold  # data is at threshold
        assert_allclose(param.value(timestep, si), values[value_eq])
        assert(param.index(timestep, si) == value_eq)
        rec.data[...] = threshold + 5  # data is above threshold
        assert_allclose(param.value(timestep, si), values[value_gt])
        assert(param.index(timestep, si) == value_gt)


def test_constant_from_df(solver):
    """
    Test that a dataframe can be used to provide data to ConstantParameter (single values).
    """
    model = load_model('simple_df.json', solver=solver)

    assert isinstance(model.nodes['demand1'].max_flow, ConstantParameter)
    assert isinstance(model.nodes['demand1'].cost, ConstantParameter)

    ts = model.timestepper.next()
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    np.testing.assert_allclose(model.nodes['demand1'].max_flow.value(ts, si), 10.0)
    np.testing.assert_allclose(model.nodes['demand1'].cost.value(ts, si), -10.0)


def test_constant_from_shared_df(solver):
    """
    Test that a shared dataframe can be used to provide data to ConstantParameter (single values).
    """
    model = load_model('simple_df_shared.json', solver=solver)

    assert isinstance(model.nodes['demand1'].max_flow, ConstantParameter)
    assert isinstance(model.nodes['demand1'].cost, ConstantParameter)

    ts = model.timestepper.next()
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    np.testing.assert_allclose(model.nodes['demand1'].max_flow.value(ts, si), 10.0)
    np.testing.assert_allclose(model.nodes['demand1'].cost.value(ts, si), -10.0)


def test_constant_from_multiindex_df(solver):
    """
    Test that a dataframe can be used to provide data to ConstantParameter (single values).
    """
    model = load_model('multiindex_df.json', solver=solver)


    assert isinstance(model.nodes['demand1'].max_flow, ConstantParameter)
    assert isinstance(model.nodes['demand1'].cost, ConstantParameter)

    ts = model.timestepper.next()
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    np.testing.assert_allclose(model.nodes['demand1'].max_flow.value(ts, si), 10.0)
    np.testing.assert_allclose(model.nodes['demand1'].cost.value(ts, si), -100.0)

def test_parameter_registry_overwrite(model):
    # define a parameter
    class NewParameter(Parameter):
        DATA = 42
    NewParameter.register()

    # re-define a parameter
    class NewParameter(IndexParameter):
        DATA = 43
        def __init__(self, *args, **kwargs):
            IndexParameter.__init__(self)
    NewParameter.register()

    data = {
        "type": "new",
        "values": 0
    }
    parameter = load_parameter(model, data)

    # parameter is instance of new class, not old class
    assert(isinstance(parameter, NewParameter))
    assert(parameter.DATA == 43)


def test_invalid_parameter_values():
    """
    Test that `load_parameter_values` returns a ValueError rather than KeyError.

    This is useful to catch and give useful messages when no valid reference to
    a data location is given.

    Regression test for Issue #247 (https://github.com/pywr/pywr/issues/247)
    """

    from pywr.parameters._parameters import load_parameter_values

    m = Model()
    data = {'name': 'my_parameter', 'type': 'AParameterThatShouldHaveValues'}
    with pytest.raises(ValueError):
        load_parameter_values(model, data)


class Test1DPolynomialParameter:
    """ Tests for `Polynomial1DParameter` """
    def test_init(self, simple_storage_model):
        """ Test initialisation raises error with too many keywords """
        stg = simple_storage_model.nodes['Storage']

        with pytest.raises(ValueError):
            # Passing both "parameter" and "storage_node" is invalid
            Polynomial1DParameter([0.5, np.pi], parameter=ConstantParameter(2.0), storage_node=stg)

    def test_1st_order_with_parameter(self, model):
        """ Test 1st order with a `Parameter` """
        x = 2.0
        p1 = Polynomial1DParameter([0.5, np.pi], parameter=ConstantParameter(x))
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        ts = model.timestepper.current
        np.testing.assert_allclose(p1.value(ts, si), 0.5 + np.pi*x)

    def test_2nd_order_with_parameter(self, model):
        """ Test 2nd order with a `Parameter` """
        x = 2.0
        px = ConstantParameter(x)
        p1 = Polynomial1DParameter([0.5, np.pi, 3.0], parameter=px)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        ts = model.timestepper.current
        np.testing.assert_allclose(p1.value(ts, si), 0.5 + np.pi*x + 3.0*x**2)

    def test_1st_order_with_storage(self, simple_storage_model):
        """ Test with a `Storage` node """
        model = simple_storage_model
        stg = model.nodes['Storage']
        x = stg.initial_volume
        p1 = Polynomial1DParameter([0.5, np.pi], storage_node=stg)
        p2 = Polynomial1DParameter([0.5, np.pi], storage_node=stg, use_proportional_volume=True)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))

        model.setup()

        ts = model.timestepper.current
        np.testing.assert_allclose(p1.value(ts, si), 0.5 + np.pi*x)
        np.testing.assert_allclose(p2.value(ts, si), 0.5 + np.pi * x/stg.max_volume)

    def test_load(self, model):
        x = 1.5
        data = {
            "type": "polynomial1d",
            "coefficients": [0.5, 2.5],
            "parameter": {
                "type": "constant",
                "value": x
            }
        }

        p1 = load_parameter(model, data)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        for ts in model.timestepper:
            np.testing.assert_allclose(p1.value(ts, si), 0.5 + 2.5*x)

    def test_load_with_scaling(self, model):
        x = 1.5
        data = {
            "type": "polynomial1d",
            "coefficients": [0.5, 2.5],
            "parameter": {
                "type": "constant",
                "value": x
            },
            "scale": 1.25,
            "offset": 0.75
        }
        xscaled = x*1.25 + 0.75

        p1 = load_parameter(model, data)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        for ts in model.timestepper:
            np.testing.assert_allclose(p1.value(ts, si), 0.5 + 2.5*xscaled)


class Test2DStoragePolynomialParameter:

    def test_1st(self, simple_storage_model):
        """ Test 1st order """
        model = simple_storage_model
        stg = model.nodes['Storage']

        x = 2.0
        y = stg.initial_volume
        coefs = [[0.5, np.pi], [2.5, 0.3]]

        p1 = Polynomial2DStorageParameter(coefs, stg, ConstantParameter(x))
        model.setup()
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        ts = model.timestepper.current
        np.testing.assert_allclose(p1.value(ts, si), 0.5 + np.pi*x + 2.5*y+ 0.3*x*y)

    def test_load(self, simple_storage_model):
        model = simple_storage_model
        stg = model.nodes['Storage']

        x = 2.0
        y = stg.initial_volume/stg.max_volume
        data = {
            "type": "polynomial2dstorage",
            "coefficients": [[0.5, np.pi], [2.5, 0.3]],
            "use_proportional_volume": True,
            "parameter": {
                "type": "constant",
                "value": x
            },
            "storage_node": "Storage"
        }

        p1 = load_parameter(model, data)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        model.setup()
        ts = model.timestepper.current
        np.testing.assert_allclose(p1.value(ts, si), 0.5 + np.pi*x + 2.5*y+ 0.3*x*y)

    def test_load_wth_scaling(self, simple_storage_model):
        model = simple_storage_model
        stg = model.nodes['Storage']

        x = 2.0
        y = stg.initial_volume/stg.max_volume
        data = {
            "type": "polynomial2dstorage",
            "coefficients": [[0.5, np.pi], [2.5, 0.3]],
            "use_proportional_volume": True,
            "parameter": {
                "type": "constant",
                "value": x
            },
            "storage_node": "Storage",
            "storage_scale": 1.3,
            "storage_offset": 0.75,
            "parameter_scale": 1.25,
            "parameter_offset": -0.5
        }

        p1 = load_parameter(model, data)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        model.setup()
        ts = model.timestepper.current

        # Scaled parameters
        x = x*1.25 - 0.5
        y = y*1.3 + 0.75

        np.testing.assert_allclose(p1.value(ts, si), 0.5 + np.pi*x + 2.5*y+ 0.3*x*y)


def test_max_parameter(simple_linear_model):
    """ Test `MaxParameter` """
    m = simple_linear_model

    data = {
        "type": "max",
        "threshold": 3,
        "parameter": {
            "type": "dailyprofile",
            "values": list(range(-10, 356))
        }
    }

    m.nodes["Input"].max_flow = load_parameter(model, data)
    m.nodes["Output"].max_flow = 9999
    m.nodes["Output"].cost = -100
    m.setup()
    # Don't go through the whole series because leap years make it harder
    for v in range(-10, 20):
        m.step()
        assert_allclose(m.nodes["Input"].flow, max(v, 3))


def test_min_parameter(simple_linear_model):
    """ Test `MinParameter` """
    m = simple_linear_model

    data = {
        "type": "min",
        "threshold": 3,
        "parameter": {
            "type": "dailyprofile",
            "values": list(range(0, 366))
        }
    }

    m.nodes["Input"].max_flow = load_parameter(model, data)
    m.nodes["Output"].max_flow = 9999
    m.nodes["Output"].cost = -100
    m.setup()
    # Don't go through the whole series because leap years make it harder
    for v in range(0, 20):
        m.step()
        assert_allclose(m.nodes["Input"].flow, min(v, 3))


def test_negative_parameter(simple_linear_model):
    """ Test `NegativeParameter` """
    m = simple_linear_model

    data = {
        "type": "negative",
        "parameter": {
            "type": "dailyprofile",
            "values": list(range(-366, 0))
        }
    }

    m.nodes["Input"].max_flow = load_parameter(model, data)
    m.nodes["Output"].max_flow = 9999
    m.nodes["Output"].cost = -100
    m.setup()
    # Don't go through the whole series because leap years make it harder
    for v in range(-366, -346):
        m.step()
        assert_allclose(m.nodes["Input"].flow, -v)