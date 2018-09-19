"""
Test for individual Parameter classes
"""
from __future__ import division
from pywr.core import Model, Timestep, Scenario, ScenarioIndex, Storage, Link, Input, Output
from pywr.parameters import (Parameter, ArrayIndexedParameter, ConstantScenarioParameter,
    ArrayIndexedScenarioMonthlyFactorsParameter, MonthlyProfileParameter, DailyProfileParameter,
    DataFrameParameter, AggregatedParameter, ConstantParameter, ConstantScenarioIndexParameter,
    IndexParameter, AggregatedIndexParameter, RecorderThresholdParameter, ScenarioMonthlyProfileParameter,
    Polynomial1DParameter, Polynomial2DStorageParameter, ArrayIndexedScenarioParameter,
    InterpolatedParameter, WeeklyProfileParameter, InterpolatedQuadratureParameter,
    FunctionParameter, AnnualHarmonicSeriesParameter, load_parameter)
from pywr.recorders import AssertionRecorder, assert_rec
from pywr.model import OrphanedParameterWarning

from pywr.recorders import Recorder
from fixtures import simple_linear_model, simple_storage_model
from helpers import load_model
import json
import os
import datetime
import numpy as np
import pandas as pd
import pytest
import itertools
from numpy.testing import assert_allclose

TEST_DIR = os.path.dirname(__file__)

@pytest.fixture
def model():
    return Model()


class TestConstantParameter:
    """ Tests for `ConstantParameter` """
    def test_basic_use(self, simple_linear_model):
        """ Test the basic use of `ConstantParameter` using the Python API """
        model = simple_linear_model
        # Add two scenarios
        scA = Scenario(model, 'Scenario A', size=2)
        scB = Scenario(model, 'Scenario B', size=5)

        p = ConstantParameter(model, np.pi, name='pi', comment='Mmmmm Pi!')

        assert not p.is_variable
        assert p.double_size == 1
        assert p.integer_size == 0

        model.setup()
        ts = model.timestepper.current
        # Now ensure the appropriate value is returned for all scenarios
        for i, (a, b) in enumerate(itertools.product(range(scA.size), range(scB.size))):
            si = ScenarioIndex(i, np.array([a, b], dtype=np.int32))
            np.testing.assert_allclose(p.value(ts, si), np.pi)

    def test_being_a_variable(self, simple_linear_model):
        """ Test the basic use of `ConstantParameter` when `is_variable=True` """
        model = simple_linear_model
        p = ConstantParameter(model, np.pi, name='pi', comment='Mmmmm Pi!', is_variable=True,
                              lower_bounds=np.pi/2, upper_bounds=2*np.pi)
        model.setup()

        assert p.is_variable
        assert p.double_size == 1
        assert p.integer_size == 0

        np.testing.assert_allclose(p.get_double_lower_bounds(), np.array([np.pi/2]))
        np.testing.assert_allclose(p.get_double_upper_bounds(), np.array([2*np.pi]))

        np.testing.assert_allclose(p.get_double_variables(), np.array([np.pi]))

        # No test updating the variables
        p.set_double_variables(np.array([1.5*np.pi, ]))
        np.testing.assert_allclose(p.get_double_variables(), np.array([1.5*np.pi]))

        # None of the integer functions should be implemented because this parameter
        # has no integer variables
        with pytest.raises(NotImplementedError):
            p.get_integer_lower_bounds()

        with pytest.raises(NotImplementedError):
            p.get_integer_upper_bounds()

        with pytest.raises(NotImplementedError):
            p.get_integer_variables()


def test_parameter_array_indexed(simple_linear_model):
    """
    Test ArrayIndexedParameter

    """
    model = simple_linear_model
    A = np.arange(len(model.timestepper), dtype=np.float64)
    p = ArrayIndexedParameter(model, A)
    model.setup()
    # scenario indices (not used for this test)
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    for v, ts in zip(A, model.timestepper):
        np.testing.assert_allclose(p.value(ts, si), v)

    # Now check that IndexError is raised if an out of bounds Timestep is given.
    ts = Timestep(pd.Period('2016-01-01', freq='1D'), 366)
    with pytest.raises(IndexError):
        p.value(ts, si)


def test_parameter_array_indexed_json_load(simple_linear_model, tmpdir):
    """Test ArrayIndexedParameter can be loaded from json dict"""
    model = simple_linear_model
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
    model.setup()

    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    for v, ts in enumerate(model.timestepper):
        np.testing.assert_allclose(p.value(ts, si), v)

def test_parameter_constant_scenario(simple_linear_model):
    """
    Test ConstantScenarioParameter

    """
    model = simple_linear_model
    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    p = ConstantScenarioParameter(model, scB, np.arange(scB.size, dtype=np.float64))
    model.setup()
    ts = model.timestepper.current
    # Now ensure the appropriate value is returned for the Scenario B indices.
    for i, (a, b) in enumerate(itertools.product(range(scA.size), range(scB.size))):
        si = ScenarioIndex(i, np.array([a, b], dtype=np.int32))
        np.testing.assert_allclose(p.value(ts, si), float(b))


def test_parameter_constant_scenario(simple_linear_model):
    """
    Test ConstantScenarioIndexParameter

    """
    model = simple_linear_model
    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    p = ConstantScenarioIndexParameter(model, scB, np.arange(scB.size, dtype=np.int32))
    model.setup()
    ts = model.timestepper.current
    # Now ensure the appropriate value is returned for the Scenario B indices.
    for i, (a, b) in enumerate(itertools.product(range(scA.size), range(scB.size))):
        si = ScenarioIndex(i, np.array([a, b], dtype=np.int32))
        np.testing.assert_allclose(p.index(ts, si), b)


def test_parameter_array_indexed_scenario_monthly_factors(simple_linear_model):
    """
    Test ArrayIndexedParameterScenarioMonthlyFactors

    """
    model = simple_linear_model
    # Baseline timeseries data
    values = np.arange(len(model.timestepper), dtype=np.float64)

    # Add two scenarios
    scA = Scenario(model, 'Scenario A', size=2)
    scB = Scenario(model, 'Scenario B', size=5)

    # Random factors for each Scenario B value per month
    factors = np.random.rand(scB.size, 12)

    p = ArrayIndexedScenarioMonthlyFactorsParameter(model, scB, values, factors)
    model.setup()

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


def test_parameter_monthly_profile(simple_linear_model):
    """
    Test MonthlyProfileParameter

    """
    model = simple_linear_model
    values = np.arange(12, dtype=np.float64)
    p = MonthlyProfileParameter(model, values)
    model.setup()

    # Iterate in time
    for ts in model.timestepper:
        imth = ts.datetime.month - 1
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        np.testing.assert_allclose(p.value(ts, si), values[imth])


class TestScenarioMonthlyProfileParameter:

    def test_init(self, simple_linear_model):
        model = simple_linear_model
        scenario = Scenario(model, 'A', 10)
        values = np.random.rand(10, 12)

        p = ScenarioMonthlyProfileParameter(model, scenario, values)

        model.setup()
        # Iterate in time
        for ts in model.timestepper:
            imth = ts.datetime.month - 1
            for i in range(scenario.size):
                si = ScenarioIndex(i, np.array([i], dtype=np.int32))
                np.testing.assert_allclose(p.value(ts, si), values[i, imth])

    def test_json(self):
        model = load_model('scenario_monthly_profile.json')

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

def test_parameter_daily_profile(simple_linear_model):
    """
    Test DailyProfileParameter

    """
    model = simple_linear_model
    values = np.arange(366, dtype=np.float64)
    p = DailyProfileParameter(model, values)
    model.setup()

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
    inpt.max_flow = DailyProfileParameter(model, np.arange(0, 366, dtype=np.float64))

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

def test_weekly_profile(simple_linear_model):
    model = simple_linear_model

    model.timestepper.start = "2004-01-01"
    model.timestepper.end = "2005-05-01"
    model.timestepper.delta = 7

    values = np.arange(0, 52) ** 2 + 27.5

    p = WeeklyProfileParameter.load(model, {"values": values})

    @assert_rec(model, p)
    def expected_func(timestep, scenario_index):
        week = int(min((timestep.dayofyear - 1) // 7, 51))
        value = week ** 2 + 27.5
        return value

    model.run()

class TestAnnualHarmonicSeriesParameter:
    """ Tests for `AnnualHarmonicSeriesParameter` """
    def test_single_harmonic(self, model):

        p1 = AnnualHarmonicSeriesParameter(model, 0.5, [0.25], [np.pi/4])
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))

        for ts in model.timestepper:
            doy = (ts.datetime.dayofyear - 1)/365
            np.testing.assert_allclose(p1.value(ts, si), 0.5 + 0.25*np.cos(doy*2*np.pi + np.pi/4))

    def test_double_harmonic(self, model):
        p1 = AnnualHarmonicSeriesParameter(model, 0.5, [0.25, 0.3], [np.pi/4, np.pi/3])
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

    def test_variable(self, model):
        """ Test that variable updating works. """
        p1 = AnnualHarmonicSeriesParameter(model, 0.5, [0.25], [np.pi/4], is_variable=True)

        assert p1.double_size == 3
        assert p1.integer_size == 0

        new_var = np.array([0.6, 0.1, np.pi/2])
        p1.set_double_variables(new_var)
        np.testing.assert_allclose(p1.get_double_variables(), new_var)

        with pytest.raises(NotImplementedError):
            p1.set_integer_variables(np.arange(3, dtype=np.int32))

        with pytest.raises(NotImplementedError):
            p1.get_integer_variables()

        si = ScenarioIndex(0, np.array([0], dtype=np.int32))

        for ts in model.timestepper:
            doy = (ts.datetime.dayofyear - 1)/365
            np.testing.assert_allclose(p1.value(ts, si), 0.6 + 0.1*np.cos(doy*2*np.pi + np.pi/2))


def custom_test_func(array, axis=None):
    return np.sum(array**2, axis=axis)


class TestAggregatedParameter:
    """Tests for AggregatedParameter"""
    funcs = {"min": np.min, "max": np.max, "mean": np.mean, "median": np.median, "sum": np.sum}

    @pytest.mark.parametrize("agg_func", ["min", "max", "mean", "median", "sum"])
    def test_agg(self, simple_linear_model, agg_func):
        model = simple_linear_model
        model.timestepper.delta = 15

        scenarioA = Scenario(model, "Scenario A", size=2)
        scenarioB = Scenario(model, "Scenario B", size=5)

        values = np.arange(366, dtype=np.float64)
        p1 = DailyProfileParameter(model, values)
        p2 = ConstantScenarioParameter(model, scenarioB, np.arange(scenarioB.size, dtype=np.float64))

        p = AggregatedParameter(model, [p1, p2], agg_func=agg_func)

        func = TestAggregatedParameter.funcs[agg_func]

        @assert_rec(model, p)
        def expected_func(timestep, scenario_index):
            x = p1.get_value(scenario_index)
            y = p2.get_value(scenario_index)
            return func(np.array([x,y]))

        model.run()

    def test_load(self, simple_linear_model):
        """ Test load from JSON dict"""
        model = simple_linear_model
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

        @assert_rec(model, p)
        def expected(timestep, scenario_index):
            return (timestep.month - 1) * 0.8

        model.run()

    @pytest.mark.parametrize("agg_func", ["min", "max", "mean", "sum", "custom"])
    def test_agg_func_get_set(self, model, agg_func):
        if agg_func == "custom":
            agg_func = custom_test_func
        p = AggregatedParameter(model, [], agg_func=agg_func)
        assert p.agg_func == agg_func
        p.agg_func = "product"
        assert p.agg_func == "product"


class DummyIndexParameter(IndexParameter):
    """A simple IndexParameter which returns a constant value"""
    def __init__(self, model, index, **kwargs):
        super(DummyIndexParameter, self).__init__(model, **kwargs)
        self._index = index
    def index(self, timestep, scenario_index):
        return self._index
    def __repr__(self):
        return "<DummyIndexParameter \"{}\">".format(self.name)

class TestAggregatedIndexParameter:
    """Tests for AggregatedIndexParameter"""
    funcs = {"min": np.min, "max": np.max, "sum": np.sum, "product": np.product}

    @pytest.mark.parametrize("agg_func", ["min", "max", "sum", "product"])
    def test_agg(self, simple_linear_model, agg_func):
        model = simple_linear_model
        model.timestepper.delta = 1
        model.timestepper.start = "2017-01-01"
        model.timestepper.end = "2017-01-03"

        scenarioA = Scenario(model, "Scenario A", size=2)
        scenarioB = Scenario(model, "Scenario B", size=5)

        p1 = DummyIndexParameter(model, 2)
        p2 = DummyIndexParameter(model, 3)

        p = AggregatedIndexParameter(model, [p1, p2], agg_func=agg_func)

        func = TestAggregatedIndexParameter.funcs[agg_func]

        @assert_rec(model, p)
        def expected_func(timestep, scenario_index):
            x = p1.get_index(scenario_index)
            y = p2.get_index(scenario_index)
            return func(np.array([x,y], np.int32))

        model.run()

    def test_agg_anyall(self, simple_linear_model):
        """Test the "any" and "all" aggregation functions"""
        model = simple_linear_model
        model.timestepper.delta = 1
        model.timestepper.start = "2017-01-01"
        model.timestepper.end = "2017-01-03"

        scenarioA = Scenario(model, "Scenario A", size=2)
        scenarioB = Scenario(model, "Scenario B", size=5)
        num_comb = len(model.scenarios.get_combinations())

        parameters = {
            0: DummyIndexParameter(model, 0, name="p0"),
            1: DummyIndexParameter(model, 1, name="p1"),
            2: DummyIndexParameter(model, 2, name="p2"),
        }

        data = [(0, 0), (1, 0), (0, 1), (1, 1), (1, 1, 1), (0, 2)]
        data_parameters = [[parameters[i] for i in d] for d in data]
        expected = [(np.any(d), np.all(d)) for d in data]

        for n, params in enumerate(data_parameters):
            for m, agg_func in enumerate(["any", "all"]):
                p = AggregatedIndexParameter(model, params, agg_func=agg_func)
                e = np.ones([len(model.timestepper), num_comb]) * expected[n][m]
                r = AssertionRecorder(model, p, expected_data=e, name="assertion {}-{}".format(n, agg_func))

        model.run()

    @pytest.mark.parametrize("agg_func", ["min", "max", "mean", "sum", "custom"])
    def test_agg_func_get_set(self, model, agg_func):
        if agg_func == "custom":
            agg_func = custom_test_func
        p = AggregatedIndexParameter(model, [], agg_func=agg_func)
        assert p.agg_func == agg_func
        p.agg_func = "product"
        assert p.agg_func == "product"


def test_parameter_child_variables(model):

    p1 = Parameter(model)
    # Default parameter
    assert len(p1.parents) == 0
    assert len(p1.children) == 0

    c1 = Parameter(model)
    c1.parents.add(p1)
    assert len(p1.children) == 1
    assert c1 in p1.children
    assert p1 in c1.parents

    # Test third level
    c2 = Parameter(model)
    c2.parents.add(c1)

    # Disable parent
    c1.parents.clear()

    assert len(p1.children) == 0


def test_scaled_profile_nested_load(model):
    """ Test `ScaledProfileParameter` loading with `AggregatedParameter` """
    model.timestepper.delta = 15

    s = Storage(model, 'Storage', max_volume=100.0, initial_volume=50.0, num_outputs=0)
    d = Output(model, 'Link')
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
                    'type': 'constant',
                    'value': 1.5,
                }
            ]
        }
    }

    s.connect(d)

    d.max_flow = p = load_parameter(model, data)

    @assert_rec(model, p)
    def expected_func(timestep, scenario_index):
        return 50.0 * 0.5 * 1.5

    model.run()


def test_parameter_df_upsampling(model):
    """ Test that the `DataFrameParameter` can upsample data from a `pandas.DataFrame` and return that correctly
    """
    # scenario indices (not used for this test)
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    # Use a 7 day timestep for this test and run 2015
    model.timestepper.delta = datetime.timedelta(7)
    model.timestepper.start = pd.to_datetime('2015-01-01')
    model.timestepper.end = pd.to_datetime('2015-12-31')
    model.timestepper.setup()

    # Daily time-step
    index = pd.date_range('2015-01-01', periods=365, freq='D')
    series = pd.Series(np.arange(365), index=index)

    p = DataFrameParameter(model, series)
    p.setup()

    A = series.resample('7D').mean()
    for v, ts in zip(A, model.timestepper):
        np.testing.assert_allclose(p.value(ts, si), v)

    model.reset()
    # Daily time-step that requires aligning
    index = pd.date_range('2014-12-31', periods=366, freq='D')
    series = pd.Series(np.arange(366), index=index)

    p = DataFrameParameter(model, series)
    p.setup()

    # offset the resample appropriately for the test
    A = series[1:].resample('7D').mean()
    for v, ts in zip(A, model.timestepper):
        np.testing.assert_allclose(p.value(ts, si), v)

    model.reset()
    # Daily time-step that is not covering the require range
    index = pd.date_range('2015-02-01', periods=365, freq='D')
    series = pd.Series(np.arange(365), index=index)

    p = DataFrameParameter(model, series)
    with pytest.raises(ValueError):
        p.setup()

    model.reset()
    # Daily time-step that is not covering the require range
    index = pd.date_range('2014-11-01', periods=365, freq='D')
    series = pd.Series(np.arange(365), index=index)

    p = DataFrameParameter(model, series)
    with pytest.raises(ValueError):
        p.setup()


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
    model.timestepper.setup()

    # Daily time-step
    index = pd.date_range('2015-01-01', periods=365, freq='D')
    df = pd.DataFrame(np.random.rand(365, 20), index=index)

    p = DataFrameParameter(model, df, scenario=scA)
    p.setup()

    A = df.resample('7D', axis=0).mean()
    for v, ts in zip(A.values, model.timestepper):
        np.testing.assert_allclose([p.value(ts, ScenarioIndex(i, np.array([i], dtype=np.int32))) for i in range(20)], v)

    p = DataFrameParameter(model, df, scenario=scB)
    with pytest.raises(ValueError):
        p.setup()


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
    p.setup()


def test_parameter_df_embed_load(model):

    # Daily time-step
    index = pd.date_range('2015-01-01', periods=365, freq='D', name='date')
    df = pd.DataFrame(np.random.rand(365), index=index, columns=['data'])

    # Save to JSON and load. This is the format we support loading as embedded data
    df_data = df.to_json(date_format="iso")
    # Removing the time information from the dataset for testing purposes
    df_data = df_data.replace('T00:00:00.000Z', '')
    df_data = json.loads(df_data)

    data = {
        'type': 'dataframe',
        'data': df_data,
        'parse_dates': True,
    }

    p = load_parameter(model, data)
    p.setup()


def test_simple_json_parameter_reference():
    # note that parameters in the "parameters" section cannot be literals
    model = load_model("parameter_reference.json")
    max_flow = model.nodes["supply1"].max_flow
    assert(isinstance(max_flow, ConstantParameter))
    assert(max_flow.value(None, None) == 125.0)
    cost = model.nodes["demand1"].cost
    assert(isinstance(cost, ConstantParameter))
    assert(cost.value(None, None) == -10.0)

    assert(len(model.parameters) == 4)  # 4 parameters defined


def test_threshold_parameter(simple_linear_model):
    model = simple_linear_model
    model.timestepper.delta = 150

    scenario = Scenario(model, "Scenario", size=2)

    class DummyRecorder(Recorder):
        def __init__(self, model, value, *args, **kwargs):
            super(DummyRecorder, self).__init__(model, *args, **kwargs)
            self.val = value
        def setup(self):
            super(DummyRecorder, self).setup()
            num_comb = len(model.scenarios.combinations)
            self.data = np.empty([len(model.timestepper), num_comb], dtype=np.float64)
        def after(self):
            timestep = model.timestepper.current
            self.data[timestep.index, :] = self.val

    threshold = 10.0
    values = [50.0, 60.0]

    rec1 = DummyRecorder(model, threshold-5, name="rec1")  # below
    rec2 = DummyRecorder(model, threshold, name="rec2")    # equal
    rec3 = DummyRecorder(model, threshold+5, name="rec3")  # above

    expected = [
        ("LT", (1, 0, 0)),
        ("GT", (0, 0, 1)),
        ("EQ", (0, 1, 0)),
        ("LE", (1, 1, 0)),
        ("GE", (0, 1, 1)),
    ]

    for predicate, (value_lt, value_eq, value_gt) in expected:
        for rec in (rec1, rec2, rec3):
            param = RecorderThresholdParameter(model, rec, threshold, values=values, predicate=predicate)
            e_val = values[getattr(rec.val, "__{}__".format(predicate.lower()))(threshold)]
            e = np.ones([len(model.timestepper), len(model.scenarios.get_combinations())]) * e_val
            e[0, :] = values[1] # first timestep is always "on"
            r = AssertionRecorder(model, param, expected_data=e)
            r.name = "assert {} {} {}".format(rec.val, predicate, threshold)

    model.run()


def test_constant_from_df():
    """
    Test that a dataframe can be used to provide data to ConstantParameter (single values).
    """
    model = load_model('simple_df.json')

    assert isinstance(model.nodes['demand1'].max_flow, ConstantParameter)
    assert isinstance(model.nodes['demand1'].cost, ConstantParameter)

    ts = model.timestepper.next()
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    np.testing.assert_allclose(model.nodes['demand1'].max_flow.value(ts, si), 10.0)
    np.testing.assert_allclose(model.nodes['demand1'].cost.value(ts, si), -10.0)


def test_constant_from_shared_df():
    """
    Test that a shared dataframe can be used to provide data to ConstantParameter (single values).
    """
    model = load_model('simple_df_shared.json')

    assert isinstance(model.nodes['demand1'].max_flow, ConstantParameter)
    assert isinstance(model.nodes['demand1'].cost, ConstantParameter)

    ts = model.timestepper.next()
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    np.testing.assert_allclose(model.nodes['demand1'].max_flow.value(ts, si), 10.0)
    np.testing.assert_allclose(model.nodes['demand1'].cost.value(ts, si), -10.0)


def test_constant_from_multiindex_df():
    """
    Test that a dataframe can be used to provide data to ConstantParameter (single values).
    """
    model = load_model('multiindex_df.json')


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
        def __init__(self, model, values, *args, **kwargs):
            super(NewParameter, self).__init__(model, *args, **kwargs)
            self.values = values
    NewParameter.register()

    # re-define a parameter
    class NewParameter(IndexParameter):
        DATA = 43
        def __init__(self, model, values, *args, **kwargs):
            super(NewParameter, self).__init__(model, *args, **kwargs)
            self.values = values
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
        param = ConstantParameter(simple_storage_model, 2.0)
        with pytest.raises(ValueError):
            # Passing both "parameter" and "storage_node" is invalid
            Polynomial1DParameter(simple_storage_model, [0.5, np.pi], parameter=param, storage_node=stg)

    def test_1st_order_with_parameter(self, simple_linear_model):
        """ Test 1st order with a `Parameter` """
        model = simple_linear_model

        x = 2.0
        p1 = Polynomial1DParameter(model, [0.5, np.pi], parameter=ConstantParameter(model, x))

        @assert_rec(model, p1)
        def expected_func(timestep, scenario_index):
            return 0.5 + np.pi * x
        model.run()

    def test_2nd_order_with_parameter(self, simple_linear_model):
        """ Test 2nd order with a `Parameter` """
        model = simple_linear_model

        x = 2.0
        px = ConstantParameter(model, x)
        p1 = Polynomial1DParameter(model, [0.5, np.pi, 3.0], parameter=px)

        @assert_rec(model, p1)
        def expected_func(timestep, scenario_index):
            return 0.5 + np.pi*x + 3.0*x**2
        model.run()

    def test_1st_order_with_storage(self, simple_storage_model):
        """ Test with a `Storage` node """
        model = simple_storage_model
        stg = model.nodes['Storage']
        x = stg.initial_volume
        p1 = Polynomial1DParameter(model, [0.5, np.pi], storage_node=stg)
        p2 = Polynomial1DParameter(model, [0.5, np.pi], storage_node=stg, use_proportional_volume=True)

        # Test with absolute storage
        @assert_rec(model, p1)
        def expected_func(timestep, scenario_index):
            return 0.5 + np.pi*x

        # Test with proportional storage
        @assert_rec(model, p2, name="proportionalassertion")
        def expected_func(timestep, scenario_index):

            return 0.5 + np.pi * x/stg.max_volume

        model.setup()
        model.step()


    def test_load(self, simple_linear_model):
        model = simple_linear_model

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

        @assert_rec(model, p1)
        def expected_func(timestep, scenario_index):
            return 0.5 + 2.5*x
        model.run()

    def test_load_with_scaling(self, simple_linear_model):
        model = simple_linear_model
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

        @assert_rec(model, p1)
        def expected_func(timestep, scenario_index):
            return 0.5 + 2.5*xscaled
        model.run()


def test_interpolated_parameter(simple_linear_model):
    model = simple_linear_model
    model.timestepper.start = "1920-01-01"
    model.timestepper.end = "1920-01-12"

    p1 = ArrayIndexedParameter(model, [0,1,2,3,4,5,6,7,8,9,10,11])
    p2 = InterpolatedParameter(model, p1, [0, 5, 10, 11], [0, 5*2, 10*3, 2])

    @assert_rec(model, p2)
    def expected_func(timestep, scenario_index):
        values = [0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 2]
        return values[timestep.index]
    model.run()


class TestInterpolatedQuadratureParameter:

    @pytest.mark.parametrize("lower_interval", [None, 0, 1])
    def test_calc(self, simple_linear_model, lower_interval):
        model = simple_linear_model
        model.timestepper.start = "1920-01-01"
        model.timestepper.end = "1920-01-12"

        b = ArrayIndexedParameter(model, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        a = None
        if lower_interval is not None:
            a = ConstantParameter(model, lower_interval)

        p2 = InterpolatedQuadratureParameter(model, b, [-5, 0, 5, 10, 11], [0, 0, 5 * 2, 10 * 3, 2],
                                             lower_parameter=a)

        def area(i):
            if i < 0:
                value = 0
            elif i < 6:
                value = 2*i**2 / 2
            elif i < 11:
                value = 25 + 4*(i - 5)**2 / 2 + (i - 5) * 10
            else:
                value = 25 + 50 + 50 + 28 / 2 + 2
            return value

        @assert_rec(model, p2)
        def expected_func(timestep, scenario_index):
            i = timestep.index
            value = area(i)
            if lower_interval is not None:
                value -= area(lower_interval)
            return value

        model.run()

    def test_load(self, simple_linear_model):
        model = simple_linear_model
        model.timestepper.start = "1920-01-01"
        model.timestepper.end = "1920-01-12"

        p1 = ArrayIndexedParameter(model, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], name='p1')

        p2 = {
            'type': 'interpolatedquadrature',
            'upper_parameter': 'p1',
            'x': [0, 5, 10, 11],
            'y': [0, 5 * 2, 10 * 3, 2]
        }

        p2 = load_parameter(model, p2)

        @assert_rec(model, p2)
        def expected_func(timestep, scenario_index):
            i = timestep.index
            if i < 6:
                value = 2 * i ** 2 / 2
            elif i < 11:
                value = 25 + 4 * (i - 5) ** 2 / 2 + (i - 5) * 10
            else:
                value = 25 + 50 + 50 + 28 / 2 + 2
            return value

        model.run()


class Test2DStoragePolynomialParameter:

    def test_1st(self, simple_storage_model):
        """ Test 1st order """
        model = simple_storage_model
        stg = model.nodes['Storage']

        x = 2.0
        y = stg.initial_volume
        coefs = [[0.5, np.pi], [2.5, 0.3]]

        p1 = Polynomial2DStorageParameter(model, coefs, stg, ConstantParameter(model, x))

        @assert_rec(model, p1)
        def expected_func(timestep, scenario_index):
            return 0.5 + np.pi*x + 2.5*y+ 0.3*x*y
        model.setup()
        model.step()

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

        @assert_rec(model, p1)
        def expected_func(timestep, scenario_index):
            return 0.5 + np.pi*x + 2.5*y+ 0.3*x*y
        model.setup()
        model.step()

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

        # Scaled parameters
        x = x*1.25 - 0.5
        y = y*1.3 + 0.75

        @assert_rec(model, p1)
        def expected_func(timestep, scenario_index):
            return 0.5 + np.pi*x + 2.5*y+ 0.3*x*y
        model.setup()
        model.step()


class TestDivisionParameter:

    def test_divsion(self, simple_linear_model):
        model = simple_linear_model
        model.timestepper.start = "2017-01-01"
        model.timestepper.end = "2017-01-15"

        profile = list(range(1, 367))

        data = {
            "type": "division",
            "numerator": {
                "name": "raw",
                "type": "dailyprofile",
                "values": profile,
            },
            "denominator": {
                "type": "constant",
                "value": 123.456
            }
        }

        model.nodes["Input"].max_flow = parameter = load_parameter(model, data)
        model.nodes["Output"].max_flow = 9999
        model.nodes["Output"].cost = -100

        daily_profile = model.parameters["raw"]

        @assert_rec(model, parameter)
        def expected(timestep, scenario_index):
            value = daily_profile.get_value(scenario_index)
            return value / 123.456
        model.run()


class TestMinMaxNegativeParameter:
    @pytest.mark.parametrize("ptype,profile", [
        ("max", list(range(-10, 356))),
        ("min", list(range(0, 366))),
        ("negative", list(range(-366, 0))),
        ("negativemax", list(range(-366, 0))),
    ])
    def test_parameter(cls, simple_linear_model, ptype,profile):
        model = simple_linear_model
        model.timestepper.start = "2017-01-01"
        model.timestepper.end = "2017-01-15"

        data = {
            "type": ptype,
            "parameter": {
                "name": "raw",
                "type": "dailyprofile",
                "values": profile,
            }
        }


        if ptype in ("max", "min"):
            data["threshold"] = 3

        func = {"min": min, "max": max, "negative": lambda t,x: -x, "negativemax": lambda t,x: max(t, -x)}[ptype]

        model.nodes["Input"].max_flow = parameter = load_parameter(model, data)
        model.nodes["Output"].max_flow = 9999
        model.nodes["Output"].cost = -100

        daily_profile = model.parameters["raw"]

        @assert_rec(model, parameter)
        def expected(timestep, scenario_index):
            value = daily_profile.get_value(scenario_index)
            return func(3, value)
        model.run()

def test_ocptt(simple_linear_model):
    model = simple_linear_model
    inpt = model.nodes["Input"]
    s1 = Scenario(model, "scenario 1", size=3)
    s2 = Scenario(model, "scenario 1", size=2)
    x = np.arange(len(model.timestepper)).reshape([len(model.timestepper), 1]) + 5
    y = np.arange(s1.size).reshape([1, s1.size])
    z = x * y ** 2
    p = ArrayIndexedScenarioParameter(model, s1, z)
    inpt.max_flow = p
    model.setup()
    model.reset()
    model.step()

    values1 = [p.get_value(scenario_index) for scenario_index in model.scenarios.combinations]
    values2 = list(p.get_all_values())
    assert_allclose(values1, [0, 0, 5, 5, 20, 20])
    assert_allclose(values2, [0, 0, 5, 5, 20, 20])


class TestThresholdParameters:

    def test_storage_threshold_parameter(self, simple_storage_model):
        """ Test StorageThresholdParameter """
        m = simple_storage_model

        data = {
            "type": "storagethreshold",
            "storage_node": "Storage",
            "threshold": 10.0,
            "predicate": ">"
        }

        p1 = load_parameter(m, data)

        si = ScenarioIndex(0, np.array([0], dtype=np.int32))

        m.nodes['Storage'].initial_volume = 15.0
        m.setup()
        # Storage > 10
        assert p1.index(m.timestepper.current, si) == 1

        m.nodes['Storage'].initial_volume = 5.0
        m.setup()
        # Storage < 10
        assert p1.index(m.timestepper.current, si) == 0

    def test_node_threshold_parameter2(self, simple_linear_model):
        model = simple_linear_model
        model.nodes["Input"].max_flow = ArrayIndexedParameter(model, np.arange(0, 20))
        model.nodes["Output"].cost = -10.0
        model.timestepper.start = "1920-01-01"
        model.timestepper.end = "1920-01-15"
        model.timestepper.delta = 1

        threshold = 5.0

        parameters = {}
        for predicate in (">", "<", "="):
            data = {
                "type": "nodethreshold",
                "node": "Output",
                "threshold": 5.0,
                "predicate": predicate,
                # we need to define values so AssertionRecorder can be used
                "values": [0.0, 1.0],
            }
            parameter = load_parameter(model, data)
            parameter.name = "nodethresold {}".format(predicate)
            parameters[predicate] = parameter

            if predicate == ">":
                expected_data = (np.arange(-1, 20) > threshold).astype(int)
            elif predicate == "<":
                expected_data = (np.arange(-1, 20) < threshold).astype(int)
            else:
                expected_data = (np.arange(-1, 20) == threshold).astype(int)
            expected_data[0] = 0 # previous flow in initial timestep is undefined
            expected_data = expected_data[:, np.newaxis]

            rec = AssertionRecorder(model, parameter, expected_data=expected_data, name="assertion recorder {}".format(predicate))

        model.run()

    @pytest.mark.parametrize("threshold, ratchet", [
        [5.0, False],
        [{"type": "constant", "value": 5.0}, False],
        [{"type": "constant", "value": 5.0}, True],
    ], ids=["double", "parameter", "parameter-ratchet"])
    def test_parameter_threshold_parameter(self, simple_linear_model, threshold, ratchet):
        """ Test ParameterThresholdParameter """
        m = simple_linear_model
        m.nodes['Input'].max_flow = 10.0
        m.nodes['Output'].cost = -10.0

        data = {
            "type": "parameterthreshold",
            "parameter": {
                "type": "constant",
                "value": 3.0
            },
            "threshold": threshold,
            "predicate": "<",
            "ratchet": ratchet
        }

        p1 = load_parameter(m, data)

        si = ScenarioIndex(0, np.array([0], dtype=np.int32))

        # Triggered initial 3 < 5
        m.setup()
        m.step()
        assert p1.index(m.timestepper.current, si) == 1

        # Update parameter, now 8 > 5; not triggered.
        p1.param.set_double_variables(np.array([8.0,]))
        m.step()
        # If using a ratchet the trigger remains on.
        assert p1.index(m.timestepper.current, si) == (1 if ratchet else 0)

        # Resetting the model resets the ratchet too.
        m.reset()
        m.step()
        # flow < 5
        assert p1.index(m.timestepper.current, si) == 0

def test_orphaned_components(simple_linear_model):
    model = simple_linear_model
    model.nodes["Input"].max_flow = ConstantParameter(model, 10.0)

    result = model.find_orphaned_parameters()
    assert(not result)
    # assert that warning not raised by check
    with pytest.warns(None) as record:
        model.check()
    for w in record:
        if isinstance(w, OrphanedParameterWarning):
            pytest.fail("OrphanedParameterWarning raised unexpectedly!")

    # add some orphans
    orphan1 = ConstantParameter(model, 5.0)
    orphan2 = ConstantParameter(model, 10.0)
    orphans = {orphan1, orphan2}
    result = model.find_orphaned_parameters()
    assert(orphans == result)

    with pytest.warns(OrphanedParameterWarning):
        model.check()

def test_deficit_parameter():
    """Test DeficitParameter

    Here we test both uses of the DeficitParameter:
      1) Recording the deficit for a node each timestep
      2) Using yesterday's deficit to control today's flow
    """
    model = load_model("deficit.json")

    model.run()

    max_flow = np.array([5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8])
    demand = 10.0
    supplied = np.minimum(max_flow, demand)
    expected = demand - supplied
    actual = model.recorders["deficit_recorder"].data
    assert_allclose(expected, actual[:,0])

    expected_yesterday = [0]+list(expected[0:-1])
    actual_yesterday = model.recorders["yesterday_recorder"].data
    assert_allclose(expected_yesterday, actual_yesterday[:,0])


def test_flow_parameter():
    """test FlowParameter

    """
    model = load_model("flow_parameter.json")

    model.run()

    max_flow = np.array([5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8])
    demand = 10.0
    supplied = np.minimum(max_flow, demand)

    actual = model.recorders["flow_recorder"].data
    assert_allclose(supplied, actual[:,0])

    expected_yesterday = [3.1415]+list(supplied[0:-1])
    actual_yesterday = model.recorders["yesterday_flow_recorder"].data
    assert_allclose(expected_yesterday, actual_yesterday[:,0])


class TestHydroPowerTargets:
    def test_target_json(self):
        """ Test loading a HydropowerTargetParameter from JSON. """
        model = load_model("hydropower_target_example.json")
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))

        # 30 time-steps are run such that the head gets so flow to hit the max_flow
        # constraint. The first few time-steps are also bound by the min_flow constraint.
        for i in range(30):
            model.step()

            rec = model.recorders["turbine1_energy"]
            param = model.parameters["turbine1_discharge"]

            turbine1 = model.nodes["turbine1"]
            assert turbine1.flow[0] > 0

            if np.allclose(turbine1.flow[0], 500.0):
                # If flow is bounded by min_flow then more HP is produced.
                assert rec.data[i, 0] > param.target.get_value(si)
            elif np.allclose(turbine1.flow[0], 1000.0):
                # If flow is bounded by max_flow then less HP is produced.
                assert rec.data[i, 0] < param.target.get_value(si)
            else:
                # If flow is within the bounds target is met exactly.
                assert_allclose(rec.data[i, 0], param.target.get_value(si))
