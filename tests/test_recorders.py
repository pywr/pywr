# -*- coding: utf-8 -*-
"""
Test the Recorder object API

"""
import pywr.core
from pywr.core import Model, Input, Output, Scenario, AggregatedNode
import numpy as np
import pandas
import pytest
import tables
import json
from numpy.testing import assert_allclose, assert_equal
from fixtures import simple_linear_model, simple_storage_model
from pywr.recorders import (
    Recorder,
    NumpyArrayNodeRecorder,
    NumpyArrayStorageRecorder,
    NumpyArrayAreaRecorder,
    NumpyArrayLevelRecorder,
    AggregatedRecorder,
    CSVRecorder,
    TablesRecorder,
    TotalDeficitNodeRecorder,
    TotalFlowNodeRecorder,
    RollingMeanFlowNodeRecorder,
    MeanFlowNodeRecorder,
    NumpyArrayParameterRecorder,
    NumpyArrayIndexParameterRecorder,
    RollingWindowParameterRecorder,
    AnnualCountIndexParameterRecorder,
    RootMeanSquaredErrorNodeRecorder,
    MeanAbsoluteErrorNodeRecorder,
    MeanSquareErrorNodeRecorder,
    PercentBiasNodeRecorder,
    RMSEStandardDeviationRatioNodeRecorder,
    NashSutcliffeEfficiencyNodeRecorder,
    EventRecorder,
    Event,
    StorageThresholdRecorder,
    NodeThresholdRecorder,
    EventDurationRecorder,
    EventStatisticRecorder,
    FlowDurationCurveRecorder,
    FlowDurationCurveDeviationRecorder,
    StorageDurationCurveRecorder,
    HydropowerRecorder,
    TotalHydroEnergyRecorder,
    TotalParameterRecorder,
    MeanParameterRecorder,
    NumpyArrayNodeCostRecorder,
    NumpyArrayNodeDeficitRecorder,
    NumpyArrayNodeSuppliedRatioRecorder,
    NumpyArrayNodeCurtailmentRatioRecorder,
    SeasonalFlowDurationCurveRecorder,
    load_recorder,
    ParameterNameWarning,
    NumpyArrayDailyProfileParameterRecorder,
    AnnualTotalFlowRecorder,
    AnnualCountIndexThresholdRecorder,
    TimestepCountIndexParameterRecorder,
    GaussianKDEStorageRecorder,
    NormalisedGaussianKDEStorageRecorder,
    NumpyArrayNormalisedStorageRecorder,
)

from pywr.recorders.progress import ProgressRecorder

from pywr.parameters import (
    DailyProfileParameter,
    FunctionParameter,
    ArrayIndexedParameter,
    ConstantParameter,
    InterpolatedVolumeParameter,
    ConstantScenarioParameter,
    StorageThresholdParameter,
)
from helpers import load_model
import os
import sys


class TestRecorder:
    """Tests for Recorder base class."""

    def test_default_no_constraint(self, simple_linear_model):
        """Test the constraint properties in the default instance (i.e. not a constraint)."""
        r = Recorder(simple_linear_model)
        assert r.constraint_lower_bounds is None
        assert r.constraint_upper_bounds is None
        assert not r.is_constraint
        assert not r.is_lower_bounded_constraint
        assert not r.is_double_bounded_constraint
        assert not r.is_upper_bounded_constraint
        assert not r.is_equality_constraint

    def test_equality_constraint(self, simple_linear_model):
        """Test equality constraint identification."""
        r = Recorder(
            simple_linear_model,
            constraint_lower_bounds=10.0,
            constraint_upper_bounds=10.0,
        )
        assert r.is_constraint
        assert not r.is_lower_bounded_constraint
        assert not r.is_double_bounded_constraint
        assert not r.is_upper_bounded_constraint
        assert r.is_equality_constraint

    def test_lower_bounded_constraint(self, simple_linear_model):
        """Test lower bounded constraint identification."""
        r = Recorder(
            simple_linear_model,
            constraint_lower_bounds=10.0,
            constraint_upper_bounds=None,
        )
        assert r.is_constraint
        assert r.is_lower_bounded_constraint
        assert not r.is_double_bounded_constraint
        assert not r.is_upper_bounded_constraint
        assert not r.is_equality_constraint

    def test_upper_bounded_constraint(self, simple_linear_model):
        """Test upper bounded constraint identification."""
        r = Recorder(
            simple_linear_model,
            constraint_lower_bounds=None,
            constraint_upper_bounds=10.0,
        )
        assert r.is_constraint
        assert not r.is_lower_bounded_constraint
        assert not r.is_double_bounded_constraint
        assert r.is_upper_bounded_constraint
        assert not r.is_equality_constraint

    def test_double_bounded_constraint(self, simple_linear_model):
        """Test upper bounds constraint identification."""
        r = Recorder(
            simple_linear_model,
            constraint_lower_bounds=2.0,
            constraint_upper_bounds=10.0,
        )
        assert r.is_constraint
        assert not r.is_lower_bounded_constraint
        assert r.is_double_bounded_constraint
        assert not r.is_upper_bounded_constraint
        assert not r.is_equality_constraint

    def test_invalid_bounds_constraint(self, simple_linear_model):
        """Test lower bounds greater than upper bounds."""
        with pytest.raises(ValueError):
            r = Recorder(
                simple_linear_model,
                constraint_lower_bounds=10.0,
                constraint_upper_bounds=2.0,
            )

    @pytest.mark.parametrize(
        "lb, ub",
        (
            (None, None),
            (5.0, 10.0),  # Feasible double bounds
            (15.0, 20.0),  # Infeasible double bounds
            (0.0, 2.0),  # Infeasible double bounds
            (0.0, None),  # Feasible lower bounds
            (15.0, None),  # Infeasible lower bounds
            (None, 15.0),  # Feasible upper bounds
            (None, 5.0),  # Infeasible upper bounds
        ),
    )
    def test_is_constraint_violated(self, simple_linear_model, lb, ub):
        """Test the calculation of a violated constraint and model feasibility."""
        m = simple_linear_model

        class TestRecorder(Recorder):
            def aggregated_value(self):
                return 10.0

        r = TestRecorder(m, constraint_lower_bounds=lb, constraint_upper_bounds=ub)

        if lb is None and ub is None:
            assert not r.is_constraint
            with pytest.raises(ValueError):
                r.is_constraint_violated()
        elif lb is None and ub is not None:
            # Upper bounded only
            if ub >= 10.0:
                assert not r.is_constraint_violated()
                assert m.is_feasible()
            else:
                assert r.is_constraint_violated()
                assert not m.is_feasible()
        elif lb is not None and ub is None:
            # Lower bounded only
            if lb <= 10.0:
                assert not r.is_constraint_violated()
                assert m.is_feasible()
            else:
                assert r.is_constraint_violated()
                assert not m.is_feasible()
        else:
            # Double bounds
            if lb <= 10.0 <= ub:
                assert not r.is_constraint_violated()
                assert m.is_feasible()
            else:
                assert r.is_constraint_violated()
                assert not m.is_feasible()


def test_numpy_recorder(simple_linear_model):
    """
    Test the NumpyArrayNodeRecorder
    """
    model = simple_linear_model
    otpt = model.nodes["Output"]

    model.nodes["Input"].max_flow = 10.0
    otpt.cost = -2.0
    rec = NumpyArrayNodeRecorder(model, otpt)

    # test retrieval of recorder
    assert model.recorders["numpyarraynoderecorder.Output"] == rec
    # test changing name of recorder
    rec.name = "timeseries.Output"
    assert model.recorders["timeseries.Output"] == rec
    with pytest.raises(KeyError):
        model.recorders["numpyarraynoderecorder.Output"]

    model.run()

    assert rec.data.shape == (365, 1)
    assert np.all((rec.data - 10.0) < 1e-12)

    df = rec.to_dataframe()
    assert df.shape == (365, 1)
    assert np.all((df.values - 10.0) < 1e-12)


def test_numpy_recorder_from_json(simple_linear_model):
    """Test loading NumpyArrayNodeRecorder from JSON style data"""

    model = simple_linear_model

    data = {"type": "numpyarraynode", "node": "Output"}

    rec = load_recorder(model, data)
    assert isinstance(rec, NumpyArrayNodeRecorder)


def test_numpy_recorder_factored(simple_linear_model):
    """Test the optional factor applies correctly"""

    model = simple_linear_model
    otpt = model.nodes["Output"]
    otpt.max_flow = 30.0
    model.nodes["Input"].max_flow = 10.0
    otpt.cost = -2

    factor = 2.0
    rec_fact = NumpyArrayNodeRecorder(model, otpt, factor=factor)

    model.run()

    assert rec_fact.data.shape == (365, 1)
    assert_allclose(20, rec_fact.data, atol=1e-7)


class TestNumpyArrayCostRecorder:
    def test_simple(self, simple_linear_model):
        """Test NumpyArrayCostRecorder with fixed cost"""
        model = simple_linear_model
        model.nodes["Input"].max_flow = 10.0
        otpt = model.nodes["Output"]
        otpt.cost = -2.0

        cost_rec = NumpyArrayNodeCostRecorder(model, otpt)
        # test retrieval of recorder
        assert model.recorders["numpyarraynodecostrecorder.Output"] == cost_rec
        model.run()

        assert cost_rec.data.shape == (365, 1)
        np.testing.assert_allclose(cost_rec.data, np.ones_like(cost_rec.data) * -2.0)

        df = cost_rec.to_dataframe()
        assert df.shape == (365, 1)
        np.testing.assert_allclose(df.values, np.ones_like(cost_rec.data) * -2.0)

    def test_scenarios(self, simple_linear_model):
        """Test NumpyArrayNodeCostRecorder with scenario varying costs."""
        model = simple_linear_model
        model.nodes["Input"].max_flow = 10.0
        otpt = model.nodes["Output"]
        scenario = Scenario(model, name="A", size=2)
        otpt.cost = ConstantScenarioParameter(
            model, values=[-2, -10], scenario=scenario
        )

        cost_rec = NumpyArrayNodeCostRecorder(model, otpt)
        # test retrieval of recorder
        assert model.recorders["numpyarraynodecostrecorder.Output"] == cost_rec
        model.run()

        assert cost_rec.data.shape == (365, 2)
        expected = np.empty_like(cost_rec.data)
        expected[:, 0] = -2.0
        expected[:, 1] = -10.0
        np.testing.assert_allclose(cost_rec.data, expected)

        df = cost_rec.to_dataframe()
        assert df.shape == (365, 2)
        np.testing.assert_allclose(df.values, expected)

    def test_time_varying(self, simple_linear_model):
        """Test NumpyArrayNodeCostRecorder with time varying costs."""
        model = simple_linear_model
        model.nodes["Input"].max_flow = 10.0
        otpt = model.nodes["Output"]

        costs = -2.0 - np.arange(365) * 0.1
        otpt.cost = ArrayIndexedParameter(model, costs)

        cost_rec = NumpyArrayNodeCostRecorder(model, otpt)
        # test retrieval of recorder
        assert model.recorders["numpyarraynodecostrecorder.Output"] == cost_rec
        model.run()

        assert cost_rec.data.shape == (365, 1)
        np.testing.assert_allclose(cost_rec.data, costs[:, np.newaxis])

        df = cost_rec.to_dataframe()
        assert df.shape == (365, 1)
        np.testing.assert_allclose(df.values, costs[:, np.newaxis])


class TestFlowDurationCurveRecorders:
    funcs = {"min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum}

    @pytest.mark.parametrize("agg_func", ["min", "max", "mean", "sum"])
    def test_fdc_recorder(self, agg_func):
        """
        Test the FlowDurationCurveRecorder
        """
        model = load_model("timeseries2.json")
        input = model.nodes["catchment1"]

        percentiles = np.linspace(20.0, 100.0, 5)
        rec = FlowDurationCurveRecorder(
            model, input, percentiles, temporal_agg_func=agg_func, agg_func="min"
        )

        # test retrieval of recorder
        assert model.recorders["flowdurationcurverecorder.catchment1"] == rec
        # test changing name of recorder
        rec.name = "timeseries.Input"
        assert model.recorders["timeseries.Input"] == rec
        with pytest.raises(KeyError):
            model.recorders["flowdurationcurverecorder.catchment1"]

        model.run()

        func = TestAggregatedRecorder.funcs[agg_func]

        assert_allclose(rec.fdc[:, 0], [20.42, 21.78, 23.22, 26.47, 29.31])
        assert_allclose(func(rec.fdc, axis=0), rec.values())
        assert_allclose(np.min(func(rec.fdc, axis=0)), rec.aggregated_value())

        assert rec.fdc.shape == (len(percentiles), len(model.scenarios.combinations))
        df = rec.to_dataframe()
        assert df.shape == (len(percentiles), len(model.scenarios.combinations))

    def test_seasonal_fdc_recorder(self):
        """
        Test the FlowDurationCurveRecorder
        """
        model = load_model("timeseries4.json")

        df = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), "models", "timeseries3.csv"),
            parse_dates=True,
            dayfirst=True,
            index_col=0,
        )

        percentiles = np.linspace(20.0, 100.0, 5)

        summer_flows = df.loc[
            pandas.Timestamp("2014-06-01") : pandas.Timestamp("2014-08-31"), :
        ]
        summer_fdc = np.percentile(summer_flows, percentiles, axis=0)

        model.run()

        rec = model.recorders["seasonal_fdc"]
        assert_allclose(rec.fdc, summer_fdc)

    @pytest.mark.parametrize(
        "agg_func, aggregate",
        [
            ("min", False),
            ("max", False),
            ("mean", False),
            ("mean", True),
            ("sum", False),
        ],
    )
    def test_fdc_dev_recorder(self, agg_func, aggregate):
        """
        Test the FlowDurationCurveDeviationRecorder
        """
        model = load_model("timeseries2.json")
        input = model.nodes["catchment1"]
        term = model.nodes["term1"]
        scenarioA = model.scenarios["scenario A"]

        natural_flow = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), "models", "timeseries2.csv"),
            parse_dates=True,
            dayfirst=True,
            index_col=0,
        )
        percentiles = np.linspace(20.0, 100.0, 5)

        natural_fdc = np.percentile(natural_flow, percentiles, axis=0)

        # Lower target is 20% below natural
        lower_input_fdc = natural_fdc * 0.8
        # Upper is 10% above
        upper_input_fdc = natural_fdc * 1.1

        if aggregate:
            # Setup only a single target for all scenarios.
            lower_input_fdc = lower_input_fdc.mean(axis=1)
            upper_input_fdc = upper_input_fdc.mean(axis=1)
            scenarioA = None

        rec = FlowDurationCurveDeviationRecorder.load(
            model,
            {
                "node": "term1",
                "percentiles": percentiles,
                "lower_target_fdc": lower_input_fdc,
                "upper_target_fdc": upper_input_fdc,
                "temporal_agg_func": agg_func,
                "agg_func": "mean",
                "scenario": "scenario A",
            },
        )

        # test retrieval of recorder
        assert model.recorders["flowdurationcurvedeviationrecorder.term1"] == rec
        # test changing name of recorder
        rec.name = "timeseries.Input"
        assert model.recorders["timeseries.Input"] == rec
        with pytest.raises(KeyError):
            model.recorders["flowdurationcurvedeviationrecorder.term1"]

        model.run()

        actual_fdc = np.maximum(natural_fdc - 23, 0.0)

        if aggregate:
            lower_input_fdc = lower_input_fdc[:, np.newaxis]
            upper_input_fdc = upper_input_fdc[:, np.newaxis]

        # Compute deviation
        lower_deviation = (lower_input_fdc - actual_fdc) / lower_input_fdc
        upper_deviation = (actual_fdc - upper_input_fdc) / upper_input_fdc
        deviation = np.maximum(
            np.maximum(lower_deviation, upper_deviation), np.zeros_like(lower_deviation)
        )

        func = TestAggregatedRecorder.funcs[agg_func]

        assert_allclose(rec.fdc_deviations[:, 0], deviation[:, 0])
        assert_allclose(func(rec.fdc_deviations, axis=0), rec.values())
        assert_allclose(
            np.mean(func(rec.fdc_deviations, axis=0)), rec.aggregated_value()
        )

        assert rec.fdc_deviations.shape == (
            len(percentiles),
            len(model.scenarios.combinations),
        )
        df = rec.to_dataframe()
        assert df.shape == (len(percentiles), len(model.scenarios.combinations))

    def test_deviation_single_target_lower(self):
        """Test deviation recorder with a lower target and no upper target"""

        model = load_model("timeseries2.json")
        input = model.nodes["catchment1"]
        term = model.nodes["term1"]
        scenarioA = model.scenarios["scenario A"]

        natural_flow = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), "models", "timeseries2.csv"),
            parse_dates=True,
            dayfirst=True,
            index_col=0,
        )
        percentiles = np.linspace(20.0, 100.0, 5)

        natural_fdc = np.percentile(natural_flow, percentiles, axis=0)

        # Lower target is 20% below natural
        lower_input_fdc = natural_fdc * 0.8

        rec = FlowDurationCurveDeviationRecorder(
            model,
            term,
            percentiles,
            lower_target_fdc=lower_input_fdc,
            scenario=scenarioA,
        )

        model.run()

        actual_fdc = np.maximum(natural_fdc - 23, 0.0)

        lower_deviation = (lower_input_fdc - actual_fdc) / lower_input_fdc
        deviation = np.maximum(lower_deviation, np.zeros_like(lower_deviation))
        assert_allclose(rec.fdc_deviations[:, 0], deviation[:, 0])

    def test_deviation_single_target_upper(self):
        """Test deviation recorder with an upper target and no lower target"""

        model = load_model("timeseries2.json")
        input = model.nodes["catchment1"]
        term = model.nodes["term1"]
        scenarioA = model.scenarios["scenario A"]

        natural_flow = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), "models", "timeseries2.csv"),
            parse_dates=True,
            dayfirst=True,
            index_col=0,
        )
        percentiles = np.linspace(20.0, 100.0, 5)

        natural_fdc = np.percentile(natural_flow, percentiles, axis=0)

        # Upper is 10% above
        upper_input_fdc = natural_fdc * 1.1

        rec = FlowDurationCurveDeviationRecorder(
            model,
            term,
            percentiles,
            upper_target_fdc=upper_input_fdc,
            scenario=scenarioA,
        )

        model.run()

        actual_fdc = np.maximum(natural_fdc - 23, 0.0)

        upper_deviation = (actual_fdc - upper_input_fdc) / upper_input_fdc
        deviation = np.maximum(upper_deviation, np.zeros_like(upper_deviation))
        assert_allclose(rec.fdc_deviations[:, 0], deviation[:, 0])

    def test_fdc_dev_from_json(self):
        """Test loading deviation recorder from json"""

        model = load_model("timeseries2_with_fdc.json")
        model.run()

        rec = model.recorders["fdc_dev1"]
        df = rec.to_dataframe()
        assert df.shape == (5, len(model.scenarios.combinations))

        rec = model.recorders["fdc_dev2"]
        df = rec.to_dataframe()
        assert df.shape == (5, len(model.scenarios.combinations))


def test_sdc_recorder():
    """
    Test the StorageDurationCurveRecorder
    """
    model = load_model("timeseries3.json")
    inpt = model.nodes["catchment1"]
    strg = model.nodes["reservoir1"]

    percentiles = np.linspace(20.0, 100.0, 5)
    flow_rec = NumpyArrayNodeRecorder(model, inpt)
    rec = StorageDurationCurveRecorder(
        model, strg, percentiles, temporal_agg_func="max", agg_func="min"
    )

    # test retrieval of recorder
    assert model.recorders["storagedurationcurverecorder.reservoir1"] == rec

    model.run()

    # Manually calculate expected storage and percentiles
    strg_volume = strg.initial_volume + np.cumsum(flow_rec.data - 23.0, axis=0)
    strg_pciles = np.percentile(strg_volume, percentiles, axis=0)

    assert_allclose(rec.sdc, strg_pciles)
    assert_allclose(np.max(rec.sdc, axis=0), rec.values())
    assert_allclose(np.min(np.max(rec.sdc, axis=0)), rec.aggregated_value())

    assert rec.sdc.shape == (len(percentiles), len(model.scenarios.combinations))
    df = rec.to_dataframe()
    assert df.shape == (len(percentiles), len(model.scenarios.combinations))


@pytest.mark.parametrize("proportional", [True, False])
def test_numpy_storage_recorder(simple_storage_model, proportional):
    """
    Test the NumpyArrayStorageRecorder
    """
    model = simple_storage_model

    res = model.nodes["Storage"]

    rec = NumpyArrayStorageRecorder(model, res, proportional=proportional)

    model.run()

    expected = np.array([[7, 4, 1, 0, 0]]).T
    if proportional:
        expected = expected / 20

    assert rec.data.shape == (5, 1)
    assert_allclose(rec.data, expected, atol=1e-7)

    df = rec.to_dataframe()
    assert df.shape == (5, 1)
    assert_allclose(df.values, expected, atol=1e-7)


def test_numpy_norm_storage_recorder(simple_storage_model):
    """
    Test the NumpyArrayNormalisedStorageRecorder
    """
    model = simple_storage_model

    res = model.nodes["Storage"]
    cc = ConstantParameter(model, value=0.2)

    rec = NumpyArrayNormalisedStorageRecorder(model, res, parameter=cc)

    model.run()

    expected = np.array(
        [
            [
                3.0 / 16,  # 3 above control curve
                0,  # at control curve
                1.0 / 4 - 1.0,  # 75% toward empty
                -1.0,  # Empty
                -1.0,  # Empty
            ]
        ]
    ).T

    assert rec.data.shape == (5, 1)
    assert_allclose(rec.data, expected, atol=1e-7)

    df = rec.to_dataframe()
    assert df.shape == (5, 1)
    assert_allclose(df.values, expected, atol=1e-7)


def test_numpy_array_level_recorder(simple_storage_model):
    model = simple_storage_model

    storage = model.nodes["Storage"]
    level_param = InterpolatedVolumeParameter(model, storage, [0, 20], [0, 100])
    storage.level = level_param
    level_rec = NumpyArrayLevelRecorder(model, storage, temporal_agg_func="min")

    model.run()

    expected = np.array([[50, 35, 20, 5, 0]]).T
    assert_allclose(level_rec.data, expected, atol=1e-7)

    df = level_rec.to_dataframe()
    assert df.shape == (5, 1)
    assert_allclose(df.values, expected, atol=1e-7)

    assert_allclose(level_rec.aggregated_value(), np.min(expected))


def test_numpy_array_area_recorder(simple_storage_model):

    model = simple_storage_model

    storage = model.nodes["Storage"]
    area_param = InterpolatedVolumeParameter(model, storage, [0, 20], [0, 100])
    storage.area = area_param
    area_rec = NumpyArrayAreaRecorder(model, storage, temporal_agg_func="min")

    model.run()

    expected = np.array([[50, 35, 20, 5, 0]]).T
    assert_allclose(area_rec.data, expected, atol=1e-7)

    df = area_rec.to_dataframe()
    assert df.shape == (5, 1)
    assert_allclose(df.values, expected, atol=1e-7)

    assert_allclose(area_rec.aggregated_value(), np.min(expected))


def test_numpy_parameter_recorder(simple_linear_model):
    """
    Test the NumpyArrayParameterRecorder
    """
    from pywr.parameters import DailyProfileParameter

    model = simple_linear_model
    # using leap year simplifies tests
    model.timestepper.start = pandas.to_datetime("2016-01-01")
    model.timestepper.end = pandas.to_datetime("2016-12-31")
    otpt = model.nodes["Output"]

    p = DailyProfileParameter(
        model,
        np.arange(366, dtype=np.float64),
    )
    p.name = "daily profile"
    model.nodes["Input"].max_flow = p
    otpt.cost = -2.0
    rec = NumpyArrayParameterRecorder(model, model.nodes["Input"].max_flow)

    # test retrieval of recorder
    assert model.recorders["numpyarrayparameterrecorder.daily profile"] == rec

    model.run()

    assert rec.data.shape == (366, 1)
    assert_allclose(rec.data, np.arange(366, dtype=np.float64)[:, np.newaxis])

    df = rec.to_dataframe()
    assert df.shape == (366, 1)
    assert_allclose(df.values, np.arange(366, dtype=np.float64)[:, np.newaxis])


def test_numpy_daily_profile_parameter_recorder(simple_linear_model):
    """
    Test the NumpyArrayDailyProfileParameterRecorder
    """
    from pywr.parameters import DailyProfileParameter

    model = simple_linear_model
    # using leap year simplifies tests
    model.timestepper.start = pandas.to_datetime("2016-01-01")
    model.timestepper.end = pandas.to_datetime("2017-12-31")
    otpt = model.nodes["Output"]

    p = DailyProfileParameter(
        model,
        np.arange(366, dtype=np.float64),
    )
    p.name = "daily profile"
    model.nodes["Input"].max_flow = p
    otpt.cost = -2.0
    rec = NumpyArrayDailyProfileParameterRecorder(model, model.nodes["Input"].max_flow)

    # test retrieval of recorder
    assert (
        model.recorders["numpyarraydailyprofileparameterrecorder.daily profile"] == rec
    )

    model.run()

    assert rec.data.shape == (366, 1)
    assert_allclose(rec.data, np.arange(366, dtype=np.float64)[:, np.newaxis])

    df = rec.to_dataframe()
    assert df.shape == (366, 1)
    assert_allclose(df.values, np.arange(366, dtype=np.float64)[:, np.newaxis])


def test_numpy_index_parameter_recorder(simple_storage_model):
    """
    Test the NumpyArrayIndexParameterRecorder

    Note the parameter is recorded at the start of the timestep, while the
    storage is recorded at the end of the timestep.
    """
    from pywr.parameters.control_curves import ControlCurveIndexParameter

    model = simple_storage_model

    res = model.nodes["Storage"]

    p = ControlCurveIndexParameter(model, res, [5.0 / 20.0, 2.5 / 20.0])

    res_rec = NumpyArrayStorageRecorder(model, res)
    lvl_rec = NumpyArrayIndexParameterRecorder(model, p)

    model.run()

    assert res_rec.data.shape == (5, 1)
    assert_allclose(res_rec.data, np.array([[7, 4, 1, 0, 0]]).T, atol=1e-7)
    assert lvl_rec.data.shape == (5, 1)
    assert_allclose(lvl_rec.data, np.array([[0, 0, 1, 2, 2]]).T, atol=1e-7)

    df = lvl_rec.to_dataframe()
    assert df.shape == (5, 1)
    assert_allclose(df.values, np.array([[0, 0, 1, 2, 2]]).T, atol=1e-7)


def test_parameter_recorder_json():
    model = load_model("parameter_recorder.json")
    rec_demand = model.recorders["demand_max_recorder"]
    rec_supply = model.recorders["supply_max_recorder"]
    model.run()
    assert_allclose(rec_demand.data, 10)
    assert_allclose(rec_supply.data, 15)


def test_nested_recorder_json():
    model = load_model("agg_recorder_nesting.json")
    rec_demand = model.recorders["demand_max_recorder"]
    rec_supply = model.recorders["supply_max_recorder"]
    rec_total = model.recorders["max_recorder"]
    model.run()
    assert_allclose(rec_demand.aggregated_value(), 10)
    assert_allclose(rec_supply.aggregated_value(), 15)
    assert_allclose(rec_total.aggregated_value(), 25)


@pytest.fixture()
def daily_profile_model(simple_linear_model):
    model = simple_linear_model
    # using leap year simplifies test
    model.timestepper.start = pandas.to_datetime("2016-01-01")
    model.timestepper.end = pandas.to_datetime("2016-12-31")

    node = model.nodes["Input"]
    values = np.arange(0, 366, dtype=np.float64)
    node.max_flow = DailyProfileParameter(model, values, name="profile")
    return model


def test_parameter_mean_recorder(daily_profile_model):
    model = daily_profile_model
    node = model.nodes["Input"]
    scenario = Scenario(model, "dummy", size=3)

    timesteps = 3
    rec_mean = RollingWindowParameterRecorder(
        model, node.max_flow, timesteps, temporal_agg_func="mean", name="rec_mean"
    )
    rec_sum = RollingWindowParameterRecorder(
        model, node.max_flow, timesteps, temporal_agg_func="sum", name="rec_sum"
    )
    rec_min = RollingWindowParameterRecorder(
        model, node.max_flow, timesteps, temporal_agg_func="min", name="rec_min"
    )
    rec_max = RollingWindowParameterRecorder(
        model, node.max_flow, timesteps, temporal_agg_func="max", name="rec_max"
    )

    model.run()

    assert_allclose(rec_mean.data[[0, 1, 2, 3, 364], 0], [0, 0.5, 1, 2, 363])
    assert_allclose(rec_max.data[[0, 1, 2, 3, 364], 0], [0, 1, 2, 3, 364])
    assert_allclose(rec_min.data[[0, 1, 2, 3, 364], 0], [0, 0, 0, 1, 362])
    assert_allclose(rec_sum.data[[0, 1, 2, 3, 364], 0], [0, 1, 3, 6, 1089])


def test_parameter_mean_recorder_json(simple_linear_model):
    model = simple_linear_model
    node = model.nodes["Input"]
    values = np.arange(0, 366, dtype=np.float64)
    parameter = DailyProfileParameter(model, values, name="input_max_flow")

    node.max_flow = parameter

    data = {
        "type": "rollingwindowparameter",
        "parameter": "input_max_flow",
        "window": 3,
        "temporal_agg_func": "mean",
    }

    rec = load_recorder(model, data)


class TestTotalParameterRecorder:
    @pytest.mark.parametrize(
        "factor, integrate", [[1.0, False], [1.0, True], [2.0, False], [0.5, True]]
    )
    def test_values(self, daily_profile_model, factor, integrate):
        model = daily_profile_model
        model.timestepper.delta = 2
        param = model.parameters["profile"]
        rec = TotalParameterRecorder(
            model, param, name="total", factor=factor, integrate=integrate
        )
        model.run()

        expected = np.arange(0, 366, dtype=np.float64)[::2].sum() * factor
        if integrate:
            expected *= 2
        assert_allclose(rec.values(), expected)

    @pytest.mark.parametrize("integrate", [None, True, False])
    def test_from_json(self, daily_profile_model, integrate):

        model = daily_profile_model

        data = {
            "type": "totalparameter",
            "parameter": "profile",
            "agg_func": "mean",
            "factor": 2.0,
        }

        if integrate is not None:
            data["integrate"] = integrate

        rec = load_recorder(model, data)
        assert rec.factor == 2.0

        if integrate is not None:
            assert rec.integrate == integrate
        else:
            assert not rec.integrate


class TestMeanParameterRecorder:
    @pytest.mark.parametrize("factor", [1.0, 2.0, 0.5])
    def test_values(self, daily_profile_model, factor):
        model = daily_profile_model
        model.timestepper.delta = 2
        param = model.parameters["profile"]
        rec = MeanParameterRecorder(model, param, name="mean", factor=factor)
        model.run()

        expected = np.arange(0, 366, dtype=np.float64)[::2].mean() * factor
        assert_allclose(rec.values(), expected)

    def test_from_json(self, daily_profile_model):

        model = daily_profile_model

        data = {
            "type": "meanparameter",
            "parameter": "profile",
            "agg_func": "mean",
            "factor": 2.0,
        }

        rec = load_recorder(model, data)
        assert rec.factor == 2.0


def test_concatenated_dataframes(simple_storage_model):
    """
    Test that Model.to_dataframe returns something sensible.

    """
    model = simple_storage_model

    scA = Scenario(model, "A", size=2)
    scB = Scenario(model, "B", size=3)

    res = model.nodes["Storage"]
    rec1 = NumpyArrayStorageRecorder(model, res)
    otpt = model.nodes["Output"]
    rec2 = NumpyArrayNodeRecorder(model, otpt)
    # The following can't return a DataFrame; is included to check
    # it doesn't cause any issues
    rec3 = TotalDeficitNodeRecorder(model, otpt)

    model.run()

    df = model.to_dataframe()
    assert df.shape == (5, 2 * 2 * 3)
    assert df.columns.names == ["Recorder", "A", "B"]


@pytest.mark.parametrize("complib", [None, "gzip", "bz2"])
def test_csv_recorder(simple_storage_model, tmpdir, complib):
    """Test the CSV Recorder"""
    model = simple_storage_model
    otpt = model.nodes["Output"]
    otpt.cost = -2.0

    # Rename output to a unicode character to check encoding to files
    otpt.name = "\u03A9"
    expected_header = ["Datetime", "Input", "Storage", "\u03A9"]

    csvfile = tmpdir.join("output.csv")
    # By default the CSVRecorder saves all nodes in alphabetical order
    # and scenario index 0.
    rec = CSVRecorder(model, str(csvfile), complib=complib, complevel=5)

    model.run()

    import csv

    kwargs = {"encoding": "utf-8"}
    mode = "rt"

    if complib == "gzip":
        import gzip

        fh = gzip.open(str(csvfile), mode, **kwargs)
    elif complib in ("bz2", "bzip2"):
        import bz2

        fh = bz2.open(str(csvfile), mode, **kwargs)
    else:
        fh = open(str(csvfile), mode, **kwargs)

    expected = [
        expected_header,
        ["2016-01-01T00:00:00", 5.0, 7.0, 8.0],
        ["2016-01-02T00:00:00", 5.0, 4.0, 8.0],
        ["2016-01-03T00:00:00", 5.0, 1.0, 8.0],
        ["2016-01-04T00:00:00", 5.0, 0.0, 6.0],
        ["2016-01-05T00:00:00", 5.0, 0.0, 5.0],
    ]

    data = fh.read(1024)
    dialect = csv.Sniffer().sniff(data)
    fh.seek(0)
    reader = csv.reader(fh, dialect)

    for irow, row in enumerate(reader):
        expected_row = expected[irow]
        if irow == 0:
            assert expected_row == row
        else:
            assert expected_row[0] == row[0]  # Check datetime
            # Check values
            np.testing.assert_allclose([float(v) for v in row[1:]], expected_row[1:])
    fh.close()


def test_loading_csv_recorder_from_json(tmpdir):
    """
    Test the CSV Recorder which is loaded from json
    """

    filename = "csv_recorder.json"

    # This is a bit horrible, but need to edit the JSON dynamically
    # so that the output.h5 is written in the temporary directory
    path = os.path.join(os.path.dirname(__file__), "models")
    with open(os.path.join(path, filename), "r") as f:
        data = f.read()
    data = json.loads(data)

    # Make an absolute, but temporary, path for the recorder
    url = data["recorders"]["model_out"]["url"]
    data["recorders"]["model_out"]["url"] = str(tmpdir.join(url))

    model = Model.load(data, path=path)

    csvfile = tmpdir.join("output.csv")
    model.run()

    periods = model.timestepper.datetime_index

    import csv

    with open(str(csvfile), "r") as fh:
        dialect = csv.Sniffer().sniff(fh.read(1024))
        fh.seek(0)
        reader = csv.reader(fh, dialect)
        for irow, row in enumerate(reader):
            if irow == 0:
                expected = ["Datetime", "inpt", "otpt"]
                actual = row
            else:
                dt = periods[irow - 1].to_timestamp()
                expected = [dt.isoformat()]
                actual = [row[0]]
                assert np.all((np.array([float(v) for v in row[1:]]) - 10.0) < 1e-12)
            assert expected == actual


class TestTablesRecorder:
    def test_create_directory(self, simple_linear_model, tmpdir):
        """Test TablesRecorder to create a new directory"""

        model = simple_linear_model
        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]
        agg_node = AggregatedNode(model, "Sum", [otpt, inpt])

        inpt.max_flow = 10.0
        otpt.cost = -2.0
        # Make a path with a new directory
        folder = tmpdir.join("outputs")
        h5file = folder.join("output.h5")
        assert not folder.exists()
        rec = TablesRecorder(model, str(h5file), create_directories=True)
        model.run()
        assert folder.exists()
        assert h5file.exists()

    def test_nodes(self, simple_linear_model, tmpdir):
        """
        Test the TablesRecorder

        """
        model = simple_linear_model
        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]
        agg_node = AggregatedNode(model, "Sum", [otpt, inpt])

        inpt.max_flow = 10.0
        otpt.cost = -2.0

        h5file = tmpdir.join("output.h5")
        import tables

        with tables.open_file(str(h5file), "w") as h5f:
            rec = TablesRecorder(model, h5f)

            model.run()

            for node_name in model.nodes.keys():
                ca = h5f.get_node("/", node_name)
                assert ca.shape == (365, 1)
                if node_name == "Sum":
                    np.testing.assert_allclose(ca, 20.0)
                else:
                    np.testing.assert_allclose(ca, 10.0)

            from datetime import date, timedelta

            d = date(2015, 1, 1)
            time = h5f.get_node("/time")
            for i in range(len(model.timestepper)):
                row = time[i]
                assert row["year"] == d.year
                assert row["month"] == d.month
                assert row["day"] == d.day

                d += timedelta(1)

            scenarios = h5f.get_node("/scenarios")
            for i, s in enumerate(model.scenarios.scenarios):
                row = scenarios[i]
                assert row["name"] == s.name.encode("utf-8")
                assert row["size"] == s.size

            model.reset()
            model.run()

            time = h5f.get_node("/time")
            assert len(time) == len(model.timestepper)

    def test_multiple_scenarios(self, simple_linear_model, tmpdir):
        """
        Test the TablesRecorder

        """
        from pywr.parameters import ConstantScenarioParameter

        model = simple_linear_model
        scA = Scenario(model, name="A", size=4)
        scB = Scenario(model, name="B", size=2)

        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]

        inpt.max_flow = ConstantScenarioParameter(model, scA, [10, 20, 30, 40])
        otpt.max_flow = ConstantScenarioParameter(model, scB, [20, 40])
        otpt.cost = -2.0

        h5file = tmpdir.join("output.h5")
        import tables

        with tables.open_file(str(h5file), "w") as h5f:
            rec = TablesRecorder(model, h5f)

            model.run()

            for node_name in model.nodes.keys():
                ca = h5f.get_node("/", node_name)
                assert ca.shape == (365, 4, 2)
                np.testing.assert_allclose(
                    ca[0, ...], [[10, 10], [20, 20], [20, 30], [20, 40]]
                )

            scenarios = h5f.get_node("/scenarios")
            for i, s in enumerate(model.scenarios.scenarios):
                row = scenarios[i]
                assert row["name"] == s.name.encode("utf-8")
                assert row["size"] == s.size

    def test_user_scenarios(self, simple_linear_model, tmpdir):
        """
        Test the TablesRecorder with user defined scenario subset

        """
        from pywr.parameters import ConstantScenarioParameter

        model = simple_linear_model
        scA = Scenario(model, name="A", size=4)
        scB = Scenario(model, name="B", size=2)

        # Use first and last combinations
        model.scenarios.user_combinations = [[0, 0], [3, 1]]

        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]

        inpt.max_flow = ConstantScenarioParameter(model, scA, [10, 20, 30, 40])
        otpt.max_flow = ConstantScenarioParameter(model, scB, [20, 40])
        otpt.cost = -2.0

        h5file = tmpdir.join("output.h5")
        import tables

        with tables.open_file(str(h5file), "w") as h5f:
            rec = TablesRecorder(model, h5f)

            model.run()

            for node_name in model.nodes.keys():
                ca = h5f.get_node("/", node_name)
                assert ca.shape == (365, 2)
                np.testing.assert_allclose(ca[0, ...], [10, 40])

            # check combinations table exists
            combinations = h5f.get_node("/scenario_combinations")
            for i, comb in enumerate(model.scenarios.user_combinations):
                row = combinations[i]
                assert row["A"] == comb[0]
                assert row["B"] == comb[1]

    def test_parameters(self, simple_linear_model, tmpdir):
        """
        Test the TablesRecorder

        """
        from pywr.parameters import ConstantParameter

        model = simple_linear_model
        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]

        p = ConstantParameter(model, 10.0, name="max_flow")
        inpt.max_flow = p

        # ensure TablesRecorder can handle parameters with a / in the name
        p_slash = ConstantParameter(model, 0.0, name="name with a / in it")
        inpt.min_flow = p_slash

        agg_node = AggregatedNode(model, "Sum", [otpt, inpt])

        inpt.max_flow = 10.0
        otpt.cost = -2.0

        h5file = tmpdir.join("output.h5")
        import tables

        with tables.open_file(str(h5file), "w") as h5f:
            with pytest.warns(ParameterNameWarning):
                rec = TablesRecorder(model, h5f, parameters=[p, p_slash])

            # check parameters have been added to the component tree
            # this is particularly important for parameters which update their
            # values in `after`, e.g. DeficitParameter (see #465)
            assert not model.find_orphaned_parameters()
            assert p in rec.children
            assert p_slash in rec.children

            with pytest.warns(tables.NaturalNameWarning):
                model.run()

            for node_name in model.nodes.keys():
                ca = h5f.get_node("/", node_name)
                assert ca.shape == (365, 1)
                if node_name == "Sum":
                    np.testing.assert_allclose(ca, 20.0)
                elif "name with a" in node_name:
                    assert node_name == "name with a _ in it"
                    np.testing.assert_allclose(ca, 0.0)
                else:
                    np.testing.assert_allclose(ca, 10.0)

    def test_nodes_with_str(self, simple_linear_model, tmpdir):
        """
        Test the TablesRecorder

        """
        from pywr.parameters import ConstantParameter

        model = simple_linear_model
        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]
        agg_node = AggregatedNode(model, "Sum", [otpt, inpt])
        p = ConstantParameter(model, 10.0, name="max_flow")
        inpt.max_flow = p

        otpt.cost = -2.0

        h5file = tmpdir.join("output.h5")
        import tables

        with tables.open_file(str(h5file), "w") as h5f:
            nodes = ["Output", "Input", "Sum"]
            where = "/agroup"
            rec = TablesRecorder(
                model,
                h5f,
                nodes=nodes,
                parameters=[
                    p,
                ],
                where=where,
            )

            model.run()

            for node_name in ["Output", "Input", "Sum", "max_flow"]:
                ca = h5f.get_node("/agroup/" + node_name)
                assert ca.shape == (365, 1)
                if node_name == "Sum":
                    np.testing.assert_allclose(ca, 20.0)
                else:
                    np.testing.assert_allclose(ca, 10.0)

    def test_demand_saving_with_indexed_array(self, tmpdir):
        """Test recording various items from demand saving example"""
        model = load_model("demand_saving2.json")

        model.timestepper.end = "2016-01-31"

        model.check()

        h5file = tmpdir.join("output.h5")
        import tables

        with tables.open_file(str(h5file), "w") as h5f:

            nodes = [
                ("/outputs/demand", "Demand"),
                ("/storage/reservoir", "Reservoir"),
            ]

            parameters = [
                ("/parameters/demand_saving_level", "demand_saving_level"),
            ]

            rec = TablesRecorder(model, h5f, nodes=nodes, parameters=parameters)

            model.run()

            max_volume = model.nodes["Reservoir"].max_volume
            rec_demand = h5f.get_node("/outputs/demand").read()
            rec_storage = h5f.get_node("/storage/reservoir").read()

            # model starts with no demand saving
            demand_baseline = 50.0
            demand_factor = 0.9  # jan-apr
            demand_saving = 1.0
            assert_allclose(
                rec_demand[0, 0], demand_baseline * demand_factor * demand_saving
            )

            # first control curve breached
            demand_saving = 0.95
            assert rec_storage[4, 0] < (0.8 * max_volume)
            assert_allclose(
                rec_demand[5, 0], demand_baseline * demand_factor * demand_saving
            )

            # second control curve breached
            demand_saving = 0.5
            assert rec_storage[11, 0] < (0.5 * max_volume)
            assert_allclose(
                rec_demand[12, 0], demand_baseline * demand_factor * demand_saving
            )

    def test_demand_saving_with_indexed_array_from_json(self, tmpdir):
        """Test recording various items from demand saving example.

        This time the TablesRecorder is defined in JSON.
        """
        filename = "demand_saving_with_tables_recorder.json"
        # This is a bit horrible, but need to edit the JSON dynamically
        # so that the output.h5 is written in the temporary directory
        path = os.path.join(os.path.dirname(__file__), "models")
        with open(os.path.join(path, filename), "r") as f:
            data = f.read()
        data = json.loads(data)

        # Make an absolute, but temporary, path for the recorder
        url = data["recorders"]["database"]["url"]
        data["recorders"]["database"]["url"] = str(tmpdir.join(url))

        model = Model.load(data, path=path)

        model.timestepper.end = "2016-01-31"
        model.check()

        # run model
        model.run()

        # run model again (to test reset behaviour)
        model.run()
        max_volume = model.nodes["Reservoir"].max_volume

        h5file = tmpdir.join("output.h5")
        with tables.open_file(str(h5file), "r") as h5f:
            assert model.metadata["title"] == h5f.title
            # Check metadata on root node
            assert h5f.root._v_attrs.author == "pytest"
            assert h5f.root._v_attrs.run_number == 0

            rec_demand = h5f.get_node("/outputs/demand").read()
            rec_storage = h5f.get_node("/storage/reservoir").read()

            # model starts with no demand saving
            demand_baseline = 50.0
            demand_factor = 0.9  # jan-apr
            demand_saving = 1.0
            assert_allclose(
                rec_demand[0, 0], demand_baseline * demand_factor * demand_saving
            )

            # first control curve breached
            demand_saving = 0.95
            assert rec_storage[4, 0] < (0.8 * max_volume)
            assert_allclose(
                rec_demand[5, 0], demand_baseline * demand_factor * demand_saving
            )

            # second control curve breached
            demand_saving = 0.5
            assert rec_storage[11, 0] < (0.5 * max_volume)
            assert_allclose(
                rec_demand[12, 0], demand_baseline * demand_factor * demand_saving
            )

    @pytest.mark.skipif(
        Model().solver.name == "glpk-edge",
        reason="Not valid for GLPK Edge based solver.",
    )
    def test_routes(self, simple_linear_model, tmpdir):
        """
        Test the TablesRecorder

        """
        model = simple_linear_model
        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]
        agg_node = AggregatedNode(model, "Sum", [otpt, inpt])

        inpt.max_flow = 10.0
        otpt.cost = -2.0

        h5file = tmpdir.join("output.h5")
        import tables

        with tables.open_file(str(h5file), "w") as h5f:
            rec = TablesRecorder(model, h5f, routes_flows="flows")

            model.run()

            flows = h5f.get_node("/flows")
            assert flows.shape == (365, 1, 1)
            np.testing.assert_allclose(flows.read(), np.ones((365, 1, 1)) * 10)

            routes = h5f.get_node("/routes")
            assert routes.shape[0] == 1
            row = routes[0]
            row["start"] = "Input"
            row["end"] = "Output"

            from datetime import date, timedelta

            d = date(2015, 1, 1)
            time = h5f.get_node("/time")
            for i in range(len(model.timestepper)):
                row = time[i]
                assert row["year"] == d.year
                assert row["month"] == d.month
                assert row["day"] == d.day

                d += timedelta(1)

            scenarios = h5f.get_node("/scenarios")
            for s in model.scenarios.scenarios:
                row = scenarios[i]
                assert row["name"] == s.name
                assert row["size"] == s.size

            model.reset()
            model.run()

            time = h5f.get_node("/time")
            assert len(time) == len(model.timestepper)

    @pytest.mark.skipif(
        Model().solver.name == "glpk-edge",
        reason="Not valid for GLPK Edge based solver.",
    )
    def test_routes_multiple_scenarios(self, simple_linear_model, tmpdir):
        """
        Test the TablesRecorder

        """
        from pywr.parameters import ConstantScenarioParameter

        model = simple_linear_model
        scA = Scenario(model, name="A", size=4)
        scB = Scenario(model, name="B", size=2)

        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]

        inpt.max_flow = ConstantScenarioParameter(model, scA, [10, 20, 30, 40])
        otpt.max_flow = ConstantScenarioParameter(model, scB, [20, 40])
        otpt.cost = -2.0

        h5file = tmpdir.join("output.h5")
        import tables

        with tables.open_file(str(h5file), "w") as h5f:
            rec = TablesRecorder(model, h5f, routes_flows="flows")

            model.run()

            flows = h5f.get_node("/flows")
            assert flows.shape == (365, 1, 4, 2)
            np.testing.assert_allclose(
                flows[0, 0], [[10, 10], [20, 20], [20, 30], [20, 40]]
            )

    @pytest.mark.skipif(
        Model().solver.name == "glpk-edge",
        reason="Not valid for GLPK Edge based solver.",
    )
    def test_routes_user_scenarios(self, simple_linear_model, tmpdir):
        """
        Test the TablesRecorder with user defined scenario subset

        """
        from pywr.parameters import ConstantScenarioParameter

        model = simple_linear_model
        scA = Scenario(model, name="A", size=4)
        scB = Scenario(model, name="B", size=2)

        # Use first and last combinations
        model.scenarios.user_combinations = [[0, 0], [3, 1]]

        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]

        inpt.max_flow = ConstantScenarioParameter(model, scA, [10, 20, 30, 40])
        otpt.max_flow = ConstantScenarioParameter(model, scB, [20, 40])
        otpt.cost = -2.0

        h5file = tmpdir.join("output.h5")
        import tables

        with tables.open_file(str(h5file), "w") as h5f:
            rec = TablesRecorder(model, h5f, routes_flows="flows")

            model.run()

            flows = h5f.get_node("/flows")
            assert flows.shape == (365, 1, 2)
            np.testing.assert_allclose(flows[0, 0], [10, 40])

            # check combinations table exists
            combinations = h5f.get_node("/scenario_combinations")
            for i, comb in enumerate(model.scenarios.user_combinations):
                row = combinations[i]
                assert row["A"] == comb[0]
                assert row["B"] == comb[1]

        # This part of the test requires IPython (see `pywr.notebook`)
        pytest.importorskip(
            "IPython"
        )  # triggers a skip of the test if IPython not found.
        from pywr.notebook.sankey import routes_to_sankey_links

        links = routes_to_sankey_links(str(h5file), "flows")
        # Value is mean of 10 and 40

        link = links[0]
        assert link["source"] == "Input"
        assert link["target"] == "Output"
        np.testing.assert_allclose(link["value"], 25.0)

        links = routes_to_sankey_links(str(h5file), "flows", scenario_slice=0)
        link = links[0]
        assert link["source"] == "Input"
        assert link["target"] == "Output"
        np.testing.assert_allclose(link["value"], 10.0)

        links = routes_to_sankey_links(
            str(h5file), "flows", scenario_slice=1, time_slice=0
        )
        link = links[0]
        assert link["source"] == "Input"
        assert link["target"] == "Output"
        np.testing.assert_allclose(link["value"], 40.0)

    def test_generate_dataframes(self, simple_linear_model, tmpdir):
        """Test TablesRecorder.generate_dataframes"""
        from pywr.parameters import ConstantScenarioParameter

        model = simple_linear_model
        scA = Scenario(model, name="A", size=4)
        scB = Scenario(model, name="B", size=2)

        otpt = model.nodes["Output"]
        inpt = model.nodes["Input"]

        inpt.max_flow = ConstantScenarioParameter(model, scA, [10, 20, 30, 40])
        otpt.max_flow = ConstantScenarioParameter(model, scB, [20, 40])
        otpt.cost = -2.0

        h5file = tmpdir.join("output.h5")
        TablesRecorder(model, h5file)
        model.run()

        dfs = {}
        for node, df in TablesRecorder.generate_dataframes(h5file):
            dfs[node] = df

        for node_name in model.nodes.keys():
            df = dfs[node_name]
            assert df.shape == (365, 8)
            np.testing.assert_allclose(df.iloc[0, :], [10, 10, 20, 20, 20, 30, 20, 40])


class TestDeficitRecorders:
    @pytest.mark.parametrize("demand", [30.0, 0.0])
    def test_total_deficit_node_recorder(self, simple_linear_model, demand):
        """
        Test TotalDeficitNodeRecorder
        """
        model = simple_linear_model
        model.timestepper.delta = 5
        otpt = model.nodes["Output"]
        otpt.max_flow = demand
        model.nodes["Input"].max_flow = 10.0
        otpt.cost = -2.0
        rec = TotalDeficitNodeRecorder(model, otpt)

        model.step()
        assert_allclose(max(demand - 10.0, 0.0) * 5, rec.aggregated_value(), atol=1e-7)

        model.step()
        assert_allclose(
            2 * max(demand - 10.0, 0.0) * 5, rec.aggregated_value(), atol=1e-7
        )

    def test_array_deficit_recoder(self, simple_linear_model):
        """Test `NumpyArrayNodeDeficitRecorder`"""
        model = simple_linear_model
        model.timestepper.delta = 1
        otpt = model.nodes["Output"]

        inflow = np.arange(365) * 0.1
        demand = np.ones_like(inflow) * 30.0

        model.nodes["Input"].max_flow = ArrayIndexedParameter(model, inflow)
        otpt.max_flow = ArrayIndexedParameter(model, demand)
        otpt.cost = -2.0

        expected_supply = np.minimum(inflow, demand)
        expected_deficit = demand - expected_supply

        rec = NumpyArrayNodeDeficitRecorder(model, otpt)

        model.run()

        assert rec.data.shape == (365, 1)
        np.testing.assert_allclose(expected_deficit[:, np.newaxis], rec.data)

        df = rec.to_dataframe()
        assert df.shape == (365, 1)
        np.testing.assert_allclose(expected_deficit[:, np.newaxis], df.values)

    @pytest.mark.parametrize("demand_factor", [30.0, 0.0])
    def test_array_supplied_ratio_recoder(self, simple_linear_model, demand_factor):
        """Test `NumpyArrayNodeSuppliedRatioRecorder`"""
        model = simple_linear_model
        model.timestepper.delta = 1
        otpt = model.nodes["Output"]

        inflow = np.arange(365) * 0.1
        demand = np.ones_like(inflow) * demand_factor

        model.nodes["Input"].max_flow = ArrayIndexedParameter(model, inflow)
        otpt.max_flow = ArrayIndexedParameter(model, demand)
        otpt.cost = -2.0

        expected_supply = np.minimum(inflow, demand)
        if demand_factor == 0.0:
            expected_ratio = np.ones_like(expected_supply)
        else:
            expected_ratio = expected_supply / demand

        rec = NumpyArrayNodeSuppliedRatioRecorder(model, otpt)

        model.run()

        assert rec.data.shape == (365, 1)
        np.testing.assert_allclose(expected_ratio[:, np.newaxis], rec.data)

        df = rec.to_dataframe()
        assert df.shape == (365, 1)
        np.testing.assert_allclose(expected_ratio[:, np.newaxis], df.values)

    @pytest.mark.parametrize("demand_factor", [30.0, 0.0])
    def test_array_curtailment_ratio_recoder(self, simple_linear_model, demand_factor):
        """Test `NumpyArrayNodeCurtailmentRatioRecorder`"""
        model = simple_linear_model
        model.timestepper.delta = 1
        otpt = model.nodes["Output"]

        inflow = np.arange(365) * 0.1
        demand = np.ones_like(inflow) * demand_factor

        model.nodes["Input"].max_flow = ArrayIndexedParameter(model, inflow)
        otpt.max_flow = ArrayIndexedParameter(model, demand)
        otpt.cost = -2.0

        expected_supply = np.minimum(inflow, demand)
        if demand_factor == 0.0:
            expected_curtailment_ratio = np.zeros_like(expected_supply)
        else:
            expected_curtailment_ratio = 1 - expected_supply / demand

        rec = NumpyArrayNodeCurtailmentRatioRecorder(model, otpt)

        model.run()

        assert rec.data.shape == (365, 1)
        np.testing.assert_allclose(expected_curtailment_ratio[:, np.newaxis], rec.data)

        df = rec.to_dataframe()
        assert df.shape == (365, 1)
        np.testing.assert_allclose(expected_curtailment_ratio[:, np.newaxis], df.values)


def test_timestep_count_index_parameter_recorder(simple_storage_model):
    """
    The test uses a simple reservoir model with different inputs that
    trigger a control curve failure after a different number of years.
    """
    from pywr.parameters import ConstantScenarioParameter, ConstantParameter
    from pywr.parameters.control_curves import ControlCurveIndexParameter

    model = simple_storage_model
    scenario = Scenario(model, "A", size=2)
    # Simulate 5 years
    model.timestepper.start = "2015-01-01"
    model.timestepper.end = "2019-12-31"
    # Control curve parameter
    param = ControlCurveIndexParameter(
        model, model.nodes["Storage"], ConstantParameter(model, 0.25)
    )

    # Storage model has a capacity of 20, but starts at 10 Ml
    # Demand is roughly 2 Ml/d per year
    #  First ensemble balances the demand
    #  Second ensemble should fail during 3rd year
    demand = 2.0 / 365
    model.nodes["Input"].max_flow = ConstantScenarioParameter(
        model, scenario, [demand, 0]
    )
    model.nodes["Output"].max_flow = demand

    # Create the recorder with a threshold of 1
    rec = TimestepCountIndexParameterRecorder(model, param, 1)

    model.run()

    assert_allclose([0, 183 + 365 + 365], rec.values(), atol=1e-7)


@pytest.mark.parametrize(
    ("params", "exclude_months"),
    [
        [1, None],
        [2, None],
        [1, [1, 2, 12]],
    ],
)
def test_annual_count_index_threshold_recorder(
    simple_storage_model, params, exclude_months
):
    """
    The test sets uses a simple reservoir model with different inputs that
    trigger a control curve failure after different numbers of years.
    """
    from pywr.parameters import ConstantScenarioParameter, ConstantParameter
    from pywr.parameters.control_curves import ControlCurveIndexParameter

    model = simple_storage_model
    scenario = Scenario(model, "A", size=2)
    # Simulate 5 years
    model.timestepper.start = "2015-01-01"
    model.timestepper.end = "2019-12-31"
    # Control curve parameter
    param = ControlCurveIndexParameter(
        model, model.nodes["Storage"], ConstantParameter(model, 0.25)
    )

    # Storage model has a capacity of 20, but starts at 10 Ml
    # Demand is roughly 2 Ml/d per year
    #  First ensemble balances the demand
    #  Second ensemble should fail during 3rd year
    demand = 2.0 / 365
    model.nodes["Input"].max_flow = ConstantScenarioParameter(
        model, scenario, [demand, 0]
    )
    model.nodes["Output"].max_flow = demand

    # Create the recorder with a threshold of 1
    rec = AnnualCountIndexThresholdRecorder(
        model, [param] * params, "TestRec", 1, exclude_months=exclude_months
    )

    model.run()

    # We expect no failures in the first ensemble, the reservoir starts failing halfway through
    # the 3rd year
    if exclude_months is None:
        expected_data = [[0, 0], [0, 0], [0, 183], [0, 365], [0, 365]]
    else:
        # Ignore counts for Jan, Feb and Dec
        assert exclude_months == [
            1,
            2,
            12,
        ]  # Test is hard-coded for these exclusion months.
        expected_data = [
            [0, 0],
            [0, 0],
            [0, 183 - 31],
            [0, 365 - 31 - 28 - 31],
            [0, 365 - 31 - 28 - 31],
        ]

    assert_allclose(expected_data, rec.data, atol=1e-7)
    df = rec.to_dataframe()
    assert_allclose(expected_data, df.values, atol=1e-7)


class TestAnnualTotalFlowRecorder:
    def test_annual_total_flow_recorder(self, simple_linear_model):
        """
        Test AnnualTotalFlowRecorder
        """

        model = simple_linear_model
        otpt = model.nodes["Output"]
        otpt.max_flow = 30.0
        model.nodes["Input"].max_flow = 10.0
        otpt.cost = -2
        rec = AnnualTotalFlowRecorder(model, "Total Flow", [otpt])

        model.run()

        assert_allclose(3650.0, rec.data, atol=1e-7)
        df = rec.to_dataframe()
        assert_allclose([[3650.0]], df.values)

    def test_annual_total_flow_recorder_factored(self, simple_linear_model):
        """
        Test AnnualTotalFlowRecorder with factors applied
        """
        model = simple_linear_model
        otpt = model.nodes["Output"]
        otpt.max_flow = 30.0
        inpt = model.nodes["Input"]
        inpt.max_flow = 10.0
        otpt.cost = -2

        factors = [2.0, 1.0]
        rec_fact = AnnualTotalFlowRecorder(
            model, "Total Flow", [otpt, inpt], factors=factors
        )

        model.run()

        assert_allclose(3650.0 * 3, rec_fact.data, atol=1e-7)
        df = rec_fact.to_dataframe()
        assert_allclose([[3650.0 * 3]], df.values)

    @pytest.mark.parametrize(
        "end_date, expected",
        [
            ("2013-01-04", [30.0, 40.0]),
            ("2012-12-31", [30.0]),
        ],
    )
    def test_annual_total_flow_recorder_year_end(
        self, simple_linear_model, end_date, expected
    ):
        """
        Test AnnualTotalFlowRecorder when timestep crosses year end.

        The two parameterisations of this test run a single timestep of 7 days that crosses into the next year. In the
        first, the model end date is set to the end of the timestep, so flow for the recorder is assigned to each year
        according to the numbers of days of the timestep that are in each year (3 in the current, 4 in the next). In the
        second parameterisation, the model end date is set to the end of year, so flow for the recorder is only assigned
        for the first 3 days of the timestep that are in the current year. The subsequent 4 days of the timestep are not
        recorded because they are beyond the model end date.
        """
        model = simple_linear_model
        simple_linear_model.timestepper.start = "2012-12-29"
        simple_linear_model.timestepper.end = end_date
        simple_linear_model.timestepper.delta = "7D"

        otpt = model.nodes["Output"]
        otpt.max_flow = 30.0
        model.nodes["Input"].max_flow = 10.0
        otpt.cost = -2
        rec = AnnualTotalFlowRecorder(model, "Total Flow", [otpt])

        model.run()

        assert_allclose(expected, rec.data.flatten())


def test_total_flow_node_recorder(simple_linear_model):
    """
    Test TotalDeficitNodeRecorder
    """
    model = simple_linear_model
    otpt = model.nodes["Output"]
    otpt.max_flow = 30.0
    model.nodes["Input"].max_flow = 10.0
    otpt.cost = -2.0
    rec = TotalFlowNodeRecorder(model, otpt)

    model.step()
    assert_allclose(10.0, rec.aggregated_value(), atol=1e-7)

    model.step()
    assert_allclose(20.0, rec.aggregated_value(), atol=1e-7)


def test_mean_flow_node_recorder(simple_linear_model):
    """
    Test MeanFlowNodeRecorder
    """
    model = simple_linear_model
    nt = len(model.timestepper)

    otpt = model.nodes["Output"]
    otpt.max_flow = 30.0
    model.nodes["Input"].max_flow = 10.0
    otpt.cost = -2.0
    rec = MeanFlowNodeRecorder(model, otpt)

    model.run()
    assert_allclose(10.0, rec.aggregated_value(), atol=1e-7)


def custom_test_func(array, axis=None):
    return np.sum(array ** 2, axis=axis)


class TestAggregatedRecorder:
    """Tests for AggregatedRecorder"""

    funcs = {"min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum}

    @pytest.mark.parametrize("agg_func", ["min", "max", "mean", "sum"])
    def test_aggregated_recorder(self, simple_linear_model, agg_func):
        model = simple_linear_model
        otpt = model.nodes["Output"]
        otpt.max_flow = 30.0
        model.nodes["Input"].max_flow = 10.0
        otpt.cost = -2.0
        rec1 = TotalFlowNodeRecorder(model, otpt)
        rec2 = TotalDeficitNodeRecorder(model, otpt)

        func = TestAggregatedRecorder.funcs[agg_func]

        rec = AggregatedRecorder(model, [rec1, rec2], agg_func=agg_func)

        assert rec in rec1.parents
        assert rec in rec2.parents

        model.step()
        assert_allclose(func([10.0, 20.0]), rec.aggregated_value(), atol=1e-7)

        model.step()
        assert_allclose(func([20.0, 40.0]), rec.aggregated_value(), atol=1e-7)

    @pytest.mark.parametrize("agg_func", ["min", "max", "mean", "sum", "custom"])
    def test_agg_func_get_set(self, simple_linear_model, agg_func):
        """Test getter and setter for AggregatedRecorder.agg_func"""
        if agg_func == "custom":
            agg_func = custom_test_func
        model = simple_linear_model
        rec = AggregatedRecorder(model, [], agg_func=agg_func)
        assert rec.agg_func == agg_func
        rec.agg_func = "product"
        assert rec.agg_func == "product"


def test_reset_timestepper_recorder():
    model = Model(
        start=pandas.to_datetime("2016-01-01"), end=pandas.to_datetime("2016-01-01")
    )

    inpt = Input(model, "input", max_flow=10)
    otpt = Output(model, "output", max_flow=50, cost=-10)
    inpt.connect(otpt)

    rec = NumpyArrayNodeRecorder(model, otpt)

    model.run()

    model.timestepper.end = pandas.to_datetime("2016-01-02")

    model.run()


def test_mean_flow_recorder():
    model = Model()
    model.timestepper.start = pandas.to_datetime("2016-01-01")
    model.timestepper.end = pandas.to_datetime("2016-01-04")

    inpt = Input(model, "input")
    otpt = Output(model, "output")
    inpt.connect(otpt)

    rec_flow = NumpyArrayNodeRecorder(model, inpt)
    rec_mean = RollingMeanFlowNodeRecorder(model, node=inpt, timesteps=3)

    scenario = Scenario(model, "dummy", size=2)

    inpt.max_flow = inpt.min_flow = FunctionParameter(
        model, inpt, lambda model, t, si: 2 + t.index
    )
    model.run()

    expected = [
        2.0,
        (2.0 + 3.0) / 2,
        (2.0 + 3.0 + 4.0) / 3,
        (3.0 + 4.0 + 5.0) / 3,  # zeroth day forgotten
    ]

    for value, expected_value in zip(rec_mean.data[:, 0], expected):
        assert_allclose(value, expected_value)


def test_mean_flow_recorder_days():
    model = Model()
    model.timestepper.delta = 7

    inpt = Input(model, "input")
    otpt = Output(model, "output")
    inpt.connect(otpt)

    rec_mean = RollingMeanFlowNodeRecorder(model, node=inpt, days=31)

    model.run()
    assert rec_mean.timesteps == 4


def test_mean_flow_recorder_json():
    model = load_model("mean_flow_recorder.json")

    # TODO: it's not possible to define a FunctionParameter in JSON yet
    supply1 = model.nodes["supply1"]
    supply1.max_flow = supply1.min_flow = FunctionParameter(
        model, supply1, lambda model, t, si: 2 + t.index
    )

    assert len(model.recorders) == 3

    rec_flow = model.recorders["Supply"]
    rec_mean = model.recorders["Mean Flow"]
    rec_check = model.recorders["Supply 2"]

    model.run()

    assert_allclose(rec_flow.data[:, 0], [2.0, 3.0, 4.0, 5.0])
    assert_allclose(rec_mean.data[:, 0], [2.0, 2.5, 3.0, 4.0])
    assert_allclose(rec_check.data[:, 0], [50.0, 50.0, 60.0, 60.0])


def test_annual_count_index_parameter_recorder(simple_storage_model):
    """Test AnnualCountIndexParameterRecord

    The test sets uses a simple reservoir model with different inputs that
    trigger a control curve failure after different numbers of years.
    """
    from pywr.parameters import ConstantScenarioParameter, ConstantParameter
    from pywr.parameters.control_curves import ControlCurveIndexParameter

    model = simple_storage_model
    scenario = Scenario(model, "A", size=2)
    # Simulate 5 years
    model.timestepper.start = "2015-01-01"
    model.timestepper.end = "2019-12-31"
    # Control curve parameter
    param = ControlCurveIndexParameter(
        model, model.nodes["Storage"], ConstantParameter(model, 0.25)
    )

    # Storage model has a capacity of 20, but starts at 10 Ml
    # Demand is roughly 2 Ml/d per year
    #  First ensemble balances the demand
    #  Second ensemble should fail during 3rd year
    demand = 2.0 / 365
    model.nodes["Input"].max_flow = ConstantScenarioParameter(
        model, scenario, [demand, 0]
    )
    model.nodes["Output"].max_flow = demand

    # Create the recorder with a threshold of 1
    rec = AnnualCountIndexParameterRecorder(model, param, 1)

    model.run()
    # We expect no failures in the first ensemble, but 3 out of 5 in the second
    assert_allclose(rec.values(), [0, 3])


# The following fixtures are used for testing the recorders in
#  pywr.recorders.calibration which require an observed data set
#  to compare with the model prediction.


@pytest.fixture
def timeseries2_model():
    return load_model("timeseries2.json")


@pytest.fixture
def timeseries2_observed():
    path = os.path.join(os.path.dirname(__file__), "models")
    df = pandas.read_csv(
        os.path.join(path, "timeseries2.csv"),
        parse_dates=True,
        dayfirst=True,
        index_col=0,
    )
    df = df.asfreq(pandas.infer_freq(df.index))
    # perturb a bit
    df += np.random.normal(size=df.shape)
    return df


class TestCalibrationRecorders:
    data = [
        (
            RootMeanSquaredErrorNodeRecorder,
            lambda sim, obs: np.sqrt(np.mean((sim - obs) ** 2, axis=0)),
        ),
        (
            MeanAbsoluteErrorNodeRecorder,
            lambda sim, obs: np.mean(np.abs(sim - obs), axis=0),
        ),
        (
            MeanSquareErrorNodeRecorder,
            lambda sim, obs: np.mean((sim - obs) ** 2, axis=0),
        ),
        (
            PercentBiasNodeRecorder,
            lambda sim, obs: np.sum(obs - sim, axis=0) * 100 / np.sum(obs, axis=0),
        ),
        (
            RMSEStandardDeviationRatioNodeRecorder,
            lambda sim, obs: np.sqrt(np.mean((obs - sim) ** 2, axis=0))
            / np.std(obs, axis=0),
        ),
        (
            NashSutcliffeEfficiencyNodeRecorder,
            lambda sim, obs: 1.0
            - np.sum((obs - sim) ** 2, axis=0)
            / np.sum((obs - obs.mean()) ** 2, axis=0),
        ),
    ]
    ids = ["rmse", "mae", "mse", "pbias", "rmse", "ns"]

    @pytest.mark.parametrize("cls,func", data, ids=ids)
    def test_calibration_recorder(
        self, timeseries2_model, timeseries2_observed, cls, func
    ):
        model = timeseries2_model
        observed = timeseries2_observed
        node = model.nodes["river1"]
        recorder = cls(model, node, observed)

        model.run()

        simulated = model.nodes["catchment1"].max_flow.dataframe
        metric = func(simulated, observed)
        values = recorder.values()
        assert values.shape[0] == len(model.scenarios.combinations)
        assert values.ndim == 1
        assert_allclose(metric, values)


@pytest.fixture
def cyclical_storage_model(simple_storage_model):
    """Extends simple_storage_model to have a cyclical boundary condition"""
    from pywr.parameters import AnnualHarmonicSeriesParameter, ConstantScenarioParameter

    m = simple_storage_model
    s = Scenario(m, name="Scenario A", size=3)

    m.timestepper.end = "2017-12-31"
    m.timestepper.delta = 5

    inpt = m.nodes["Input"]
    inpt.max_flow = AnnualHarmonicSeriesParameter(
        m, 5, [0.1, 0.0, 0.25], [0.0, 0.0, 0.0], name="inpt_flow"
    )

    otpt = m.nodes["Output"]
    otpt.max_flow = ConstantScenarioParameter(m, s, [5, 6, 2])

    return m


@pytest.fixture
def cyclical_linear_model(simple_linear_model):
    """Extends simple_storage_model to have a cyclical boundary condition"""
    from pywr.parameters import AnnualHarmonicSeriesParameter, ConstantScenarioParameter

    m = simple_linear_model
    s = Scenario(m, name="Scenario A", size=3)

    m.timestepper.end = "2017-12-31"
    m.timestepper.delta = 5

    inpt = m.nodes["Input"]
    inpt.max_flow = AnnualHarmonicSeriesParameter(
        m, 5, [1.0, 0.0, 0.5], [0.0, 0.0, 0.0]
    )

    otpt = m.nodes["Output"]
    otpt.max_flow = ConstantScenarioParameter(m, s, [5, 6, 2])
    otpt.cost = -10.0

    return m


class TestEventRecorder:
    """Tests for EventRecorder"""

    funcs = {
        "min": np.min,
        "max": np.max,
        "mean": np.mean,
        "median": np.median,
        "sum": np.sum,
    }

    @pytest.mark.parametrize(
        "threshold_component", [StorageThresholdRecorder, StorageThresholdParameter]
    )
    def test_load(self, cyclical_storage_model, threshold_component):
        """Test load method"""
        m = cyclical_storage_model
        strg = m.nodes["Storage"]
        param = threshold_component(m, strg, 4.0, predicate="<=", name="trigger")
        EventRecorder.load(
            m,
            {
                "name": "event_rec",
                "threshold": "trigger",
                "tracked_parameter": "inpt_flow",
            },
        )
        EventDurationRecorder.load(m, {"event_recorder": "event_rec"})
        EventStatisticRecorder.load(m, {"event_recorder": "event_rec"})
        m.run()

    @pytest.mark.parametrize(
        "recorder_agg_func", ["min", "max", "mean", "median", "sum"]
    )
    def test_event_capture_with_storage(
        self, cyclical_storage_model, recorder_agg_func
    ):
        """Test Storage events using a StorageThresholdRecorder"""
        m = cyclical_storage_model

        strg = m.nodes["Storage"]
        arry = NumpyArrayStorageRecorder(m, strg)

        # Create the trigger using a threhsold parameter
        trigger = StorageThresholdRecorder(m, strg, 4.0, predicate="<=")
        evt_rec = EventRecorder(m, trigger)
        evt_dur = EventDurationRecorder(
            m, evt_rec, recorder_agg_func=recorder_agg_func, agg_func="max"
        )

        m.run()

        # Ensure there is at least one event
        assert evt_rec.events

        # Build a timeseries of when the events say an event is active
        triggered = np.zeros_like(arry.data, dtype=np.int32)
        for evt in evt_rec.events:
            triggered[evt.start.index : evt.end.index, evt.scenario_index.global_id] = 1

            # Check the duration
            td = evt.end.datetime - evt.start.datetime
            assert evt.duration == td.days

        #   Test that the volumes in the Storage node during the event periods match
        assert_equal(triggered, arry.data <= 4)

        df = evt_rec.to_dataframe()

        assert len(df) == len(evt_rec.events)

        func = TestEventRecorder.funcs[recorder_agg_func]

        # Now check the EventDurationRecorder does the aggregation correctly
        expected_durations = []
        for si in m.scenarios.combinations:
            event_durations = []
            for evt in evt_rec.events:
                if evt.scenario_index.global_id == si.global_id:
                    event_durations.append(evt.duration)

            # If there are no events then the metric is zero
            if len(event_durations) > 0:
                expected_durations.append(func(event_durations))
            else:
                expected_durations.append(0.0)

        assert_allclose(evt_dur.values(), expected_durations)
        assert_allclose(evt_dur.aggregated_value(), np.max(expected_durations))

    def test_event_capture_with_node(self, cyclical_linear_model):
        """Test Node flow events using a NodeThresholdRecorder"""
        m = cyclical_linear_model

        otpt = m.nodes["Output"]
        arry = NumpyArrayNodeRecorder(m, otpt)

        # Create the trigger using a threhsold parameter
        trigger = NodeThresholdRecorder(m, otpt, 4.0, predicate=">")
        evt_rec = EventRecorder(m, trigger)

        m.run()

        # Ensure there is at least one event
        assert evt_rec.events

        # Build a timeseries of when the events say an event is active
        triggered = np.zeros_like(arry.data, dtype=np.int32)
        for evt in evt_rec.events:
            triggered[evt.start.index : evt.end.index, evt.scenario_index.global_id] = 1

            # Check the duration
            td = evt.end.datetime - evt.start.datetime
            assert evt.duration == td.days

        # Test that the volumes in the Storage node during the event periods match
        assert_equal(triggered, arry.data > 4)

    @pytest.mark.parametrize(
        "recorder_agg_func", ["min", "max", "mean", "median", "sum"]
    )
    def test_no_event_capture_with_storage(
        self, cyclical_storage_model, recorder_agg_func
    ):
        """Test Storage events using a StorageThresholdRecorder"""
        m = cyclical_storage_model

        strg = m.nodes["Storage"]
        arry = NumpyArrayStorageRecorder(m, strg)

        # Create the trigger using a threhsold parameter
        trigger = StorageThresholdRecorder(m, strg, -1.0, predicate="<")
        evt_rec = EventRecorder(m, trigger)
        evt_dur = EventDurationRecorder(
            m, evt_rec, recorder_agg_func=recorder_agg_func, agg_func="max"
        )

        m.run()

        # Ensure there are no events in this test
        assert len(evt_rec.events) == 0
        df = evt_rec.to_dataframe()
        assert len(df) == 0

        assert_allclose(evt_dur.values(), np.zeros(len(m.scenarios.combinations)))
        assert_allclose(evt_dur.aggregated_value(), 0)

    @pytest.mark.parametrize("minimum_length", [1, 2, 3, 4])
    def test_hysteresis(self, simple_linear_model, minimum_length):
        """Test the minimum_event_length keyword of EventRecorder"""
        m = simple_linear_model

        flow = np.zeros(len(m.timestepper))

        flow[:10] = [0, 0, 10, 0, 10, 10, 10, 0, 0, 0]
        # With min event length of 1. There are two events with lengths (1, 3)
        #                 |---|   |---------|
        # With min event length up to 4. There is one event with length 3
        #                         |---------|
        # Min event length >= 4 gives no events

        inpt = m.nodes["Input"]
        inpt.max_flow = ArrayIndexedParameter(m, flow)

        # Force through whatever flow Input can provide
        otpt = m.nodes["Output"]
        otpt.max_flow = 100
        otpt.cost = -100

        # Create the trigger using a threhsold parameter
        trigger = NodeThresholdRecorder(m, otpt, 4.0, predicate=">")
        evt_rec = EventRecorder(m, trigger, minimum_event_length=minimum_length)

        m.run()

        if minimum_length == 1:
            assert len(evt_rec.events) == 2
            assert_equal([1, 3], [e.duration for e in evt_rec.events])
        elif minimum_length < 4:
            assert len(evt_rec.events) == 1
            assert_equal(
                [
                    3,
                ],
                [e.duration for e in evt_rec.events],
            )
        else:
            assert len(evt_rec.events) == 0

    @pytest.mark.parametrize(
        "recorder_agg_func", ["min", "max", "mean", "median", "sum"]
    )
    def test_statistic_recorder(self, cyclical_storage_model, recorder_agg_func):
        """Test EventStatisticRecorder"""
        m = cyclical_storage_model

        strg = m.nodes["Storage"]
        inpt = m.nodes["Input"]
        arry = NumpyArrayNodeRecorder(m, inpt)

        # Create the trigger using a threhsold parameter
        trigger = StorageThresholdRecorder(m, strg, 4.0, predicate="<=")
        evt_rec = EventRecorder(m, trigger, tracked_parameter=inpt.max_flow)
        evt_stat = EventStatisticRecorder(
            m,
            evt_rec,
            agg_func="max",
            event_agg_func="min",
            recorder_agg_func=recorder_agg_func,
        )

        m.run()

        # Ensure there is at least one event
        assert evt_rec.events

        evt_values = {si.global_id: [] for si in m.scenarios.combinations}
        for evt in evt_rec.events:
            evt_values[evt.scenario_index.global_id].append(
                np.min(
                    arry.data[
                        evt.start.index : evt.end.index, evt.scenario_index.global_id
                    ]
                )
            )

        func = TestEventRecorder.funcs[recorder_agg_func]

        agg_evt_values = []
        for k, v in sorted(evt_values.items()):
            if len(v) > 0:
                agg_evt_values.append(func(v))
            else:
                agg_evt_values.append(np.nan)

        # Test that the
        assert_allclose(evt_stat.values(), agg_evt_values)
        assert_allclose(evt_stat.aggregated_value(), np.max(agg_evt_values))


def test_progress_recorder(simple_linear_model):
    model = simple_linear_model
    rec = ProgressRecorder(model)
    model.run()


class TestHydroPowerRecorder:
    @pytest.mark.parametrize("efficiency", [1.0, 0.85])
    def test_constant_level(self, simple_storage_model, efficiency):
        """Test HydropowerRecorder"""
        m = simple_storage_model

        strg = m.nodes["Storage"]
        otpt = m.nodes["Output"]

        elevation = ConstantParameter(m, 100)
        rec = HydropowerRecorder(m, otpt, elevation, efficiency=efficiency)
        rec_total = TotalHydroEnergyRecorder(m, otpt, elevation, efficiency=efficiency)

        m.setup()
        m.step()

        # First step
        # Head: 100m
        # Flow: 8 m3/day
        # Power: 1000 * 9.81 * 8 * 100
        # Energy: power * 1 day = power
        np.testing.assert_allclose(
            rec.data[0, 0], 1000 * 9.81 * 8 * 100 * 1e-6 * efficiency
        )
        # Second step has the same answer in this model
        m.step()
        np.testing.assert_allclose(
            rec.data[1, 0], 1000 * 9.81 * 8 * 100 * 1e-6 * efficiency
        )
        np.testing.assert_allclose(
            rec_total.values()[0], 2 * 1000 * 9.81 * 8 * 100 * 1e-6 * efficiency
        )

    def test_varying_level(self, simple_storage_model):
        """Test HydropowerRecorder with varying level on Storage node"""
        from pywr.parameters import InterpolatedVolumeParameter

        m = simple_storage_model

        strg = m.nodes["Storage"]
        otpt = m.nodes["Output"]

        elevation = InterpolatedVolumeParameter(m, strg, [0, 10, 20], [0, 100, 200])
        rec = HydropowerRecorder(m, otpt, elevation)
        rec_total = TotalHydroEnergyRecorder(m, otpt, elevation)

        m.setup()
        m.step()

        # First step
        # Head: 100m
        # Flow: 8 m3/day
        # Power: 1000 * 9.81 * 8 * 100
        # Energy: power * 1 day = power
        np.testing.assert_allclose(rec.data[0, 0], 1000 * 9.81 * 8 * 100 * 1e-6)
        # Second step is at a lower head
        # Head: 70m
        m.step()
        np.testing.assert_allclose(rec.data[1, 0], 1000 * 9.81 * 8 * 70 * 1e-6)
        np.testing.assert_allclose(rec_total.values()[0], 1000 * 9.81 * 8 * 170 * 1e-6)

    def test_varying_level_with_turbine_level(self, simple_storage_model):
        """Test HydropowerRecorder with varying level on Storage and defined level on the recorder"""
        from pywr.parameters import InterpolatedVolumeParameter

        m = simple_storage_model

        strg = m.nodes["Storage"]
        otpt = m.nodes["Output"]

        elevation = InterpolatedVolumeParameter(m, strg, [0, 10, 20], [0, 100, 200])
        rec = HydropowerRecorder(m, otpt, elevation, turbine_elevation=80)
        rec_total = TotalHydroEnergyRecorder(m, otpt, elevation, turbine_elevation=80)

        m.setup()
        m.step()

        # First step
        # Head: 100 - 80 = 20m
        # Flow: 8 m3/day
        # Power: 1000 * 9.81 * 8 * 100
        # Energy: power * 1 day = power
        np.testing.assert_allclose(rec.data[0, 0], 1000 * 9.81 * 8 * 20 * 1e-6)
        # Second step is at a lower head
        # Head: 70 - 80: -10m (i.e. not sufficient)
        m.step()
        np.testing.assert_allclose(rec.data[1, 0], 1000 * 9.81 * 8 * 0 * 1e-6)
        np.testing.assert_allclose(rec_total.values()[0], 1000 * 9.81 * 8 * 20 * 1e-6)

    def test_load_from_json(
        self,
    ):
        """Test example hydropower model loads and runs."""
        model = load_model("hydropower_example.json")

        r = model.recorders["turbine1_energy"]

        # Check the recorder has loaded correctly
        assert r.water_elevation_parameter == model.parameters["reservoir1_level"]
        assert r.node == model.nodes["turbine1"]

        assert_allclose(r.turbine_elevation, 35.0)
        assert_allclose(r.efficiency, 0.85)
        assert_allclose(r.flow_unit_conversion, 1e3)

        # Finally, check model runs with the loaded recorder.
        model.run()


class TestGaussianKDEStorageRecorder:
    def test_kde_recorder(self, simple_storage_model):
        """A basic functional test of `GaussianKDEStorageRecorder`"""
        model = simple_storage_model
        res = model.nodes["Storage"]

        kde = GaussianKDEStorageRecorder(model, res, target_volume_pc=0.2)

        model.run()

        pdf = kde.to_dataframe()
        p = kde.aggregated_value()
        assert pdf.shape == (101, 1)
        assert 0 < p < 1
        np.testing.assert_allclose(pdf.values, kde.values())

    def test_kde_from_json(self, simple_storage_model):
        """Test loading KDE recorder from JSON data."""
        model = simple_storage_model

        kde = load_recorder(
            model,
            {
                "type": "GaussianKDEStorageRecorder",
                "node": "Storage",
                "target_volume_pc": 0.2,
            },
        )

        model.run()

        pdf = kde.to_dataframe()
        p = kde.aggregated_value()
        assert pdf.shape == (101, 1)
        assert 0 < p < 1
        np.testing.assert_allclose(pdf.values, kde.values())

    def test_norm_kde_recorder(self, simple_storage_model):
        """A basic functional test of `NormalisedGaussianKDEStorageRecorder`"""
        model = simple_storage_model
        res = model.nodes["Storage"]
        cc = ConstantParameter(model, 0.2)

        kde = NormalisedGaussianKDEStorageRecorder(model, res, parameter=cc)

        model.run()

        pdf = kde.to_dataframe()
        p = kde.aggregated_value()
        assert pdf.shape == (101, 1)
        assert 0 < p < 1
        np.testing.assert_allclose(pdf.values, kde.values())

    def test_norm_kde_from_json(self, simple_storage_model):
        """Test loading normalised KDE recorder from JSON data."""
        model = simple_storage_model
        cc = ConstantParameter(model, 0.2, name="my-parameter")

        kde = load_recorder(
            model,
            {
                "type": "NormalisedGaussianKDEStorageRecorder",
                "node": "Storage",
                "parameter": "my-parameter",
            },
        )

        model.run()

        pdf = kde.to_dataframe()
        p = kde.aggregated_value()
        assert pdf.shape == (101, 1)
        assert 0 < p < 1
        np.testing.assert_allclose(pdf.values, kde.values())
