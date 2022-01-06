from pywr.core import Model, Storage, Link, ScenarioIndex, Timestep, Output
from pywr.parameters import ConstantParameter, DailyProfileParameter, load_parameter
from pywr.parameters.control_curves import (
    ControlCurveParameter,
    ControlCurveInterpolatedParameter,
    ControlCurvePiecewiseInterpolatedParameter,
)
from pywr.parameters._control_curves import _interpolate
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayStorageRecorder, assert_rec
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
import pytest
import datetime
import os
from fixtures import simple_linear_model, simple_storage_model
from helpers import load_model


@pytest.fixture
def model(simple_storage_model):
    """Modified simple_storage_model to be steady-state."""
    i = simple_storage_model.nodes["Input"]
    i.max_flow = 0
    o = simple_storage_model.nodes["Output"]
    o.max_flow = 0
    s = simple_storage_model.nodes["Storage"]
    s.max_volume = 100.0
    return simple_storage_model


@pytest.mark.parametrize(
    "configuration, expected_value",
    [
        ((0.0, 0.0, 1.0, 50.0, 100.0), 50.0),
        ((1.0, 0.0, 1.0, 50.0, 100.0), 100.0),
        ((0.5, 0.0, 1.0, 50.0, 100.0), 75.0),
        ((0.0, 0.5, 1.0, 50.0, 100.0), 50.0),
        ((0.75, 0.5, 1.0, 50.0, 100.0), 75.0),
    ],
)
def test_interpolation(configuration, expected_value):
    current_position, lower_bound, upper_bound, lower_value, upper_value = configuration
    assert (
        _interpolate(
            current_position, lower_bound, upper_bound, lower_value, upper_value
        )
        == expected_value
    )


class TestPiecewiseControlCurveParameter:
    """Tests for ControlCurveParameter"""

    @staticmethod
    def _assert_results(m, s):
        """Correct results for the following tests"""

        @assert_rec(m, s.cost)
        def expected_func(timestep, scenario_index):
            v = s.initial_volume
            if v >= 80.0:
                expected = 1.0
            elif v >= 60:
                expected = 0.7
            else:
                expected = 0.4
            return expected

        for initial_volume in (90, 70, 30):
            s.initial_volume = initial_volume
            m.run()

    def test_with_values(self, model):
        """Test with `values` keyword argument"""
        m = model
        s = m.nodes["Storage"]

        # Return 10.0 when above 0.0 when below
        s.cost = ControlCurveParameter(m, s, [0.8, 0.6], [1.0, 0.7, 0.4])
        self._assert_results(m, s)

    def test_with_parameters(self, model):
        """Test with `parameters` keyword argument."""
        m = model
        s = m.nodes["Storage"]

        # Two different control curves
        cc = [ConstantParameter(model, 0.8), ConstantParameter(model, 0.6)]
        # Three different parameters to return
        params = [
            ConstantParameter(model, 1.0),
            ConstantParameter(model, 0.7),
            ConstantParameter(model, 0.4),
        ]
        s.cost = ControlCurveParameter(model, s, cc, parameters=params)

        self._assert_results(m, s)

    def test_values_load(self, model):
        """Test load of float lists."""

        m = model
        s = m.nodes["Storage"]

        data = {
            "type": "controlcurve",
            "control_curves": [0.8, 0.6],
            "values": [1.0, 0.7, 0.4],
            "storage_node": "Storage",
        }

        s.cost = p = load_parameter(model, data)
        assert isinstance(p, ControlCurveParameter)
        self._assert_results(m, s)

    def test_parameters_load(self, model):
        """Test load of parameter lists for 'control_curves' and 'parameters' keys."""

        m = model
        s = m.nodes["Storage"]

        data = {
            "type": "controlcurve",
            "storage_node": "Storage",
            "control_curves": [
                {"type": "constant", "value": 0.8},
                {"type": "monthlyprofile", "values": [0.6] * 12},
            ],
            "parameters": [
                {
                    "type": "constant",
                    "value": 1.0,
                },
                {"type": "constant", "value": 0.7},
                {"type": "constant", "value": 0.4},
            ],
        }

        s.cost = p = load_parameter(model, data)
        assert isinstance(p, ControlCurveParameter)
        self._assert_results(m, s)

    def test_single_cc_load(self, model):
        """Test load from dict with 'control_curve' key

        This is different to the above test by using singular 'control_curve' key in the dict
        """
        m = model
        s = m.nodes["Storage"]

        data = {
            "type": "controlcurve",
            "storage_node": "Storage",
            "control_curve": 0.8,
        }

        s.cost = p = load_parameter(model, data)
        assert isinstance(p, ControlCurveParameter)

        @assert_rec(m, p)
        def expected_func(timestep, scenario_index):
            v = s.initial_volume
            if v >= 80.0:
                expected = 0
            else:
                expected = 1
            return expected

        for initial_volume in (90, 70):
            s.initial_volume = initial_volume
            m.run()

    def test_with_nonstorage(self, model):
        """Test usage on non-`Storage` node."""
        # Now test if the parameter is used on a non storage node
        m = model
        s = m.nodes["Storage"]

        l = Link(m, "Link")
        # Connect the link node to the network to create a valid model
        o = m.nodes["Output"]
        s.connect(l)
        l.connect(o)

        cc = ConstantParameter(model, 0.8)
        l.cost = ControlCurveParameter(model, s, cc, [10.0, 0.0])

        @assert_rec(m, l.cost)
        def expected_func(timestep, scenario_index):
            v = s.initial_volume
            if v >= 80.0:
                expected = 10.0
            else:
                expected = 0.0
            return expected

        for initial_volume in (90, 70):
            s.initial_volume = initial_volume
            m.run()

    def test_with_nonstorage_load(self, model):
        """Test load from dict with 'storage_node' key."""
        m = model
        s = m.nodes["Storage"]
        l = Link(m, "Link")
        # Connect the link node to the network to create a valid model
        o = m.nodes["Output"]
        s.connect(l)
        l.connect(o)

        data = {
            "type": "controlcurve",
            "control_curve": 0.8,
            "values": [10.0, 0.0],
            "storage_node": "Storage",
        }

        l.cost = p = load_parameter(model, data)
        assert isinstance(p, ControlCurveParameter)

        @assert_rec(m, l.cost)
        def expected_func(timestep, scenario_index):
            v = s.initial_volume
            if v >= 80.0:
                expected = 10.0
            else:
                expected = 0.0
            return expected

        for initial_volume in (90, 70):
            s.initial_volume = initial_volume
            m.run()


@pytest.mark.parametrize("use_parameters", [False, True])
def test_control_curve_interpolated(model, use_parameters):
    m = model
    m.timestepper.delta = 200

    s = m.nodes["Storage"]
    o = m.nodes["Output"]
    s.connect(o)

    cc = ConstantParameter(model, 0.8)
    values = [20.0, 5.0, 0.0]

    if use_parameters:
        # Create the parameter using parameters for the values
        parameters = [ConstantParameter(model, v) for v in values]
        s.cost = p = ControlCurveInterpolatedParameter(
            model, s, cc, parameters=parameters
        )
    else:
        # Create the parameter using a list of values
        s.cost = p = ControlCurveInterpolatedParameter(model, s, cc, values)

    @assert_rec(model, p)
    def expected_func(timestep, scenario_index):
        v = s.initial_volume
        c = cc.value(timestep, scenario_index)
        if c == 1.0 and v == 100.0:
            expected = values[1]
        elif c == 0.0 and v == 0.0:
            expected = values[1]
        else:
            expected = np.interp(v / 100.0, [0.0, c, 1.0], values[::-1])
        return expected

    for control_curve in (0.0, 0.8, 1.0):
        cc.set_double_variables(
            np.array(
                [
                    control_curve,
                ]
            )
        )
        for initial_volume in (0.0, 10.0, 50.0, 80.0, 90.0, 100.0):
            s.initial_volume = initial_volume
            model.run()


@pytest.mark.parametrize("use_parameters", [False, True])
def test_control_curve_interpolated_json(use_parameters):
    # this is a little hack-y, as the parameters don't provide access to their
    # data once they've been initalised
    if use_parameters:
        model = load_model("reservoir_with_cc_param_values.json")
    else:
        model = load_model("reservoir_with_cc.json")
    reservoir1 = model.nodes["reservoir1"]
    model.setup()
    path = os.path.join(os.path.dirname(__file__), "models", "control_curve.csv")
    control_curve = pd.read_csv(path)["Control Curve"].values
    values = [-8, -6, -4]

    @assert_rec(model, reservoir1.cost)
    def expected_cost(timestep, si):
        # calculate expected cost manually and compare to parameter output
        volume_factor = reservoir1._current_pc[si.global_id]
        cc = control_curve[timestep.index]
        return np.interp(volume_factor, [0.0, cc, 1.0], values[::-1])

    model.run()


def test_circular_control_curve_interpolated_json():
    # this is a little hack-y, as the parameters don't provide access to their
    # data once they've been initalised
    model = load_model("reservoir_with_circular_cc.json")
    reservoir1 = model.nodes["reservoir1"]
    model.setup()
    path = os.path.join(os.path.dirname(__file__), "models", "control_curve.csv")
    control_curve = pd.read_csv(path)["Control Curve"].values
    values = [-8, -6, -4]

    @assert_rec(model, reservoir1.cost)
    def expected_cost(timestep, si):
        # calculate expected cost manually and compare to parameter output
        volume_factor = reservoir1._current_pc[si.global_id]
        cc = control_curve[timestep.index]
        return np.interp(volume_factor, [0.0, cc, 1.0], values[::-1])

    model.run()


def test_demand_saving_with_indexed_array():
    """Test demand saving based on reservoir control curves

    This is a relatively complex test to pass due to the large number of
    dependencies of the parameters actually being tested. The test is an
    example of how demand savings can be applied in times of drought based
    on the state of a reservoir.
    """

    model = load_model("demand_saving2.json")

    model.timestepper.end = pd.Timestamp("2016-01-31")

    rec_demand = NumpyArrayNodeRecorder(model, model.nodes["Demand"])
    rec_storage = NumpyArrayStorageRecorder(model, model.nodes["Reservoir"])

    model.check()
    model.run()

    max_volume = model.nodes["Reservoir"].max_volume

    # model starts with no demand saving
    demand_baseline = 50.0
    demand_factor = 0.9  # jan-apr
    demand_saving = 1.0
    assert_allclose(
        rec_demand.data[0, 0], demand_baseline * demand_factor * demand_saving
    )

    # first control curve breached
    demand_saving = 0.95
    assert rec_storage.data[4, 0] < (0.8 * max_volume)
    assert_allclose(
        rec_demand.data[5, 0], demand_baseline * demand_factor * demand_saving
    )

    # second control curve breached
    demand_saving = 0.5
    assert rec_storage.data[11, 0] < (0.5 * max_volume)
    assert_allclose(
        rec_demand.data[12, 0], demand_baseline * demand_factor * demand_saving
    )


def test_demand_saving_with_indexed_array_from_hdf():
    """Test demand saving based on a predefined demand saving level in a HDF file."""
    model = load_model("demand_saving_hdf.json")

    model.timestepper.end = pd.Timestamp("2016-01-31")

    rec_demand = NumpyArrayNodeRecorder(model, model.nodes["Demand"])
    rec_storage = NumpyArrayStorageRecorder(model, model.nodes["Reservoir"])

    model.check()
    model.run()

    max_volume = model.nodes["Reservoir"].max_volume

    # model starts with no demand saving
    demand_baseline = 50.0
    demand_saving = 1.0
    assert_allclose(rec_demand.data[0, 0], demand_baseline * demand_saving)

    # first control curve breached
    demand_saving = 0.8
    assert_allclose(rec_demand.data[11, 0], demand_baseline * demand_saving)

    # second control curve breached
    demand_saving = 0.5
    assert_allclose(rec_demand.data[12, 0], demand_baseline * demand_saving)

    # second control curve breached
    demand_saving = 0.25
    assert_allclose(rec_demand.data[13, 0], demand_baseline * demand_saving)


class TestControlCurvePiecewiseInterpolatedParameter:
    """Tests for `ControlCurvePiecewiseInterpolatedParameter`"""

    def test_single_control_curve(self, simple_storage_model):
        """Test `ControlCurvePiecewiseInterpolatedParameter` with one control curve."""
        model = simple_storage_model
        storage_node = model.nodes["Storage"]
        input_node = model.nodes["Input"]
        output_node = model.nodes["Output"]

        control_curves = [
            ConstantParameter(model, 0.5),
        ]

        parameter = ControlCurvePiecewiseInterpolatedParameter(
            model, storage_node, control_curves, [(500, 200), (100, 50)], name="CCPIP"
        )
        assert parameter.minimum == 0.0
        assert parameter.maximum == 1.0

        input_node.max_flow = 1.0
        input_node.cost = 0
        output_node.max_flow = 0.0
        storage_node.initial_volume = 0.0
        storage_node.max_volume = 100.0
        storage_node.cost = -10

        model.timestepper.start = "1920-01-01"
        model.timestepper.delta = 1
        model.timestepper.end = model.timestepper.start + model.timestepper.offset * 100

        @assert_rec(model, parameter)
        def expected_func(timestep, scenario_index):
            volume = timestep.index
            control_curve = 0.5
            current_position = volume / storage_node.max_volume
            if current_position >= control_curve:
                factor = (volume - 50) / 50
                value = 200 + factor * (500 - 200)
            else:
                factor = volume / 50
                value = 50 + factor * (100 - 50)
            return value

        model.run()

    def test_two_control_curves(self, simple_storage_model):
        """Test `ControlCurvePiecewiseInterpolatedParameter` with two control curves."""
        model = simple_storage_model
        storage_node = model.nodes["Storage"]
        input_node = model.nodes["Input"]
        output_node = model.nodes["Output"]

        control_curves = [
            ConstantParameter(model, 0.75),
            ConstantParameter(model, 0.25),
        ]

        parameter = ControlCurvePiecewiseInterpolatedParameter(
            model,
            storage_node,
            control_curves,
            [(500, 200), (100, 50), (0, -100)],
            name="CCPIP",
        )
        assert parameter.minimum == 0.0
        assert parameter.maximum == 1.0

        input_node.max_flow = 1.0
        input_node.cost = 0
        output_node.max_flow = 0.0
        storage_node.initial_volume = 0.0
        storage_node.max_volume = 100.0
        storage_node.cost = -10

        model.timestepper.start = "1920-01-01"
        model.timestepper.delta = 1
        model.timestepper.end = model.timestepper.start + model.timestepper.offset * 100

        @assert_rec(model, parameter)
        def expected_func(timestep, scenario_index):
            volume = timestep.index

            current_position = volume / storage_node.max_volume
            if current_position >= 0.75:
                factor = (volume - 75) / 25
                value = 200 + factor * (500 - 200)
            elif current_position >= 0.25:
                factor = (volume - 25) / 50
                value = 50 + factor * (100 - 50)
            else:
                factor = volume / 25
                value = -100 + factor * 100
            return value

        model.run()

    def test_json(self, simple_storage_model):
        """Test loading from JSON data."""
        model = simple_storage_model
        control_curve1 = ConstantParameter(model, 0.5, name="cc1")
        control_curve2 = ConstantParameter(model, 0.25, name="cc2")
        parameter_data = {
            "type": "controlcurvepiecewiseinterpolated",
            "storage_node": "Storage",
            "control_curves": ["cc1", "cc2"],
            "minimum": 0.2,
            "maximum": 0.7,
            "values": [[200, 100], [10, 5], [0, -10]],
        }
        parameter = load_parameter(model, parameter_data)
        assert parameter.minimum == 0.2
        assert parameter.maximum == 0.7
        np.testing.assert_allclose(parameter.values, [[200, 100], [10, 5], [0, -10]])
        assert parameter.control_curves == [control_curve1, control_curve2]
        assert parameter.storage_node is model.nodes["Storage"]
