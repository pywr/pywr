from pywr.core import Model, Storage, Link, ScenarioIndex, Timestep
from pywr.parameters import ConstantParameter, DailyProfileParameter, load_parameter
from pywr.parameters.control_curves import ControlCurveParameter, ControlCurveInterpolatedParameter, MonthlyProfileControlCurveParameter
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayStorageRecorder
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
import pytest
import datetime
import os

from helpers import load_model

@pytest.fixture
def model(solver):
    return Model(solver=solver)


class TestPiecewiseControlCurveParameter:
    """Tests for ControlCurveParameter """

    @staticmethod
    def _assert_results(m, s):
        """ Correct results for the following tests """
        m.scenarios.setup()
        s.setup(m)  # Init memory view on storage (bypasses usual `Model.setup`)

        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        s.initial_volume = 90.0
        m.reset()
        ts = next(m.timestepper)
        assert_allclose(s.get_cost(m.timestepper.current, si), 1.0)

        s.initial_volume = 70.0
        m.reset()
        ts = next(m.timestepper)
        assert_allclose(s.get_cost(m.timestepper.current, si), 0.7)

        s.initial_volume = 40.0
        m.reset()
        ts = next(m.timestepper)
        assert_allclose(s.get_cost(m.timestepper.current, si), 0.4)

    def test_with_values(self, model):
        """Test with `values` keyword argument"""
        m = model
        s = Storage(m, 'Storage', max_volume=100.0)

        # Return 10.0 when above 0.0 when below
        s.cost = ControlCurveParameter(s, [0.8, 0.6], [1.0, 0.7, 0.4])
        self._assert_results(m, s)

    def test_with_parameters(self, model):
        """ Test with `parameters` keyword argument. """
        m = model

        s = Storage(m, 'Storage', max_volume=100.0)

        # Two different control curves
        cc = [ConstantParameter(0.8), ConstantParameter(0.6)]
        # Three different parameters to return
        params = [
            ConstantParameter(1.0), ConstantParameter(0.7), ConstantParameter(0.4)
        ]
        s.cost = ControlCurveParameter(s, cc, parameters=params)

        self._assert_results(m, s)

    def test_values_load(self, model):
        """ Test load of float lists. """

        m = model
        s = Storage(m, 'Storage', max_volume=100.0)

        data = {
            "type": "controlcurve",
            "control_curves": [0.8, 0.6],
            "values": [1.0, 0.7, 0.4],
            "storage_node": "Storage"
        }

        s.cost = p = load_parameter(model, data)
        assert isinstance(p, ControlCurveParameter)
        self._assert_results(m, s)

    def test_parameters_load(self, model):
        """ Test load of parameter lists for 'control_curves' and 'parameters' keys. """

        m = model
        s = Storage(m, 'Storage', max_volume=100.0)

        data = {
            "type": "controlcurve",
            "storage_node": "Storage",
            "control_curves": [
                {
                    "type": "constant",
                    "value": 0.8
                },
                {
                    "type": "monthlyprofile",
                    "values": [0.6]*12
                }
            ],
            "parameters": [
                {
                    "type": "constant",
                    "value": 1.0,
                },
                {
                    "type": "constant",
                    "value": 0.7
                },
                {
                    "type": "constant",
                    "value": 0.4
                }
            ]
        }

        s.cost = p = load_parameter(model, data)
        assert isinstance(p, ControlCurveParameter)
        self._assert_results(m, s)

    def test_single_cc_load(self, model):
        """ Test load from dict with 'control_curve' key

        This is different to the above test by using singular 'control_curve' key in the dict
        """
        m = model
        m.scenarios.setup()
        s = Storage(m, 'Storage', max_volume=100.0)

        data = {
            "type": "controlcurve",
            "storage_node": "Storage",
            "control_curve": 0.8,
        }

        s.cost = p = load_parameter(model, data)
        assert isinstance(p, ControlCurveParameter)

        s.setup(m)  # Init memory view on storage (bypasses usual `Model.setup`)

        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        s.initial_volume = 90.0
        m.reset()
        assert_allclose(s.get_cost(m.timestepper.current, si), 0)

        s.initial_volume = 70.0
        m.reset()
        assert_allclose(s.get_cost(m.timestepper.current, si), 1)

    def test_with_nonstorage(self, model):
        """ Test usage on non-`Storage` node. """
        # Now test if the parameter is used on a non storage node
        m = model
        m.scenarios.setup()
        s = Storage(m, 'Storage', max_volume=100.0)

        l = Link(m, 'Link')
        cc = ConstantParameter(0.8)
        l.cost = ControlCurveParameter(s, cc, [10.0, 0.0])

        s.setup(m)  # Init memory view on storage (bypasses usual `Model.setup`)
        print(s.volume)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        assert_allclose(l.get_cost(m.timestepper.current, si), 0.0)
        # When storage volume changes, the cost of the link changes.
        s.initial_volume = 90.0
        m.reset()
        print(s.volume)
        assert_allclose(l.get_cost(m.timestepper.current, si), 10.0)

    def test_with_nonstorage_load(self, model):
        """ Test load from dict with 'storage_node' key. """
        m = model
        m.scenarios.setup()
        s = Storage(m, 'Storage', max_volume=100.0)
        l = Link(m, 'Link')

        data = {
            "type": "controlcurve",
            "control_curve": 0.8,
            "values": [10.0, 0.0],
            "storage_node": "Storage"
        }

        l.cost = p = load_parameter(model, data)
        assert isinstance(p, ControlCurveParameter)

        s.setup(m)  # Init memory view on storage (bypasses usual `Model.setup`)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        print(s.volume)
        assert_allclose(l.get_cost(m.timestepper.current, si), 0.0)
        # When storage volume changes, the cost of the link changes.
        s.initial_volume = 90.0
        m.reset()
        assert_allclose(l.get_cost(m.timestepper.current, si), 10.0)


def test_control_curve_interpolated(model):
    m = model
    m.scenarios.setup()
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    s = Storage(m, 'Storage', max_volume=100.0)

    cc = ConstantParameter(0.8)
    values = [20.0, 5.0, 0.0]
    s.cost = ControlCurveInterpolatedParameter(s, cc, values)
    s.setup(m)

    for v in (0.0, 10.0, 50.0, 80.0, 90.0, 100.0):
        s.initial_volume = v
        s.reset()
        assert_allclose(s.get_cost(m.timestepper.current, si), np.interp(v/100.0, [0.0, 0.8, 1.0], values[::-1]))

    # special case when control curve is 100%
    cc.update(np.array([1.0,]))
    s.initial_volume == 100.0
    s.reset()
    assert_allclose(s.get_cost(m.timestepper.current, si), values[1])

    # special case when control curve is 0%
    cc.update(np.array([0.0,]))
    s.initial_volume == 0.0
    s.reset()
    assert_allclose(s.get_cost(m.timestepper.current, si), values[0])


def test_control_curve_interpolated_json(solver):
    # this is a little hack-y, as the parameters don't provide access to their
    # data once they've been initalised
    model = load_model("reservoir_with_cc.json", solver=solver)
    reservoir1 = model.nodes["reservoir1"]
    model.setup()
    ts = next(model.timestepper)
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    path = os.path.join(os.path.dirname(__file__), "models", "control_curve.csv")
    control_curve = pd.read_csv(path)["Control Curve"].values
    max_volume = reservoir1.max_volume
    values = [-8, -6, -4]
    for n in range(0, 10):
        # calculate expected cost manually and compare to parameter output
        volume = reservoir1._volume[si.global_id]
        volume_factor = reservoir1._current_pc[si.global_id]
        cc = control_curve[model.timestepper.current.index]
        expected_cost = np.interp(volume_factor, [0.0, cc, 1.0], values[::-1])
        cost = reservoir1.get_cost(model.timestepper.current, si)
        assert_allclose(expected_cost, cost)
        model.step()


class TestMonthlyProfileControlCurveParameter:
    """ Test `MonthlyProfileControlCurveParameter` """
    def _assert_results(self, model, s, p, scale=1.0):
        # Test correct aggregation is performed

        s.setup(model)  # Init memory view on storage (bypasses usual `Model.setup`)

        s.initial_volume = 90.0
        model.reset()  # Set initial volume on storage
        ts = next(model.timestepper)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        for mth in range(1, 13):
            ts = Timestep(datetime.datetime(2016, mth, 1), 366, 1.0)
            np.testing.assert_allclose(p.value(ts, si), 1.0*scale)

        s.initial_volume = 70.0
        model.reset()  # Set initial volume on storage
        ts = next(model.timestepper)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        for mth in range(1, 13):
            ts = Timestep(datetime.datetime(2016, mth, 1), 366, 1.0)
            np.testing.assert_allclose(p.value(ts, si), 0.7 * (mth - 1)*scale)

        s.initial_volume = 30.0
        model.reset()  # Set initial volume on storage
        ts = next(model.timestepper)
        si = ScenarioIndex(0, np.array([0], dtype=np.int32))
        for mth in range(1, 13):
            ts = Timestep(datetime.datetime(2016, mth, 1), 366, 1.0)
            np.testing.assert_allclose(p.value(ts, si), 0.3*scale)

    def test_no_scale_no_profile(self, model):
        """ No scale or profile specified """

        s = Storage(model, 'Storage', max_volume=100.0)
        l = Link(model, 'Link')

        data = {
            'type': 'monthlyprofilecontrolcurve',
            'control_curves': [0.8, 0.6],
            'values': [[1.0]*12, [0.7]*np.arange(12), [0.3]*12],
            'storage_node': 'Storage'
        }

        l.max_flow = p = load_parameter(model, data)
        p.setup(model)
        model.scenarios.setup()
        self._assert_results(model, s, p)

    def test_scale_no_profile(self, model):
        """ Test `MonthlyProfileControlCurveParameter` """

        s = Storage(model, 'Storage', max_volume=100.0)
        l = Link(model, 'Link')

        data = {
            'type': 'monthlyprofilecontrolcurve',
            'control_curves': [0.8, 0.6],
            'values': [[1.0] * 12, [0.7] * np.arange(12), [0.3] * 12],
            'storage_node': 'Storage',
            'scale': 1.5
        }

        l.max_flow = p = load_parameter(model, data)
        p.setup(model)
        model.scenarios.setup()
        self._assert_results(model, s, p, scale=1.5)

    def test_no_scale_profile_param(self, model):
        """ No scale, but profile `Parameter` specified """

        s = Storage(model, 'Storage', max_volume=100.0)
        l = Link(model, 'Link')

        data = {
            'type': 'monthlyprofilecontrolcurve',
            'control_curves': [0.8, 0.6],
            'values': [[1.0] * 12, [0.7] * np.arange(12), [0.3] * 12],
            'storage_node': 'Storage',
            'profile': {
                'type': 'dailyprofile',
                'values': [1.5]*366
            }
        }

        l.max_flow = p = load_parameter(model, data)
        p.setup(model)
        model.scenarios.setup()
        self._assert_results(model, s, p, scale=1.5)

    def test_no_scale_profile(self, model):
        """ No scale, but profile array specified """

        s = Storage(model, 'Storage', max_volume=100.0)
        l = Link(model, 'Link')

        data = {
            'type': 'monthlyprofilecontrolcurve',
            'control_curves': [0.8, 0.6],
            'values': [[1.0] * 12, [0.7] * np.arange(12), [0.3] * 12],
            'storage_node': 'Storage',
            'profile': [1.5]*12
        }

        l.max_flow = p = load_parameter(model, data)
        p.setup(model)
        model.scenarios.setup()
        self._assert_results(model, s, p, scale=1.5)

    def test_json_load(self, solver):

        model = load_model("demand_saving.json", solver=solver)

        storage = model.nodes["supply1"]
        demand = model.nodes["demand1"]
        assert (isinstance(demand.max_flow, MonthlyProfileControlCurveParameter))

        model.setup()

        profile = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0]) * 10.0
        saving = np.array([
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8]
                ])

        scenario_index = ScenarioIndex(0, np.array([], dtype=np.int32))

        for i in range(12):
            model.step()
            # First two timesteps should result in storage above 50% control curve
            # Therefore no demand saving
            if i < 2:
                expected_max_flow = profile[i] * saving[0, i]
            else:
                expected_max_flow = profile[i] * saving[1, i]

            value = demand.max_flow.value(model.timestepper.current, scenario_index)
            assert_allclose(value, expected_max_flow)



def test_daily_profile_control_curve(model):
    """ Test `DailyProfileControlCurveParameter` """

    s = Storage(model, 'Storage', max_volume=100.0)
    l = Link(model, 'Link')

    data = {
        'type': 'dailyprofilecontrolcurve',
        'control_curves': [0.8, 0.6],
        'values': [[1.0]*366, [0.7]*np.arange(366), [0.3]*366],
        'storage_node': 'Storage'
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
        np.testing.assert_allclose(p.value(ts, si), 1.0)

    s.initial_volume = 70.0
    model.reset()  # Set initial volume on storage
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    for mth in range(1, 13):
        ts = Timestep(datetime.datetime(2016, mth, 1), 366, 1.0)
        doy = ts.datetime.dayofyear
        np.testing.assert_allclose(p.value(ts, si), 0.7*(doy - 1))

    s.initial_volume = 30.0
    model.reset()  # Set initial volume on storage
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    for mth in range(1, 13):
        ts = Timestep(datetime.datetime(2016, mth, 1), 366, 1.0)
        np.testing.assert_allclose(p.value(ts, si), 0.3)

def test_demand_saving_with_indexed_array(solver):
    """Test demand saving based on reservoir control curves

    This is a relatively complex test to pass due to the large number of
    dependencies of the parameters actually being tested. The test is an
    example of how demand savings can be applied in times of drought based
    on the state of a reservoir.
    """

    model = load_model("demand_saving2.json", solver=solver)

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
    assert_allclose(rec_demand.data[0, 0], demand_baseline * demand_factor * demand_saving)

    # first control curve breached
    demand_saving = 0.95
    assert(rec_storage.data[4, 0] < (0.8 * max_volume) )
    assert_allclose(rec_demand.data[5, 0], demand_baseline * demand_factor * demand_saving)

    # second control curve breached
    demand_saving = 0.5
    assert(rec_storage.data[11, 0] < (0.5 * max_volume) )
    assert_allclose(rec_demand.data[12, 0], demand_baseline * demand_factor * demand_saving)


def test_demand_saving_with_indexed_array_from_hdf(solver):
    """Test demand saving based on a predefined demand saving level in a HDF file."""
    model = load_model("demand_saving_hdf.json", solver=solver)

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

    # second control c2urve breached
    demand_saving = 0.25
    assert_allclose(rec_demand.data[13, 0], demand_baseline * demand_saving)