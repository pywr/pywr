from pywr.core import Model, Storage, Link, ScenarioIndex
from pywr.parameters import ConstantParameter, DailyProfileParameter
from pywr.parameters.control_curves import ControlCurvePiecewiseParameter, ControlCurveInterpolatedParameter
import numpy as np
from numpy.testing import assert_allclose
import pytest

from helpers import load_model

@pytest.fixture
def model(solver):
    return Model(solver=solver)


def test_control_curve_piecewise(model):
    m = model
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    s = Storage(m, 'Storage', max_volume=100.0)

    cc = ConstantParameter(0.8)
    # Return 10.0 when above 0.0 when below
    s.cost = ControlCurvePiecewiseParameter(cc, 10.0, 0.0)
    s.setup(m)

    s.initial_volume = 90.0
    m.reset()
    assert_allclose(s.get_cost(m.timestepper.current, si), 10.0)

    s.initial_volume = 50.0
    m.reset()
    assert_allclose(s.get_cost(m.timestepper.current, si), 0.0)

    # Now test if the parameter is used on a non storage node

    l = Link(m, 'Link')
    cc = ConstantParameter(0.8)
    l.cost = ControlCurvePiecewiseParameter(cc, 10.0, 0.0, storage_node=s)
    assert_allclose(l.get_cost(m.timestepper.current, si), 0.0)
    # When storage volume changes, the cost of the link changes.
    s.initial_volume = 90.0
    m.reset()
    assert_allclose(l.get_cost(m.timestepper.current, si), 10.0)


def test_control_curve_interpolated(model):
    m = model
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))


    s = Storage(m, 'Storage', max_volume=100.0)

    cc = ConstantParameter(0.8)
    values = [0.0, 5.0, 20.0]
    s.cost = ControlCurveInterpolatedParameter(cc, values)
    s.setup(m)

    for v in (0.0, 10.0, 50.0, 80.0, 90.0, 100.0):
        s.initial_volume = v
        s.reset()
        assert_allclose(s.get_cost(m.timestepper.current, si), np.interp(v/100.0, [0.0, 0.8, 1.0], values))

    # special case when control curve is 100%
    cc._value = 1.0
    s.initial_volume == 100.0
    s.reset()
    assert_allclose(s.get_cost(m.timestepper.current, si), values[1])

def test_control_curve_interpolated_json(solver):
    """Test loading a reservoir with a daily profile control curve from JSON"""
    model = load_model("reservoir_with_cc.json", solver=solver)
    
    storage = model.nodes["reservoir1"]
    assert(isinstance(storage.cost, ControlCurveInterpolatedParameter))
    assert(isinstance(storage.cost.control_curve, DailyProfileParameter))

    model.setup()

    cc_values = [0.604, 0.608, 0.612]
    cost_values = [True, False, False]
    
    scenario_index = ScenarioIndex(0, np.array([], dtype=np.int32))
    
    for expected_cc, expected_cost in zip(cc_values, cost_values):
        model.step()
        value = storage.cost.control_curve.value(model.timestep, scenario_index)
        cost = storage.cost.value(model.timestep, scenario_index)
        assert_allclose(value, expected_cc)
        assert((cost >= -6) == expected_cost)
