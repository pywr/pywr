from pywr.core import Model, Storage, Link, ScenarioIndex
from pywr.parameters import ConstantParameter
from pywr.parameters.control_curves import ControlCurvePiecewiseParameter, ControlCurveInterpolatedParameter
import numpy as np
from numpy.testing import assert_allclose

def test_control_curve_piecewise():
    m = Model()
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


def test_control_curve_interpolated():

    m = Model()
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