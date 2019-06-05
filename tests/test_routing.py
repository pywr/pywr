"""Routing tests.
"""
from __future__ import division
from pywr.core import Model, Input, Output
from pywr.recorders import NumpyArrayNodeRecorder
from pywr.domains.river import RoutedRiver
import numpy as np
import pytest


@pytest.fixture()
def simple_routed_model(request):
    """
    Make a simple model with a single Input and Output.

    Input -> Link -> Output

    """
    model = Model()
    inpt = Input(model, name="Input", max_flow=10, min_flow=10)

    lnk = RoutedRiver(model, name="River", weighting=0.5, time_of_travel=1)
    inpt.connect(lnk)

    otpt = Output(model, name="Output", cost=-10)
    lnk.connect(otpt)

    return model


class TestLinearRoutingParameter:
    """ Tests for `LinearRoutingParameter` """

    @pytest.mark.parametrize("weighting,time_of_travel", [(0.5, 1), (0.1, 2), (0, 5)])
    def test_basic_use(self, simple_routed_model, weighting, time_of_travel):
        """ Test the basic use of `LinearRoutingParameter` using the Python API """
        model = simple_routed_model

        inpt = model.nodes['Input']
        inpt_rec = NumpyArrayNodeRecorder(model, inpt)

        otpt = model.nodes['Output']
        otpt_rec = NumpyArrayNodeRecorder(model, otpt)

        river = model.nodes['River']
        p = river.agg_node.max_flow  # LinearRoutingParameter is applied to the aggregated node in RoutedRiver

        river.weighting = weighting
        river.time_of_travel = time_of_travel

        model.run()

        np.testing.assert_allclose(inpt_rec.data, 10)
        dt = model.timestepper.delta.days

        prev_in_flow = 0.0
        prev_out_flow = 0.0
        for i, (in_flow, out_flow) in enumerate(zip(inpt_rec.data, otpt_rec.data)):
            # Calculate Muskingham coefficients & expected flow
            d = (time_of_travel * (1 - weighting) + 0.5 * dt)
            c0 = -(time_of_travel * weighting - 0.5 * dt) / d
            c1 = (time_of_travel * weighting + 0.5 * dt) / d
            c2 = (time_of_travel * (1 - weighting) - 0.5 * dt) / d
            expected_flow = in_flow * c0 + prev_in_flow * c1 + prev_out_flow * c2

            np.testing.assert_allclose(out_flow, expected_flow)
            # Update previous flows for next time-step
            prev_in_flow = in_flow
            prev_out_flow = out_flow



