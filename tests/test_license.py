#!/usr/bin/env python

import pytest
from datetime import datetime
from pywr._core import Timestep
from pywr.licenses import License, TimestepLicense, AnnualLicense
from fixtures import simple_linear_model
from numpy.testing import assert_allclose

def test_base_license():
    with pytest.raises(TypeError):
        lic = License()

def test_daily_license():
    '''Test daily licence'''

    lic = TimestepLicense(42.0)
    assert(isinstance(lic, License))
    assert(lic.value(Timestep(datetime(2015, 1, 1), 0, 0)) == 42.0)

    # daily licences don't have resource state
    assert(lic.resource_state(Timestep(datetime(2015, 1, 1), 0, 0)) is None)


def test_simple_model_with_annual_licence(simple_linear_model):
    m = simple_linear_model

    annual_total = 365
    lic = AnnualLicense(annual_total)
    # Apply licence to the model
    m.node["Input"].max_flow = lic
    m.node["Output"].max_flow = 10.0
    m.node["Output"].cost = -10.0
    m.setup()
    print(lic)
    # timestepper.current is the next time step (because model has not begun)
    assert_allclose(lic.value(m.timestepper.current), annual_total/365)
    m.step()

    # Licence is a hard constraint of 1.0
    # timestepper.current is no end of the first day
    assert_allclose(m.node["Output"].flow, 1.0)
    # Check the constraint for the next timestep.
    assert_allclose(lic.value(m.timestepper._next), 1.0)

    # Now constrain the demand so that licence is not fully used
    m.node["Output"].max_flow = 0.5
    m.step()

    assert_allclose(m.node["Output"].flow, 0.5)
    # Check the constraint for the next timestep. The available amount should now be large
    # due to the reduced use
    remaining = (annual_total-1.5)
    assert_allclose(lic.value(m.timestepper._next), remaining / (365 - 2))

    # Unconstrain the demand
    m.node["Output"].max_flow = 10.0
    m.step()
    assert_allclose(m.node["Output"].flow, remaining / (365 - 2))
    # Licence should now be on track for an expected value of 1.0
    remaining -= remaining / (365 - 2)
    assert_allclose(lic.value(m.timestepper._next), remaining / (365 - 3))