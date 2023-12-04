#!/usr/bin/env python

import pytest
from datetime import datetime
from pywr.core import Timestep, ScenarioIndex, Model
from pywr.parameters.licenses import (
    License,
    TimestepLicense,
    AnnualLicense,
    AnnualExponentialLicense,
    AnnualHyperbolaLicense,
)
from pywr.recorders import NumpyArrayNodeRecorder
from fixtures import simple_linear_model
from helpers import load_model
from numpy.testing import assert_allclose
import numpy as np
import pandas


def test_base_license():
    with pytest.raises(TypeError):
        lic = License()


def test_daily_license(simple_linear_model):
    """Test daily licence"""
    m = simple_linear_model
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))
    lic = TimestepLicense(m, None, 42.0)
    assert isinstance(lic, License)
    assert lic.value(Timestep(pandas.Period("2015-1-1"), 0, 1), si) == 42.0

    # daily licences don't have resource state
    assert lic.resource_state(Timestep(pandas.Period("2015-1-1"), 0, 1)) is None


@pytest.mark.skipif(
    Model().solver.name.startswith("glpk") and Model().solver.set_fixed_flows_once,
    reason="This test changes constant constraints between steps.",
)
def test_simple_model_with_annual_licence(simple_linear_model):
    m = simple_linear_model
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    annual_total = 365
    lic = AnnualLicense(m, m.nodes["Input"], annual_total)
    # Apply licence to the model
    m.nodes["Input"].max_flow = lic
    m.nodes["Output"].max_flow = 10.0
    m.nodes["Output"].cost = -10.0
    m.setup()

    m.step()

    # Licence is a hard constraint of 1.0
    # timestepper.current is now end of the first day
    assert_allclose(m.nodes["Output"].flow, 1.0)
    # Check the constraint for the next timestep.
    assert_allclose(lic.value(m.timestepper._next, si), 1.0)

    # Now constrain the demand so that licence is not fully used
    m.nodes["Output"].max_flow = 0.5
    m.step()

    assert_allclose(m.nodes["Output"].flow, 0.5)
    # Check the constraint for the next timestep. The available amount should now be larger
    # due to the reduced use
    remaining = annual_total - 1.5
    assert_allclose(lic.value(m.timestepper._next, si), remaining / (365 - 2))

    # Unconstrain the demand
    m.nodes["Output"].max_flow = 10.0
    m.step()
    assert_allclose(m.nodes["Output"].flow, remaining / (365 - 2))
    # Licence should now be on track for an expected value of 1.0
    remaining -= remaining / (365 - 2)
    assert_allclose(lic.value(m.timestepper._next, si), remaining / (365 - 3))


def test_annual_license_json():
    """
    This test demonstrates how an annual licence can be forceably distributed
    evenly across a year. The licence must build up a surplus before it can
    use more than the average.
    """
    model = load_model("annual_license.json")

    model.timestepper.start = "2001-01-01"
    model.timestepper.end = "2001-01-31"
    model.timestepper.delta = 5

    rec = NumpyArrayNodeRecorder(model, model.nodes["supply1"])

    model.run()

    initial_amount = 200.0
    # first day evenly apportions initial amount for each day of year
    first_day = initial_amount / 365
    assert_allclose(rec.data[0], first_day)
    # second day does the same, minus yesterday and with less days remaining
    remaining_days = 365 - 5
    second_day = (initial_amount - first_day * 5) / remaining_days
    assert_allclose(rec.data[1], second_day)
    # actual amount is the same as maximum was taken
    assert_allclose(first_day, second_day)
    # third day nothing is taken (no demand), so licence is saved
    assert_allclose(rec.data[2], 0.0)
    # fourth day more can be supplied as we've built up a surplus
    remaining_days = 365 - 5 * 3
    fourth_day = (initial_amount - (first_day + second_day) * 5) / remaining_days
    assert_allclose(rec.data[3], fourth_day)
    assert fourth_day > first_day


def test_simple_model_with_annual_licence_multi_year(simple_linear_model):
    """Test the AnnualLicense over multiple years"""
    import pandas as pd
    import datetime, calendar

    m = simple_linear_model
    # Modify model to run for 3 years of non-leap years at 30 day time-step.
    m.timestepper.start = pd.to_datetime("2017-1-1")
    m.timestepper.end = pd.to_datetime("2020-1-1")
    m.timestepper.delta = datetime.timedelta(30)

    annual_total = 365.0
    lic = AnnualLicense(m, m.nodes["Input"], annual_total)
    # Apply licence to the model
    m.nodes["Input"].max_flow = lic
    m.nodes["Output"].max_flow = 10.0
    m.nodes["Output"].cost = -10.0
    m.setup()

    for i in range(len(m.timestepper)):
        m.step()
        days_in_year = 365 + int(calendar.isleap(m.timestepper.current.datetime.year))
        assert_allclose(m.nodes["Output"].flow, annual_total / days_in_year)


@pytest.mark.skipif(
    Model().solver.name.startswith("glpk") and Model().solver.set_fixed_flows_once,
    reason="This test changes constant constraints between steps.",
)
def test_simple_model_with_exponential_license(simple_linear_model):
    m = simple_linear_model
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    annual_total = 365
    # Expoential licence with max_value of e should give a hard constraint of 1.0 when on track
    lic = AnnualExponentialLicense(m, m.nodes["Input"], annual_total, np.e)
    # Apply licence to the model
    m.nodes["Input"].max_flow = lic
    m.nodes["Output"].max_flow = 10.0
    m.nodes["Output"].cost = -10.0
    m.setup()

    m.step()

    # Licence is a hard constraint of 1.0
    # timestepper.current is now end of the first day
    assert_allclose(m.nodes["Output"].flow, 1.0)
    # Check the constraint for the next timestep.
    assert_allclose(lic.value(m.timestepper._next, si), 1.0)

    # Now constrain the demand so that licence is not fully used
    m.nodes["Output"].max_flow = 0.5
    m.step()

    assert_allclose(m.nodes["Output"].flow, 0.5)
    # Check the constraint for the next timestep. The available amount should now be larger
    # due to the reduced use
    remaining = annual_total - 1.5
    assert_allclose(
        lic.value(m.timestepper._next, si), np.exp(-remaining / (365 - 2) + 1)
    )

    # Unconstrain the demand
    m.nodes["Output"].max_flow = 10.0
    m.step()
    assert_allclose(m.nodes["Output"].flow, np.exp(-remaining / (365 - 2) + 1))
    # Licence should now be on track for an expected value of 1.0
    remaining -= np.exp(-remaining / (365 - 2) + 1)
    assert_allclose(
        lic.value(m.timestepper._next, si), np.exp(-remaining / (365 - 3) + 1)
    )


@pytest.mark.skipif(
    Model().solver.name.startswith("glpk") and Model().solver.set_fixed_flows_once,
    reason="This test changes constant constraints between steps.",
)
def test_simple_model_with_hyperbola_license(simple_linear_model):
    m = simple_linear_model
    si = ScenarioIndex(0, np.array([0], dtype=np.int32))

    annual_total = 365
    # Expoential licence with max_value of e should give a hard constraint of 1.0 when on track
    lic = AnnualHyperbolaLicense(m, m.nodes["Input"], annual_total, 1.0)
    # Apply licence to the model
    m.nodes["Input"].max_flow = lic
    m.nodes["Output"].max_flow = 10.0
    m.nodes["Output"].cost = -10.0
    m.setup()

    m.step()

    # Licence is a hard constraint of 1.0
    # timestepper.current is now end of the first day
    assert_allclose(m.nodes["Output"].flow, 1.0)
    # Check the constraint for the next timestep.
    assert_allclose(lic.value(m.timestepper._next, si), 1.0)

    # Now constrain the demand so that licence is not fully used
    m.nodes["Output"].max_flow = 0.5
    m.step()

    assert_allclose(m.nodes["Output"].flow, 0.5)
    # Check the constraint for the next timestep. The available amount should now be larger
    # due to the reduced use
    remaining = annual_total - 1.5
    assert_allclose(lic.value(m.timestepper._next, si), (365 - 2) / remaining)

    # Unconstrain the demand
    m.nodes["Output"].max_flow = 10.0
    m.step()
    assert_allclose(m.nodes["Output"].flow, (365 - 2) / remaining)
    # Licence should now be on track for an expected value of 1.0
    remaining -= (365 - 2) / remaining
    assert_allclose(lic.value(m.timestepper._next, si), (365 - 3) / remaining)
