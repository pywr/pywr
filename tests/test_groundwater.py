from pywr.domains.groundwater import KeatingAquifer
from pywr.parameters.groundwater import KeatingStreamFlowParameter
from pywr.core import Model, Input, Output, Link
from pywr.recorders import (
    NumpyArrayNodeRecorder,
    NumpyArrayLevelRecorder,
    NumpyArrayStorageRecorder,
)
import pandas
import numpy as np
import pytest

num_streams = 1
num_additional_inputs = 1
stream_flow_levels = [[100.0, 125.0]]  # m
transmissivity = [1000, 20000]  # m2/d
transmissivity = [t * 0.001 for t in transmissivity]  # m3 to Ml
coefficient = 1  # no units
storativity = [0.05]  # %
levels = [0.0, 1000.0]  # m
area = 50000 * 50000  # m2


def test_keating_aquifer():
    model = Model(
        start=pandas.to_datetime("2016-01-01"),
        end=pandas.to_datetime("2016-01-01"),
    )

    aqfer = KeatingAquifer(
        model,
        "keating",
        num_streams,
        num_additional_inputs,
        stream_flow_levels,
        transmissivity,
        coefficient,
        levels,
        area=area,
        storativity=storativity,
    )

    catchment = Input(model, "catchment", max_flow=0)
    stream = Output(model, "stream", max_flow=np.inf, cost=0)
    abstraction = Output(model, "abstraction", max_flow=15, cost=-999)

    catchment.connect(aqfer)
    aqfer.connect(stream, from_slot=0)
    aqfer.connect(abstraction, from_slot=1)

    rec_level = NumpyArrayLevelRecorder(model, aqfer)
    rec_volume = NumpyArrayStorageRecorder(model, aqfer)
    rec_stream = NumpyArrayNodeRecorder(model, stream)
    rec_abstraction = NumpyArrayNodeRecorder(model, abstraction)

    model.check()

    assert len(aqfer.inputs) == (num_streams + num_additional_inputs)

    for initial_level in (50, 100, 110, 150):
        # set the inital aquifer level and therefor the initial volume
        aqfer.initial_level = initial_level
        initial_volume = aqfer.initial_volume
        assert initial_volume == (area * storativity[0] * initial_level * 0.001)
        # run the model (for one timestep only)
        model.run()
        # manually calculate keating streamflow and check model flows are OK
        Qp = (
            2
            * transmissivity[0]
            * max(initial_level - stream_flow_levels[0][0], 0)
            * coefficient
        )
        Qe = (
            2
            * transmissivity[1]
            * max(initial_level - stream_flow_levels[0][1], 0)
            * coefficient
        )
        delta_storage = initial_volume - rec_volume.data[0, 0]
        abs_flow = rec_abstraction.data[0, 0]
        stream_flow = rec_stream.data[0, 0]
        assert delta_storage == (stream_flow + abs_flow)
        assert stream_flow == (Qp + Qe)

    A_VERY_LARGE_NUMBER = 9999999999999
    model.timestepper.end = pandas.to_datetime("2016-01-02")

    # fill the aquifer completely
    # there is no spill for the storage so it should find no feasible solution
    with pytest.raises(RuntimeError):
        catchment.max_flow = A_VERY_LARGE_NUMBER
        catchment.min_flow = A_VERY_LARGE_NUMBER
        model.run()

    # drain the aquifer completely
    catchment.min_flow = 0
    catchment.max_flow = 0
    abstraction.max_flow = A_VERY_LARGE_NUMBER
    model.setup()
    model.run()
    assert rec_volume.data[1, 0] == 0
    abs_flow = rec_abstraction.data[1, 0]
    stream_flow = rec_stream.data[1, 0]
    assert stream_flow == 0
    assert abs_flow == 0
