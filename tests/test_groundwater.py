from pywr.domains.groundwater import KeatingAquifer
from pywr.core import Model, Input, Output, Link


def test_keating_model(solver):
    model = Model(solver=solver)

    stream_flow_levels = [75.0, 80.0]
    transmissivity = [1000.0, 10000.0]
    storage = [0.01, 0.05]
    levels = [70.0, 78.0, 90.0]
    area = 10000**2  # 100km2
    volumes = [0.0]
    for i in range(len(levels)-1):
        volumes.append(
            volumes[-1] + (levels[i+1] - levels[i])*area*storage[i]
        )

    # Create a simple river system
    inpt = Input(model, 'input')
    lnk = Link(model, 'link')
    otpt = Output(model, 'output')

    inpt.connect(lnk)
    lnk.connect(otpt)

    aqfer = KeatingAquifer(model, 'aquifer', levels, volumes, stream_flow_levels, transmissivity)
    aqfer.initial_volume = volumes[-2]
    aqfer.connect(lnk)

    model.run()

    # TODO test something!
