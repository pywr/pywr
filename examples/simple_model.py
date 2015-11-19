#!/bin/python
"""
This script sets up and runs the most simple model possible.
"""
import pywr.core
import pandas

def make_model(solver='glpk'):
    """
    Make a simple model with a single Input and Output.

    Input -> Link -> Output

    """
    model = pywr.core.Model(solver=solver, parameters={
            'timestamp_start': pandas.to_datetime('2015-01-01'),
            'timestamp_finish': pandas.to_datetime('2115-12-31')
    })
    inpt = pywr.core.Input(model, name="Input", max_flow=10.0)
    lnk = pywr.core.Link(model, name="Link", cost=1.0)
    inpt.connect(lnk)
    otpt = pywr.core.Output(model, name="Output", max_flow=5.0, cost=-2.0)
    lnk.connect(otpt)

    return model


if __name__ == '__main__':
    model = make_model()
    model.run()