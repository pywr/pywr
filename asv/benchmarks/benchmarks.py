# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from pywr.core import Model, Input, Output, Link, Storage
import pandas

solver = 'glpk'

class Meh:
    def setup(self):
        pass
    
    def time_meh(self):
        model = Model(solver=solver, parameters={
                'timestamp_start': pandas.to_datetime('2015-01-01'),
                'timestamp_finish': pandas.to_datetime('2115-12-31')
        })
        inpt = Input(model, name="Input", max_flow=10.0)
        lnk = Link(model, name="Link", cost=1.0)
        inpt.connect(lnk)
        otpt = Output(model, name="Output", max_flow=5.0, cost=-2.0)
        lnk.connect(otpt)
        model.run()
