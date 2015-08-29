from pywr.core import Model, Input, Output, Storage
from pywr._core import NumpyArrayRecorder
from pywr.solvers.cython_glpk import CythonGLPKSolver

import pandas
import numpy as np
import datetime

import time

def test_me():
    model = Model(
        parameters={
            'timestamp_start': pandas.to_datetime('1888-01-01'),
            'timestamp_finish': pandas.to_datetime('1888-01-05'),
            'timestep': datetime.timedelta(1),
        },
        solver='cyglpk'
    )

    supply1 = Input(model, 'supply1')
    supply1.max_flow = 3.0
    supply1.cost = 10
    supply1.recorder = NumpyArrayRecorder(5)

    reservoir1 = Storage(model, name='reservoir1')
    reservoir1.min_volume = 0.0
    reservoir1.max_volume = 100.0
    reservoir1._volume = 16.0
    reservoir1.cost = 5
    reservoir1.recorder = NumpyArrayRecorder(5)

    demand1 = Output(model, 'demand1')
    demand1.max_flow = 5.0
    demand1.cost = -100
    demand1.recorder = NumpyArrayRecorder(5)

    supply1.connect(reservoir1)
    reservoir1.connect(demand1)

    #t0 = time.time()
    model.run()
    #print(time.time()-t0)

    assert(np.array_equal(supply1.recorder.data, np.array([0, 0, 0, 3, 3])))
    assert(np.array_equal(reservoir1.recorder.data, np.array([11, 6, 1, 0, 0])))
    assert(np.array_equal(demand1.recorder.data, np.array([5, 5, 5, 4, 3])))
