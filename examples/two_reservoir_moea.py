"""
This example shows the trade-off of deficit against cost by altering a reservoir control curve.
"""

import os
import pandas
import numpy as np
import inspyred
from pywr.core import Model, Input, Output, Link, Storage
from pywr.parameters import ArrayIndexedParameter, MonthlyProfileParameter, AnnualHarmonicSeriesParameter
from pywr.parameters.control_curves import ControlCurvePiecewiseParameter
from pywr.recorders import TotalDeficitNodeRecorder, TotalFlowRecorder, AggregatedRecorder
from pywr.optimisation.moea import InspyredOptimisationModel


def create_model(harmonic=True):
    # import flow timeseries for catchments
    flow = pandas.read_csv(os.path.join('data', 'Not a real flow series.csv'))

    flow['Date'] = flow['Date'].apply(pandas.to_datetime)
    flow.set_index('Date', inplace=True)
    flow_parameter = ArrayIndexedParameter(flow['Flow'].values)

    model = InspyredOptimisationModel(
        solver='glpk',
        parameters={
            'timestamp_start': flow.index[0],
            'timestamp_finish': flow.index[-1],
        }
    )

    catchment1 = Input(model, 'catchment1', min_flow=flow_parameter, max_flow=flow_parameter)
    catchment2 = Input(model, 'catchment2', min_flow=flow_parameter, max_flow=flow_parameter)

    reservoir1 = Storage(model, 'reservoir1', min_volume=3000, max_volume=20000, volume=16000)
    reservoir2 = Storage(model, 'reservoir2', min_volume=3000, max_volume=20000, volume=16000)

    if harmonic:
        control_curve = AnnualHarmonicSeriesParameter(0.5, [0.5], [0.0], mean_upper_bounds=1.0, amplitude_upper_bounds=1.0)
    else:
        control_curve = MonthlyProfileParameter(np.array([0.0]*12, np.float32), lower_bounds=0.0, upper_bounds=1.0)

    control_curve.is_variable = True
    controller = ControlCurvePiecewiseParameter(control_curve, 10.0, 0.0, storage_node=reservoir1)
    transfer = Link(model, 'transfer', max_flow=controller, cost=-500)

    demand1 = Output(model, 'demand1', max_flow=100.0, cost=-101)
    demand2 = Output(model, 'demand2', max_flow=81.0, cost=-100)

    river1 = Link(model, 'river1')
    river2 = Link(model, 'river2')

    # compensation flows from reservoirs
    compensation1 = Link(model, 'compensation1', max_flow=5.0, cost=-9999)
    compensation2 = Link(model, 'compensation2', max_flow=5.0, cost=-9998)

    terminator = Output(model, 'terminator', cost=1.0)

    catchment1.connect(reservoir1)
    catchment2.connect(reservoir2)
    reservoir1.connect(demand1)
    reservoir2.connect(demand2)
    reservoir2.connect(transfer)
    transfer.connect(reservoir1)
    reservoir1.connect(river1)
    reservoir2.connect(river2)
    river1.connect(terminator)
    river2.connect(terminator)
    compensation1.connect(terminator)
    compensation2.connect(terminator)

    r1 = TotalDeficitNodeRecorder(model, demand1)
    r2 = TotalDeficitNodeRecorder(model, demand2)
    r3 = AggregatedRecorder(model, [r1, r2])
    r3.is_objective = True
    r4 = TotalFlowRecorder(model, transfer)
    r4.is_objective = True

    return model



def main(prng=None, display=False, harmonic=False):
    from random import Random
    from time import time

    if prng is None:
        prng = Random()
        prng.seed(time())

    problem = create_model(harmonic=harmonic)
    problem.setup()
    ea = inspyred.ec.emo.NSGA2(prng)
    ea.variator = [inspyred.ec.variators.blend_crossover,
                   inspyred.ec.variators.gaussian_mutation]
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = [
        inspyred.ec.observers.file_observer,
    ]
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          pop_size=25,
                          bounder=problem.bounder,
                          maximize=False,
                          max_generations=20)

    if display:
        from matplotlib import pylab
        final_arc = ea.archive
        print('Best Solutions: \n')
        for f in final_arc:
            print(f)
        import pylab
        x = []
        y = []

        for f in final_arc:
            x.append(f.fitness[0])
            y.append(f.fitness[1])

        pylab.scatter(x, y, c='b')
        pylab.xlabel('Maximum demand deficit [Ml/d]')
        pylab.ylabel('Transferred volume [Ml/d]')
        title = 'Harmonic Control Curve' if harmonic else 'Monthly Control Curve'
        pylab.savefig('{0} Example ({1}).pdf'.format(ea.__class__.__name__, title), format='pdf')
        pylab.show()
    return ea

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--harmonic', action='store_true', help='Use an harmonic control curve.')
    args = parser.parse_args()

    main(display=True, harmonic=args.harmonic)