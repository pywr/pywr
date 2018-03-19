"""
This example shows the trade-off (pareto frontier) of deficit against cost by altering a reservoir control curve.

Two types of control curve are possible. The first is a monthly control curve containing one value for each
month. The second is a harmonic control curve with cosine terms around a mean. Both Parameter objects
are part of pywr.parameters.

Inspyred is used in this example to perform a multi-objective optimisation using the NSGA-II algorithm. The
script should be run twice (once with --harmonic) to generate results for both types of control curve. Following
this --plot can be used to generate an animation and PNG of the pareto frontier.

"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_model(klass, harmonic=True):

    with open('two_reservoir.json') as fh:
        data = json.load(fh)

    if harmonic:
        # Patch the control curve parameter
        data['parameters']['control_curve'] = {
            'type': 'AnnualHarmonicSeries',
            'mean': 0.5,
            'amplitudes': [0.5, 0.0],
            'phases': [0.0, 0.0],
            'mean_upper_bounds': 1.0,
            'amplitude_upper_bounds': 1.0,
            'is_variable': True
        }

    return klass.load(data)


def platypus_main(harmonic=False):
    import platypus
    from pywr.optimisation.platypus import PlatypusOptimisationModel

    model = load_model(PlatypusOptimisationModel, harmonic=harmonic)
    problem = model.setup()

    algorithm = platypus.NSGAII(problem)
    algorithm.run(10000)

    plt.scatter([s.objectives[0] for s in algorithm.result],
                [s.objectives[1] for s in algorithm.result])

    plt.xlabel(model._objectives[0].name)
    plt.ylabel(model._objectives[1].name)
    plt.grid(True)
    plt.ylim(0, 4000)
    plt.xlim(215000, 215500)
    title = 'Harmonic Control Curve' if harmonic else 'Monthly Control Curve'
    plt.savefig('{} Example ({}).pdf'.format('platypus', title), format='pdf')
    plt.show()


def inspyred_main(prng=None, display=False, harmonic=False):
    import inspyred
    from pywr.optimisation.inspyred import InspyredOptimisationModel
    from random import Random
    from time import time

    if prng is None:
        prng = Random()
        prng.seed(time())

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    stats_file = open('{}-{}-statistics-file.csv'.format(script_name, 'harmonic' if harmonic else 'monthly'), 'w')
    individuals_file = open('{}-{}-individuals-file.csv'.format(script_name, 'harmonic' if harmonic else 'monthly'), 'w')

    problem = load_model(InspyredOptimisationModel, harmonic=harmonic)
    problem.setup()
    ea = inspyred.ec.emo.NSGA2(prng)
    #ea.variator = [inspyred.ec.variators.blend_crossover,
    #               inspyred.ec.variators.gaussian_mutation]
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = [
        inspyred.ec.observers.file_observer,
    ]
    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          pop_size=100,
                          bounder=problem.bounder,
                          maximize=False,
                          max_generations=200,
                          statistics_file=stats_file,
                          individuals_file=individuals_file)

    # Save the final population archive  to CSV files
    stats_file = open('{}-{}-final-statistics-file.csv'.format(script_name, 'harmonic' if harmonic else 'monthly'), 'w')
    individuals_file = open('{}-{}-final-individuals-file.csv'.format(script_name, 'harmonic' if harmonic else 'monthly'), 'w')
    inspyred.ec.observers.file_observer(ea.archive, 'final', None,
                                        args={'statistics_file': stats_file, 'individuals_file': individuals_file})

    if display:
        final_arc = ea.archive
        print('Best Solutions: \n')
        for f in final_arc:
            print(f)

        x = []
        y = []

        for f in final_arc:
            x.append(f.fitness[0])
            y.append(f.fitness[1])

        plt.scatter(x, y, c='b')
        plt.xlabel('Total demand deficit [Ml/d]')
        plt.ylabel('Total Transferred volume [Ml/d]')
        plt.ylim(0, 4000)
        plt.xlim(215000, 215500)
        plt.grid()
        title = 'Harmonic Control Curve' if harmonic else 'Monthly Control Curve'
        plt.savefig('{0} Example ({1}).pdf'.format(ea.__class__.__name__, title), format='pdf')
        plt.show()
    return ea


def load_individuals(filename):
    """ Read an inspyred individuals file in to two pandas.DataFrame objects.

    There is one DataFrame for the objectives and another for the variables.
    """
    import ast

    index = []
    all_objs = []
    all_vars = []
    with open(filename, 'r') as f:
        for row in f.readlines():
            gen, pop_id, objs, vars = ast.literal_eval(row.strip())
            index.append((gen, pop_id))
            all_objs.append(objs)
            all_vars.append(vars)

    index = pd.MultiIndex.from_tuples(index, names=['generation', 'individual'])
    return pd.DataFrame(all_objs, index=index), pd.DataFrame(all_vars, index=index)


def animate_generations(objective_data, colors):
    """
    Animate the pareto frontier plot over the saved generations.
    """
    import matplotlib.animation as animation

    def update_line(gen, dfs, ax, xmax, ymax):
        ax.cla()
        artists = []
        for i in range(gen+1):
            for c, key in zip(colors, sorted(dfs.keys())):
                df = dfs[key]

                scat = ax.scatter(df.loc[i][0], df.loc[i][1], alpha=0.8**(gen-i), color=c,
                              label=key if i == gen else None, clip_on=True, zorder=100)
                artists.append(scat)

        ax.set_title('Generation: {:d}'.format(gen))
        ax.set_xlabel('Total demand deficit [Ml/d]')
        ax.set_ylabel('Total Transferred volume [Ml/d]')
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.legend()
        ax.grid()
        return artists

    fig, ax = plt.subplots(figsize=(10, 10))
    last_gen = list(objective_data.values())[0].index[-1][0]
    last_gen = int(last_gen)

    xmax = max(df.loc[last_gen][0].max() for df in objective_data.values())
    ymax = max(df.loc[last_gen][1].max() for df in objective_data.values())

    line_ani = animation.FuncAnimation(fig, update_line, last_gen+1,
                                       fargs=(objective_data, ax, xmax, ymax), interval=400, repeat=False)

    line_ani.save('generations.mp4', bitrate=1024,)
    fig.savefig('generations.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--harmonic', action='store_true', help='Use an harmonic control curve.')
    parser.add_argument('--plot', action='store_true', help='Plot the pareto frontier.')
    parser.add_argument('--library', type=str, choices=['inspyred', 'platypus'])
    args = parser.parse_args()

    if args.plot:
        objs, vars = {}, {}
        for cctype in ('monthly', 'harmonic'):
            objs[cctype], vars[cctype] = load_individuals('two_reservoir_moea-{}-individuals-file.csv'.format(cctype))

        animate_generations(objs, ('b', 'r'))
        plt.show()
    else:
        if args.library == 'inspyred':
            inspyred_main(display=True, harmonic=args.harmonic)
        elif args.library == 'platypus':
            platypus_main(harmonic=args.harmonic)
