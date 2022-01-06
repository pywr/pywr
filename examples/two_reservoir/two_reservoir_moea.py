"""
This example shows the trade-off (pareto frontier) of deficit against cost by altering a reservoir control curve.

Two types of control curve are possible. The first is a monthly control curve containing one value for each
month. The second is a harmonic control curve with cosine terms around a mean. Both Parameter objects
are part of pywr.parameters.

The example demonstrates the use of two different optimisation libraries: pygmo and platypus. The
choice of library can be made with a command line argument.

"""
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_model_data(harmonic=True):

    with open("two_reservoir.json") as fh:
        data = json.load(fh)

    if harmonic:
        # Patch the control curve parameter
        data["parameters"]["control_curve"] = {
            "type": "AnnualHarmonicSeries",
            "mean": 0.5,
            "amplitudes": [0.5, 0.0],
            "phases": [0.0, 0.0],
            "mean_upper_bounds": 1.0,
            "amplitude_upper_bounds": 1.0,
            "is_variable": True,
        }

    return data


def platypus_main(harmonic=False):
    import platypus
    from pywr.optimisation.platypus import PlatypusWrapper

    wrapper = PlatypusWrapper(get_model_data(harmonic=harmonic))

    with platypus.ProcessPoolEvaluator() as evaluator:
        algorithm = platypus.NSGAIII(
            wrapper.problem, population_size=50, evaluator=evaluator, divisions_outer=12
        )
        algorithm.run(10000)

    objectives = pd.DataFrame(
        data=np.array([s.objectives for s in algorithm.result]),
        columns=[o.name for o in wrapper.model_objectives],
    )

    title = "harmonic" if harmonic else "monthly"
    objectives.to_hdf("two_reservoir_moea.h5", f"platypus_{title}")


def pygmo_main(harmonic=False):
    import pygmo as pg
    from pywr.optimisation.pygmo import PygmoWrapper

    def update_archive(pop, archive=None):
        if archive is None:
            combined_f = pop.get_f()
        else:
            combined_f = np.r_[archive, pop.get_f()]
        indices = pg.select_best_N_mo(combined_f, 50)
        new_archive = combined_f[indices, :]
        new_archive = np.unique(new_archive.round(4), axis=0)
        return new_archive

    wrapper = PygmoWrapper(get_model_data(harmonic=harmonic))
    prob = pg.problem(wrapper)
    print(prob)
    algo = pg.algorithm(pg.moead(gen=1))
    # algo = pg.algorithm(pg.nsga2(gen=1))

    pg.mp_island.init_pool(2)
    isl = pg.island(algo=algo, prob=prob, size=50, udi=pg.mp_island())

    ref_point = [216500, 4000]
    pop = isl.get_population()

    print("Evolving!")
    archive = update_archive(pop)
    hv = pg.hypervolume(archive)
    vol = hv.compute(ref_point)
    hvs = [
        vol,
    ]
    print(
        "Gen: {:03d}, Hypervolume: {:6.4g}, Archive size: {:d}".format(
            0, vol, len(archive)
        )
    )

    for gen in range(20):
        isl.evolve(1)
        isl.wait_check()

        pop = isl.get_population()
        archive = update_archive(pop, archive)

        hv = pg.hypervolume(archive)
        vol = hv.compute(ref_point)
        print(
            "Gen: {:03d}, Hypervolume: {:6.4g}, Archive size: {:d}".format(
                gen + 1, vol, len(archive)
            )
        )
        hvs.append(vol)

    hvs = pd.Series(hvs)

    print("Finished!")

    plt.scatter(archive[:, 0], archive[:, 1])

    objectives = wrapper.model_objectives
    plt.xlabel(objectives[0].name)
    plt.ylabel(objectives[1].name)
    plt.grid(True)
    title = "Harmonic Control Curve" if harmonic else "Monthly Control Curve"
    plt.savefig("{} Example ({}).pdf".format("pygmo", title), format="pdf")

    fig, ax = plt.subplots()
    ax.plot(hvs / 1e6, marker="o")
    ax.grid(True)
    ax.set_ylabel("Hypervolume")
    ax.set_xlabel("Generation")
    plt.savefig("{} Example Hypervolume ({}).pdf".format("pygmo", title), format="pdf")

    plt.show()


if __name__ == "__main__":
    import argparse

    # Setup logging
    import logging

    logger = logging.getLogger("pywr")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--harmonic", action="store_true", help="Use an harmonic control curve."
    )
    parser.add_argument("--library", type=str, choices=["platypus", "pygmo"])
    args = parser.parse_args()

    if args.library == "platypus":
        platypus_main(harmonic=args.harmonic)
    elif args.library == "pygmo":
        pygmo_main(harmonic=args.harmonic)
    else:
        raise ValueError(
            'Optimisation library "{}" not recognised.'.format(args.library)
        )
