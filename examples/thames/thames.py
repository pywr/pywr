""" Example Pywr model using a simple reservoir system.



"""
from pywr.model import Model
from pywr.recorders import TablesRecorder
import numpy as np
from matplotlib import pyplot as plt
import click
import time
import json
import pandas

MODEL_FILENAME = "thames.json"


@click.group()
def cli():
    pass


@cli.command()
def run():

    # Run the model
    model = Model.load(MODEL_FILENAME)

    # Add a storage recorder
    TablesRecorder(model, "thames_output.h5", parameters=[p for p in model.parameters])

    # Run the model
    stats = model.run()
    print(stats)
    stats_df = stats.to_dataframe()
    print(stats_df)

    keys_to_plot = (
        "time_taken_before",
        "solver_stats.bounds_update_nonstorage",
        "solver_stats.bounds_update_storage",
        "solver_stats.objective_update",
        "solver_stats.lp_solve",
        "solver_stats.result_update",
        "time_taken_after",
    )

    keys_to_tabulate = (
        "timesteps",
        "time_taken",
        "solver",
        "num_scenarios",
        "speed",
        "solver_name" "solver_stats.total",
        "solver_stats.number_of_rows",
        "solver_stats.number_of_cols",
        "solver_stats.number_of_nonzero",
        "solver_stats.number_of_routes",
        "solver_stats.number_of_nodes",
    )

    values = []
    labels = []
    explode = []
    solver_sub_total = 0.0
    for k in keys_to_plot:
        v = stats_df.loc[k][0]
        values.append(v)
        label = k.split(".", 1)[-1].replace("_", " ").capitalize()
        explode.append(0.0)
        if k.startswith("solver_stats"):
            labels.append("Solver - {}".format(label))
            solver_sub_total += v
        else:

            labels.append(label)

    values.append(stats_df.loc["solver_stats.total"][0] - solver_sub_total)
    labels.append("Solver - Other")
    explode.append(0.0)

    values.append(stats_df.loc["time_taken"][0] - sum(values))
    values = np.array(values) / sum(values)
    labels.append("Other")
    explode.append(0.0)

    fig, (ax1, ax2) = plt.subplots(
        figsize=(12, 4), ncols=2, sharey="row", gridspec_kw={"width_ratios": [2, 1]}
    )

    print(values, labels)
    ax1.pie(values, explode=explode, labels=labels, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    cell_text = []
    for index, value in stats_df.iterrows():
        if index not in keys_to_tabulate:
            continue
        v = value[0]
        if isinstance(v, (float, np.float64, np.float32)):
            v = f"{v:.2f}"

        cell_text.append([index, v])

    tbl = ax2.table(cellText=cell_text, colLabels=["Statistic", "Value"], loc="center")
    tbl.scale(1.5, 1.5)  # may help
    tbl.set_fontsize(14)
    ax2.axis("off")

    fig.savefig("run_statistics_w_tables.png", dpi=300)
    fig.savefig("run_statistics_w_tables.eps")

    plt.show()


@cli.command()
@click.option("--ext", default="png")
@click.option("--show/--no-show", default=False)
def figures(ext, show):

    for name, df in TablesRecorder.generate_dataframes("thames_output.h5"):
        df.columns = ["Very low", "Low", "Central", "High", "Very high"]

        fig, (ax1, ax2) = plt.subplots(
            figsize=(12, 4), ncols=2, sharey="row", gridspec_kw={"width_ratios": [3, 1]}
        )
        df["2100":"2125"].plot(ax=ax1)
        df.quantile(np.linspace(0, 1)).plot(ax=ax2)

        if name.startswith("reservoir"):
            ax1.set_ylabel("Volume [$Mm^3$]")
        else:
            ax1.set_ylabel("Flow [$Mm^3/day$]")

        for ax in (ax1, ax2):
            ax.set_title(name)
            ax.grid(True)
        plt.tight_layout()

        if ext is not None:
            fig.savefig(f"{name}.{ext}", dpi=300)

    if show:
        plt.show()


@cli.command("plot-res")
@click.option("--ext", default="png")
@click.option("--show/--no-show", default=False)
def plot_res(ext, show):

    end_year = "2105"

    data = {}
    for name, df in TablesRecorder.generate_dataframes("thames_output.h5"):
        df.columns = ["Very low", "Low", "Central", "High", "Very high"]
        data[name] = df

    fig1, ax1 = plt.subplots(figsize=(16, 5), dpi=300)
    data["reservoir1"].loc[:end_year, "Central"].plot(ax=ax1)
    ax1.set_ylabel("Volume [$Mm^3$]")
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(16, 5), dpi=300)
    data["demand_saving_level"].loc[:end_year, "Central"].plot(ax=ax2)
    ax2.set_ylabel("Demand saving level")
    plt.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(16, 5), dpi=300)
    data["demand_max_flow"].loc[:end_year, "Central"].plot(ax=ax3)
    ax3.set_ylabel("Demand [$Mm^3/day$]")
    plt.tight_layout()

    for ax in (ax1, ax2, ax3):
        ax.grid(True)

    if ext is not None:
        fig1.savefig(f"Reservoir.{ext}", dpi=300)
        fig2.savefig(f"Demand saving level.{ext}", dpi=300)
        fig3.savefig(f"Demand.{ext}", dpi=300)

    if show:
        plt.show()


@cli.command("plot-res2")
@click.option("--ext", default="png")
@click.option("--show/--no-show", default=False)
def plot_res2(ext, show):

    end_year = "2105"

    data = {}
    for name, df in TablesRecorder.generate_dataframes("thames_output.h5"):
        df.columns = ["Very low", "Low", "Central", "High", "Very high"]
        data[name] = df

    fig1, ax1 = plt.subplots(figsize=(16, 5), dpi=300)
    data["reservoir1"].loc[:end_year].plot(ax=ax1)
    ax1.set_ylabel("Volume [$Mm^3$]")
    plt.legend()
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(16, 5), dpi=300)
    data["reservoir1"].quantile(np.linspace(0, 1)).plot(ax=ax2)
    ax2.set_ylabel("Volume [$Mm^3$]")
    ax2.set_xlabel("Quantile")
    plt.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(16, 5), dpi=300)
    df = data["demand_saving_level"].apply(pandas.Series.value_counts)
    df /= df.sum(axis=0)
    df.plot.bar(ax=ax3)
    ax3.set_ylabel("Proportion of time.")
    ax3.set_xlabel("Demand saving level")
    plt.tight_layout()

    for ax in (ax1, ax2, ax3):
        ax.grid(True)

    if ext is not None:
        fig1.savefig(f"Reservoir (scenarios).{ext}", dpi=300)
        fig2.savefig(f"Reservoir SDC (scenarios).{ext}", dpi=300)
        fig3.savefig(f"Demand saving level count (scenarios).{ext}", dpi=300)

    if show:
        plt.show()


@cli.command("plot-cc")
@click.option("--ext", default="png")
@click.option("--show/--no-show", default=False)
def plot_control_curves(ext, show):

    with open(MODEL_FILENAME) as fh:
        data = json.load(fh)

    parameters = data["parameters"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    dates = pandas.date_range("2015-01-01", "2015-12-31")

    L1_values = np.array([parameters["level1"]["values"][d.month - 1] for d in dates])
    L2_values = np.array([parameters["level2"]["values"][d.month - 1] for d in dates])

    x = np.arange(0, len(dates)) + 1

    ax.fill_between(x, 1.0, L1_values, label="Level 0", alpha=0.8)
    ax.fill_between(x, L1_values, L2_values, label="Level 1", alpha=0.8)
    ax.fill_between(x, L2_values, 0.0, label="Level 2", alpha=0.8)

    plt.xlabel("Day of year")
    plt.ylabel("Reservoir volume [%]")

    plt.grid(True)
    plt.ylim([0.0, 1.0])
    plt.xlim(1, 365)
    plt.legend(["Level 0", "Level 1", "Level 2"], loc="upper right")
    ax.plot(x, L1_values, color="k", label=None)
    ax.plot(x, L2_values, color="k", label=None)
    plt.tight_layout()

    if ext is not None:
        fig.savefig(f"Control curve zones.{ext}", dpi=300)

    if show:
        plt.show()


@cli.command("plot-dsf")
@click.option("--ext", default="png")
@click.option("--show/--no-show", default=False)
def plot_demand_saving_factor(ext, show):

    with open(MODEL_FILENAME) as fh:
        data = json.load(fh)

    parameters = data["parameters"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    dates = pandas.date_range("2015-01-01", "2015-12-31")

    L0_values = np.array([parameters["level0_factor"]["values"] for d in dates])
    L1_values = np.array(
        [parameters["level1_factor"]["values"][d.month - 1] for d in dates]
    )
    L2_values = np.array(
        [parameters["level2_factor"]["values"][d.month - 1] for d in dates]
    )

    x = np.arange(0, len(dates)) + 1

    ax.plot(x, L0_values, label="Level 0")
    ax.plot(x, L1_values, label="Level 0")
    ax.plot(x, L2_values, label="Level 0")

    plt.xlabel("Day of year")
    plt.ylabel("Demand restriction factor")

    plt.grid(True)
    plt.ylim(0.6, 1.2)
    plt.xlim(1, 365)
    plt.legend(["Level 0", "Level 1", "Level 2"], loc="upper right")

    plt.tight_layout()

    if ext is not None:
        fig.savefig(f"Demand restriction factors.{ext}", dpi=300)

    if show:
        plt.show()


if __name__ == "__main__":
    cli()
