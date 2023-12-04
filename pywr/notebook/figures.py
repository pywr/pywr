import numpy as np
import scipy.stats
import pandas
import matplotlib
import matplotlib.pyplot as plt

c = {
    "Afill": "#ee1111",
    "Aedge": "#660000",
    "Bfill": "#1111ee",
    "Bedge": "#000066",
    "Cfill": "#11bb11",
    "Cedge": "#008800",
}


def align_series(A, B, names=None, start=None, end=None):
    """Align two series for plotting / comparison

    Parameters
    ----------
    A : `pandas.Series`
    B : `pandas.Series`
    names : list of strings
    start : `pandas.Timestamp` or timestamp string
    end : `pandas.Timestamp` or timestamp string

    Example
    -------
    >>> A, B = align_series(A, B, ["Pywr", "Aquator"], start="1920-01-01", end="1929-12-31")
    >>> plot_standard1(A, B)

    """
    # join series B to series A
    # TODO: better handling of heterogeneous frequencies
    df = pandas.concat([A, B], join="inner", axis=1)

    # apply names
    if names is not None:
        df.columns = names
    else:
        names = list(df.columns)

    # clip start and end to user-specified dates
    idx = [df.index[0], df.index[-1]]
    if start is not None:
        idx[0] = pandas.Timestamp(start)
    if end is not None:
        idx[1] = pandas.Timestamp(end)

    if start or end:
        df = df.loc[idx[0] : idx[-1], :]

    A = df[names[0]]
    B = df[names[1]]
    return A, B


def plot_standard1(A, B):
    fig, axarr = plt.subplots(3, figsize=(10, 12), facecolor="white")
    plot_timeseries(A, B, axarr[0])
    plot_QQ(A, B, axarr[1])
    plot_percentiles(A, B, axarr[2])
    return fig, axarr


def set_000formatter(axis):
    axis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )


def plot_timeseries(A, B, ax=None):
    if ax is None:
        ax = plt.gca()
    B.plot(ax=ax, color=c["Bfill"], clip_on=False)
    A.plot(ax=ax, color=c["Afill"], clip_on=False)
    ax.grid(True)
    ax.set_ylim(0, None)
    set_000formatter(ax.get_yaxis())
    ax.set_xlabel("")
    ax.legend([B.name, A.name], loc="best")
    return ax


def plot_QQ(A, B, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(
        B.values, A.values, color=c["Cfill"], edgecolor=c["Cedge"], clip_on=False
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    limit = max(xlim[1], ylim[0])
    ax.plot([0, limit], [0, limit], "-k")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.grid(True)
    set_000formatter(ax.get_xaxis())
    set_000formatter(ax.get_yaxis())
    ax.set_xlabel(B.name)
    ax.set_ylabel(A.name)
    ax.legend(["Equality"], loc="best")
    return ax


def plot_percentiles(A, B, ax=None):
    if ax is None:
        ax = plt.gca()
    percentiles = np.linspace(0.001, 0.999, 1000) * 100
    A_pct = scipy.stats.scoreatpercentile(A.values, percentiles)
    B_pct = scipy.stats.scoreatpercentile(B.values, percentiles)
    percentiles = percentiles / 100.0
    ax.plot(percentiles, B_pct[::-1], color=c["Bfill"], clip_on=False, linewidth=2)
    ax.plot(percentiles, A_pct[::-1], color=c["Afill"], clip_on=False, linewidth=2)
    ax.set_xlabel("Cumulative frequency")
    ax.grid(True)
    ax.xaxis.grid(True, which="both")
    set_000formatter(ax.get_yaxis())
    ax.set_xscale("logit")
    xticks = ax.get_xticks()
    xticks_minr = ax.get_xticks(minor=True)
    ax.set_xticklabels([], minor=True)
    ax.set_xticks([0.01, 0.1, 0.5, 0.9, 0.99])
    ax.set_xticklabels(["1", "10", "50", "90", "99"])
    ax.set_xlim(0.001, 0.999)
    ax.legend([B.name, A.name], loc="best")
    return ax
