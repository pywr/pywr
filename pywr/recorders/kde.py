from typing import Tuple
import numpy as np
import pandas
from scipy import stats
from ._recorders import NumpyArrayStorageRecorder, NumpyArrayNormalisedStorageRecorder


class GaussianKDEStorageRecorder(NumpyArrayStorageRecorder):
    """This recorder fits a [Kernel Density Estimation (KDE)](https://en.wikipedia.org/wiki/Kernel_density_estimation)
    to a time-series of storage node's volume.

    This recorder inherits from [pywr.recorders.NumpyArrayStorageRecorder][] and, at the end of a simulation,
    uses the KDE to estimate the probability density function of the storage time-series.
    It returns the probability of being at or below a specified target volume in the `aggregated_value()` method
    (i.e. used for optimisation). The recorder flattens the data from all scenarios before computing the KDE. Therefore,
    a single PDF is produced and returned via `.to_dataframe()`.

    The user can specify an optional resampling (e.g. to create annual minimum time-series) before fitting
    the KDE using the Pandas library. By default, the KDE is reflected at the proportional storage bounds (0.0 and 1.0) to represent the
    boundedness of the distribution (i.e. the distribution may leak outside the bounded domain when
    not reflected). This can be disabled if required.

    Examples
    -------
    Python
    ======
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.recorders import GaussianKDEStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        max_volume=500,
        cost=-20.0,
        initial_volume_pc=0.8
    )
    GaussianKDEStorageRecorder(
        model=model,
        name="KDE storage",
        node=storage,
        resample_freq="Y",
        target_volume_pc=0.3
    )
    ```

    JSON
    ======
    ```json
    {
        "KDE storage": {
            "type": "GaussianKDEStorageRecorder",
            "node": "Reservoir",
            "resample_freq": "Y",
            "target_volume_pc": 0.3
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Storage
        Storage instance to record.
    resample_freq : Optional[str]
        The resampling frequency used by prior to distribution fitting.
    resample_func : Optional[str]
        The resampling function used prior to distribution fitting.
    target_volume_pc : float
        The proportional target volume for which a probability of being at or lower is estimated.
    num_pdf : int
        Number of points in the PDF estimate
    use_reflection : bool
        Whether to reflect the PDF at the upper and lower bounds (i.e. 0% and 100% volume) to account for
        the boundedness of the distribution.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def __init__(self, *args, **kwargs):
        """Initialize recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Storage
            Storage instance to record.
        resample_freq : Optional[str]
            The resampling frequency used by prior to distribution fitting. See [this page](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
            for a list of available resampling frequencies.
        resample_func : Optional[str]
            The resampling function used prior to distribution fitting. See
            [this page](https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats)
            for a list of the function names you can use.
        target_volume_pc : float
            The proportional target volume for which a probability of being at or lower is estimated.
        num_pdf : Optional[int], default=101
            Number of points in the PDF estimate
        use_reflection : Optional[bool], default=True
            Whether to reflect the PDF at the upper and lower bounds (i.e. 0% and 100% volume) to
            correct the density function near its boundaries.
        name : Optional[str]
            Name of the recorder.
        comment : Optional[str]
            Comment or description of the recorder.
        ignore_nan : Optional[bool]
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float]
            Epsilon distance used by some optimisation algorithms.
        constraint_lower_bounds : Optional[float | Iterable[float]]
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        """
        self.resample_freq = kwargs.pop("resample_freq", None)
        self.resample_func = kwargs.pop("resample_func", None)
        self.target_volume_pc = kwargs.pop("target_volume_pc")
        self.use_reflection = kwargs.pop("use_reflection", True)
        self.num_pdf = kwargs.pop("num_pdf", 101)

        super().__init__(*args, proportional=True, **kwargs)
        self._probability_of_target_volume = None
        self._pdf = None

    def reset(self):
        """Reset the internal state of the recorder."""
        super().reset()
        self._probability_of_target_volume = None
        self._pdf = None

    def finish(self):
        """Calculate the PDF."""
        super().finish()

        df = super().to_dataframe()
        # Apply resampling if defined
        if self.resample_func is not None and self.resample_freq is not None:
            df = df.resample(self.resample_freq).agg(self.resample_func)

        x = np.linspace(0.0, 1.0, self.num_pdf)

        # Compute the probability for the target volume
        p, pdf = self._estimate_pdf_and_target_probability(df.values.flatten(), x)

        self._probability_of_target_volume = p
        self._pdf = pandas.DataFrame(data=pdf, index=x)

    def values(self):
        """Return the estimated PDF values."""
        return self._pdf.values

    def to_dataframe(self):
        """Return a `pandas.DataFrame` of the estimated PDF.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains the probability between 0 and 1 and the relative storage
            of being at or lower.
        """
        return self._pdf

    def aggregated_value(self):
        """Get the probability of being at or below the target volume."""
        return self._probability_of_target_volume

    def _estimate_pdf_and_target_probability(
        self, values, x
    ) -> Tuple[float, np.ndarray]:
        """Return a probability of being at below `target_volume_pc` and an estimate of the PDF. This method can
        (if `use_reflection` is `True`) reflect the PDF at the lower and upper boundaries to stop the
        PDF leaking in to infeasible space.

        Returns
        -------
        tuple[float, numpy.typing.NDArray[numpy.number]]
            The probability of being at or below `target_volume_pc` and the PDF estimate.
        """
        # Fit a Gaussian KDE
        kernel = stats.gaussian_kde(values)
        p = kernel.integrate_box_1d(0.0, self.target_volume_pc)
        pdf = kernel(x)

        if self.use_reflection:
            # Reflection at the lower boundary
            kernel_lb = stats.gaussian_kde(-values)
            p += kernel_lb.integrate_box_1d(0.0, self.target_volume_pc)
            pdf += kernel_lb(x)

            # Reflection at the upper boundary
            kernel_ub = stats.gaussian_kde(2.0 - values)
            p += kernel_ub.integrate_box_1d(0.0, self.target_volume_pc)
            pdf += kernel_ub(x)

        return p, pdf


GaussianKDEStorageRecorder.register()


class NormalisedGaussianKDEStorageRecorder(NumpyArrayNormalisedStorageRecorder):
    """This recorder fits a [Kernel Density Estimation (KDE)](https://en.wikipedia.org/wiki/Kernel_density_estimation)
    to a time-series of storage node's volume.

    This recorder inherits from [pywr.recorders.NumpyArrayNormalisedStorageRecorder][] and, at the end of a simulation,
    uses the KDE to estimate the probability density function of the storage time-series.
    It returns the probability of being at or below zero of the normalised values in the `aggregated_value()` method
    (i.e. used for optimisation). The normalisation is relative to a `Parameter` which defines a control curve.
    The recorder flattens the data from all scenarios before computing the KDE. Therefore,
    a single PDF is produced and returned via `.to_dataframe()`.

    The user can specify an optional resampling (e.g. to create annual minimum time-series) before fitting
    the KDE. By default, the KDE is reflected at the normalised storage bounds (-1.0 and 1.0) to represent the
    boundedness of the distribution (i.e. the distribution may leak outside the bounded domain when
    not reflected). This can be disabled if required.

    Examples
    -------
    Python
    ======
    ```python
    import numpy as np
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import ConstantParameter
    from pywr.recorders import NormalisedGaussianKDEStorageRecorder

    model = Model()
    storage = Storage(
        model,
        name="Reservoir",
        max_volume=500,
        cost=-20.0,
        initial_volume_pc=0.8
    )
    NormalisedGaussianKDEStorageRecorder(
        model=model,
        name="KDE storage",
        node=storage,
        resample_freq="Y",
        parameter=ConstantParameter(model, 0.8),
        target_volume_pc=0.3
    )
    ```

    JSON
    ======
    ```json
    {
        "KDE storage": {
            "type": "NormalisedGaussianKDEStorageRecorder",
            "node": "Reservoir",
            "resample_freq": "Y",
            "target_volume_pc": 0.3,
            "parameter": 0.8
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    node : Storage
        Storage instance to record.
    parameter : Parameter
        The control curve parameter to use to normalise the storage between -1.0 and 1.0.
    resample_freq : Optional[str]
        The resampling frequency used by prior to distribution fitting.
    resample_func : Optional[str]
        The resampling function used prior to distribution fitting.
    target_volume_pc : float
        The proportional target volume for which a probability of being at or lower is estimated.
    num_pdf : int
        Number of points in the PDF estimate
    use_reflection : bool
        Whether to reflect the PDF at the upper and lower bounds (i.e. 0% and 100% volume) to account for
        the boundedness of the distribution.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : Optional[bool]
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def __init__(self, *args, **kwargs):
        """Initialize recorder.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Storage
            Storage instance to record.
        parameter : Parameter
            The control curve parameter to use to normalise the storage between -1.0 and 1.0.
        resample_freq : Optional[str]
            The resampling frequency used by prior to distribution fitting.
        resample_func : Optional[str]
            The resampling function used prior to distribution fitting.
        target_volume_pc : float
            The proportional target volume for which a probability of being at or lower is estimated.
        num_pdf : Optional[int], default=101
            Number of points in the PDF estimate
        use_reflection : Optional[bool], default=True
            Whether to reflect the PDF at the upper and lower normalised bounds (i.e. -1.0 and 1.0 volume) to
            correct the density function near its boundaries.
        name : Optional[str]
            Name of the recorder.
        comment : Optional[str]
            Comment or description of the recorder.
        ignore_nan : Optional[bool]
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float]
            Epsilon distance used by some optimisation algorithms.
        constraint_lower_bounds : Optional[float | Iterable[float]]
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        """
        self.resample_freq = kwargs.pop("resample_freq", None)
        self.resample_func = kwargs.pop("resample_func", None)
        self.use_reflection = kwargs.pop("use_reflection", True)
        self.num_pdf = kwargs.pop("num_pdf", 101)

        super().__init__(*args, **kwargs)
        self._probability_of_target_volume = None
        self._pdf = None

    def reset(self):
        """Reset the internal state of the recorder."""
        super().reset()
        self._probability_of_target_volume = None
        self._pdf = None

    def finish(self):
        """Calculate the PDF."""
        super().finish()

        df = super().to_dataframe()
        # Apply resampling if defined
        if self.resample_func is not None and self.resample_freq is not None:
            df = df.resample(self.resample_freq).agg(self.resample_func)

        x = np.linspace(0.0, 1.0, self.num_pdf)

        # Compute the probability for the target volume
        p, pdf = self._estimate_pdf_and_target_probability(df.values.flatten(), x)

        self._probability_of_target_volume = p
        self._pdf = pandas.DataFrame(data=pdf, index=x)

    def values(self):
        """Return the estimated PDF values."""
        return self._pdf.values

    def to_dataframe(self):
        """Return a `pandas.DataFrame` of the estimated PDF.

        Returns
        -------
        pandas.DataFrame
            This DataFrame contains the probability between 0 and 1 and the normalised storage
            of being at or lower.
        """
        return self._pdf

    def aggregated_value(self):
        """Get the probability of being at or below the target volume."""
        return self._probability_of_target_volume

    def _estimate_pdf_and_target_probability(
        self, values, x
    ) -> Tuple[float, np.ndarray]:
        """Return a probability of being at below `target_volume_pc` and an estimate of the PDF. This method can
        (if `use_reflection` is `True`) reflect the PDF at the lower and upper boundaries to stop the
        PDF leaking in to infeasible space.

        Returns
        -------
        tuple[float, numpy.typing.NDArray[numpy.number]]
            The probability of being at or below `target_volume_pc` and the PDF estimate.
        """
        # Fit a Gaussian KDE
        kernel = stats.gaussian_kde(values)
        p = kernel.integrate_box_1d(-1.0, 0.0)
        pdf = kernel(x)

        if self.use_reflection:
            # Reflection at the lower boundary
            kernel_lb = stats.gaussian_kde(-2.0 - values)
            p += kernel_lb.integrate_box_1d(-1.0, 0.0)
            pdf += kernel_lb(x)

            # Reflection at the upper boundary
            kernel_ub = stats.gaussian_kde(2.0 - values)
            p += kernel_ub.integrate_box_1d(-1.0, 0.0)
            pdf += kernel_ub(x)

        return p, pdf


NormalisedGaussianKDEStorageRecorder.register()
