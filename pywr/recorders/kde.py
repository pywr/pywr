from typing import Tuple
import numpy as np
import pandas
from scipy import stats
from ._recorders import NumpyArrayStorageRecorder, NumpyArrayNormalisedStorageRecorder


class GaussianKDEStorageRecorder(NumpyArrayStorageRecorder):
    """A recorder that fits a KDE to a time-series of volume.

    This recorder is an extension to `NumpyArrayStorageRecorder` which, at the end of a simulation,
    uses kernel density estimation (KDE) to estimate the probability density function of the storage time-series.
    It returns the probability of being at or below a specified target volume in the `aggregated_value()` method
    (i.e. used for optimisation). The recorder flattens data from all scenarios before computing the KDE. Therefore,
    a single PDF is produced and returned via `.to_dataframe()`.

    The user can specify an optional resampling (e.g. to create annual minimum time-series) prior to fitting
    the KDE. By default the KDE is reflected at the proportional storage bounds (0.0 and 1.0) to represent the
    boundedness of the distribution. This can be disabled if required.

    Parameters
    ==========
    resample_freq : str or None
        If not None the resampling frequency used by prior to distribution fitting.
    resample_func : str or None
        If not None the resampling function used prior to distribution fitting.
    target_volume_pc : float
        The proportional target volume for which a probability of being at or lower is estimated.
    num_pdf : int
        Number of points in the PDF estimate. Defaults to 101.
    use_reflection : bool
        Whether to reflect the PDF at the upper and lower bounds (i.e. 0% and 100% volume) to account for
        the boundedness of the distribution. Defaults to true.
    """

    def __init__(self, *args, **kwargs):
        self.resample_freq = kwargs.pop("resample_freq", None)
        self.resample_func = kwargs.pop("resample_func", None)
        self.target_volume_pc = kwargs.pop("target_volume_pc")
        self.use_reflection = kwargs.pop("use_reflection", True)
        self.num_pdf = kwargs.pop("num_pdf", 101)

        super().__init__(*args, proportional=True, **kwargs)
        self._probability_of_target_volume = None
        self._pdf = None

    def reset(self):
        super().reset()
        self._probability_of_target_volume = None
        self._pdf = None

    def finish(self):
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
        """Return a `pandas.DataFrame` of the estimated PDF."""
        return self._pdf

    def aggregated_value(self):
        return self._probability_of_target_volume

    def _estimate_pdf_and_target_probability(
        self, values, x
    ) -> Tuple[float, np.ndarray]:
        """Return a probability of being at below `self.target_volume_pc` and a estimate of the PDF

        This method can (if `self.use_reflection` is truthy) reflect the PDF at the lower and upper boundaries
        to stop the PDF leaking in to infeasible space.
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
    """A recorder that fits a KDE to a normalised time-series of volume.

    This recorder is an extension to `NumpyArrayNormalisedStorageRecorder` which, at the end of a simulation,
    uses kernel density estimation (KDE) to estimate the probability density function of the storage time-series.
    It returns the probability of being at or below zero of the normalised values in the `aggregated_value()` method
    (i.e. used for optimisation). The recorder flattens data from all scenarios before computing the KDE. Therefore,
    a single PDF is produced and returned via `.to_dataframe()`.

    The user can specify an optional resampling (e.g. to create annual minimum time-series) prior to fitting
    the KDE. By default the KDE is reflected at the normalised storage bounds (-1.0 and 1.0) to represent the
    boundedness of the distribution. This can be disabled if required.

    Parameters
    ==========
    resample_freq : str or None
        If not None the resampling frequency used by prior to distribution fitting.
    resample_func : str or None
        If not None the resampling function used prior to distribution fitting.
    num_pdf : int
        Number of points in the PDF estimate. Defaults to 101.
    use_reflection : bool
        Whether to reflect the PDF at the upper and lower normalised bounds (i.e. -1.0 and 1.0 volume) to account for
        the boundedness of the distribution. Defaults to true.
    """

    def __init__(self, *args, **kwargs):
        self.resample_freq = kwargs.pop("resample_freq", None)
        self.resample_func = kwargs.pop("resample_func", None)
        self.use_reflection = kwargs.pop("use_reflection", True)
        self.num_pdf = kwargs.pop("num_pdf", 101)

        super().__init__(*args, **kwargs)
        self._probability_of_target_volume = None
        self._pdf = None

    def reset(self):
        super().reset()
        self._probability_of_target_volume = None
        self._pdf = None

    def finish(self):
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
        """Return a `pandas.DataFrame` of the estimated PDF."""
        return self._pdf

    def aggregated_value(self):
        return self._probability_of_target_volume

    def _estimate_pdf_and_target_probability(
        self, values, x
    ) -> Tuple[float, np.ndarray]:
        """Return a probability of being at below `self.target_volume_pc` and a estimate of the PDF

        This method can (if `self.use_reflection` is truthy) reflect the PDF at the lower and upper boundaries
        to stop the PDF leaking in to infeasible space.
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
