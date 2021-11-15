"""
This module contains `Recorder` subclasses useful in the evaluation of model
 performance and calibration.

Several different evaluation metrics are implemented in this module. Many are
 well known to modellers. However an important reference for this implementation
 is `[1]_` which reviews the available metrics for watershed models.

..  [1] Moriasi, D.N., et al. (2007) Model Evaluation Guidelines for Systematic
    Quantification of Accuracy in Watershed Simulations. Transactions of the ASABE, 50, 885-900.
    http://dx.doi.org/10.13031/2013.23153


"""
from ._recorders import NumpyArrayNodeRecorder
import numpy as np


class AbstractComparisonNodeRecorder(NumpyArrayNodeRecorder):
    """Base class for all Recorders performing timeseries comparison of `Node` flows"""

    def __init__(self, model, node, observed, **kwargs):
        super(AbstractComparisonNodeRecorder, self).__init__(model, node, **kwargs)
        self.observed = observed
        self._aligned_observed = None

    def setup(self):
        super(AbstractComparisonNodeRecorder, self).setup()
        # Align the observed data to the model
        from pywr.parameters import align_and_resample_dataframe

        self._aligned_observed = align_and_resample_dataframe(
            self.observed, self.model.timestepper.datetime_index
        )


class RootMeanSquaredErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the RMSE between model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        return np.sqrt(np.mean((obs - mod) ** 2, axis=0))


class MeanAbsoluteErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the MAE between model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        return np.mean(np.abs(obs - mod), axis=0)


class MeanSquareErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the MSE between model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        return np.mean((obs - mod) ** 2, axis=0)


class PercentBiasNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the percent bias between model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        return np.sum(obs - mod, axis=0) * 100 / np.sum(obs, axis=0)


class RMSEStandardDeviationRatioNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the RMSE-observations standard deviation ratio between model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        return np.sqrt(np.mean((obs - mod) ** 2, axis=0)) / np.std(obs, axis=0)


class NashSutcliffeEfficiencyNodeRecorder(AbstractComparisonNodeRecorder):
    """Recorder evaluates the Nash-Sutcliffe efficiency model and observed"""

    def values(self):
        mod = self.data
        obs = self._aligned_observed
        obs_mean = np.mean(obs, axis=0)
        return 1.0 - np.sum((obs - mod) ** 2, axis=0) / np.sum(
            (obs - obs_mean) ** 2, axis=0
        )
