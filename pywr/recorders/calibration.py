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

import pandas

from ._recorders import NumpyArrayNodeRecorder
import numpy as np
from typing_extensions import TYPE_CHECKING


if TYPE_CHECKING:
    from ..core import Model, Node


class AbstractComparisonNodeRecorder(NumpyArrayNodeRecorder):
    """Base class for all recorders performing timeseries comparison of `Node` flows."""

    def __init__(
        self, model: "Model", node: "Node", observed: pandas.DataFrame, **kwargs
    ):
        """
        Initialize the recorder.

        Parameters
        ----------
        model : Model
            The model instance
        node: Node
            The node instance to compare against.
        observed : pandas.DataFrame
            A pandas DataFrame containing the observed values for `node`. This
            will be aligned and resampled to the provided timestepper index.

        Other parameters
        ----------------
        factor : Optional[int], default=1
            A factor can be provided to scale the total flow (e.g. for calculating operational costs).
        agg_func : Optional[str | Callable], default="mean"
            Scenario aggregation function to use when `aggregated_value()` is called.
        ignore_nan : Optional[bool], default=False
            Flag to ignore NaN values when calling `aggregated_value()`.
        is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']], default=None
            Flag to denote the direction, if any, of optimisation undertaken with this recorder.
        epsilon : Optional[float], default=1.0
            Epsilon distance used by some optimisation algorithms.
        name : Optional[str], default=None
            Name of the recorder.
        constraint_lower_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        constraint_upper_bounds : Optional[float | Iterable[float]], default=None
            The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
             constraint during an optimisation problem.
        comment : Optional[str], default=None
            Comment or description of the recorder.
        """
        super(AbstractComparisonNodeRecorder, self).__init__(model, node, **kwargs)
        self.observed = observed
        self._aligned_observed = None

    def setup(self):
        """Align the DataFrame to the timestepper date index."""
        super(AbstractComparisonNodeRecorder, self).setup()
        # Align the observed data to the model
        from pywr.parameters import align_and_resample_dataframe

        self._aligned_observed = align_and_resample_dataframe(
            self.observed, self.model.timestepper.datetime_index
        )


class RootMeanSquaredErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """This recorder evaluates the RMSE between the node's flow and the observed timeseries using:

    $$ RMSE = \sqrt{ \sum_i{(F_i - O_i)^2} / n } $$

    where:

    - F<sub>i</sub> is the simulated flow for the i<sup>th</sup> timestep.
    - O<sub>i</sub> is the observed flow for the i<sup>th</sup> timestep.
    - `n` is the number of timesteps.

    !!!warning "Python only"
        You cannot load this parameter from a JSON document.

    >  **Reference**: Moriasi, D.N., et al. (2007) Model Evaluation Guidelines for Systematic
      Quantification of Accuracy in Watershed Simulations. Transactions of the ASABE, 50, 885-900.
      http://dx.doi.org/10.13031/2013.23153

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : bool
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def values(self):
        """
        Get the root mean squared error from the data.

        Returns
        -------
        float
            The metric.
        """
        mod = self.data
        obs = self._aligned_observed
        return np.sqrt(np.mean((obs - mod) ** 2, axis=0))


class MeanAbsoluteErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """This recorder evaluates the MAE between model and observed timeseries using:

    $$ MAE = {\sum_i{|F_i - O_i|}} / n $$

    where:

    - F<sub>i</sub> is the simulated flow for the i<sup>th</sup> timestep.
    - O<sub>i</sub> is the observed flow for the i<sup>th</sup> timestep.
    - `n` is the number of timesteps.

    !!!warning "Python only"
        You cannot load this parameter from a JSON document.

    >  **Reference**: Moriasi, D.N., et al. (2007) Model Evaluation Guidelines for Systematic
      Quantification of Accuracy in Watershed Simulations. Transactions of the ASABE, 50, 885-900.
      http://dx.doi.org/10.13031/2013.23153

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : bool
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def values(self):
        """
        Get the mean absolute error from the data.

        Returns
        -------
        float
            The metric.
        """
        mod = self.data
        obs = self._aligned_observed
        return np.mean(np.abs(obs - mod), axis=0)


class MeanSquareErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """This recorder evaluates the MSE between modelled flow and observed timeseries using:

    $$ MSE = {\sum_i{(F_i - O_i)^2}} / n $$

    where:

    - F<sub>i</sub> is the simulated flow for the i<sup>th</sup> timestep.
    - O<sub>i</sub> is the observed flow for the i<sup>th</sup> timestep.
    - `n` is the number of timesteps.

    !!!warning "Python only"
        You cannot load this parameter from a JSON document.

    >  **Reference**: Moriasi, D.N., et al. (2007) Model Evaluation Guidelines for Systematic
      Quantification of Accuracy in Watershed Simulations. Transactions of the ASABE, 50, 885-900.
      http://dx.doi.org/10.13031/2013.23153

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : bool
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def values(self):
        """
        Get the mean squared error from the data.

        Returns
        -------
        float
            The metric.
        """
        mod = self.data
        obs = self._aligned_observed
        return np.mean((obs - mod) ** 2, axis=0)


class PercentBiasNodeRecorder(AbstractComparisonNodeRecorder):
    """This recorder evaluates the percent bias between modelled flow and observed timeseries using:

    $$ B = {\sum_i{(F_i - O_i)}} \div {\sum_i{O_i}} * 100 $$

    where:

    - F<sub>i</sub> is the simulated flow for the i<sup>th</sup> timestep.
    - O<sub>i</sub> is the observed flow for the i<sup>th</sup> timestep.

    !!!warning "Python only"
        You cannot load this parameter from a JSON document.

    >  **Reference**: Moriasi, D.N., et al. (2007) Model Evaluation Guidelines for Systematic
      Quantification of Accuracy in Watershed Simulations. Transactions of the ASABE, 50, 885-900.
      http://dx.doi.org/10.13031/2013.23153

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : bool
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def values(self):
        """
        Get the percent bias from the data.

        Returns
        -------
        float
            The metric.
        """
        mod = self.data
        obs = self._aligned_observed
        return np.sum(obs - mod, axis=0) * 100 / np.sum(obs, axis=0)


class RMSEStandardDeviationRatioNodeRecorder(AbstractComparisonNodeRecorder):
    """This recorder evaluates the RMSE-observations standard deviation ratio between the
    modelled flow and observed timeseries using:

    $$ RMSE_{std} = \sqrt{ \sum_i{(F_i - O_i)^2} / n } \div \sqrt{\sum_i{|O_i - O_m|^2 } / n} $$

    where:

    - F<sub>i</sub> is the simulated flow for the i<sup>th</sup> timestep.
    - O<sub>i</sub> is the observed flow for the i<sup>th</sup> timestep.
    - O<sub>m</sub> is the mean of the observed flow.
    - `n` is the number of timesteps.

    !!!warning "Python only"
        You cannot load this parameter from a JSON document.

    >  **Reference**: Moriasi, D.N., et al. (2007) Model Evaluation Guidelines for Systematic
      Quantification of Accuracy in Watershed Simulations. Transactions of the ASABE, 50, 885-900.
      http://dx.doi.org/10.13031/2013.23153

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : bool
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def values(self):
        """
        Get the RMSE standard deviation ratio from the data.

        Returns
        -------
        float
            The metric.
        """
        mod = self.data
        obs = self._aligned_observed
        return np.sqrt(np.mean((obs - mod) ** 2, axis=0)) / np.std(obs, axis=0)


class NashSutcliffeEfficiencyNodeRecorder(AbstractComparisonNodeRecorder):
    """This recorder evaluates the [Nash-Sutcliffe efficiency](https://en.wikipedia.org/wiki/Nash–Sutcliffe_model_efficiency_coefficient)
    between the modelled flow data and the observed timeseries.

    !!!warning "Python only"
        You cannot load this parameter from a JSON document.

    >  **Reference**: Moriasi, D.N., et al. (2007) Model Evaluation Guidelines for Systematic
      Quantification of Accuracy in Watershed Simulations. Transactions of the ASABE, 50, 885-900.
      http://dx.doi.org/10.13031/2013.23153

    Attributes
    ----------
    model : Model
        The model instance.
    node : Node
        Node instance to record.
    factor: float
        The factor used to scale the total flow.
    agg_func : str | Callable
        Scenario aggregation function to use when `aggregated_value()` is called.
    name : Optional[str]
        Name of the recorder.
    comment : Optional[str]
        Comment or description of the recorder.
    ignore_nan : bool
        Flag to ignore NaN values when calling `aggregated_value()`.
    is_objective : Optional[Literal[ 'maximize', 'maximise', 'max', 'minimize', 'minimise', 'min']]
        Flag to denote the direction, if any, of optimisation undertaken with this recorder.
    epsilon : Optional[float]
        Epsilon distance used by some optimisation algorithms.
    constraint_lower_bounds : Optional[float | Iterable[float]]
        The value(s) to use for lower bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    constraint_upper_bounds : Optional[float | Iterable[float]]
        The value(s) to use for upper bound for the recorder value. When given, the recorder instance is marked as a
         constraint during an optimisation problem.
    """

    def values(self):
        """
        Get the Nash-Sutcliffe efficiency from the data.

        Returns
        -------
        float
            The metric.
        """
        mod = self.data
        obs = self._aligned_observed
        obs_mean = np.mean(obs, axis=0)
        return 1.0 - np.sum((obs - mod) ** 2, axis=0) / np.sum(
            (obs - obs_mean) ** 2, axis=0
        )
