Recorders
==========

.. currentmodule:: pywr.recorders


Base Recorder classes
---------------------

All the `Recorder` subclasses in `pywr` are descended from a common base class.

.. autosummary::
   :toctree: generated/

   Recorder
   NodeRecorder
   StorageRecorder
   ParameterRecorder
   IndexParameterRecorder

Numpy array recorders
---------------------

The following parameters are used for storing results in memory. The data can be
accessed following a model run before the model instances is destroyed.

.. autosummary::
   :toctree: generated/

   NumpyArrayNodeRecorder
   NumpyArrayStorageRecorder
   NumpyArrayLevelRecorder
   NumpyArrayAreaRecorder
   NumpyArrayParameterRecorder
   NumpyArrayIndexParameterRecorder

Flow duration curve recorders
-----------------------------

.. autosummary::
   :toctree: generated/

   FlowDurationCurveRecorder
   StorageDurationCurveRecorder
   FlowDurationCurveDeviationRecorder
   SeasonalFlowDurationCurveRecorder

Deficit recorders
-----------------

.. autosummary::
   :toctree: generated/

   TotalDeficitNodeRecorder
   DeficitFrequencyNodeRecorder

Statistical recorders
---------------------

.. autosummary::
   :toctree: generated/

   AggregatedRecorder
   MeanFlowNodeRecorder
   TotalFlowNodeRecorder
   MeanParameterRecorder
   TotalParameterRecorder
   RollingMeanFlowNodeRecorder
   MinimumVolumeStorageRecorder
   MinimumThresholdVolumeStorageRecorder
   RollingWindowParameterRecorder


Index recorders
---------------

.. autosummary::
   :toctree: generated/

   AnnualCountIndexParameterRecorder

File recorders
--------------

.. autosummary::
   :toctree: generated/

   CSVRecorder
   TablesRecorder


Hydro-power recorders
---------------------

.. autosummary::
   :toctree: generated/

   HydropowerRecorder
   TotalHydroEnergyRecorder
