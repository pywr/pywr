Parameters
==========

.. currentmodule:: pywr.parameters


Base Parameter class
--------------------

All the `Parameter` subclasses in `pywr` are descended from a common base class.

.. autosummary::
   :toctree: generated/

   Parameter
   IndexParameter


Simple parameters
-----------------

.. autosummary::
   :toctree: generated/

   ConstantParameter
   ConstantScenarioParameter
   ConstantScenarioIndexParameter


Combining multiple parameters
-----------------------------

.. autosummary::
   :toctree: generated/

   AggregatedParameter
   AggregatedIndexParameter
   DivisionParameter
   NegativeParameter
   MaxParameter
   NegativeMaxParameter
   MinParameter
   NegativeMinParameter
   OffsetParameter

Annual profile parameters
-------------------------

.. autosummary::
   :toctree: generated/

   DailyProfileParameter
   WeeklyProfileParameter
   MonthlyProfileParameter
   UniformDrawdownProfileParameter
   ScenarioDailyProfileParameter
   ScenarioWeeklyProfileParameter
   ScenarioMonthlyProfileParameter
   ArrayIndexedScenarioMonthlyFactorsParameter
   RbfProfileParameter

Dataframe parameter
-------------------

.. autosummary::
   :toctree: generated/

   DataFrameParameter


HDF5 Parameter
--------------
.. autosummary::
   :toctree: generated/

   TablesArrayParameter


Array based parameters
----------------------

.. autosummary::
   :toctree: generated/

   ArrayIndexedParameter
   ArrayIndexedScenarioParameter
   IndexedArrayParameter


Threshold parameters
--------------------

.. autosummary::
   :toctree: generated/

    AbstractThresholdParameter
    StorageThresholdParameter
    NodeThresholdParameter
    ParameterThresholdParameter
    MultipleThresholdIndexParameter
    MultipleThresholdParameterIndexParameter
    RecorderThresholdParameter
    CurrentYearThresholdParameter
    CurrentOrdinalDayThresholdParameter


Activation function parameters
------------------------------

.. autosummary::
   :toctree: generated/

    BinaryStepParameter
    RectifierParameter
    LogisticParameter


Interpolation parameters
------------------------

.. autosummary::
   :toctree: generated/

    InterpolatedParameter
    InterpolatedVolumeParameter
    InterpolatedQuadratureParameter
    InterpolatedFlowParameter


Control curve parameters
------------------------

.. autosummary::
    :toctree: generated/

    control_curves.BaseControlCurveParameter
    control_curves.ControlCurveParameter
    control_curves.ControlCurveInterpolatedParameter
    control_curves.ControlCurveIndexParameter
    control_curves.ControlCurvePiecewiseInterpolatedParameter


Hydropower parameters
----------------------

.. autosummary::
   :toctree: generated/

   HydropowerTargetParameter


Other parameters
----------------

.. autosummary::
   :toctree: generated/

   AnnualHarmonicSeriesParameter
   DeficitParameter
   ScenarioWrapperParameter
   PiecewiseIntegralParameter
   FlowParameter
   FlowDelayParameter
   DiscountFactorParameter


