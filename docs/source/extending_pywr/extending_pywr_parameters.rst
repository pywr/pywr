.. _extending-pywr-parameters:

Extending Pywr with custom Parameters
-------------------------------------

Pywr provides many existing Parameters that can be used to model complex behaviours. This includes parameters that
read data in different formats (e.g. the dataframe parameter), parameters that solve specific problems (e.g. reservoir
control curves) and general purpose parameters (e.g. the ``AggregatedParameter``).

Custom parameters can be used to model specific behaviours otherwise not possible with the existing parameters. To
write a custom parameter you must (as a minimim) inherit from the base ``pywr.parameter.Parameter`` class and implement
the ``Parameter.value`` method. The arguments to this method represent the current timestep and scenario allowing the
value of the parameter to be dynamic.

A simple parameter
==================

The example below shows a minimal parameter which returns the same value every timestep. The parameter is initialised
with the value to return, which it stores in the ``_value`` attribute on the instance [*]_. This value is returned
whenever the ``value`` method is queries. The ``load`` class method is used to create an instance of the parameter
from JSON.

.. code-block:: python

    from pywr.parameters import Parameter

    class MyParameter(Parameter):
        def __init__(self, model, value, **kwargs):
            # called once when the parameter is created
            super().__init__(model, **kwargs)
            self._value = value

        def value(self, timestep, scenario_index):
            # called once per timestep for each scenario
            return self._value

        @classmethod
        def load(cls, model, data):
            # called when the parameter is loaded from a JSON document
            value = data.pop("value")
            return cls(model, value, **data)

    MyParameter.register()  # register the name so it can be loaded from JSON

.. [*] ``_value`` is used instead of ``value`` to avoid overloading the method of the same name.

An instance of the parameter can be created from a JSON model as below. The type of the parameter is the class name,
with the word "parameter" optionally removed (e.g. ``"constant"`` and ``"constantparameter"`` both create an instance
of ``ConstantParameter``).

.. code-block:: yaml

    "max_flow": {
        "type": "myparameter",
        "value": 123.0
    }

Timesteps and scenarios
=======================

The ``Parameter.value`` method is called once per timestep for each scenario. The value it returns can be varied using
the ``timestep`` and ``scenario_index`` arguments. For example, a simple version of a ``MonthlyProfileParameter`` could be
created using the following:

.. code-block:: python

    class MonthlyProfileParameter(Parameter):
        def __init__(self, model, profile, **kwargs):
            super().__init__(model, **kwargs)
            self.profile = profile  # a 12-element list of floats

        def value(self, timestep, scenario_index):
            index = timestep.month - 1  # convert to zero-based index
            value = self.profile[index]

        @dataclass
        def load(cls, model, data):
            profile = data.pop(profile)
            return cls(model, profile, **data)

Tracking state with setup, reset, before and after
==================================================

The ``Parameter.setup`` and ``Parameter.reset`` methods are called once at the start of a model run before the first
timestep. The ``reset`` method is called for every run, while the `setup` method is only called if the structure of
the model has changed [*]_.

The ``Parameter.before`` and ``Parameter.after`` methods are called before and after each timestep, respectively. These
methods can be used when a parameter needs to track state between timesteps. For example, a licence parameter needs
to track the volume remaining in the licence. It's important to remember that when using scenarios the model has
multiple states. It's good practice to write stateful parameters with this in mind, even if you aren't using scenarios
initially, so that you can in the future without rewriting anything. The example below shows a very simplistic licence
parameter which has a finite volume.

.. code-block:: python

    class LicenceParameter(Parameter):
        def __init__(self, model, total_volume, **kwargs):
            super().__init__(self, model, **kwargs)
            self.total_volume = total_volume

        def setup(self):
            # allocate an array to hold the parameter state
            num_scenarios = len(self.model.scenarios.combinations)
            self._volume_remaining = np.empty([num_scenarios], np.float64)

        def reset(self):
            # reset the amount remaining in all states to the initial value
            self._volume_remaining[...] = self.total_volume

        def value(self, timestep, scenario_index):
            # return the current volume remaining for the scenario
            return self._volume_remaining[scenario_index.global_id]

        def after(self):
            # update the state
            timestep = self.model.timestepper.current  # get current timestep
            flow_during_timestep = self._node.flow * timestep.days  # see explanation below
            self._remaining -= flow_during_timestep
            self._remaining[self._remaining < 0] = 0  # volume remaining cannot be less than zero

        def load(self, model, data):
            total_volume = data.pop("total_volume")
            return cls(model, total_volume, **data)

The example above uses the `_node` attribute of the parameter, which is automatically set when the parameter is attached
to a node. The `flow` attribute of the node represents the flow (per day) via that node. To get the total flow for the
timestep it must be multipled by the number of days in the timestep, available as `timestep.days`.

.. [*] The model is said to be "dirty" if nodes or edges are added or removed, resulting in a change to the structure
       of the linear programme used to solve the model. This usually requires Parameters which track state to
       reallocate memory, instead of just resetting values.

Dependency on other parameters
==============================

The value of each parameter is calculated at the start of every timestep. A dependency tree is used to ensure that
parameters are evaluated in the correct order and that there are no circular dependencies [*]_. For example, the
``AggregatedParameter`` returns the aggregated value of a set of parameters using a user-defined function. In the
terminology of the dependency tree the ``AggregatedParameter`` is the parent of the other parameters, which are it's
children. When writing a parameter these dependencies need to be defined explicitly by modifying the
``Parameter.parents`` or ``Parameter.children`` attributes.

To get the value of a child parameter use the ``Parameter.get_value`` method, or for the index use
``Parameter.get_index``. These methods return the value/index for the current timestep and scenario. To access the
value from previous timesteps you must manually track the state of the child parameters.

The ``pywr.parameters.load_parameter`` function is used to load parameters from JSON. This works with both references to
parameters and nested parameters.

As an example, see a simplified version of ``AggregatedParameter`` that returns the sum value of it's child parameters.

.. code-block:: python

    class SumParameter(Parameter):
        def __init__(self, model, parameters, **kwargs):
            super().__init__(model, **kwargs)
            self.parameters = parameters
            for parameter in self.parameters:
                self.children.add(parameter)

        def value(self, timestep, scenario_index):
            total_value = sum([parameter.get_value(scenario_index) for parameter in parameters])
            return total_value

        @classmethod
        def load(self, model, data):
            parameters = [load_parameter(parameter_data)
                          for parameter_data in data.pop("parameters")]
            return cls(model, parameters, **data)

.. [*] A circular dependency is when two (or more) parameters depend on each other. This can be direct (e.g A depends
       on B, B depends on A) or indirect (e.g. A depends on B, B depends on C, C depends on A). Pywr is unable to resolve the
       order in which to calculate these parameters and will raise an error at runtime.

Improving performance with Cython
=================================

Parameters are evaluated many times and can be a significant part of the model run time. Many of the parameters in the
core library have been written in Cython to improve performance. Custom parameters can be written in Cython too. Cython
can also be used to link to external C/C++ libraries.

A full tutorial in Cython is beyond the scope of this documentation - see the
`Cython Documentation <https://cython.readthedocs.io/en/latest/>`_.

The easiest way to compile and run custom parameters written in Cython is using the ``pyximport`` command, which
compiles pyx modules at runtime. If the parameter is linking to a foreign library you may need to compile using a
``setup.py`` in order to pass linker arguments.

The example below demonstrates a custom parameter which uses a function from a foreign library (the ``pow`` function
from ``libm``). There are a few differences from the Python equivalent:

* Use of the ``cimport`` statement
* Inherit from ``pywr.parameters._parameters.Parameter``
* The ``value`` method is defined as a cpdef function. This signature must match exactly.

.. code-block:: cython
    :caption: custom_parameters.pyx

    from pywr.parameters._parameters cimport Parameter
    from pywr._core cimport Timestep, ScenarioIndex

    cdef extern from "math.h":
        double pow(double x, double y)

    cdef class SquaredParameter(Parameter):
        cdef double _value

        def __init__(self, model, value, **kwargs):
            super().__init__(model, **kwargs)
            self._value = value

        cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
            return pow(self._value, 2.0)

        @classmethod
        def load(cls, model, data):
            # called when the parameter is loaded from a JSON document
            value = data.pop("value")
            return cls(model, value, **data)

    SquaredParameter.register()


.. code-block:: python
    :caption: run_model.py

    import pyximport
    pyximport.install()

    from pywr.model import Model
    import custom_parameters

    model = Model.load("simple.json")
    model.run()
