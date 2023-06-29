GLPK error handling and performance
===================================

In complex models it may be possible to create combinations of nodes, custom parameter outputs and other data that
does not translate in to a sensible linear programme. GLPK's C interface by default performs little check of input values
(e.g. NaNs are unchecked). Since v1.17 Pywr has enabled additional data checking and error handling by default.

Version 1.17 and above
----------------------

Since version v1.17 Pywr has included additional value checking and error handling by default. These checks will
ensure that NaN values are caught before they reach GLPK. The error handling ensures that any internal errors within
GLPK are caught gracefully and translated to Python exceptions. Two exceptions `GLPKError` and `GLPKInternalError`
are raised for these cases respectively.

Internal errors (i.e. `GLPKInternalError`) will invalidate the entire GLPK environment and is very difficult to recover
from. In general if either of these errors occur users are recommended to debug their causes and re-load (ideally with
an new process) the models. It is *not* recommended to attempt to catch these exceptions, recover and continue with
a simulation.

The addition of the data checking and error handling is not zero cost in terms of runtime performance. Benchmarks on
random test models and real world models have shown the cost is minimal in the context of an entire simulation.
However, since version 1.17 a new runtime time option has been added which uses GLPK without any data
checks or error handling. This should be comparable with Pywr's default behaviour prior to version 1.17. This option
can be enabled with the `use_unsafe_api=True` keyword argument to a GLPK solver, or via the environment variable
`PYWR_SOLVER_GLPK_UNSAFE_API`


Older versions
--------------

Prior to version v1.17 checks of NaN values could be enabled using the `--enabled-debug` option at build time. Any
errors would result in an `AssertionError` from the GLPK solver interface. Internal GLPK errors are not handled in
earlier versions of Pywr. If an error occurs within GLPK it is very likely that this will result in a segmentation
fault of the calling Python process.
