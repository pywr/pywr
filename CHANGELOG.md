# Changelog

All notable changes to this project will be documented in this file.

## [1.27.0] - 2024-10-17

### 🚀 Features

- Fallback to loading from values or tables. ([#1133](https://github.com/pywr/pywr/issues/1133))
- Allow a columns attribute to be sepecified for tables and dataframes ([#1140](https://github.com/pywr/pywr/issues/1140))

### ⚙️ Miscellaneous Tasks

- Upgrade upload-artifact and download-artifact to v4 ([#1141](https://github.com/pywr/pywr/issues/1141))
- *(release)* Update changelog for 1.26.0 release.

## [1.26.0] - 2024-06-25

### 🐛 Bug Fixes

- Use NaN when calculating duration of no events. ([#1131](https://github.com/pywr/pywr/issues/1131))

### ⚙️ Miscellaneous Tasks

- *(release)* Update changelog for 1.26.0 release. ([#1132](https://github.com/pywr/pywr/issues/1132))

## [1.25.0] - 2024-06-20

### 🚀 Features

- Add support for defining net or gross loss in LossLink. ([#1124](https://github.com/pywr/pywr/issues/1124))

### 📚 Documentation

- Add OtherModelXXX parameters to API docs. ([#1126](https://github.com/pywr/pywr/issues/1126))

### ⚙️ Miscellaneous Tasks

- Use delvewheel to repair Windows wheels. ([#1125](https://github.com/pywr/pywr/issues/1125))
- Pin numpy<2 for now. ([#1130](https://github.com/pywr/pywr/issues/1130))
- *(release)* Update changelog for 1.25.0 release. ([#1128](https://github.com/pywr/pywr/issues/1128))

## [1.24.0] - 2024-03-20

### 🚀 Features

- Cache file hashes to avoid recomputation when the same file is referenced multiple times ([#1118](https://github.com/pywr/pywr/issues/1118))

### 🐛 Bug Fixes

- Update warnings check in orphaned components test

### ⚙️ Miscellaneous Tasks

- Black formatting updates
- *(release)* Update changelog for 1.24.0 release. ([#1120](https://github.com/pywr/pywr/issues/1120))

## [1.23.0] - 2023-11-13

### 🚀 Features

- Support parameter loss_factor in LossLink. ([#1113](https://github.com/pywr/pywr/issues/1113))

### ⚙️ Miscellaneous Tasks

- *(release)* Update changelog for 1.23.0 release. ([#1114](https://github.com/pywr/pywr/issues/1114))

## [1.22.1] - 2023-10-20

### 🐛 Bug Fixes

- Fix issue with dataframe scenario indexing ([#1111](https://github.com/pywr/pywr/issues/1111))

### ⚙️ Miscellaneous Tasks

- *(release)* Update changelog for 1.22.1 release. ([#1112](https://github.com/pywr/pywr/issues/1112))

## [1.22.0] - 2023-10-06

### ⚙️ Miscellaneous Tasks

- Add Zulip link to README. ([#1108](https://github.com/pywr/pywr/issues/1108))
- Support Cython 3.x ([#1104](https://github.com/pywr/pywr/issues/1104))
- Updates for Python 3.12 in CI ([#1107](https://github.com/pywr/pywr/issues/1107))
- *(release)* Update changelog for v1.22.0 release. ([#1109](https://github.com/pywr/pywr/issues/1109))

## [1.21.0] - 2023-07-24

### 🚀 Features

- Add metadata to TablesRecorder arrays. ([#1083](https://github.com/pywr/pywr/issues/1083))
- Pass kwargs to model load via optimisation wrappers ([#1096](https://github.com/pywr/pywr/issues/1096))

### 🐛 Bug Fixes

- Add pywr.parameter imports from control_curves module. ([#1084](https://github.com/pywr/pywr/issues/1084))
- Overload get_xxx_flow for PiecewiseLink. ([#1088](https://github.com/pywr/pywr/issues/1088))

### ⚙️ Miscellaneous Tasks

- Use ubuntu-latest image for building wheels. ([#1090](https://github.com/pywr/pywr/issues/1090))
- Support pandas v2.x ([#1089](https://github.com/pywr/pywr/issues/1089))
- Pin Cython<3 until support is resolved. ([#1100](https://github.com/pywr/pywr/issues/1100))
- *(release)* Update changelog for v1.21.0 release. ([#1099](https://github.com/pywr/pywr/issues/1099))

## [1.20.1] - 2023-02-23

### 🐛 Bug Fixes

- Fix bug loading timestep_offset in DataFrameParameter from JSON. ([#1078](https://github.com/pywr/pywr/issues/1078))

### ⚙️ Miscellaneous Tasks

- Apply black formatting to examples. ([#1079](https://github.com/pywr/pywr/issues/1079))
- Update changelog for v1.20.1 release. ([#1080](https://github.com/pywr/pywr/issues/1080))

## [1.20.0] - 2023-01-19

### 🚀 Features

- Allow storage max volume to be set as a constant parameter without specifying both initial volume and initial volume pc. ([#1071](https://github.com/pywr/pywr/issues/1071))
- Add timestep offset to DataFrameParameter and TablesArrayParameter ([#1073](https://github.com/pywr/pywr/issues/1073))
- Update TablesRecorder to save scenario slice data ([#1072](https://github.com/pywr/pywr/issues/1072))
- Add RollingMeanFlowNodeParameter ([#1074](https://github.com/pywr/pywr/issues/1074))

### 🐛 Bug Fixes

- Fix Cython typing warning in MonthlyProfile ([#1075](https://github.com/pywr/pywr/issues/1075))

### Other

- *(other)* Add Python 3.11 to CI. ([#1065](https://github.com/pywr/pywr/issues/1065))
- *(other)* Update changelog for v1.20.0 release. ([#1076](https://github.com/pywr/pywr/issues/1076))

## [1.19.0] - 2022-11-10

### Other

- *(other)* Remove outdated comment on constant factors in AggregatedNode. ([#1060](https://github.com/pywr/pywr/issues/1060))

* Remove outdated comment on constant factors in AggregatedNode.

Updates the cookbook comment that implies factors must be
constants. This is no longer the case.

* Properly change the documentation.
- *(other)* Update to AnnualCountIndexThresholdRecorder to include a range of days to record ([#1061](https://github.com/pywr/pywr/issues/1061))

* initial commit to add inclusion of days in a year to record

* Fixes to rec, test and black

* blacked with updated version of black

* Response to JET review

* Newline in docstring.

Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Remove half-finished recorder. ([#1064](https://github.com/pywr/pywr/issues/1064))
- *(other)* Pass model, path and solver args to super call to _load_from_dict method ([#1067](https://github.com/pywr/pywr/issues/1067))
- *(other)* Fix LicenceParameter example. ([#1068](https://github.com/pywr/pywr/issues/1068))

Fix the LicenceParameter example in the docs. Thanks to
@ahamilton144 for the contribution.

Fixes #1062.
- *(other)* Move project dependencies to pyproject.toml ([#1040](https://github.com/pywr/pywr/issues/1040))

Remove ipython, jinja2 and matplotlib as core dependency (i.e. install_requires).
- *(other)* Add WeightedAverageProfileParameter ([#1066](https://github.com/pywr/pywr/issues/1066))
- *(other)* Update changelog for v1.19.0 release ([#1069](https://github.com/pywr/pywr/issues/1069))

## [1.18.0] - 2022-08-08

### Other

- *(other)* Update to check hash function to make it not case sensitive.  ([#1046](https://github.com/pywr/pywr/issues/1046))

* update to check has function to make it not case sensitive. Test case added

* black formatting

Co-authored-by: Johnson <Jack.Johnson@atkinsglobal.com>
- *(other)* Update load method for several parameters to pass all data dict to class initialisation ([#1048](https://github.com/pywr/pywr/issues/1048))

* update load method for several parameters to pass data to initialisation + use interp_kwargs keyword for interpolation param tests
- *(other)* MonthlyVirtualStorage ([#1049](https://github.com/pywr/pywr/issues/1049))

* initial work on MonthlyVirtualStorage

* add test model

* black linting

* black linting on test file

* add param descriptions to doc string

* state default param values in the node docstring

* Add MonthlyVirtualStorage to API docs.

Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Add StorageParameter ([#1057](https://github.com/pywr/pywr/issues/1057))

* Add StorageParameter

Fixes #1056.

* Add tests for use_proportional_volume=True in StorageParameter.
- *(other)* Model inception ([#1041](https://github.com/pywr/pywr/issues/1041))

* Initial commit of MultiModel.

Initial work on a `MultiModel` class to facilitate running
several Pywr models in the same simulation. Includes a basic
parameter for using getting a Parameter value from another
model during the simulation.

The coupled simulation is taken at the time-step level. All
sub-models perform their "before" steps before moving to "solve",
and then to "after".

Additional work is required to allow getting node properties
(e.g. current volume, flow) from other models.

* Add missing multi model tests.

* Minor fix to multi model test.

* Add additional tests for MultiModel; refactor approach and parameters a little.

* Fix formatting.

* Complete test for three dependent models.

* Support sub-model specific paths.

* Removed some unused imports and add debug logger messages.

* Add OtherModelIndexParameterValueIndexParameter

* Initial commit of ShadowStorage and ShadowNode.

* Correct issue with copying memoryviews in Shadow nodes.

* Add a runtime check that all timesteps in sub-models are equal.

* Correct typo in comments of multi-model parameters.

* Add scenario consistency checking for MultiModel.

MultiModel now checks scenario consistency during setup. Added
tests to check for handling inconsistent configurations.

* Update minimum version in multi-model test models.

* Correct formatting.

* Improve docstrings, comments and debug logging.

* Add a resource profile to model.setup()

Tracks performance counter and MAXRSS for each call to
setup of a node and component. Recording the entries and
eventually dumping them to a CSV.

* Support setup profiler in MultiModel.

* Add reset profiling.

* Apply Black formatting.

* Improve resource Profiler

- Support platforms that don't implement resource.
- Better docstrings.
- Better tests.
- *(other)* Update changelog for v1.18.0 release ([#1059](https://github.com/pywr/pywr/issues/1059))

## [1.17.2] - 2022-02-21

### Other

- *(other)* Add factors to AggregatedNode component_attrs ([#1037](https://github.com/pywr/pywr/issues/1037))

This changes fixes a bug where there Parameters attached to an AggregatedNode are incorrectly identified as being orphaned.
- *(other)* Update RollingVirtualStorage to calculate initial utilisation using the initial volume ([#1036](https://github.com/pywr/pywr/issues/1036))

* update rolling vs to calculate initial utilisation based on the initial volume.

* correct test formatting

* response to review comments: improve docstring and deal with cases where inital vol is zero

* move check of 0.0 initial vol/pc and max volume param into setup
- *(other)* Pass cutoff arg to call to nx.all_simple_paths ([#1039](https://github.com/pywr/pywr/issues/1039))
- *(other)* Update to black v22 format. ([#1044](https://github.com/pywr/pywr/issues/1044))
- *(other)* Correct the use of keyword arguments in Model.load ([#1043](https://github.com/pywr/pywr/issues/1043))

* Correct the use of keyword arguments in Model.load

This fixes a bug with how keyword arguments are passed between
the various load methods. The bug meant that any solver passed
to .load() was ignored. This fixes that and adds tests. The
keyword argument overrides both the solver settings in the JSON
data and PYWR_SOLVER environment variable.

Fixes #1042

* Address test failures and fix test capture.
- *(other)* Update changelog for v1.17.2 release ([#1045](https://github.com/pywr/pywr/issues/1045))

Bug fix release now #1042 is fixed.

## [1.17.1] - 2022-01-04

### Other

- *(other)* Build & test Python 3.10 wheels. ([#1007](https://github.com/pywr/pywr/issues/1007))

* Build & test Python 3.10 wheels.

* Always install latest cibuildwheel.

* Enable CI builds for 3.9 and 3.10 under Windows.

* Excluded MUSL Linux builds for now.
- *(other)* Update changelog for 1.17.1 release ([#1034](https://github.com/pywr/pywr/issues/1034))

This release is just to release the new wheels for new Python versions.

## [1.17.0] - 2021-12-20

### Other

- *(other)* Use Python 3.9 for building the documentation ([#1029](https://github.com/pywr/pywr/issues/1029))

Python 3.7 is a bit old and not all packages will support it in future.
- *(other)* Add ControlCurveParameter to the API docs ([#1027](https://github.com/pywr/pywr/issues/1027))

* Add ControlCurveParameter to the API docs

* Fix ControlCurveParameter docstring.
- *(other)* Add section on checksums to the external data documentation. ([#1028](https://github.com/pywr/pywr/issues/1028))

* Add section on checksums to the external data documentation.
- *(other)* Improve RiverSplitWithGauge docstring. ([#1030](https://github.com/pywr/pywr/issues/1030))
- *(other)* Improve GLPK error handling. ([#1021](https://github.com/pywr/pywr/issues/1021))

* Improve GLPK error handling.

This picks up some work from #759 and the work done in #762 to
improve the error handling in GLPK. Best to read the new page
in the documentation that's been added about this. The TLDR
is that this adds NaN checks and GLPK error handling to Pywr
by default at the cost of some performance. The older
behaviour can be created with `--glpk-unsafe` compile time flag.

There are two new tests that cause seg faults with the unsafe
interface (i.e. current behaviour). Raising exceptions is better
from a handling point of view for users wrapping models in
multi-processing, etc.

Queries:
 - Should the unsafe interface be a runtime option instead of
a compile time option?
- Is changing the default error handling behaviour a good idea?

* Fix format of setup.py

* Actually add the new documentation.

* Skip NaN tests for lpsolve.

* Fix test formatting.

* Fix except instead of except?

* Remove print statements.

* Tweaks to docs in response to review comments.

* Handle exceptions from GLPK low-level methods.

This should ensure the setup methods do not leak any memory
and that a useful error message is logged.

* Make unsafe GLPK API a runtime option.

* Remove unsafe flags from setup.py

* Fix bug in unsafe API and skip some tests with unsafe API.

* Correct keyword argument name for using unsafe GLPK API.
- *(other)* Improve handling of NaN current pc in storage nodes. ([#1020](https://github.com/pywr/pywr/issues/1020))

* Improve handling of NaN current pc in storage nodes.

The current approach requires Parameter writers to handle
NaN values for current_pc. NaNs can occur when storage nodes
have zero maximum volume. This PR adds a convenience method
to get a finite and bounded current_pc for all storage nodes.
Pywr's internal parameters are updated to use this in preference
to direct access of the ._current_pc memory view.

Parameter writers should also use this new cpdef function instead
of using .current_pc directly.

There are two recorders which still use the raw ._current_pc
memory view for their data outputs. Those could be updated too.

We should also add some explicit tests for cases where using this
new internal function makes a difference.

* Use isnan from libc.math.

* Address typos and excess space from review comments.
- *(other)* Add min_output argument to RectifierParameter ([#1031](https://github.com/pywr/pywr/issues/1031))

* add min_output arg to rectifier param
- *(other)* Improve solver error messages for invalid networks. ([#1025](https://github.com/pywr/pywr/issues/1025))

* Improve solver error messages for invalid networks.

If the user doesn't run `Model.check()` then the error from
the solver is quite cryptic. This change adds some sense
checks to the number of columns for the basic network constraints.

* Add missing test models.
- *(other)* Fix test name conflict ([#1033](https://github.com/pywr/pywr/issues/1033))

* Fix test name conflict

Rename a test that shares the same name as another test.

* Fix the test itself.
- *(other)* Release v1.17.0 ([#1032](https://github.com/pywr/pywr/issues/1032))

* Release v1.17.0

- Updated the CHANGELOG with recent changes.
- Corrected new documentation regarding GLPK runtime option.

## [1.16.1] - 2021-12-08

### Other

- *(other)* Update README install instructions ([#1023](https://github.com/pywr/pywr/issues/1023))

Point new users to PyPi instead of compiling Pywr from source.
- *(other)* Fix presolve test skipif filter to include any solver starting with 'glpk'. ([#1022](https://github.com/pywr/pywr/issues/1022))

* Fix the skipif filter to include any solver starting with 'glpk'.

* Fix test formatting.
- *(other)* Fix internal node names for PiecewiseLink. ([#1024](https://github.com/pywr/pywr/issues/1024))

It was using `self.name` before it was initialised resulting in
all internal nodes being called `None <something>`. Use the name
argument directly instead.
- *(other)* Update README for v1.16.1 release. ([#1026](https://github.com/pywr/pywr/issues/1026))

## [1.16.0] - 2021-12-02

### Other

- *(other)* Fix doc error about aggregated parameters ([#1009](https://github.com/pywr/pywr/issues/1009))

Docs incorrectly include a "threshold" for a `NegativeParameter` and misses some commas in the JSON definitions.
- *(other)* Switch to black code formatter. ([#1010](https://github.com/pywr/pywr/issues/1010))

* Switch to black code formatter.

* Add more files reformatted with black.
- *(other)* Remove circular import in calibration recorders module. ([#1012](https://github.com/pywr/pywr/issues/1012))

Circular import causes a problem with some tools (e.g. mypy). It is unnecessary in this case by importing directly from the Cython module.
- *(other)* Remove old and unused travis folder. ([#1016](https://github.com/pywr/pywr/issues/1016))

This folder is not used since we moved to Github Actions. It
contains out of date Dockerfiles and build scripts which we no
longer use or support.
- *(other)* Uniform drawdown residual licence ([#1013](https://github.com/pywr/pywr/issues/1013))

* wip adding a residual days attribute to uniformdrawdownparameter

* update calculation of drawdown target and fix tests

* update param and test docstrings

* run black formatter on changes
- *(other)* Support inital_volume_pc in VirtualStorage nodes. ([#1015](https://github.com/pywr/pywr/issues/1015))

Add support for initial_volume_pc keyword argument when
initialising VirtualStorage nodes and its subclasses. This allows
these nodes to use a parameter for maximum volume, which requires
specifying both initial_volume and initial_volume_pc. Previously
an error message indicated the user should provide
initial_volume_pc, but when they tried this was not an valid
keyword argument.

The tests handle the case where an AnnualVirtualStorage node
resets of the first time-step (i.e. before the maximum volume
parameter is evaluated).
- *(other)* Update changelog for v1.16.0 release. ([#1019](https://github.com/pywr/pywr/issues/1019))

## [1.15.3] - 2021-10-04

### Other

- *(other)* Add CI steps to publish wheels and sdist to test PyPI. ([#999](https://github.com/pywr/pywr/issues/999))

Publishing only happens on tagged commits. If this works as
expected we can follow-up with another commit to move to
live PyPI.
- *(other)* Assert entries when creating rows in the GLPK matrix. ([#1002](https://github.com/pywr/pywr/issues/1002))

* Assert entries when creating rows in the GLPK matrix.

In response to #1001 this commit implements a check to the
`glp_set_mat_row` call to ensure that there is at least one
entry in the row. Zero entry rows are likely caused by invalid
or unsupported model / network configuration (e.g. those
reported in #1001). For the time being this change will ensure
such models fail to run with an `AssertionError` instead of
failing to applying expected constraints silently.

A proper fix to enable support for a `PiecewiseLink` or other
complex nodes with `VirtualStorage` is still required.

* Add test for virtual storage with aggregated node.
- *(other)* Update changelog for v1.15.3 release. ([#1003](https://github.com/pywr/pywr/issues/1003))
- *(other)* Update the PyPI publishing in GA pipeline. ([#1004](https://github.com/pywr/pywr/issues/1004))

The current approach failed because the pypi-publish action
does not run on Windows. The recommended approach is to use
a separate job that downloads the final artifacts (wheels and
sdist) before uploading to PyPI.
- *(other)* Add artifact names for upload and download. ([#1005](https://github.com/pywr/pywr/issues/1005))
- *(other)* Only upload to PyPI on tags. ([#1006](https://github.com/pywr/pywr/issues/1006))

## [1.15.2] - 2021-09-03

### Other

- *(other)* Update _parameters.pyx ([#997](https://github.com/pywr/pywr/issues/997))

Fixed attribute name in remove method of AggregatedParameter and AggregatedIndexParameter classes
- *(other)* Update changelog for v1.15.2 release. ([#998](https://github.com/pywr/pywr/issues/998))

## [1.15.1] - 2021-08-25

### 🐛 Bug Fixes

- Fix FlowDurationCurveDeviationRecorder load method and update existing test to check change ([#989](https://github.com/pywr/pywr/issues/989))

### Other

- *(other)* Unpin <5 decorator pin as networkx now requires >5 ([#990](https://github.com/pywr/pywr/issues/990))
- *(other)* Fix divide by zero error in two recorders. ([#993](https://github.com/pywr/pywr/issues/993))

`NumpyArrayNodeSuppliedRatioRecorder` and `NumpyArrayNodeCurtailmentRatioRecorder`
could both error with a divide by zero if the max flow of the parameter
they monitor returned zero.

Fixes #991
- *(other)* Add methods to Timestep to return days in current and next year ([#994](https://github.com/pywr/pywr/issues/994))

* Add `days_in_current_year` and `days_in_next_year` methods to the Timestep object

* Fix AnnualTotalFlowRecorder so that number on days in a timestep is taken account for in flow calculation

* improve typing of timestep methods and AnnualTotalFlowRecorder after method

* response to review comments

* revert return type of days_in_current_year and days_in_next_yearto float and use epoch time to more accurately calaculate number of days
- *(other)* Update changelog for v1.15.1 release. ([#995](https://github.com/pywr/pywr/issues/995))

## [1.15.0] - 2021-05-20

### Other

- *(other)* Fix use of deprecated np.int and np.float functions. ([#982](https://github.com/pywr/pywr/issues/982))

* Fix use of deprecated np.int function.

* Remove use of deprecated np.float function.
- *(other)* Selective bounds and objective updates for the glpk-edge solver ([#912](https://github.com/pywr/pywr/issues/912))

* Initial commit of more selective update of flow bounds.

Adds an optional setting to the glpk-edge based solver that,
when active, only sets constraint bounds for nodes with fixed
min_flow and max_flow values once. This happens during setup,
other nodes with parameters for either min_flow or max_flow
are updated as usual.

* Initial commit of a more selective update of costs.

Adds an optional setting to the glpk-edge based solver that,
when active, only gets costs from nodes that have a non-fixed
cost attribute. The fixed element of the costs is pre-calculated
in the setup if the model.

* Improve efficiency of fixed costs assignment.

* Remove typo in has_fixed_cost docstrings.

* Clarify glpk-edge comment re costs calculations.

* Add support for passing new glpk-edge parameters by environment variables.

* Improve fixed cost and flow constraint perf improvements.

- Improvements applied to regular route based GLPK solver.
- Solver settings reported to ModelResult object.
- Settings tested in CI and tests modified to work with
new settings.

* Address comment typos.
- *(other)* Update constant Parameters in reset rather than every timestep. ([#983](https://github.com/pywr/pywr/issues/983))

* Move bounds update for non-storage nodes with constants to .reset()

* Implement `is_constant` for Parameters and use it GLPK edge solver.

New is_constant attribute and associated methods that can be
used if a parameter does not vary with time or scenario. This
allows the solver to only set these constraints during reset.

* Implement constant only updating for GLPK path based solver.

* Swap logic in get_constant_value() and value() for activation parameters.

* Fix bug not applying scale and offset in ConstantParameter.
- *(other)* Update constant aggregated node factors only during rest. ([#985](https://github.com/pywr/pywr/issues/985))

* Update constant aggregated node factors only during rest.

This is a performance improvement that follows on from #983
by only updating aggregated node factors in the GLPK solvers
during reset instead of every timestep. It should have most
benefit for models which use many of aggregated nodes with factors
as the code both for calculating the normalised factors is
not very efficient.

A new solver option and corresponding environment variable is
added to turn this feature on. It is off by default.

Modified the CI test suite to run only with and without the
extra options. As opposed to running each option individually.

* Fix bug with glpk route solver updating all aggregated factors every timestep.

* Add set_fixed_factors_once to solver settings.
- *(other)* Make DF parameter only load required data. ([#981](https://github.com/pywr/pywr/issues/981))

* WIP making DF parameter only load required data.

* Add test and test model for DF sub-sampling.
- *(other)* Updated MonthlyProfileParameter and RbfProfileParameter to take list … ([#986](https://github.com/pywr/pywr/issues/986))

* updated MonthlyProfileParameter and RbfProfileParameter to take list of values for upper and lower bounds, instead of only a constant across the year

* changed upper/lower type from list to array_type. updated docstring

* Tidy up docstrings.

* Add tests for variable bounds.

Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Update changelog ready for v1.15.0 release. ([#987](https://github.com/pywr/pywr/issues/987))

* Update changelog ready for v1.15.0 release.

* Fix typo in changelog.

## [1.14.0] - 2021-04-15

### Other

- *(other)* Pin decorator<5 to satisfy networkx dependency in doc builds. ([#978](https://github.com/pywr/pywr/issues/978))

NetworkX issue here:
  https://github.com/networkx/networkx/issues/4732
- *(other)* Register event recorders and add load methods for them ([#976](https://github.com/pywr/pywr/issues/976))

* register event recs and add load methods

* response to review comments

* add default values for aggregation args of EventDuration and EventStatistic recs
- *(other)* Load fdc deviation targets from file ([#977](https://github.com/pywr/pywr/issues/977))

* add functionality to fdc dev recorders to load fdc targets from file

* update fdc deviation recorder to allow only one of upper or lower targets to be specified

* fix cls creation call in load method and add comment to doc string about indexing external target data files
- *(other)* Update changelog ready for v1.14.0 release. ([#980](https://github.com/pywr/pywr/issues/980))

## [1.13.1] - 2021-03-26

### Other

- *(other)* Fix bug loading NumpyArrayNormalisedStorageRecorder from JSON. ([#974](https://github.com/pywr/pywr/issues/974))

Added regression tests.
- *(other)* Update changelog for v1.13.1 release. ([#975](https://github.com/pywr/pywr/issues/975))

## [1.13.0] - 2021-03-22

### Other

- *(other)* Add Python-3.9 ManyLinux builds to CI. ([#954](https://github.com/pywr/pywr/issues/954))
- *(other)* Add an optional tags dictionary to Component ([#968](https://github.com/pywr/pywr/issues/968))

Parameters and Recorders can now optionally take a dict of tags.
The component just stores these in a public attribute. It will
be useful for various tools to provide and save metadata about
components. E.g. for filtering and grouping.
- *(other)* Add GaussianKDEStorageRecorder. ([#970](https://github.com/pywr/pywr/issues/970))

* Add GaussianKDEStorageRecorder.

Added an implementation of a recorder that uses a KDE to estimate
storage probabilities. It uses a reflected KDE to estimate the
bounded distribution of proportional storage. It returns a PDF
and provides a probability of being at or below a target storage
via the `.aggregated_value()` method.

* Implement .values() method.

* Fix typo in test docstring.
- *(other)* MultiThresholdIndexParameter ([#969](https://github.com/pywr/pywr/issues/969))

* added MultiThresholdIndexParameter to return an index against multiple thresholds for previous days flow at a node

* added MultipleThresholdParameterIndexParameter for use with parameters

* response to review comments

* response to review: registered self.parameter as a child of self

* added parameters to the API

* Minor docstring fixes.

Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Reinstate a circular loading test that should now work. ([#972](https://github.com/pywr/pywr/issues/972))
- *(other)* Add NormalisedGaussianKDEStorageRecorder ([#971](https://github.com/pywr/pywr/issues/971))

* Add NormalisedGaussianKDEStorageRecorder

Works in a similar way to GaussianKDEStorageRecorder, but
derives from a new array recorder. NormalisedGaussianKDEStorageRecorder
keeps an array of storage that is normalised between -1 and 1
where the 0 value is at a control curve. The new KDE recorder
estimates a probability of being at or below zero in the normalised
series.

* Fix keyword description in docstring.
- *(other)* Add LossLink node. ([#960](https://github.com/pywr/pywr/issues/960))

* Add LossLink node.

* Allow loss_factor to be zero and support supplying data from a table.

* Make LossLink apply constraints to and record net outflow (previously was gross inflow).

* Add condition for loss_factor of 1.0

* Add tests for other loss factors. Fix returning loss_factor.
- *(other)* Update changelog and API docs for v1.13.0 release. ([#973](https://github.com/pywr/pywr/issues/973))

## [1.12.0] - 2021-02-23

### Other

- *(other)* Several activation functions useful for optimisation. ([#965](https://github.com/pywr/pywr/issues/965))

* Several activation functions useful for optimisation.

Initial commit of 3 functions - binary step, rectifier and
logistics. No tests and only basic docstrings at the moment.
Commit mainly for comment.

* Add tests for activation parameters.

* Add new parameters to API documentation.
- *(other)* Update changelog for v1.12.0 release. ([#967](https://github.com/pywr/pywr/issues/967))

## [1.11.0] - 2021-01-08

### Other

- *(other)* Switch to openpyxl instead of xlrd.
- *(other)* Drop support for Python 3.6.
- *(other)* Merge pull request #959 from pywr/drop-xlrd

Switch to openpyxl instead of xlrd and drop Python 3.6 support.
- *(other)* Add get_all_x methods to nodes.
- *(other)* Add commit_all to StorageInput and StorageOutput.
- *(other)* Merge pull request #958 from pywr/get-all-methods

Add `get_all_xxx` methods to core nodes.
- *(other)* Update changelog for v1.11.0 release.
- *(other)* Merge pull request #961 from pywr/release_v1.11

Update changelog for v1.11.0 release.

## [1.11.0-beta] - 2020-12-09

### Other

- *(other)* Fixed documentation links in README ([#956](https://github.com/pywr/pywr/issues/956))
- *(other)* Implement a 2-stage loading scheme for nodes. ([#945](https://github.com/pywr/pywr/issues/945))

* Implement a 2-stage loading scheme for nodes.

The changes here remove the existing `load` classmethod for
nodes. These are replaced with a new classmethod `pre_load`
which performs the object instantiation. After this itial
stage the parameters and recorders are fully loaded. Finally,
any parameter references are assigned to node attributes.

This resolves the circular loading between nodes and parameters.
However, it has required somworking of node init methods
which has impacted on the JSON schema for the nodes. Overall
this should make the JSON schema and init method arguments
more consistent.

Fixes #380.

* Re-instate Model._get_node_from_ref

To aid transition to the new node loading functional
it`_get_node_from_ref` is retained and defers to `model.nodes[]`
while issuing a DeprationWarning. Custom parameters that require
references to nodes will use this function during loading, but
should be updated in the future.

* Add Loadable to AggregatedStorage.

* Add test for RVS from JSON. Fix bug with failing to load.

* Remove unused __deferred_parameter_list_attributes__

* Add cost property to RiverGauge.

## [1.10.0] - 2020-12-07

### Other

- *(other)* Change basestring to str ([#928](https://github.com/pywr/pywr/issues/928))

This was only needed for Python 2 support.
- *(other)* Update install docs ([#931](https://github.com/pywr/pywr/issues/931))

- Use `pytest` instead of the deprecated `py.test`
- Fix Appveyor URL.
- *(other)* Some flake8 compliance changes. ([#929](https://github.com/pywr/pywr/issues/929))

* Some flake8 compliance changes.

This tackled the files in the root of the package, but none
if the sub-packages.

* Import ScenarioIndex in pywr.core
- *(other)* Added NumpyArrayNodeCostRecorder ([#932](https://github.com/pywr/pywr/issues/932))

* Added NumpyArrayNodeCostRecorder

* Improve NumpyArrayNodeCostRecorder tests.

Tests for fixed as well as scenario and time varying costs.
- *(other)* Add reading external data option to InterpolateVolumeParameter. ([#926](https://github.com/pywr/pywr/issues/926)) ([#930](https://github.com/pywr/pywr/issues/930))

* Add reading external data option to InterpolateVolumeParameter. ([#926](https://github.com/pywr/pywr/issues/926))

* Add a test for interpolatedvolumeparameter, in order to read files

* Add test for interpolatevolumes, corrected an unnecessary print
- *(other)* Add flow param and discount factor param to docs ([#934](https://github.com/pywr/pywr/issues/934))
- *(other)* A few bug fixes for draw and save graph schematic functions: ([#938](https://github.com/pywr/pywr/issues/938))

- fix save_graph for models with virtual nodes so that positions are assigned to correct nodes
- add seasonalvirtualstorage to pywr_json_to_d3_json exclusion list
- make component nodes of delaynode children of the main node so that they are not plotted
- *(other)* Add to_dataframe methods to two annual count recorders. ([#939](https://github.com/pywr/pywr/issues/939))

`AnnualTotalFlowRecorder` and `AnnualCountIndexThresholdRecorder`
store their data internally in an annual timeseries. This commit
exposes that data as a pandas DataFrame.
- *(other)* Fix bug with RbfProfileParameter to using rbf_kwargs. ([#946](https://github.com/pywr/pywr/issues/946))
- *(other)* Remove deprecated PiecewiseLinearControlCurve parameter. ([#947](https://github.com/pywr/pywr/issues/947))
- *(other)* Update load methods of interpolation parameters so that they accept interp1d kwargs ([#943](https://github.com/pywr/pywr/issues/943))

* update load methods of interpolation params so that they accept kwargs to pass to interp1d

* remove code in load funcs of interp params that deals with 'kind' interp kwarg and add property to base class that ensures that 'fill_value' is converted to a tuple
- *(other)* Add optional exclude_months to AnnualCountIndexThreshRecorder ([#950](https://github.com/pywr/pywr/issues/950))

This new option means the counts are ignored for the excluded
months if they are defined. This allows counting only a sub-set
of months during a year.
- *(other)* Initial commit of Github Action using cibuildhweel. ([#948](https://github.com/pywr/pywr/issues/948))

* Initial commit of Github Action using cibuildhweel.

* Only run Python-3.6+

* Remove Appveyor and Travis configs.

* Install GLPK library for Linux builds.

* Install GLPK library for Linux builds (2).

* Use manylinux2014 for CI builds.

* Disable Windows and MACOS builds for now.

* Add platypus-opt test dependency.

* Skip 32-bit builds.

* Add pyproject.toml

* Checkout with full git metadata so setuptools_scm works.

* Fix Windows builds.

* Fix checkout depth for sdist.

* Add script to run tests with all solvers.

* Fix build.yml error.

* Run test script using sh.

* Add Windows specific tests script and +x permission on Linux test script.

* Fix Windows paths in test command.

* Debug Windows test script directory.

* Fix Windows pytest path.

* Add error handling to test scripts.

* Clarify some comments in Github actions config.

* Add Github Actions job to build documentation.

* Install glpk and lpsolve in build docs job.

* Update glpk & lpsolve install for build docs job.

* Update glpk & lpsolve install for build docs job (2).

* Disable lpsolve for docs build.

* Add add some debugging output to Windows wheel repair script.

* Repair Windows wheel with packaged DLL name.

* Add documentation deployment.

* Add sphinx.ext.githubpages extension to documentation build.

* Update documentation and badge links.

* Add sphinx.ext.githubpages extension to documentation build (2).
- *(other)* Update README.rst ([#952](https://github.com/pywr/pywr/issues/952))
- *(other)* Build docs only on master ([#951](https://github.com/pywr/pywr/issues/951))
- *(other)* Make run-tests.sh executable ([#953](https://github.com/pywr/pywr/issues/953))

* Make run-tests.sh script executable.
- *(other)* Updated changelog ready for v1.10.0 release. ([#955](https://github.com/pywr/pywr/issues/955))

## [1.9.0] - 2020-09-15

### Other

- *(other)* Fix timeseries1_weekly.h5 to work with pandas v1.1.0 ([#917](https://github.com/pywr/pywr/issues/917))

Recent versions of pandas were unable to load this file. The
file has been recreated from the equivalent CSV with a 7D
frequency.

Also tidied up a few unneeded attributes in the associated test
models.

Fixes #916.
- *(other)* Add BisectionSearchModel ([#915](https://github.com/pywr/pywr/issues/915))

* Initial commit of BisectionSearchModel.

This is a new subclass of Model that performs a bisection search
by varying a single parameter's value to find its largest value
that satisfies all model constraints.

There are updates to allow loading of different Model subc-classes
from JSON and also in the optimisation wrapper.

* Add new pywr.utils sub-package to setup.py

* Improve BisectionSearchModel

- Improve docstring.
- Add logging information.
- Add infeasibility check and corresponding test.

* Add utils.bisect to API docs.

* Add missing test model.

* Add error_on_infeasible keyword BisectionSearchModel.

New keyword argument determines whether the bisection process
raises an error if no feaisble solution is found.

* Allow specifying error_on_infeasible in the JSON.
- *(other)* Update appveyor badge to refer to master branch ([#920](https://github.com/pywr/pywr/issues/920))

The appveyor badge in the README was referring to the latest build rather than the latest master build (i.e. could be a PR). This points the badge at master.
- *(other)* Reset RollingMeanFlowNodeRecorder's memory position on Model.reset() ([#893](https://github.com/pywr/pywr/issues/893))
- *(other)* Fix warnings about small aggregated node factors. ([#921](https://github.com/pywr/pywr/issues/921))

Make the warning trigger when the absolute value is less than
 the threshold.
- *(other)* Fix AggregatedStorage initial volume when defined as proportions. ([#922](https://github.com/pywr/pywr/issues/922))

This refactors the reset storage code to provide two helper
functions that return the initial volume in absolute and
proprtional terms. These calculations depend on how the initial
and maximum volume is defined.

The AggregatedStorage node uses one these new functions to
correctly calculate its initial volume.

Fixes #918.
- *(other)* Dynamic factors for Aggregated Node ([#919](https://github.com/pywr/pywr/issues/919))

* Dynamic factors for agg node working for simple test for glpk solver

* attempt to minimise updates required for each solve by dynamic agg factors. cvarrays used to cache indicies and factors vals

* added get_factors_norm method to agg node + some refactoring of application of dynamic factors in glpk solver

* update agg node load function and factor property + add test to checking loading of factor parameters

* update agg node factors property so that it accepts parameters

* added dynamic factors support to glpk-edge solver + added warning when trying to use them with lpsolve

* add factor constraint update time to solver stats

* responses to review comments. Main change is removal of factor_parameters property

* update lpsolve script to account for changes to agg node properties

* Refactored dynamic aggregated node factors.

Simplified the internal code by always updating the factors'
cefficient in the constraint matrix. This is inefficient when
the factors are constants, but means the API is consistent
with other `get_xxx` methods. The solver code is also
simplified with no distinction between fixed and dynamic
factors.

Also removed a now deprecated skipped test.

* Fix error in idx_row_aggregated calculation.

* Fix lpsolve AggregatedNode factors calculatiion.

The lpsolve solver does not support dynamic aggregated node
factors. It assumes factor parameters are of type
ConstantParameter and retrieves the factor value using
get_double_variables()[0].

Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Initial commit of RollingVirtualStorage node. ([#891](https://github.com/pywr/pywr/issues/891))

* Initial commit of RollingVirtualStorage node.

* Apply Storage volume bounds capping only if an adjustment is given.

* Add weekly timestep test for RollingVirtualStorage.

- Fix RollingVirtualStorage bug with non-daily timestep.
- Fix a couple of doc typos.
- *(other)* Add NullSolver for debugging. ([#924](https://github.com/pywr/pywr/issues/924))
- *(other)* Add a threshold when checking GLPK constraints bounds equality. ([#925](https://github.com/pywr/pywr/issues/925))

Use a small threshold in the difference between upper bounds
and lower bounds for a constraint to qualify as GLP_FX. This is
a practical consideration where exact floating point equality
might not be true in all cases.
- *(other)* Add SeasonalVirtualStorage Node ([#923](https://github.com/pywr/pywr/issues/923))

* Add SeasonalVirtualStorage Node

* fix mistake in lpsolve script

* added docstring and added node to api docs

* fixed couple of bugs then meant node did not operate correctly in initial year. Also:
    - Added tesst to check initial year edge cases
    - responsed to review comments

* expanded docstring of seasonalvirtualstorage + set active attribute to true in reset method

* further refinements to docstring and formatting

* Minor typos and flake8 fixes.

Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Updated changelog ready for v1.9.0 release. ([#927](https://github.com/pywr/pywr/issues/927))

## [1.8.0] - 2020-07-16

### Other

- *(other)* Wrap setup and parameter functions in try: except ([#896](https://github.com/pywr/pywr/issues/896))

* Wrap setup and parameter functions in try: except, adding contextual
data to exceptions to aid in debugging.
Specifically:
1. Throw an exception when 'type' is not found on a parameter and throw
a custom exception when this occurs with a summarised version of the
erroneous data value in the message.
2. Wrap various setup functions in model.pyx in try: : except, logging
the component name in the before re-raising the exception.

* Wrap reset() functions in try: except, fix some typos and use json.dumps() instead of str() when handling the 'type' error

(cherry picked from commit 8b4024fe1ba8ac7616b23c024abf9a84e9fdc524)

Co-authored-by: Stephen Knox <stephen.knox@manchester.ac.uk>
- *(other)* Add profile recorder and refactor day of year and week indices. ([#903](https://github.com/pywr/pywr/issues/903))

* Add NumpyArrayDailyProfileParameterRecorder.

A new recorder for storing a calculated daily profile. Useful
for optimisation where a new profile might be calculated from
non-daily profiles and/or combinations of parameters.

* Add .is_leap_year and .dayofyear_index to Timestep.

Refactor the is_leap_year function from the _parameters module.
The Timestep object now contains computed attributes for
.is_leap_year and the day of year index (0 - 366) for use with
annual profiles. Refactored parameters to use these new attribues
where neccessary.

The new NumpyArrayDailyProfileParameterRecorder uses the refactored
attributes instead of redefining the is_leap_year inline function.

* Remove super() arguments for NumpyArrayDailyProfileParameterRecorder.

* Add .week attribute to Timestep.

This can be used directly from the Timestep making a consistent
approach. This replaces specific calculations in individual
parameters.

Fixes #804.

* Change Timestep.week to .week_index for consistency.
- *(other)* Add support for optimising the days of year in RbfProfileParameter. ([#908](https://github.com/pywr/pywr/issues/908))

* Add support for optimising the days of year in RbfProfileParameter.

Support optimising the days of year used for interpolation in
RbfProfileParameter. Support is provided using a new keyword
`variable_days_of_year_range` which defines the maximum deviation
from the given days of year that the optimisation use.

In addition new keywords, `min_value` and `max_value`, are added
which allowing capping the overall interpolated profile.

* Fix bug with the RbfProfileParameter integer bounds.

* Add additional days of year validation to RbfProfileParameter.

- Check for strictly monotonic days of year.
- Check for no overlap with variable DoY range.

* Fix typo in error message.
- *(other)* Some documentation improvements ([#905](https://github.com/pywr/pywr/issues/905))

* Improve the API documentation.

Add core section with core classes and fix some headings.

* Add verbosity to sphinx build.

* More documentation fixes.

* Fix docstring typos
- *(other)* Adding discounting capability  ([#901](https://github.com/pywr/pywr/issues/901))

* added test for discount parameter and model

* fixed description

* moved parameter to parameters.py and added test in test_parameters.py

* fixed test for discount parameter, cleaned up JSON file and doc of discount parameter

* fixed indentation error and cythonised discount parameter

* attempt to cythonise, not yet working

* passed test with cythonised discount parameter

* removed warning

* added simplifications James suggested

Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Add DelayNode and FlowDelayParameter ([#904](https://github.com/pywr/pywr/issues/904))

* add DelayNode and FlowDelayParameter

* changed how delay parameter deals with the days kwarg + added a couple of tests

* response to review comments

* switch to using pointer in FlowDelayParameter rather than np.roll

* allow an initial flow value to be set for DelayNode and FlowDelayParameter

* response to review comments
- *(other)* A better workaround for #470 ([#690](https://github.com/pywr/pywr/issues/690))

* A better workaround for #470

This change the reservoir volume initialisation in the case where
max_volume is parameter. Now both `initial_volume` and
`initial_volume_pc` must be specified as floats, and are used as
is. This removes the current whitelist of acceptable parameters.

It does however mean that if the user specifies initial conditions
that are not compatible with the actual max_volume (i.e.
initial_volume != initial_volume_pc/max_volume) then there is the
possibility that parameter values will be incorrect on the first
time-step. This changes effectively makes that the users
responsibility.

* Add parameters section docstring of Storage class.
- *(other)* Update changelog ready for v1.8.0 release. ([#911](https://github.com/pywr/pywr/issues/911))

## [1.7.2] - 2020-06-19

### Other

- *(other)* Update changelog ready for v1.7.2 release. ([#902](https://github.com/pywr/pywr/issues/902))

## [1.7.1] - 2020-06-18

### Other

- *(other)* Skip AggregatedNode and AggregatedStorage when creating d3 graph data from json ([#895](https://github.com/pywr/pywr/issues/895))
- *(other)* Cast values to np.float64 to ensure correct type. ([#897](https://github.com/pywr/pywr/issues/897))

This allows passing integer values in the JSON.
- *(other)* Fix PywrSchematic handling of a dictionary of data. ([#899](https://github.com/pywr/pywr/issues/899))

Added a check for whether loading from a str and load JSON data
only if a path is given. If given a dict then the JSON data is
the argument.

Fixes #898.
- *(other)* Update changelog ready for v1.7.1 release. ([#900](https://github.com/pywr/pywr/issues/900))

## [1.7.0] - 2020-06-08

### Other

- *(other)* Add IPython to install dependencies. ([#870](https://github.com/pywr/pywr/issues/870))

* Add IPython to install dependencies.

* Make notebook tests unskippable because of missing imports.
- *(other)* Rework the Recorder constraint functionality to be more flexible. ([#869](https://github.com/pywr/pywr/issues/869))

* Rework the Recorder constraint functionality to be more flexible.

Upper and lower bounds are now specified on Recorders instead of a
binary `is_constraint` flag. This allows the optimisaiton wrappers
to handle equality, single bounded and double bounded constraints.
It also allows the user to specify the constraint threshold. The
previous implementation relied on the default setting of the
respective optimisation wrappers (which weren't consistent).

* Fix Recorder docstring re new bounds kwargs.

* Fix error in pygmo wrapper.

* Add constrained problem to pygmo tests.

* Add constraint violation and model feasibility methods.

A new method Recorder.is_constraint_violated() that computes
whether the value from the current simulation violates the
defined constraints. A Model.is_feasible() method returns
true if all constraints are not violated.

* Fix internal representation of undefined constraint upper bounds.

* Address review re docstrings and comments.

* Extend Model.is_feasible() docstring.
- *(other)* New notebook graph functionality  ([#868](https://github.com/pywr/pywr/issues/868))

* Moved draw_graph function into an object and created new method to save model json with updated schematic positions from d3 graph

* added to_html method to PywrSchematic class

* add .html to pkg_data for pywr.notebook

* prevent nodes and links moving beyond svg dimensions

* added warning if trying to save node positions when schematic object created using model object

* moved python funcs for creating d3 graph data back out of schematic class

* removed reference to self in calls to d3 data funcs in PywrSchematic

* add option to save node positions to csv

* update draw_graph to save to csv and optionally save positions of unfixed nodes

* re-add standalone draw_graph function that uses PywrSchematic

* update PywrSchematic.draw_graph json warning to check input filetype

* Loop through graph nodes in save_graph.js rather than model json data so positions can be saved to csv when schematic is created using model object

* response to review comments

Co-authored-by: James Batchelor <james.batchelor@atkinsglobal.com>
- *(other)* Add OffsetParameter ([#874](https://github.com/pywr/pywr/issues/874))

* Initial commit of OffsetParameter.

* Add NegativeMinParameter to tests.

* Add test for OffsetParameter variable API.

* Fix docstring typos.
- *(other)* Add PywrRandomGenerator as a Platypus generator. ([#867](https://github.com/pywr/pywr/issues/867))

This generator seeds the initial population with a Solution that
corresponds to the model's current configuration.
- *(other)* Remove inspyred optimisation wrapper. ([#878](https://github.com/pywr/pywr/issues/878))

Not aware of anyone using it, and it has a number of issues.

This closes #871 and closes #459.
- *(other)* Expose run statistics to optimisation wrappers ([#877](https://github.com/pywr/pywr/issues/877))

* Add simulation time and speed to optimisation log messages.

* Save simulation run stats to optimisation wrappers.
- *(other)* Add RbfProfileParameter. ([#873](https://github.com/pywr/pywr/issues/873))

* Initial commit of RbfProfileParameter.

* Fix 1-based indexing and add check input checking to RbfParameter.

Adds tests for invalid days of the year and the variable API.
- *(other)* Remove unreachable code segment when loading CSV dataframes. ([#880](https://github.com/pywr/pywr/issues/880))

Fixes #827.
- *(other)* Add threshold parameters to the API docs. ([#881](https://github.com/pywr/pywr/issues/881))

Fixes #799.
- *(other)* Correct MeanParameterRecorder docstring. ([#883](https://github.com/pywr/pywr/issues/883))

Fixes #679.
- *(other)* Remove incorrect parsing of position keyword in several nodes. ([#884](https://github.com/pywr/pywr/issues/884))

Fixes #733.
- *(other)* Change the signature of recorder aggregation functions to catch exceptions. ([#879](https://github.com/pywr/pywr/issues/879))

Update the Cython function signature to `except *` so that
exception checking is always done. This is helpful if there are
errors during the computation of metrics during optimisation.
- *(other)* Add optional dependencies for docs and dev. ([#882](https://github.com/pywr/pywr/issues/882))

* Add optional dependencies for docs and dev.

There was an existing test extra which I renamed to dev.

Fixes #755.

* Reinstate "test" and add "optimisation" extras in setup.py
- *(other)* Skip virtual nodes when creating d3 graph data from model json ([#885](https://github.com/pywr/pywr/issues/885))

* skip virtual nodes when creating d3 graph data from model json

* added test to check that the from_model and from_json funcs return similar data

* added docstring to graph data test

* Tidy up test docstring.

Co-authored-by: James Batchelor <james.batchelor@atkinsglobal.com>
Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Update docstings and arg names of InterpolatedVolumeParameter and InterpolatedFlowParameter ([#890](https://github.com/pywr/pywr/issues/890))

* update docstring and input arg names of InterpolatedVolumeParameter and InterpolatedFlowParameter

* response to review comments

Co-authored-by: James Batchelor <james.batchelor@atkinsglobal.com>
- *(other)* Add ability to provide other solutions in PywrRandomGenerator. ([#892](https://github.com/pywr/pywr/issues/892))

* Add ability to provide other solutions in PywrRandomGenerator.

This extends the ability of PywrRandomGenerator to return
alternative initial solutions instead of just the current
model configuration. User's provide a list of dictionaries
containing variable values for each of the variable Parameters.

Fixes #887.

* Update docstring of PlatypusRandomGenerator.
- *(other)* Update changelog for 1.7.0 release. ([#894](https://github.com/pywr/pywr/issues/894))

## [1.6.0] - 2020-04-07

### Other

- *(other)* Updated pywr_json_to_d3_json func so that it does not modify a dict when one is passed as an arg ([#833](https://github.com/pywr/pywr/issues/833))
- *(other)* AnnualTotalFlowRecorder can now take a Factor. Test Added. ([#837](https://github.com/pywr/pywr/issues/837))

* AnnualTotalFlowRecorder can now take a Factor. Test Added.

* changes following JET review

* Add factors property to make values accessible from Python.

Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Added UniformDrawdownProfileParameter ([#836](https://github.com/pywr/pywr/issues/836))

* Added UniformDrawdownProfileParameter

This commit adds a new parameter that returns a recurring profile
that drawns down from 1.0 to 0.0 over the course of a year. The day
on which to start the drawdown is configurable with the same keywords
as to AnnualVirtualStorage.

Basic tests included. Also added a more complex AnnualVirtualStorage
test model that uses the new UniformDrawdownProfileParameter as
a control curve in a dynamic cost calculation.

* Fix typo in UniformDrawdownProfileParameter docstring.
- *(other)* Optional factor added to NumpyArrayNode Recorder, default=1.0. Test a… ([#838](https://github.com/pywr/pywr/issues/838))

* Optional factor added to NumpyArrayNode Recorder, default=1.0. Test added and passing

* property removed as not required. Should allow succesful build
- *(other)* Fix the application of scaling factors in AnnualTotalFlowRecorder. ([#840](https://github.com/pywr/pywr/issues/840))

This fixes a bug with the application of scaling factors when there
are multiple nodes.
- *(other)* Add paper reference to README and documentation. ([#846](https://github.com/pywr/pywr/issues/846))

Fixes #761.
- *(other)* Fix citation quote in README.rst
- *(other)* Fix reset of AbstractNode._prev_flow. ([#855](https://github.com/pywr/pywr/issues/855))
- *(other)* Fix a bug calculating of AggregatedStorage's initial volume. ([#854](https://github.com/pywr/pywr/issues/854))

The previous calculation did not reset the max volume calculation
on each loop of the scenarios. Consequently the max volume
accumulated with each scenario, and made the current_pc calculation
incorrect. The bug would only impact the first timestep.
- *(other)* D3 update ([#834](https://github.com/pywr/pywr/issues/834))

* Updated draw_graph.js to use d3.v5

* Add x and y coords to node attribute table + allow node position to be fixed again

* Removed IE specific js code

Co-authored-by: James Batchelor <james.batchelor@atkinsglobal.com>
- *(other)* Add `ControlCurvePiecewiseInterpolatedParameter` ([#857](https://github.com/pywr/pywr/issues/857))

* Add `ControlCurvePiecewiseInterpolatedParameter`

A new parameter the interpolates between multiple control curves in
a piecewise fashion. A pair of values is given for each "band"
between the storage bounds and the control curves. This pair is
used to set linear interpolation based on the current storage.

* Fix docstring typo in ControlCurvePiecewiseInterpolatedParameter
and check the dimensions of `values` inside the property setter
rather than `__init__` function.

* Add DeprecationWarning to PiecewiseLinearControlCurve.
- *(other)* Fix the __init__ method of BreakLink. ([#850](https://github.com/pywr/pywr/issues/850))

Existing behaviour meant that the keywords min_flow, max_flow
and cost were all set to the internal Link. However, these
were then overwritten by the call to super().__init__ and its
default values. Now we pass the keywords (in **kwargs) to be
set by super().__init__ rather than directly in local __init__.

Fixes #642.
- *(other)* Fix issue with using AbstractStorage in control curve parameters. ([#861](https://github.com/pywr/pywr/issues/861))

Removed local type (cdef Storage) in favour of using attribute
directly.
- *(other)* Fix resetting licences to maximum volume by default. ([#860](https://github.com/pywr/pywr/issues/860))

This commit adds a new keyword argument to the internal storage
reset method. The keyword determines whether initial volume or
maximum volume are used to reset the current volume. The default
remains to use initial volume. However, the AnnualVirtualStorage
node uses the keyword to reset to either value based on its own
keyword setting. This gives the user the choice whether the
AVS node should reset to initial or maximum volume. Tests included.

Fixes #859.
- *(other)* Registering ArrayIndexedScenarioParameter ([#863](https://github.com/pywr/pywr/issues/863))

Co-authored-by: Jmiguel17 <[jose.gcabrera@postgrad.manchester.ac.uk]>
- *(other)* Paper examples ([#852](https://github.com/pywr/pywr/issues/852))

* Move two_reservoir example to sub-folder.

* WIP on two reservoir example.

* Completed two reservoir example for paper.

* Initial commit of thames-like example.

* Add timings for before and after.

* Add generate_dataframes methods to TablesRecorder.

* Update analytical test for paper.

* Update Thames example from UGM.

* Improve TablesRecorder.generate_dataframes

Add a test for it too.
- *(other)* Add count_nonzero aggregation function. ([#866](https://github.com/pywr/pywr/issues/866))

* Add count_nonzero aggregation function.

* Cast count_nonzero return to double array.

* Fix it properly this time!

* Add docstring to Aggregator.

* Add Aggregator to API docs.
- *(other)* Release v1.6 ([#858](https://github.com/pywr/pywr/issues/858))

* Update changelog for most recent changes.

* Fix building and publishing docs on tags.

Fixes #831.

* Added `ControlCurvePiecewiseInterpolatedParameter` to API docs.

* Updated changelog for most recent changes.

* Updated changelog for most recent changes. (2)

## [1.5.0] - 2020-01-26

### 🐛 Bug Fixes

- Fixed bug in draw_graph function where position attributes are mistakenly assigned to the node object and modified existing test to check this ([#821](https://github.com/pywr/pywr/issues/821))

### Other

- *(other)* Remove warning ignore ([#803](https://github.com/pywr/pywr/issues/803))
- *(other)* Add additional annual recorders ([#784](https://github.com/pywr/pywr/issues/784))

Added AnnualTotalFlowRecorder, AnnualCountIndexThresholdRecorder and DailyCountIndexParameterRecorder

Closes #491
- *(other)* An daily interpolation support to MonthlyProfileParameter.
- *(other)* Improve docstring and comments for MonthlyProfileParameter.
- *(other)* Pin coverage to <5.0 because of a bug when using --cov-append.
- *(other)* Merge pull request #807 from pywr/interp_montly_profile

Add daily interpolation support to MonthlyProfileParameter.
- *(other)* Added  ScenarioDailyProfileParameter
- *(other)* Add ScenarioWeeklyProfileParameter + update docstring of ScenarioDailyProfileParameter
- *(other)* Review updates
- *(other)* Remove repeated test and add to new parameters to API docs.
- *(other)* Merge pull request #802 from pywr/scenario_profiles

daily and weekly scenario profiles
- *(other)* Some general documentation fixes ([#771](https://github.com/pywr/pywr/issues/771))

* Fix deploying documentation for tags to sub-folders.

* Update documentation link in README to master folder.

* Fix invalid escape charatcers and rendering ASCII diagrams in docstrings.

* Import DivisionParameter into main parameters namespace.

* Use proper pywr package version for documentation.

* Several documentation build and warning fixes.
- *(other)* Refactor of setup.py ([#811](https://github.com/pywr/pywr/issues/811))

* Refactor of setup.py

* Require modern-ish setuptools

* Link lpsolve55

* Don't overwrite existing include_dirs

* Handle include_dirs as string or list

* Build --without-lpsolve on Windows

* Added lpsolve to appveyor
- *(other)* Refactor of GLPK solvers ([#812](https://github.com/pywr/pywr/issues/812))

* Refactor of GLPK solvers

* Use GLPKSolver.__init__

* Don't redefine inf

* Move dump methods to parent class

* Move presolve into parent class

Co-authored-by: James Tomlinson <tomo.bbe@gmail.com>
- *(other)* Added __contains__ to NamedIterator
- *(other)* Added __contains__ to NodeIterator
- *(other)* Add getting node reference and membership testing to tutorial.
- *(other)* Merge pull request #813 from snorfalorpagus/iterator-contains

Added __contains__ method to NamedIterator and NodeIterator
- *(other)* .labels is deprecated, use .codes instead ([#814](https://github.com/pywr/pywr/issues/814))
- *(other)* Fix warning when incrementing timestep by integers ([#815](https://github.com/pywr/pywr/issues/815))
- *(other)* Fix resetting progress of ProgressRecorder. ([#816](https://github.com/pywr/pywr/issues/816))

Fixes #806.
- *(other)* Add missing new line from code-block sections in json.rst ([#817](https://github.com/pywr/pywr/issues/817))
- *(other)* Fix duplicated test function names. ([#818](https://github.com/pywr/pywr/issues/818))

Storage max_volume test function name was repeated. Also tidied
up the docstring for the test functions.
- *(other)* Add Python 3.8 to CI builds. ([#796](https://github.com/pywr/pywr/issues/796))

* Add Python 3.8 to CI builds.

* Py3.8 doesn't have 'm' in directory name.

* Allow runtime dependency install to fail in build-wheels.sh

* Fix DLL resolution on Windows in Python 3.8

* As previous, only better

* Remove jupyter test from appveyor

Co-authored-by: Joshua Arnott <josh@snorfalorpagus.net>
- *(other)* Merge all GLPK solvers into one module. ([#822](https://github.com/pywr/pywr/issues/822))

This commit merges all the GLPK related code into a single
extension module. This addresses an issue ([#819](https://github.com/pywr/pywr/issues/819)) found when
static linking to GLPK used. The GLPK library relies on some global
 memory for its environment. When static linking is used this
global memory is created for each extension module. This breaks the
ability to share GLPK problem pointers (`*glp_prob`) between
extension modules.

Rather than require dynamic linking this commit merges all GLPK
related code to a single module. Therefore static linking can be
used safely.

Fixes #819.
- *(other)* Add .pxd files to package_data in setup.py ([#824](https://github.com/pywr/pywr/issues/824))

* Add .pxd files to package_data in setup.py

Fixes #823.
- *(other)* Release v1.5 ([#830](https://github.com/pywr/pywr/issues/830))

* Add new recorders to API docs.

* Update changelog for v1.5.0

## [1.4.0] - 2019-11-11

### Other

- *(other)* Support timesteps based on pandas offsets ([#675](https://github.com/pywr/pywr/issues/675))

* Support for better timesteps using pandas.

* Added a new dataframe alignment and resampling function.

This function supports up and down sampling (the old one only
support downsampling). It works with pandas.PeriodIndex and
passes the existing tests along with several new ones. A new
ResamplingError exception has been added also.

* Fix the final Timestep to retain existing behaviour.

* Missed some changes to index accessor during rebase.

* Fix Timestep attribute access in glpk-edge solver.

* Make the logic of converting to a PeriodIndex more robust.

* Only calculate datetime from period once on Timestep init.

* Move read_dataframe and load_dataframe to pywr.dataframe_tools

* Add some docstrings in pywr.dataframe_tools

* Clarity fix in pywr.timestepper.

* Rework the dataframe alignment and resampling func and tests.

 - Renamed resample_func keyword arg to down_sample_func to
   better reflect how it is used in the model.
 - Tests now pass different down_sample_func where relevant.
 - Up and Down sampling test classes names corrected.

* Fix bug with resampling date offsets to other date offsets.

Users must provide the appropriate resampling function here.

* Consistent use of @property decorator in timestepper module.

* Added tests and check for zero timestep length.

* Fix description of simple1_monthly test model.

* Define and use constant for seconds in day in timestepper.

* Revert back to use of property for Timestamp.datetime.

Calculation of the datetime from the Period object in all cases
is very slow. We can calculate the required attributes (day, month,
year, etc) directly from the Period object.

Also improved some calls to Timestamp.datetime.year that could
just be Timestamp.year. Mainly in the licences code.
- *(other)* Fix string equality checking in build-wheels.sh ([#769](https://github.com/pywr/pywr/issues/769))
- *(other)* Interpolated Flow Parameter ([#740](https://github.com/pywr/pywr/issues/740))

* Added a generic parameter that interpolates based on flow of a non-storage node

* Added a test for InterpolatedFlowParameter

* Minor style changes to InterpolateFlowParameter and its test.
- *(other)* Added some docs for AggregatedNode ([#756](https://github.com/pywr/pywr/issues/756))

* Added some docs for AggregatedNode
- *(other)* Add support for percentile and percentileofscore aggregation functions. ([#777](https://github.com/pywr/pywr/issues/777))
- *(other)* Add new PiecewiseIntegralParameter. ([#772](https://github.com/pywr/pywr/issues/772))
- *(other)* Initial implementation of including python modules. ([#765](https://github.com/pywr/pywr/issues/765))

This approach simply detects the file extension of filenames
referenced in the existing "includes" section. For .py files
runpy is used to execute the code.
- *(other)* Pop comment from data passed to pandas.read_xxxxx ([#788](https://github.com/pywr/pywr/issues/788))

This prevents passing a "comment" argument to Pandas when reading from an external source.
- *(other)* Add year and ordinal threshold params ([#789](https://github.com/pywr/pywr/issues/789))

* Add support for asserting index values in AssertionRecorder.

* Add CurrentYearThresholdParameter and CurrentOrdinalDayThresholdParameter

These two parameters allow for a comparison against the current year and
ordinal day of the simulation respectively. They are useful for triggering
things at specific dates in the simulation.

* Remove old basestring referrence in _thresholds.pyx
- *(other)* Update Appveyor badge in README.rst ([#795](https://github.com/pywr/pywr/issues/795))

The current badge is pointing to the old snorf organisation on Appveyor. This now points to the pywr-admin organisation which is where the builds are done and linked to in the PR.

@snorfalorpagus you might want to disable the Appveyor builds on your account? I think they are being duplicated at the moment.
- *(other)* Remove use of past ([#778](https://github.com/pywr/pywr/issues/778))

* Remove use of past.basestring.

* Remove future dependency and reference in install docs.

* Remove __future__ imports in tests files.
- *(other)* Update changelog and docs version for release v1.4.0 ([#798](https://github.com/pywr/pywr/issues/798))

## [1.3.0] - 2019-09-11

### Other

- *(other)* Fix pandas keyword argument for read_excel(). ([#747](https://github.com/pywr/pywr/issues/747))
- *(other)* Removed six dependency by using metaclass kwarg for node classes that directly inherit NodeMeta ([#745](https://github.com/pywr/pywr/issues/745))
- *(other)* Allow use of parameters as values in ControlCurveInterpolatedParameter ([#750](https://github.com/pywr/pywr/issues/750))

* Allow use of parameters as values in ControlCurveInterpolatedParameter

This copies the approach of ControlCurveParameter and allows the user
to provide either values as an iterable of floats or parameters as
an iterable of Parameter instances. Each parameter's value is used
during the calculation in the same way as the current list of values.

Tests extended to check this behaviour and support loading from JSON.

* Tidy up use of i variable in ControlCurveInterpolatedParameter.
* Add comment about registering child parameters.
* Add some comments to the interpolation calculation.
- *(other)* Fix loading PiecewiseLink with parameters from JSON. ([#749](https://github.com/pywr/pywr/issues/749))

* Fix loading PiecewiseLink with parameters from JSON.

The load method on PiecewiseLink was broken when given parameters. It
was also a little ambiguous regarding the use of 'max_flow' or 'max_flows',
and 'cost' or 'costs'. This is now made explicit to use the singular
case. That is consistent with the init methods of the class. Plural keys
are no longer supported.

Add a test for loading parameters on to PiecewiseLink sublinks.

* Tidy-up test_piecewise.py.

Remove unused imports, future print function and make pep8 compliant.
- *(other)* Remove GIT_VERSION.txt and pywr.__git_hash__ ([#752](https://github.com/pywr/pywr/issues/752))

The git hash is part of the version number since we started using
setuptools_scm. This commit removes the generation of a separate
hash file GIT_VERSION.txt as part of setup.py

Fixes #739 and #751
- *(other)* Remove Blender node. ([#757](https://github.com/pywr/pywr/issues/757))

Fixes #234.
- *(other)* Add ScenarioWrapperParameter ([#763](https://github.com/pywr/pywr/issues/763))

* Add ScenarioWrapperParameter

A new parameter for quickly using multiple parameter definitions across
a scenario's ensembles. Useful for cases where non-scenario aware
parameters need to be changed with a scenario's ensemble. For example,
reservoir release rules.

* Add missing test model JSON.

* Fix ScenarioWrapperParameter docstring.
- *(other)* Fix CSVRecorder volume bug ([#767](https://github.com/pywr/pywr/issues/767))

* Fix #766.

* Remove conditional Python 2 support in CSVRecorder that is no longer tested.

* Remove Python major version check in CSVRecorder test.
- *(other)* Update versions and changelog for v1.3.0 release. ([#768](https://github.com/pywr/pywr/issues/768))

## [1.2.0] - 2019-06-25

### 🐛 Bug Fixes

- Fixed test warnings ([#695](https://github.com/pywr/pywr/issues/695))

Some fixes to the tests to remove warnings.

* Fixed np.sum warnings in test_aggregated_nodes.py ([#678](https://github.com/pywr/pywr/issues/678))
* Removed deprecated check lower_bounds() and upper_bounds() in test_paramaters.py
* Updated use of deprecated update() method to set_double_variables()
* Removed update, lower_bounds and upper_bounds function in `Parameter` object

### Other

- *(other)* Add support for dataframe data embedded directly in to JSON. ([#700](https://github.com/pywr/pywr/issues/700))

* Add support for dataframe data embedded directly in to JSON.

* Ensure checksum only tested when "url" key given

Raise errors when "filetype" or "checksum" given when loading with embedded data.
- *(other)* Remove conda from CI ([#692](https://github.com/pywr/pywr/issues/692))

 - Drop Conda from CI
 - Build wheels instead
 - Use manylinux1 for Linux builds
 - OSX is no longer tested in CI.
- *(other)* Readd missing doctr secured env variable. ([#707](https://github.com/pywr/pywr/issues/707))
- *(other)* Fix permissions of files created in docker ([#708](https://github.com/pywr/pywr/issues/708))

Fix permissions of files created in docker
- *(other)* Enable codecov based coverage reporting. ([#705](https://github.com/pywr/pywr/issues/705))

* Enable codecov based coverage reporting.
* Add codecov badge.
* Only run coverage and codecov if tracing is on.
- *(other)* Add Dockerfile and wheel building for manylinux2010. ([#710](https://github.com/pywr/pywr/issues/710))
- *(other)* Some work on an area recorder
- *(other)* Refactored the Storage array recorders.

- Area, level and volume recorders share a common base class.
- Brings support for `to_dataframe` to area and level recorders.
- Added docstrings for area and level recorders.
- Added tests for level recorder.
- Test for dataframe creation and temporal aggregation.
- Added area recorder to API docs.
- *(other)* Add area recorder ([#684](https://github.com/pywr/pywr/issues/684))
- *(other)* Node documentation ([#668](https://github.com/pywr/pywr/issues/668))
- *(other)* Update installation documentation.

 - Remove pywr conda channel (conda-forge is now used).
 - Add section for installing using pip from pypi.
- *(other)* Merge pull request #713 from pywr/update_install_docs

Update installation documentation.
- *(other)* Added getter for Recorder.agg_func
- *(other)* Added getter for Aggregated(Index)Parameter.agg_func
- *(other)* Merge pull request #719 from pywr/agg_func_get_set

Added getter for Recorder.agg_func
- *(other)* Refactor of named parameter/recorder loading
- *(other)* Fixed recursive loading of recorders in AggregatedRecorder
- *(other)* Merge pull request #720 from snorfalorpagus/issue674-component-loaders

Refactor of named parameter/recorder loading
- *(other)* Fix a bug with AggregatedRecorder not returning the instance on load.

- Includes associated test for future regression.
- *(other)* Merge pull request #723 from pywr/recursive_agg_recorder_bug

Fix a bug with AggregatedRecorder not returning the instance on load.
- *(other)* Add DivisionParameter and tests. ([#722](https://github.com/pywr/pywr/issues/722))

* Add DivisionParameter and tests.

Fixes #651
- *(other)* Use flow instead of max_flow in two_reservoirs ([#721](https://github.com/pywr/pywr/issues/721))
- *(other)* Add flow parameter for tracking yesterday's flow. ([#724](https://github.com/pywr/pywr/issues/724))

* Add flow parameter for tracking yesterday's flow.

- Includes a test and test model.
- Add configurable initial condition to FlowParameter.
- *(other)* Add InterpolatedQuadratureParameter ([#714](https://github.com/pywr/pywr/issues/714))

* added InterpolatedIntegratedParameter

* Complete InterpolatedQuadratureParameter.

- Renamed to `InterpolatedQuadratureParameter`.
- Added test to check quadrature calculation.
- Added interpolation parameters to docs.
- Fixed bug in docstring of `InterpolatedParameter`.

* Fix line returns in test_parameters.py

* Remove unused values variable in quadrature test.

* Add load method and test for InterpolatedParameter.

* Support assigning lower interval bounds on InterpolatedQuadratureParameter.

- Added docstring with Parameters listed.
- Include lower interval in tests.
- *(other)* Add new array deficit recorders. ([#729](https://github.com/pywr/pywr/issues/729))

* Add new array deficit recorders.

- NumpyArrayNodeDeficitRecorder - timeseries of deficit.
- NumpyArrayNodeSuppliedRatioRecorder - timeseries of supply / demand
- NumpyArrayNodeCurtailmentRatioRecorder - timeseries of 1 - supply / demand
- *(other)* Fix bug with hydropower parameters & recorders not applying efficiency factor. ([#737](https://github.com/pywr/pywr/issues/737))
- *(other)* Update copyright year 2019 ([#736](https://github.com/pywr/pywr/issues/736))
- *(other)* Update versions and changelog for v1.2.0 release. ([#738](https://github.com/pywr/pywr/issues/738))

## [1.1.0] - 2019-02-20

### Other

- *(other)* Enabled cython lanugage_level=3 and embedsignature=True
- *(other)* Fixed local scope issue load method of ControlCurveIndexParameter
- *(other)* Merge pull request #645 from pywr/cython_directives

Additional cython compiler directives.
- *(other)* Updated url in setup.py metadata to GitHub project ([#661](https://github.com/pywr/pywr/issues/661))
- *(other)* Skip test_notebook if missing optional dependencies.] ([#660](https://github.com/pywr/pywr/issues/660))

Use tkagg backend for matplotlib in tests.
New function `get_parameter_from_registry`.

Closes #624.
- *(other)* Type optimisation in AggregatedIndexParameter. Closes #648. ([#662](https://github.com/pywr/pywr/issues/662))
- *(other)* Setuptools_scm is not a runtime dependency.
- *(other)* Added presentation from September meeting at UCL.
- *(other)* Fixed the fdc deviation recorder to work with no scenario ([#677](https://github.com/pywr/pywr/issues/677))

* Fixed the flow duration curve deviation recorder to work with no scenario.

- Now supports passing targets of a single dimension which are used in all scenarios.
- Remove use of np.asarray.
- Regression tests for single targets.
- Regressions tests load from JSON and use no scenario.

* Updated changelog with FDC changes.
- *(other)* Enable Python 3.7 CI for Linux and Windows.
- *(other)* Migrate numpy and vc to conda build 3 to do auto version pinning.
- *(other)* Add anaconda cxx compiler.
- *(other)* Revert move to conda build 3.
- *(other)* Try auto pinning numpy but not using the new compilers.
- *(other)* Update appveyor.yml to use conda-build 3 setup script.
- *(other)* Add snappy dependency to work around pytable feedstock bug.
- *(other)* Replace deprecated calls time time.clock with time.perf_counter.

Fixes #683.
- *(other)* Merge pull request #643 from pywr/py37

Python 3.7
- *(other)* Initial commit of an edge based solver using GLPK.
- *(other)* Revert to change to test_aggregated_node_max_flow_same_route and mark skip for glpk-edge solver.
- *(other)* Enabled testing glpk-edge solver in conda recipe.
- *(other)* Replace deprecated time.clock with time.perf_counter in GLPK edge solver.
- *(other)* Remove debug print statements in GLPK edge solver.
- *(other)* Fix bug with aggregated node in GLPK edge solver.
- *(other)* Review responses for the GLPK edge based solver.
- *(other)* Variable renames and clean up for GLPK path solver.
- *(other)* Merge pull request #672 from pywr/cy_glpk_edge

Initial commit of an edge based solver using GLPK.
- *(other)* Some documentation ([#652](https://github.com/pywr/pywr/issues/652))


* Updated copyright year. Fixes #581.
* Refer to "demand restrictions" rather than "demand savings" in the docs. Fixes #541.
* Additional documentation: control curves
* Additional documentation: Extending Pywr - custom parameters & nodes.
- *(other)* Update CHANGELOG.md
- *(other)* Release v1.1.0

## [1.0.0] - 2018-09-19

### Other

- *(other)* Version bump to v0.6dev
- *(other)* New and improved variable API for Parameters. ([#601](https://github.com/pywr/pywr/issues/601))

* New and improved variable API for Parameters.

This commit changes the API for getting and setting Parameter variables. It implements the changes discussed in #258. The changes have been made to all existing Parameters that supported the old API.
- *(other)* Adding functionality to notebook graph function labels ([#612](https://github.com/pywr/pywr/issues/612))

- Updated notebook functionality to show labels and extract node data.
- Added notebook API to docs.
- *(other)* Added flow weights to AggregatedNode constraints. ([#603](https://github.com/pywr/pywr/issues/603))

Added flow weights to AggregatedNode constraints.
- *(other)* Added flow weights to changelog
- *(other)* Use new pandas API that has been changed in v0.23 ([#620](https://github.com/pywr/pywr/issues/620))
- *(other)* Allow specifying solver name via PYWR_SOLVER environment variable. ([#619](https://github.com/pywr/pywr/issues/619))

- Env variable is used if solver is not given directly.
- If neither given it retains the current default behaviour.
- Removed the "--solver" argument from pytest and the solver fixture.
- Modified all the tests that depended on this fixture.
- Modified the conda recipe to specify solver by environment variable.

Fixes #618.
- *(other)* Add PYWR_SOLVER env variable to changelog.
- *(other)* Fix a bug in the AnnualHarmonicSeriesParameter that meant amplitude and phase were not updated correctly.
- *(other)* Add variable test for AnnualHarmonicSeriesParameter.
- *(other)* Couple more fixes to the AnnualHarmonicSeriesParameter.
- *(other)* Added #622 to CHANGELOG
- *(other)* Drop CI support for Python 2.7 and 3.4. ([#623](https://github.com/pywr/pywr/issues/623))

Good bye Pyt.on 2.7. You have served us well!

Updates README, docs and CI files.

Fixes #572.
- *(other)* Update CHANGELOG with dropped Python<3.6 support.
- *(other)* Merge branch 'master' into fix_harmonic_var
- *(other)* Merge pull request #622 from pywr/fix_harmonic_var

Fix a bug in the AnnualHarmonicSeriesParameter
- *(other)* Removed CachedParamter defition ([#626](https://github.com/pywr/pywr/issues/626))

Closes #594.
- *(other)* Add tutorial.json to docs ([#627](https://github.com/pywr/pywr/issues/627))

Closes #615.
- *(other)* Fix the math problems in the hydropower docstrings. ([#616](https://github.com/pywr/pywr/issues/616))
- *(other)* Fixed bug in Parameter returning the wrong bounds for deprecatd API. ([#625](https://github.com/pywr/pywr/issues/625))
- *(other)* Updated to use new networkx selfloops function ([#628](https://github.com/pywr/pywr/issues/628))
- *(other)* Better warning/error messages in TablesArrayParameter ([#629](https://github.com/pywr/pywr/issues/629))

* Modified data size warnings to state size of data vs scenario size and number of time steps
- *(other)* Update CHANGELOG.md
- *(other)* Added flake8 config and fix some warnings ([#634](https://github.com/pywr/pywr/issues/634))

* Added setup.cfg with flake8 config

* flake8 fixes for solvers

* flake8 fix for _core and _model

* Ignore "module level import not at top of file"

It doesn't understand cimport.
- *(other)* Add new HydroPowerTargetParameter ([#631](https://github.com/pywr/pywr/issues/631))

* Add new HydroPowerTargetParameter

This parameter can be used to calculate flow given an energy target. See docstring.

Test included and modifications to existing hydropower recorders docstring to reference this new one. Added a new section to parameters API docs for hydropower.

* Missed test model.

* Don't rely on testing floating point numbers exactly.

* Use Hydropower instead of HydroPower for class names.

* Remove hyphen in docs.
- *(other)* Removed deprecated (Monthly|Daily)ProfileControlCurve. Closes #231. ([#640](https://github.com/pywr/pywr/issues/640))

* Removed deprecated (Monthly|Daily)ProfileControlCurve. Closes #231.

* Update CHANGELOG.md
- *(other)* Add platypus optimisation support.

- Implements a new platypus model wrapper.
- Refactors the optimisation example to use either library.

The refactored approach wraps the platypus.Problem and creates a module
level cache of the Model for sub-processes to use. These module level values
are accessed through properties on the `BaseOptimisationWrapper`. This is parent
class shared by both the platypus and inspyred wrappers.

Also added tests for running an algorithm with a ProcessPoolEvaluator.
- *(other)* Add optimisation support for pygmo.

- Extends the two reservoir example to include pygmo.
- Adds basic test for pygmo wrapper.

Fixes #621.
- *(other)* Refactor the optimisation tests.

The platypus and pygmo tests are in separate modules. These are skipped if the respective
package can not be imported.
- *(other)* Added pygmo as test requirement in the conda build recipe.
- *(other)* Removed some erroneous import statements in pywr.optimisation
- *(other)* Fix Python 2 issue with class definition.
- *(other)* Merge branch 'master' into platypus2
- *(other)* Initial support for constraints in pygmo wrapper.
- *(other)* Add platypus-opt to testing section now it is in conda-forge.
- *(other)* Platypus integration to use new variable API.
- *(other)* Couple of bug fixes to platypus wrapper.
- *(other)* Force garbage collection in platypus tests.
- *(other)* Platypus requires custom variators for mixed type problems.

For the time being cast integer variables to reals. This is the approach used in the other wrappers.
- *(other)* Use a UID for model caching in pywr.optimisation.
- *(other)* Correct docstring and unused variable in inspyred wrapper.
- *(other)* Simplify the two reservoir MOEA example.

- Remove plotting routines.
- Mention all optimisation libraries in module docstring.
- *(other)* Merge pull request #610 from pywr/platypus2

Add platypus and pygmo optimisation support.
- *(other)* Add snappy dependency to work around pytable feedstock bug.
- *(other)* Added two new parameter recorders.

- TotalParameterRecorder
- MeanParameterRecorder

Including tests and docs.
- *(other)* Refactored some of the Recorder classes.

- Use a shared Aggregator class for computing aggregation across scenarios and time.
- Modify existing classes to use this shared aggregator.
- Made the naming of the temporal agg function keyword consistent across several classes.
- Added support for temporal aggregation and .values() method on the NumpyArrayXXXRecorder classes.
- *(other)* Added tests for Aggregator.

 - Removed ANY and ALL aggregation from recorders enum as they were not used.
 - Made aggregate_1d and aggregate_2d methods raise exceptions.
- *(other)* Some additions and corrections to recorder docstrings.
- *(other)* Fixed docstring typo.
- *(other)* Improved AggregatedRecorder docstring and added to API docs.
- *(other)* Fix bug in NumpyArrayParameterRecorder temporal aggregation.
- *(other)* Make aggregator accessible from Python space.
- *(other)* Add optional proportional keyword to NumpyArrayStorageRecorder.
- *(other)* Merge pull request #635 from pywr/param_recorders

Refactor of several recorders
- *(other)* Added ConstantScenarioIndexParameter.
- *(other)* Merge pull request #654 from pywr/const_scen_index_param

Added ConstantScenarioIndexParameter.
- *(other)* Removed finally caluse in _cached property of BaseOptimisationWrapper

There are instances where the `cache` variable might not be assigned. Best to remove the finally clause entirely.
- *(other)* Allow specifying a UID in the optimisation wrapper.
- *(other)* Add a reference to the wrapper in the platypus wrapper.
- *(other)* Merge pull request #649 from pywr/opt_rm_finally

Removed finally caluse in _cached property of BaseOptimisationWrapper
- *(other)* Adding area attribute ([#657](https://github.com/pywr/pywr/issues/657))

* Added area attribute to Storage class

* Added assert check for area attribute in test_reservoir_surface_area

* Small style change to match level keyword/attr
- *(other)* Add ratchet support to threshold parameters. ([#655](https://github.com/pywr/pywr/issues/655))

* Add ratchet support to threshold parameters.
* Fix bug with ratchet control not supporting scenarios.
* Use uint8 instead int memory view.
- *(other)* Update changelog with latests changes.
- *(other)* Use setuptools_scm for versioning.
- *(other)* Update conda-recipe version to 1.0.0
- *(other)* Update changelog version to 1.0.0
- *(other)* Adding missing setuptools_scm to conda-recipe.

## [0.5.1] - 2018-05-04

### Other

- *(other)* Version bump to 0.6dev0
- *(other)* Update Changelog for v0.5
- *(other)* Update Appveyor Miniconda version for 3.6

The current location (`C:\\Miniconda35-x64`) was out of date and an old version.

Use conda-forge-build-setup.

Removed some cruft from appveyor.yml
- *(other)* Request Cython<0.28 until upstream bug resolved.

See https://github.com/cython/cython/issues/2152
- *(other)* Revert "Request Cython<0.28 until upstream bug resolved."

This reverts commit e6e8fc6ca1d8084eb7d51cf9abed1dc0bc4d0a55.
- *(other)* Merge pull request #609 from pywr/cy0.28.1

Revert "Request Cython<0.28 until upstream bug resolved."
- *(other)* Fix setup.py and add MANIFEST.in ([#614](https://github.com/pywr/pywr/issues/614))

* Changes to make the sdist work correctly.
 - Adds MANIFEST.in
 - Changes setup.py to work without git (correctly catching the exception).
 - Adds long_description based on README.rst

* Added support for using environment variables to define solver build options.
* Use conda recipe to build sdist and bdist_wheel.
* Add pypi and anaconda deployment to .travis.yml.
- *(other)* Release v0.5.1
- *(other)* Add anaconda-client to .travis.yml
- *(other)* Remove deployment stuff from travis and conda recipe.
- *(other)* Add missing requirements (packaging) to setup.py

## [0.5] - 2018-02-26

### Other

- *(other)* Version bump to 0.5dev0
- *(other)* The threshold in *ThresholdParameter can now be a Parameter. ([#517](https://github.com/pywr/pywr/issues/517))

* The threshold in *ThresholdParameter can now be a Parameter.

Closes #516.
- *(other)* Improved the FDC tests by checking my aggregation functions.
- *(other)* Merge pull request #496 from pywr/better_fdc_tests

Improved the FDC tests by checking my aggregation functions.
- *(other)* WIP on hydropower example.
- *(other)* Some updates to the hydropower example.
- *(other)* Removed superfluous print.
- *(other)* Merge pull request #493 from pywr/hydropower_example

WIP: Hydropower example
- *(other)* Revert GLPK default log level to "off" ([#518](https://github.com/pywr/pywr/issues/518))
- *(other)* Load parameters for PiecewiseLink in JSON. ([#519](https://github.com/pywr/pywr/issues/519))

Closes #499.
- *(other)* Added ProgressRecorder and JupyterProgressRecorder ([#520](https://github.com/pywr/pywr/issues/520))

* Added ProgressRecorder and JupyterProgressRecorder

Closes #509.

* Added speed info to JupyterProgressRecorder

* Fixed test for ProgressRecorder.
- *(other)* Second attempt at fixing #515. ([#523](https://github.com/pywr/pywr/issues/523))
- *(other)* Initial commit of storage surface area property ([#525](https://github.com/pywr/pywr/issues/525))
- *(other)* Recursive deletion of child nodes when deleting compound nodes. ([#527](https://github.com/pywr/pywr/issues/527))

Closes #166.
- *(other)* Compatibility with NetworkX 2.x ([#529](https://github.com/pywr/pywr/issues/529))
- *(other)* Fixed dictionary modification during iteration when recursively deleting nodes.
- *(other)* Merge pull request #531 from pywr/nx2_del

Fixed dictionary modification during iteration
- *(other)* Renamed InterpolatedLevelParameter to InterpolatedParameter

Fixes #526
- *(other)* Renamed InterpolatedParameter argument from levels to values.
- *(other)* Refactored this again ready for more subclasses.
- *(other)* Merge pull request #534 from pywr/fix526

Renamed InterpolatedLevelParameter to InterpolatedParameter
- *(other)* Fix to the csv recorder as it is assumed that the nodes objects are provided but when nodes names are provided (e.g. loaded from the json), it will throw an error
- *(other)* Drop try and excepts updating self.nodes 

changing nodes_names to be private
- *(other)* Fix issue of checking node type for CSV recorder

Modify the code to checking against AbstractNode and AbstractStorage instead of node and storage
- *(other)* Adding test loading csv recorder from json
- *(other)* CSV_Recorder.json 

Adding json file to be used with test loading csv recorder from json
- *(other)* A few tweaks to the CSVRecorder and associated tests

- Made the test load and save to temporary folder.
- Fixed tabs to spaces in the JSON file.
- Changed JSON format to use "url" key instead of "csvfile"
- *(other)* Merge pull request #530 from khaledk2/patch-1

Fix to the csv recorder as it is assumed that the nodes objects are p…
- *(other)* Added optional checksum when loading DataFrameParameter and TablesArrayParameter using hashlib.
- *(other)* Merge pull request #522 from pywr/checksum

Added optional checksum
- *(other)* Added InterpolationParameter. Closes #478. ([#535](https://github.com/pywr/pywr/issues/535))

* Added InterpolationParameter. Closes #478.
* Fixed docstring typo
- *(other)* IndexedArrayParameter also accepts 'parameters' in JSON. Closes #538. ([#539](https://github.com/pywr/pywr/issues/539))
- *(other)* Added .gitattributes for timeseries2.csv. Closes #536. ([#540](https://github.com/pywr/pywr/issues/540))
- *(other)* Added WeeklyProfileParameter ([#537](https://github.com/pywr/pywr/issues/537))

* Added WeeklyProfileParameter
* Accept 53-week profile, but drop last value with a warning
- *(other)* Make py.test in conda recipe verbose.
- *(other)* Added scale and offset to ConstantParameter.
- *(other)* Merge pull request #543 from pywr/constant_scale_offset

Added scale and offset to ConstantParameter.
- *(other)* Travis now caches miniconda folder
- *(other)* Drop testing of Python 3.4 on Appveyor ([#548](https://github.com/pywr/pywr/issues/548))

conda-forge no longer produces binaries for 3.4 [vc10] which makes testing difficult.
- *(other)* Added JSON support for license parameters. ([#544](https://github.com/pywr/pywr/issues/544))

* Added JSON support for license parameters.
* Improved annual licence test
- *(other)* Added design goals to README

Partially addresses #545
- *(other)* Added GZ/BZ2 compression for CSVRecorder. Closes #532
- *(other)* Fix for bz2 csv in Python 2.x
- *(other)* 2nd attempt at bz2 csv fix
- *(other)* Fixed Python 2 issue with BZ2 compression.
- *(other)* Another python 2 fix for gzip mode
- *(other)* Added unicode character test to CSV recorder test.
- *(other)* Fixed file mode in the CSV test for Python 2&3 compat.
- *(other)* Encode node names in utf-8 for csv recorder
- *(other)* Some tweaks to unicode handling.
- *(other)* Python 2 doesn't support encoding keyword for file opening.
- *(other)* Merge pull request #542 from pywr/csv_compress

Added GZ/BZ2 compression for CSVRecorder. Closes #532
- *(other)* Support loading intial_volume_pc from JSON.
- *(other)* Merge pull request #561 from pywr/init_vol_pc_json

Support loading initial_volume_pc from JSON.
- *(other)* Added a sense check to TablesArrayParameter having non-finite values.
- *(other)* Add parameter statistic tracking to EventRecorder

 - Adds the ability for Event objects to hold a simple list of values from
   a Parameter for the duration of the event.
   - Currently only supports Parameters and not other components or nodes.
 - New EventStatisticRecorder to compute arbitrary aggregation of the
   tracked values and return via .values() and .aggregated_value()
   methods.
 - Test of the new object and functionality.
- *(other)* EventRecorder now saves value of track parameter to df ([#550](https://github.com/pywr/pywr/issues/550))
- *(other)* EventRecorder now saves value of track parameter to df
- *(other)* Responses to merge request comments
- *(other)* Deleted after method in EventDurationRecorder
- *(other)* Merge pull request #551 from Batch21/better_events

Better events
- *(other)* Updated event class docstrings with *agg_func descriptions.
- *(other)* Merge pull request #547 from pywr/better_events

 Add parameter statistic tracking to EventRecorder
- *(other)* Set freq explicitly in align_and_resample. Closes #563.
- *(other)* Merge pull request #564 from pywr/df_resample_freq

Set freq explicitly in align_and_resample. Closes #563.
- *(other)* Minor fixes from lgtm.com
- *(other)* WIP on implementing logging calls.
- *(other)* Python 2.7 fix for the arguments to ProgressRecorder.
- *(other)* Merge remote-tracking branch 'origin/logging'
- *(other)* Use conda-forge lp_solve
- *(other)* Added features to Windows conda recipe
- *(other)* So long numpy x.x
- *(other)* Vc features
- *(other)* Merge pull request #567 from pywr/lpsolve_cf

Use conda-forge lp_solve
- *(other)* Added PiecewiseLinearControlCurve
- *(other)* Fixed py2 division error and added parameter_property
- *(other)* Merge pull request #566 from pywr/new_control_curve

Added PiecewiseLinearControlCurve
- *(other)* Better docs ([#574](https://github.com/pywr/pywr/issues/574))

* Added doctr for automatic doc builds

* Added doctr for automatic doc builds (2)

* Added doctr for automatic doc builds (3)

* Added doctr for automatic doc builds (4)

* Added doctr for automatic doc builds (5)

* Added doctr for automatic doc builds (6)

* Added doctr for automatic doc builds (7)

* Added doctr for automatic doc builds (8)

* Added doctr for automatic doc builds (9)

* WIP on improving the docs. Especially the parameters.

* Added missing parameter_property.py module.

* Further WIP on the documentation.

* Added missing dependencies when building docs.

* Reconfigure the documentation build system.

* Reconfigure the documentation build system (1)

* Reconfigure the documentation build system (2)

* Reconfigure the documentation build system (3)

* Reconfigure the documentation build system (4)

* Reconfigure the documentation build system (5)

* Reconfigure the documentation build system (6)

* Reconfigure the documentation build system (7)

* Reconfigure the documentation build system (8)

* Reconfigure the documentation build system (9)
- *(other)* Cythonised `ControlCurveParameter`.
- *(other)* Ported PiecewiseLinearControlCurve to cython
- *(other)* PiecewiseLinearControlCurve accepts abstract storage
- *(other)* Merge branch 'new_cc_pyx' into combined_pyx_cc
- *(other)* Typed _interpolate function and flagged to use C style division.
- *(other)* Merge pull request #571 from pywr/combined_pyx_cc

Combined pyx cc
- *(other)* Added warnings to DataFrameParamter and TablesArrayParameter regarding unutilised data.
- *(other)* Merge pull request #578 from pywr/fix562

Added warnings to DataFrameParamter and TablesArrayParameter
- *(other)* Added gitter badge to README
- *(other)* Merge pull request #577 from pywr/gitter-badge

Added gitter badge to README
- *(other)* Added DeprecationWarning to .
- *(other)* Merge pull request #575 from pywr/fix231

Added DeprecationWarning to AbstractProfileControlCurveParameter
- *(other)* Added test for issue #380. Marked as xfail.
- *(other)* Merge pull request #576 from pywr/test380

Added test for issue #380. Marked as xfail.
- *(other)* Enabled OSX builds on travis ([#589](https://github.com/pywr/pywr/issues/589))

* Added OSX to travis.yml for Py3.6 only.
* Change travis matrix to remove BUILD_DOC=1 on OSX
* Missed the "include" subsection in "matrix"
* Install correct version of Miniconda for OS X
* Added MACOSX_DEPLOYMENT_TARGET=10.9 to travis.yml
This was to fix an error in the previous build :
`error: $MACOSX_DEPLOYMENT_TARGET mismatch: now "10.7" but "10.9" during configure`
- *(other)* Created a changelog
- *(other)* Merge pull request #590 from pywr/changelog

Created a changelog
- *(other)* Improved the mean flow recorders.

- Current MeanFlowRecorder renamed to RollingMeanFlowNodeRecorder
  - It is only a rolling mean recorder, and it is meant for nodes.
  - This is a better overall description.
- Added a MeanFlowNodeRecorder that is a simple mean of flow through an
  entire model run.
  - This doesn't give instantaneous mean flow part way through a run
    however.
- Added both to the docs.
- *(other)* Removed unneeded variable in MeanFlowNodeRecorder.
- *(other)* Merge pull request #592 from pywr/mean_rec

Improved the mean flow recorders.
- *(other)* Load parameters after instance created when loading Storage.

This should prevent circular errors on parameter loading.
- *(other)* Ensure that area parameter is loaded correctly.
- *(other)* Added a temporary test to show how evaporation works.
- *(other)* Merge pull request #587 from pywr/fix585only

Make area property load correctly in Storage
- *(other)* Suppress UnutilisedDataWarning when running the tests.
- *(other)* Merge pull request #597 from pywr/pytest_warn

Suppress UnutilisedDataWarning when running the tests.
- *(other)* Moved the HydroPowerRecorder in to pywr.recorders. ([#584](https://github.com/pywr/pywr/issues/584))

* Moved the HydroPowerRecorder in to pywr.recorders.

* Fixed HydroPowerRecorder tests.

* Updated documentation

- Added HydroPowerRecorder and improved its docstring.
- Added CSVRecorder and edit its docstring to by numpy compliant.
- Added TablesRecorder.

* Refactored hydropower recorder.

- It is now in a separate cython model "pywr.recorders._hydropower"
- There are two hydropower related recorders,
  1. HydroPowerRecorder for storing timeseries data in memory.
  2. TotalHydroEnergyRecorder for storing hydropower energy.

* Added  TotalHydroEnergyRecorder to recorders API docs

* Addressed bugs and comments in hydropower recorders and example.

* Extended HydroPowerRecorder tests from JSON to assert values.
- *(other)* Release v0.5

<!-- generated by git-cliff -->
