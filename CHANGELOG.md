# Changelog

All issue numbers are relative to https://github.com/pywr/pywr/issues unless otherwise stated.

## v1.5.0

### New Features

- Added `ScenarioDailyProfileParameter` and `ScenarioWeeklyProfileParameter` to provide different profiles per scenario. (#802)
- Added `TimestepCountIndexParameterRecorder`, `AnnualCountIndexThresholdRecorder` and `AnnualTotalFlowRecorder` (#784)
- Added daily interpolation support to `MonthlyProfileParameter`. (#807)
- Added `__contains__` method to `NamedIterator` and `NodeIterator` (#813) 

### Bug fixes

- Fix resetting progress of `ProgressRecorder` (#816)
- Fix for `draw_graph` issue error when loading model object that has schematic positions (#821)

### Miscellaneous

- Removed `FutureWarning` and `UnicodeWarning` warning filters (#803)
- Refactored `setup.py` to improve build time dependency handling and specifying build arguments (#811)
- Fix deprecated use of `.labels` in the tests (#814) 
- Fix test warning when incrementing timestep by integers (#815)
- Fix duplicated test function names (#818)
- Support Python 3.8 (#796)
- Refactored the GLPK solvers in to a single extension module. (#822)
- Add `.pxd` files to the Pywr's package data so they are distributed. (#824)

### Documentation

- Fixed some warning and deployment issues with the documentation (#771) 
- Add missing new line from code-block sections in `json.rst` (#817)

## v1.4.0

### New Features

- Added support time-steps based on Pandas offsets (#675)
- Added `InterpolatedFlowParameter` (#740)
- Added support for `percentile` and `percentileofscore` aggregation functions (#777)
- Added `PiecewiseIntegralParameter` (#772)
- Added support for including references to Python modules in JSON format (#765)
- Added `CurrentYearThresholdParameter` and `CurrentOrdinalDayThresholdParameter` parameters (#789)

### Bug fixes

- Ensure `comment` key doesn't get passed to Pandas `read_xxx` functions (#788)

### Documentation

- Added some docs for AggregatedNode (#756)

## v1.3.0

### New Features

- Allow use of parameters as values in `ControlCurveInterpolatedParameter` (#750)
- Added `ScenarioWrapper` parameter (#763)

### Bug fixes

- Fix loading PiecewiseLink with parameters from JSON (#749)
- Fixed a bug with `CSVRecorder` not saving volumes correctly (#767)

### Miscellaneous

- Removed `six` as dependency (#745)
- Removed `pywr.__git_hash__` (#752)
- Removed `Blender` node (#757)

## v1.2.0

### New Features

- Support for embedding dataframe data directly in to JSON (#700)
- Added `NumpyArrayAreaRecorder` and refactored storage recorders (#684)
- Added getter for Recorder.agg_func (#719)
- Add `DivisionParameter` and tests (#722)
- Add `FlowParameter` for tracking yesterday's flow (#724)
- Add `InterpolatedQuadratureParameter` (#714)
- Add new array deficit recorders (#729):
  - `NumpyArrayNodeDeficitRecorder` - timeseries of deficit.
  - `NumpyArrayNodeSuppliedRatioRecorder` - timeseries of supply / demand
  - `NumpyArrayNodeCurtailmentRatioRecorder` - timeseries of 1 - supply / demand

### Bug fixes

- Fix a bug with hydropower parameters & recorders not applying efficiency factor (#737)
- Refactor of the code used to load named parameters and recorders to use shared functions (as they are both components) (#720)
- Fix a bug with AggregatedRecorder not returning the instance on load (#723)
- Use `flow` instead of `max_flow` in two_reservoirs test and example (#721)

### Documentation

- Added API documentation for nodes (#668)
- Fix `PiecewiseLink` docstring's ASCII diagram (#668)

### Miscellaneous

- Clean-up various warnings in tests (#695)
- Removed conda-recipe (Pywr is now in conda-forge) (#692)
- Added codecov based coverage reporting (#705)
- Updated test builds to use manylinux2010 to build wheels (#710)
- Updated installation instructions to reflect wheels in Pypi and conda-forge installation.


## v1.1.0

### New features

- New "edge based" GLPK solver. (#672)
- Improved `FlowDurationDeviationRecorder` with JSON support and bug fixes when no scenario is given. (#677)

### Bug fixes

- Replace deprecated calls time time.clock with time.perf_counter. (#683)
- Type optimisation in AggregatedIndexParameter. (#662)

### Documentation 

- Updated documentation: control curves, extending Pywr. (#652)

### Miscellaneous

- Variable renames and clean up for GLPK path solver. (#672)
- Spport for Python 3.7. (#662)
- Updated url in setup.py metadata to GitHub project (#661)
- Additional cython compiler directives (#645)

## v1.0.0

### New features

- Added ratchet support to threshold parameters. (#655)
- Added `ConstantScenarioIndexParameter`. (#654)
- Added support for Platypus and Pygmo optimisation wrappers. Involved a refactor of the existing optimisation support. (#610)
- Added `HydropowerTargetParameter` to specify a flow target from a hydropower target. (#631)
- Renamed `HydroPowerRecorder` to `HydropowerRecorder` (#631)
- Better warning/error messages in `TablesArrayParameter` (#629)
- Allow solver to be defined by the environment variable `PYWR_SOLVER`. (#619)
- Added flow weights to `AggregatedNode`. (#603)
- Added additional labeling functionality to notebook graphing functions. (#612)
- New and improved variable API for Parameters. (#601, #258, #625)

### Bug fixes

- Fix bug setting the area property of `Storage` nodes. (#657)
- Fixed bug with finally clause in the optimisation wrapper. (#649)
- Fix a bug in `AnnualHarmonicSeriesParameter` related to updating the `amplitudes` and `phases` values with `set_double_variables` (#622)

### Miscellaneous

- Refactored several recorders and unified the use of temporal aggregation. This deprecated several keyword arguments in some existing Recorders. See the PR for details. (#635)
- Removed deprecated (Monthly|Daily)ProfileControlCurve. (#640)
- Dropped support for Python 2 and <3.6. Pywr is no longer tested against Python versions earlier than 3.6. (#623)
- Use new `networkx.nodes_with_selfloops` function. (#628)
- `AbstractProfileControlCurveParameter`, `MonthlyProfileControlCurveParameter` and `DailyProfileControlCurveParameter` have been removed after deprecation. (#231, #640)
- Improved documentation. (#616, #627)


## v0.5.1

### Miscellaneous

- Fixes to the source distribution (sdist) and inclusions of MANIFEST.in
- Changes to the build systems on travis to enable deploy to Anaconda and Pypi.

## v0.5

### New features

- Added build and testing for OS X via travis. (#588)
- Added data consumption warnings to DataFrameParamter and TablesArrayParameter. (#562)
- Added `PiecewiseLinearControlCurve` as a new cython parameter.
- Pywr now emits some logging calls during a model run. 
- Improved the event handling code to allow tracking of a `Parameter`'s value during an event.
- Added support for initialising storage volume by percentage. Can be set in json through the `"initial_volume_pc"` property.
- Added GZ2 and BZ2 compression support to `CSVRecorder`.
- Added JSON support for license parameters. (#544)
- Added a sense check to `TablesArrayParameter` having non-finite values.
- Added scale and offset to ConstantParameter.
- Added JSON support for license parameters. (#544)
- Added `WeeklyProfileParameter` (#537)
- Added `InterpolationParameter`. Closes #478. (#535)
- Added surface area property to `Storage` (#525)
- Added optional checksum when loading DataFrameParameter and TablesArrayParameter using hashlib.
- Added ProgressRecorder and JupyterProgressRecorder (#520)
- The threshold in `*ThresholdParameter` can now be a Parameter. (#517)
- Added `HydroPowerRecorder` and `TotalHydroEnergyRecorder` for hydropower studies (#584)

### Bug fixes
- Explicitly set the frequency during dataframe resampling (#563)
- `IndexedArrayParameter` also accepts 'parameters' in JSON. Closes #538. (#539)
- Recursive deletion of child nodes when deleting compound nodes. (#527)
- Compatibility with NetworkX 2.x (#529)
- Changed GLPK log level to remove printing of superfluous messages to stdout. (#523)
- Fixed loading parameters for `PiecewiseLink` in JSON. (#519)

### Deprecated features
- `AbstractProfileControlCurveParameter` marked for deprecation.

### Miscellaneous
- Improved the online documentation including the API reference. 
- Added a hydropower example. 
- General tweaks and corrections to class docstrings.
- Updated conda build recipe to use the conda-forge lp_solve package.
- Updated the conda build recipe to use MSVC features.
