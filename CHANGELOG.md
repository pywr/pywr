# Changelog

All issue numbers are relative to https://github.com/pywr/pywr/issues unless otherwise stated.

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
