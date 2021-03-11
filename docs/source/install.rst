Installing Pywr
===============

Pywr should work on Python 3.7 (or later) on Windows, Linux or OS X.

Building Pywr from source requires a working C compiler. It has been built successfully with MSVC on Windows, GCC on Linux and clang/LLVM on OS X.

The easiest way to install Pywr is using the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. See instructions below on `Installing binary packages with Anaconda`_.

Dependencies
------------

Pywr has several external dependencies, listed below.

 * Core dependencies (required)

   * `Cython <http://cython.org/>`_

   * `NumPy <http://www.numpy.org/>`_

   * `NetworkX <https://networkx.github.io/>`_

   * `Pandas <http://pandas.pydata.org/>`_

   * `packaging <https://pypi.python.org/pypi/packaging>`_

 * Linear programming solvers (at least one required)

   * `GLPK <https://www.gnu.org/software/glpk/>`_ (recommended)

   * `lpsolve <http://lpsolve.sourceforge.net/5.5/>`_

 * Optional dependencies (providing additional functionality)

   * `pytables <http://www.pytables.org/>`_

   * `xlrd <https://pypi.python.org/pypi/xlrd>`_

   * `pytest <http://pytest.org/latest/>`_ (for testing only)

   * `SciPy <http://www.scipy.org/>`_

   * `Jupyter <https://jupyter.org/>`_

   * `Matplotlib <http://matplotlib.org/>`_

Installing (in general)
-----------------------

When installing Pywr you must specific which solvers to build. This is done by passing ``--with-<solver>`` as an argument to ``setup.py``. The following command will build and install Pywr with both the GLPK and lpsolve solvers:

.. code-block:: shell

  python setup.py install --with-glpk --with-lpsolve

To install Pywr in-place in developer mode, use the ``develop`` command instead of ``install``. This is only useful if you plan to modify the Pywr source code and is not required for general use.

.. code-block:: shell

  python setup.py develop --with-glpk --with-lpsolve

Installing binary wheels with pip
---------------------------------

Binary wheel distributions of Pywr are hosted on `Pypi <https://pypi.org/project/pywr/>`_ for Windows and Linux.

.. code-block:: shell

  pip install pywr

Installing binary packages with Anaconda
----------------------------------------

A binary distribution of Pywr is provided for 3.7+ (64-bit) on Windows, Linux and OS X for the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. Note that this release may lag behind the development version.

You will need to install and configure Anaconda before proceeding. The `conda 30-minute test drive <http://conda.pydata.org/docs/test-drive.html>`_ is a good place to start.

Pywr is hosted on the conda-forge channel.

.. code-block:: shell

  conda config --add channels conda-forge
  conda install pywr

Installing from source with Anaconda
------------------------------------

It's possible to install the dependencies as Anaconda packages, but still build from source. This is only required if you want to keep up with development versions, rather than using the binaries done for releases. In this case you need to specify the include and library paths in your environment as the libraries will be installed in a non-standard location. This can be done by passing the relevant flags to setup.py. As an example, the following batch script should work on Windows (with a similar approach taken on Linux/macOS)

.. code-block:: shell

  set LIBRARY=%CONDA_PREFIX%\Library
  set LIBRARY_INC=%LIBRARY%\include
  set LIBRARY_LIB=%LIBRARY%\lib
  python setup.py build_ext -I"%LIBRARY_INC%" -L"%LIBRARY_LIB%" --inplace --with-glpk --with-lpsolve install

Installing on Windows
---------------------

To build Pywr from source on Windows you must first install and configure the MSVC compiler. See the `instructions on this blog <https://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/>`_. It is important that you install the correct version of MSVC to correspond with your Python version.

Binaries for GLPK are available from the `WinGLPK project <http://winglpk.sourceforge.net/>`_. This includes the MSVC solution files if you want to build GLPK yourself on Windows, although the prebuilt binaries are fine.

Binaries for lpsolve are available from the `lpsolve sourceforge website <https://sourceforge.net/projects/lpsolve/>`_.

Installing on Linux
-------------------

No special instructions required. Follow instructions as for `installing (in general)`_ to build from source. A conda package is also available.

Ubuntu
~~~~~~

The following commands should install the GLPK and lpsolve libraries:

.. code-block:: shell

  sudo apt-get install libgmp3-dev libglpk-dev glpk
  sudo apt-get install liblpsolve55-dev lp-solve

The Ubuntu package for lpsolve includes a static library which can confuse the compiler. The easiest work-around is to remove it:

.. code-block:: shell

  sudo rm /usr/lib/liblpsolve55.a
  sudo ln -s /usr/lib/lp_solve/liblpsolve55.so /usr/lib/liblpsolve55.so

Installing on OS X
------------------

Follow instructions as for `installing (in general)`_ to build from source. A conda package is also available.

If external libraries are located in a non-standard location you either need to set the `DYLD_LIBRARY_PATH` environment variable at runtime:

.. code-block:: shell

  export DYLD_LIBRARY_PATH=/path/to/library/directory

Alternatively (and recommended) set the `rpath` of the extension during compilation.

.. code-block:: shell

  export CFLAGS="-Wl,-rpath,/path/to/library/directory"

You may also need to specify the location of the library headers:

.. code-block:: shell

  export C_INCLUDE_PATH=/path/to/include/directory

Examples of the above can be seen in the conda recipe (see `conda-recipe/build.sh`).

The dependencies (GLPK and/or lpsolve) can be built from source manually, or installed using `Homebrew <http://brew.sh/>`_.

Development and testing
-----------------------

The source code for Pywr is managed using Git and is hosted on GitHub: https://github.com/pywr/pywr/ .

There are a collection of unit tests for Pywr written using ``pytest``. These can be run using:

.. code-block:: shell

  pytest tests

This will run all avaialble tests using the default solver. A specific solver can be tested by specifying the `PYWR_SOLVER` environment variable:

.. code-block:: shell

  PYWR_SOLVER=lpsolve pytest tests

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

Pywr is automatically built and tested on Linux and Windows using Travis-CI and AppVeyor (respectively).

Creating a pull request on GitHub will automatically trigger a build.

https://travis-ci.org/pywr/pywr

https://ci.appveyor.com/project/pywr-admin/pywr

Both services install Pywr using the Anaconda Python distribution, as this was the easiest way to install all the dependencies.
