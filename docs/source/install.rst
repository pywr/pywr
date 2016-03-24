Installing Pywr
===============

Pywr should work on Python 2.7 (or later) and 3.4 (or later) on Windows, Linux or OS X.

Building Pywr from source requires a working C compiler. It has been built successfully with MSVC on Windows, GCC on Linux and clang/LLVM on OS X.

The easiest way to install Pywr on Windows is using the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. See instructions below on how to install using Anaconda.

Dependencies
------------

Pywr has several external dependencies, listed below.

 * Core dependencies (required)

   * `Cython <http://cython.org/>`_

   * `NumPy <http://www.numpy.org/>`_

   * `NetworkX <https://networkx.github.io/>`_

   * `Pandas <http://pandas.pydata.org/>`_

 * Linear programming solvers (at least one required)

   * `GLPK <https://www.gnu.org/software/glpk/>`_ (recommended)

   * `lpsolve <http://lpsolve.sourceforge.net/5.5/>`_

 * Optional dependencies (providing additional functionality)

   * `pytables <http://www.pytables.org/>`_

   * `xlrd <https://pypi.python.org/pypi/xlrd>`_

   * `pytest <http://pytest.org/latest/>`_ (for testing only)

Installing (in general)
-----------------------

When installing Pywr you must specific which solvers to build. This is done by passing ``--with-<solver>`` as an argument to ``setup.py``. The following command will build and install Pywr with both the GLPK and lpsolve solvers:

.. code-block:: shell

  python setup.py install --with-glpk --with-lpsolve

To install Pywr in-place in developer mode, use the ``develop`` command instead of ``install``. This is only useful if you plan to modify the Pywr source code and is not required for general use.

.. code-block:: shell

  python setup.py develop --with-glpk --with-lpsolve

Installing on Windows
---------------------

Installing with Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~

A binary distribution of Pywr is provided for Python 2.7 and 3.4 (64-bit) on Windows for the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. You will need to install and configure Anaconda before proceeding. Note that this release may lag behind the development version.

The following commands add the "channel" the binary is hosted on, then install Pywr including all dependencies.

.. code-block:: shell

  conda config --add channels snorfalorpagus
  conda install pywr

At the time of writing the version of ``pytest`` available via `conda` is too old and should be installed via `pip`:

.. code-block:: shell

  pip install pytest==2.8.5

Building from source
~~~~~~~~~~~~~~~~~~~~

To build Pywr from source on Windows you must first install and configure the MSVC compiler. See the `instructions on this blog <https://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/>`_. It is important that you install the correct version of MSVC to correspond with your Python version.

Binaries for GLPK are available from the `WinGLPK project <http://winglpk.sourceforge.net/>`_. This includes the MSVC solution files if you want to build GLPK yourself on Windows, although the prebuilt binaries are fine.

Binaries for lpsolve are available from the `lpsolve sourceforge website <https://sourceforge.net/projects/lpsolve/>`_.

Installing on Linux
-------------------

No special instructions required.

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

No special instructions required.

The dependencies (GLPK and/or lpsolve) can be built from source manually, or installed using `Homebrew <http://brew.sh/>`_.

Development and testing
-----------------------

The source code for Pywr is managed using Git and is hosted on GitHub: https://github.com/pywr/pywr/ .

There are a collection of unit tests for Pywr written using ``pytest``. These can be run using:

.. code-block:: shell

  py.test tests

This will run all avaialble tests using the default solver. A specific solver can be tested by specifying at the command line:

.. code-block:: shell

  py.test tests --solver=lpsolve

Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~

Pywr is automatically built and tested on Linux and Windows using Travis-CI and AppVeyor (respectively).

Creating a pull request on GitHub will automatically trigger a build.

https://travis-ci.org/pywr/pywr

https://ci.appveyor.com/project/snorfalorpagus/pywr

Both services install Pywr using the Anaconda Python distribution, as this was the easiest way to install all the dependencies.
