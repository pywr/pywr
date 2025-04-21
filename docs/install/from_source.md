# Installing from source
This section describes how to build and install Pywr from source. You can read the sections below
that explain how to do so depending on the Operating System you are using.

!!!note
    Building Pywr from source requires a working C compiler. It has been built successfully with MSVC on 
    Windows, GCC on Linux and clang/LLVM on MacOS.

## Dependencies
Pywr has several external dependencies which are:
 
- Core dependencies (required)
    - [Cython](http://cython.org/)
    - [NumPy](http://www.numpy.org/)
    - [NetworkX](https://networkx.github.io/)
    - [Pandas](http://pandas.pydata.org/)
    - [packaging](https://pypi.python.org/pypi/packaging)
- Linear programming solvers (at least one required)
    - [GLPK](https://www.gnu.org/software/glpk/) (recommended)
    - [lpsolve](http://lpsolve.sourceforge.net/5.5/)
- Optional dependencies (providing additional functionality)
    - [pytables](http://www.pytables.org/)
    - [xlrd](https://pypi.python.org/pypi/xlrd)
    - [pytest](http://pytest.org/latest/) (for testing only)
    - [SciPy](http://www.scipy.org/)
    - [Jupyter](https://jupyter.org/)
    - [Matplotlib](http://matplotlib.org/)

## Solver options
When installing Pywr you must specify which solvers to build. This is done by passing `--with-<solver>` as
an argument to the `setup.py` file in the root folder of the project. For example, the following command will build and 
install Pywr with both the GLPK and lpsolve solvers:

    python setup.py install --with-glpk --with-lpsolve

To install Pywr in-place in developer mode, use the develop command instead of install. This is only useful if 
you plan to modify the Pywr source code and is not required for general use.

    python setup.py develop --with-glpk --with-lpsolve

## Compile on Ubuntu
The following commands install the GLPK and lpsolve libraries:

    sudo apt-get install libgmp3-dev libglpk-dev glpk
    sudo apt-get install liblpsolve55-dev lp-solve

The Ubuntu package for `lpsolve` includes a static library which can confuse the compiler. The easiest work-around is
to remove it:

    sudo rm /usr/lib/liblpsolve55.a
    sudo ln -s /usr/lib/lp_solve/liblpsolve55.so /usr/lib/liblpsolve55.so

## Compile on Windows
TODO: use Windows Kit

## Compile on MacOS
This section describes how to install the necessary dependencies
and compile pywr on MacOS with the `glpk` solver. If you wish 
to use `lpsolve`, you can alter the commands below. This guide also
assumes you have:

1. the [Brew package manager](https://brew.sh) already installed on your machine.
2. You have already created a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments).
3. You have already installed the `gcc` compiler. The easiest option is to install [Apple’s XCode](https://developer.apple.com/).

### Python
Activate the virtual environment 

    source myvenv/bin/activate

Install then the Python dependencies using `pip`:

    pip install cython pandas networkx numpy scipy tables openpyxl

!!! note

    If you have a mac with an ARM processor, there is no compiled version for the `tables` 
    wheel. `pip` will attempt to install it from source which will likely fail. Install `hdf5`
    with brew first and then install the wheel from PyPI:

        brew install hdf5
        pip install tables

### Solver
The GLPK solver with the header and library files is available via `brew`, to install it run:

    brew install glpk

Obtain the path where the solver was installed using

    brew info glpk

The path looks like: `/opt/homebrew/Cellar/glpk/5.0` but you may have a different version

### Compile pywr
Compile pywr using the command below. Remember to replace the path to the glpk solver

    CFLAGS="-I /opt/homebrew/Cellar/glpk/5.0/include" LDFLAGS="-L /opt/homebrew/Cellar/glpk/5.0/lib" python setup.py build_ext --inplace --verbose
  
The `-I` includes the `glpk.h` header file pywr needs, the `-L` option includes the dynamic library already compiled
for MacOS.
