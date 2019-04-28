#!/usr/bin/env python

try:
    from setuptools import setup
    from setuptools import Extension
    print('Using setuptools for setup!')
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
    print('Using distutils for setup!')
from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import sys
import os
from packaging.version import Version
import subprocess

with open('README.rst') as fh:
    long_description = fh.read()

setup_kwargs = {
    'name': 'pywr',
    'description': 'Python Water Resource model',
    'long_description': long_description,
    'long_description_content_type': 'text/x-rst',
    'author': 'Joshua Arnott',
    'author_email': 'josh@snorfalorpagus.net',
    'url': 'https://github.com/pywr/pywr',
    'packages': ['pywr', 'pywr.solvers', 'pywr.domains', 'pywr.parameters', 'pywr.recorders', 'pywr.notebook', 'pywr.optimisation'],
    'use_scm_version': True,
    'setup_requires': ['setuptools_scm'],
    'install_requires': [
        'pandas',
        'networkx',
        'scipy',
        'tables',
        'future',
        'xlrd',
        'packaging',
        'matplotlib',
        'jinja2'
    ]
}


define_macros = []

# HACK: optional features are too difficult to do properly
# http://stackoverflow.com/a/4056848/1300519
optional = set()
if '--with-glpk' in sys.argv:
    optional.add('glpk')
    sys.argv.remove('--with-glpk')
elif os.environ.get('PYWR_BUILD_GLPK', 'false').lower() == 'true':
    optional.add('glpk')

if '--with-lpsolve' in sys.argv:
    optional.add('lpsolve')
    sys.argv.remove('--with-lpsolve')
elif os.environ.get('PYWR_BUILD_LPSOLVE', 'false').lower() == 'true':
    optional.add('lpsolve')

if '--annotate' in sys.argv:
    annotate = True
    sys.argv.remove('--annotate')
else:
    annotate = False
if not optional:
    # default is to attempt to build everything
    optional.add('glpk')
    optional.add('lpsolve')

compiler_directives = {}
if '--enable-profiling' in sys.argv:
     compiler_directives['profile'] = True
     sys.argv.remove('--enable-profiling')

build_trace = False
if '--enable-trace' in sys.argv:
    sys.argv.remove('--enable-trace')
    build_trace = True
elif os.environ.get('PYWR_BUILD_TRACE', 'false').lower() == 'true':
    build_trace = True

if build_trace:
    print('Tracing is enabled.')
    compiler_directives['linetrace'] = True
    define_macros.append(('CYTHON_TRACE', '1'))
    define_macros.append(('CYTHON_TRACE_NOGIL', '1'))

compile_time_env = {}
if '--enable-debug' in sys.argv:
    compile_time_env['SOLVER_DEBUG'] = True
    sys.argv.remove('--enable-debug')
else:
    compile_time_env['SOLVER_DEBUG'] = False

# See the following documentation for a description of these directives
#  https://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiler-directives
compiler_directives['language_level'] = 3
compiler_directives['embedsignature'] = True


extensions = [
    Extension('pywr._core', ['pywr/_core.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr._model', ['pywr/_model.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr._component', ['pywr/_component.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),

    # Parameters sub-package

    Extension('pywr.parameters._parameters', ['pywr/parameters/_parameters.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr.parameters._polynomial', ['pywr/parameters/_polynomial.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr.parameters._thresholds', ['pywr/parameters/_thresholds.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr.parameters._control_curves', ['pywr/parameters/_control_curves.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr.parameters._hydropower', ['pywr/parameters/_hydropower.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),

    # Other modules
    Extension('pywr.recorders._recorders', ['pywr/recorders/_recorders.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr.recorders._thresholds', ['pywr/recorders/_thresholds.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr.recorders._hydropower', ['pywr/recorders/_hydropower.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),


]

extensions_optional = []
if 'glpk' in optional:
    extensions_optional.extend([
        Extension('pywr.solvers.cython_glpk', ['pywr/solvers/cython_glpk.pyx'],
                  include_dirs=[np.get_include()],
                  libraries=['glpk'],
                  define_macros=define_macros),
        Extension('pywr.solvers.cython_glpk_edge', ['pywr/solvers/cython_glpk_edge.pyx'],
                  include_dirs=[np.get_include()],
                  libraries=['glpk'],
                  define_macros=define_macros),
    ])
if 'lpsolve' in optional:
    if os.name == 'nt':
        define_macros.append(('WIN32', 1))
    extensions_optional.append(
        Extension('pywr.solvers.cython_lpsolve', ['pywr/solvers/cython_lpsolve.pyx'],
                  include_dirs=[np.get_include()],
                  libraries=['lpsolve55'],
                  define_macros=define_macros),
    )

setup_kwargs['package_data'] = {
    'pywr.notebook': ['*.js', '*.css']
}

# store the current git hash in the module
try:
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip().decode("utf-8")
except subprocess.CalledProcessError:
    pass
else:
    with open("pywr/GIT_VERSION.txt", "w") as f:
        f.write(git_hash + "\n")
    setup_kwargs["package_data"]["pywr"] = ["GIT_VERSION.txt"]

# build the core extension(s)
setup_kwargs['ext_modules'] = cythonize(extensions + extensions_optional,
                                        compiler_directives=compiler_directives, annotate=annotate,
                                        compile_time_env=compile_time_env)

if os.environ.get('PACKAGE_DATA', 'false').lower() == 'true':
    pkg_data = setup_kwargs["package_data"].get("pywr", [])
    pkg_data.extend(['.libs/*', '.libs/licenses/*'])
    setup_kwargs["package_data"]["pywr"] = pkg_data

setup(**setup_kwargs)
