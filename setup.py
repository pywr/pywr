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

# get version string from __init__.py
with open(os.path.join(os.path.dirname(__file__), "pywr", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            version = Version(line.split("=")[1].strip().strip("\"'"))

setup_kwargs = {
    'name': 'pywr',
    'version': str(version),
    'description': 'Python Water Resource model',
    'author': 'Joshua Arnott',
    'author_email': 'josh@snorfalorpagus.net',
    'url': 'http://snorf.net/pywr/',
    'packages': ['pywr', 'pywr.solvers', 'pywr.domains', 'pywr.parameters', 'pywr.notebook', 'pywr.optimisation'],
}


define_macros = []

# HACK: optional features are too difficult to do properly
# http://stackoverflow.com/a/4056848/1300519
optional = set()
if '--with-glpk' in sys.argv:
    optional.add('glpk')
    sys.argv.remove('--with-glpk')
if '--with-lpsolve' in sys.argv:
    optional.add('lpsolve')
    sys.argv.remove('--with-lpsolve')
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

if '--enable-trace' in sys.argv:
    print('Tracing is enabled.')
    compiler_directives['linetrace'] = True
    define_macros.append(('CYTHON_TRACE', '1'))
    sys.argv.remove('--enable-trace')

extensions = [
    Extension('pywr._core', ['pywr/_core.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr.parameters._parameters', ['pywr/parameters/_parameters.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr._recorders', ['pywr/_recorders.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
    Extension('pywr.parameters._control_curves', ['pywr/parameters/_control_curves.pyx'],
              include_dirs=[np.get_include()],
              define_macros=define_macros),
]

extensions_optional = []
if 'glpk' in optional:
    extensions_optional.append(
        Extension('pywr.solvers.cython_glpk', ['pywr/solvers/cython_glpk.pyx'],
                  include_dirs=[np.get_include()],
                  libraries=['glpk'],
                  define_macros=define_macros),
    )
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

# build the core extension(s)
setup_kwargs['ext_modules'] = cythonize(extensions + extensions_optional, compiler_directives=compiler_directives, annotate=annotate)
setup(**setup_kwargs)
