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

setup_kwargs = {
    'name': 'pywr',
    'version': '0.1',
    'description': 'Python Water Resource model',
    'author': 'Joshua Arnott',
    'author_email': 'josh@snorfalorpagus.net',
    'url': 'http://snorf.net/pywr/',
    'packages': ['pywr', 'pywr.solvers', 'pywr.domains', 'pywr.parameters'],
}

extensions = [
    Extension('pywr._core', ['pywr/_core.pyx'],
              include_dirs=[np.get_include()],),
    Extension('pywr.parameters._parameters', ['pywr/parameters/_parameters.pyx'],
              include_dirs=[np.get_include()],),
    Extension('pywr._recorders', ['pywr/_recorders.pyx'],
              include_dirs=[np.get_include()],),
    Extension('pywr.parameters._control_curves', ['pywr/parameters/_control_curves.pyx'],
              include_dirs=[np.get_include()],),
]

# HACK: optional features are too difficult to do properly
# http://stackoverflow.com/a/4056848/1300519
optional = set()
if '--with-glpk' in sys.argv:
    optional.add('glpk')
    sys.argv.remove('--with-glpk')
if '--with-lpsolve' in sys.argv:
    optional.add('lpsolve')
    sys.argv.remove('--with-lpsolve')
if not optional:
    # default is to attempt to build everything
    optional.add('glpk')
    optional.add('lpsolve')

compiler_directives = {}
if '--enable-profiling' in sys.argv:
     compiler_directives['profile'] = True
     sys.argv.remove('--enable-profiling')

extensions_optional = []
if 'glpk' in optional:
    extensions_optional.append(
        Extension('pywr.solvers.cython_glpk', ['pywr/solvers/cython_glpk.pyx'],
                  include_dirs=[np.get_include()],
                  libraries=['glpk'],),
    )
if 'lpsolve' in optional:
    extensions_optional.append(
        Extension('pywr.solvers.cython_lpsolve', ['pywr/solvers/cython_lpsolve.pyx'],
                  include_dirs=[np.get_include()],
                  libraries=['lpsolve55'],),
    )

# build the core extension(s)
setup_kwargs['ext_modules'] = cythonize(extensions + extensions_optional, compiler_directives=compiler_directives)
setup(**setup_kwargs)
