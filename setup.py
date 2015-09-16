#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
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
    'packages': ['pywr', 'pywr.solvers', 'pywr.domains'],
}

extensions = [
    Extension('pywr._core', ['pywr/_core.pyx'],
              include_dirs=[np.get_include()],),
]

extensions_optional = [
    Extension('pywr.solvers.cython_glpk', ['pywr/solvers/cython_glpk.pyx'],
              include_dirs=[np.get_include()],
              libraries=['glpk'],),
    Extension('pywr.solvers.cython_lpsolve', ['pywr/solvers/cython_lpsolve.pyx'],
              include_dirs=[np.get_include()],
              libraries=['lpsolve55'],),
]

# Optional extension code from Bob Ippolito's simplejson project
# https://github.com/simplejson/simplejson

if sys.platform == 'win32' and sys.version_info > (2, 6):
    # 2.6's distutils.msvc9compiler can raise an IOError when failing to
    # find the compiler
    # It can also raise ValueError http://bugs.python.org/issue7511
    ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError,
                  IOError, ValueError)
else:
    ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)

class BuildFailed(Exception):
    pass

class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors:
            raise BuildFailed()

setup_kwargs['cmdclass'] = {}

# attempt to build the cython solver extensions
success = []
failure = []
for extension in extensions_optional:
    setup_kwargs['ext_modules'] = cythonize([extension])
    setup_kwargs['cmdclass']['build_ext'] = ve_build_ext
    try:
        setup(**setup_kwargs)
    except BuildFailed:
        failure.append(extension)
    else:
        success.append(extension)

if not success:
    raise BuildFailed('None of the solvers managed to build')

# build the core extension(s)
setup_kwargs['ext_modules'] = cythonize(extensions)
del(setup_kwargs['cmdclass']['build_ext'])
setup(**setup_kwargs)

print('Successfully built pywr with the following extensions:')
for extension in success:
    print('  * {}'.format(extension.name))
