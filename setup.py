#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('pywr._core', ['pywr/_core.pyx'],
              include_dirs=[np.get_include()],),
    Extension('pywr.solvers.cython_glpk', ['pywr/solvers/cython_glpk.pyx'],
              include_dirs=[np.get_include()],
              libraries = ['glpk'],)
]

setup(
    name='pywr',
    version='0.1',
    description='Python Water Resource model',
    author='Joshua Arnott',
    author_email='josh@snorfalorpagus.net',
    url='http://snorf.net/pywr/',
    packages=['pywr', 'pywr.solvers'],
    ext_modules=cythonize(extensions)
)
