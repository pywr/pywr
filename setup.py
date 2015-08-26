#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='pywr',
    version='0.1',
    description='Python Water Resource model',
    author='Joshua Arnott',
    author_email='josh@snorfalorpagus.net',
    url='http://snorf.net/pywr/',
    packages=['pywr', 'pywr.solvers'],
    ext_modules=cythonize([
        "pywr/_core.pyx",
        ])
)
