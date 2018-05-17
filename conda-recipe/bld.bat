set LIB=%LIBRARY_LIB%;.\lib;%LIB%
set LIBPATH=%LIBRARY_LIB%;.\lib;%LIBPATH%
set INCLUDE=%LIBRARY_INC%;%INCLUDE%

:: clean up before building
:: assumes all .c files were compiled from .pyx and should be removed
del /s /q build dist
del /s __pycache__ *.pyc *.pyo *.pyd *.so *.c

"%PYTHON%" setup.py build_ext -I"%LIBRARY_INC%" -L"%LIBRARY_LIB%" install --with-glpk --with-lpsolve --single-version-externally-managed --record=record.txt

# Also build the source and wheel distributions for pypi
"%PYTHON%" setup.py sdist
"%PYTHON%" setup.py bdist_wheel

if errorlevel 1 exit 1

