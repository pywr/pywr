set LIBRARY=C:\Users\n22209mb\AppData\Local\Continuum\anaconda3\envs\py36\Library

set LIBRARY_INC=%LIBRARY%\include

set LIBRARY_LIB=%LIBRARY%\lib

python setup.py build_ext -I"%LIBRARY_INC%" -L"%LIBRARY_LIB%" --inplace --with-glpk develop