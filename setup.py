import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


def setup_package():
    compiler_directives = {
        "language_level": 3,
        "embedsignature": True,
    }

    define_macros = []
    compile_time_env = {}

    annotate = False

    class new_build_ext(_build_ext):
        def finalize_options(self):
            # Defer the import of Cython and NumPy until after setup_requires
            from Cython.Build.Dependencies import cythonize
            import numpy

            self.distribution.ext_modules[:] = cythonize(
                self.distribution.ext_modules,
                compiler_directives=compiler_directives,
                compile_time_env=compile_time_env,
                annotate=annotate,
            )
            if not self.include_dirs:
                self.include_dirs = []
            elif isinstance(self.include_dirs, str):
                self.include_dirs = [self.include_dirs]
            self.include_dirs.append(numpy.get_include())
            super().finalize_options()

    metadata = dict(
        name="pywr",
        packages=[
            "pywr",
            "pywr.solvers",
            "pywr.domains",
            "pywr.parameters",
            "pywr.recorders",
            "pywr.notebook",
            "pywr.optimisation",
            "pywr.utils",
        ],
    )

    ext_modules = [
        Extension("pywr._core", ["pywr/_core.pyx"]),
        Extension("pywr._model", ["pywr/_model.pyx"]),
        Extension("pywr._component", ["pywr/_component.pyx"]),
        Extension("pywr.parameters._parameters", ["pywr/parameters/_parameters.pyx"]),
        Extension("pywr.parameters._polynomial", ["pywr/parameters/_polynomial.pyx"]),
        Extension("pywr.parameters._thresholds", ["pywr/parameters/_thresholds.pyx"]),
        Extension(
            "pywr.parameters._control_curves", ["pywr/parameters/_control_curves.pyx"]
        ),
        Extension("pywr.parameters._hydropower", ["pywr/parameters/_hydropower.pyx"]),
        Extension(
            "pywr.parameters._activation_functions",
            ["pywr/parameters/_activation_functions.pyx"],
        ),
        Extension("pywr.recorders._recorders", ["pywr/recorders/_recorders.pyx"]),
        Extension("pywr.recorders._thresholds", ["pywr/recorders/_thresholds.pyx"]),
        Extension("pywr.recorders._hydropower", ["pywr/recorders/_hydropower.pyx"]),
    ]

    config = parse_optional_arguments()

    if config["glpk"]:
        ext_modules.append(
            Extension(
                "pywr.solvers.cython_glpk",
                ["pywr/solvers/cython_glpk.pyx"],
                libraries=["glpk"],
            )
        )

    if config["lpsolve"]:
        if os.name == "nt":
            lpsolve_macros = define_macros + [("WIN32", 1)]
        else:
            lpsolve_macros = define_macros
        ext_modules.append(
            Extension(
                "pywr.solvers.cython_lpsolve",
                ["pywr/solvers/cython_lpsolve.pyx"],
                libraries=["lpsolve55"],
                define_macros=lpsolve_macros,
            )
        )

    metadata["ext_modules"] = ext_modules
    metadata["cmdclass"] = {"build_ext": new_build_ext}

    annotate = config["annotate"]

    if config["profile"]:
        compiler_directives["profile"] = True

    if config["trace"]:
        compiler_directives["linetrace"] = True
        define_macros.extend(
            [
                ("CYTHON_TRACE", "1"),
                ("CYTHON_TRACE_NOGIL", "1"),
            ]
        )

    setup(**metadata)


def long_description():
    with open("README.rst") as f:
        return f.read()


def package_data():
    pkg_data = {
        "pywr.notebook": ["*.js", "*.css", "*.html"],
        "pywr": ["*.pxd"],
        "pywr.parameters": ["*.pxd"],
        "pywr.recorders": ["*.pxd"],
        "pywr.solvers": ["*.pxd"],
    }
    if os.environ.get("PACKAGE_DATA", "false").lower() == "true":
        pkg_data["pywr"].extend([".libs/*", ".libs/licenses/*"])
    return pkg_data


def parse_optional_arguments():
    config = {
        "glpk": True,
        "lpsolve": True,
        "annotate": False,
        "profile": False,
        "trace": False,
    }

    if "PYWR_BUILD_GLPK" in os.environ:
        config["glpk"] = os.environ["PYWR_BUILD_GLPK"].lower() in (
            "true",
            "1",
        )

    if "PYWR_BUILD_LPSOLVE" in os.environ:
        config["lpsolve"] = os.environ["PYWR_BUILD_LPSOLVE"].lower() in (
            "true",
            "1",
        )

    if "PYWR_BUILD_ANNOTATE" in os.environ:
        config["annotate"] = os.environ["PYWR_BUILD_ANNOTATE"].lower() in (
            "true",
            "1",
        )

    if "PYWR_BUILD_PROFILE" in os.environ:
        config["profile"] = os.environ["PYWR_BUILD_PROFILE"].lower() in (
            "true",
            "1",
        )

    if "PYWR_BUILD_TRACE" in os.environ:
        config["trace"] = os.environ["PYWR_BUILD_TRACE"].lower() in (
            "true",
            "1",
        )

    return config


if __name__ == "__main__":
    setup_package()
