import glob
import os
import sys

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

CVRP_CPP_SOURCES = ["FewShotCVRP/cpp/cvrp.cpp"]
CVRP_INCLUDE_DIRS = [pybind11.get_include(), pybind11.get_include(user=True)]

HILLVALL_CPP_SOURCES = ["FewShotCVRP/cpp/hillvall_bindings.cpp"] + glob.glob(
    "external/HillVallEA/HillVallEA/*.cpp"
)
HILLVALL_INCLUDE_DIRS = [pybind11.get_include(), pybind11.get_include(user=True), "./"]

# O2 on GCC/Clang, /O2 on MSVC
extra_compile_args = ["-O2"] if sys.platform != "win32" else ["/O2"]

# Optional: define DBG=1 by setting environment variable DBG=1
define_macros = []
if os.environ.get("DBG") == "1":
    define_macros.append(("DBG", "1"))

ext_modules = [
    Pybind11Extension(
        name="cvrp_cpp",
        sources=CVRP_CPP_SOURCES,
        include_dirs=CVRP_INCLUDE_DIRS,
        language="c++",
        extra_compile_args=extra_compile_args,
        define_macros=define_macros,
        cxx_std=17,
    ),
    Pybind11Extension(
        name="hillvallimpl",
        sources=HILLVALL_CPP_SOURCES,
        include_dirs=HILLVALL_INCLUDE_DIRS,
        language="c++",
        extra_compile_args=extra_compile_args,
        define_macros=define_macros,
        cxx_std=17,
    ),
]

# No metadata here; setuptools reads it from pyproject.toml (PEP 621).
setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
