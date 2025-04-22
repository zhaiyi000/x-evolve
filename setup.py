from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cpp_helper",
        ["cpp_helper.cpp"],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="cpp_helper",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)