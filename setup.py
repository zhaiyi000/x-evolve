# setup.py
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'block_cpp',
        ['block_cpp.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='block_cpp',
    ext_modules=ext_modules,
)