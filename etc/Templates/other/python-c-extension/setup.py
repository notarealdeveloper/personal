#!/usr/bin/env python3

import os
import shutil
from setuptools import setup, Extension

include_dirs = []
try:
    import numpy
    include_dirs += [numpy.get_include()]
except ModuleNotFoundError:
    print(f"numpy module not found.")

module = Extension(
    'cpython',
    sources = ['cpython.c'],
    include_dirs = include_dirs,
)

setup(
    name = 'cpython',
    version = '0.1',
    description = 'A python module called python for python.',
    ext_modules = [module],
)

def copy_so_to_cwd():
    files = os.popen("find build -type f -name '*.so'").read().splitlines()
    for file in files:
        shutil.copy(file, ".")

copy_so_to_cwd()
