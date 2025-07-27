#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

DESCRIPTION = 'A cowsay package for python.'
PYTHON_REQUIRES = '>=3.6.0'

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as fp:
    LONG_DESCRIPTION = fp.read()

setup(
    name = 'cowsay',
    version = '0.1',
    packages = find_packages(),
    author = 'Jason Wilkes',
    author_email = 'notarealdeveloper@gmail.com',
    python_requires = PYTHON_REQUIRES,
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
)
