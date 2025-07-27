#!/usr/bin/env python3

import numpy as np
import cimportlib
import slang

cimportlib.install(build_args={
    'extra_compile_args': ['-fopenmp'],
    'extra_link_args': ['-fopenmp'],
})

N = 100_000_000
import module # pyx


a = np.arange(0, N)
with slang.Timing('cython'):
    a = module.cython_mask_array(a, mod=2, value=0)
print(a)

b = np.arange(0, N)
with slang.Timing('python'):
    b = module.python_mask_array(b, mod=2, value=0)
print(b)

c = np.arange(0, N)
with slang.Timing('numpy'):
    c = np.where(c % 2 == 0, 0, c)
print(c)

d = np.arange(0, N)
with slang.Timing('cython_nogil'):
    d = module.cython_mask_array_nogil(d)
print(d)
