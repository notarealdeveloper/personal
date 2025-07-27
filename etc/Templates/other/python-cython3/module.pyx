#cython: language_level=3
#distutils: extra_compile_args=-fopenmp
#distutils: extra_link_args=-fopenmp

from libc.stdint cimport uint64_t
cimport numpy as np
cimport cython
from cython.parallel import prange

ctypedef np.int64_t DTYPE

def python_mask_array(a, mod=2, value=0):
    i = 0
    length = len(a)
    while i < length:
        if i % mod == 0:
            a[i] = value
        i += 1
    return a

cpdef np.ndarray[DTYPE, ndim=1] cython_mask_array(np.ndarray[DTYPE, ndim=1] a, int mod=2, DTYPE value=0):
    cdef:
        uint64_t i = 0
        uint64_t length = len(a)
    while i < length:
        if i % mod == 0:
            a[i] = value
        i += 1
    return a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE, ndim=1] cython_mask_array_nogil(np.ndarray[DTYPE, ndim=1] a, int mod=2, DTYPE value=0):
    cdef:
        int i = 0
        int length = len(a)
    with nogil:
        for i in prange(0, length):
            if i % mod == 0:
                a[i] = value
    return a

