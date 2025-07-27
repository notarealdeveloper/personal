# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from libc.stdlib cimport malloc, free

# This works too
# ==============
# cdef extern from "<stdlib.h>" nogil:
#     void free (void *ptr)
#     void *malloc (size_t size)

cdef char *allocate(size_t bufsize):
    """ On linux, malloc seems not to actually allocate
        any memory until it's read from or written to.
    """
    cdef char *p = <char *>malloc(bufsize)
    for n in range(0, bufsize, 1024):
        p[n] = b'\xff'
    return p

cdef void deallocate(void *ptr):
    free(ptr)
    return

def main(size_t n):

    p = allocate(n)

    print(f"Allocated {n} byte buffer.")

    done = False
    while not done:
        response = input("Free buffer? (y/N): ").strip()
        if response.lower() == 'y':
            break
    free(p)
    print(f"Freed {n} byte buffer.")
    return
