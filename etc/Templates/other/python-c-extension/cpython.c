#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

/* https://docs.python.org/3/extending/extending.html */

static PyObject *
cpython_getaddr(PyObject *self, PyObject *ptr)
{
    return PyLong_FromLong((unsigned long)ptr);
}

static PyObject *
cpython_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyObject *
cpython_arrayfunc(PyObject *self, PyObject *obj)
{
    PyArrayObject *array = (PyArrayObject *)obj;
    npy_intp size = PyArray_SIZE(array);
    npy_intp i;
    unsigned long *buffer = (unsigned long *)PyArray_BYTES(array);

    //printf("C: array at %p, size is %lu\n", array, size);

    //Py_BEGIN_ALLOW_THREADS

    // PyArray_GETCONTIGUOUS
    for (i=0; i<size; i++) {
        buffer[i] += 3;
    }

    //Py_END_ALLOW_THREADS

    return (PyObject *)array;
}

static PyObject *
cpython_overwrite(PyObject *self, PyObject *obj)
{
    printf("Overwriting: object at %p\n", obj);

    //&obj = PyLong_FromLong(42);
    return (PyObject *)obj;
}


static PyMethodDef CPythonMethods[] = {
    {"getaddr", cpython_getaddr, METH_O, "Get the address of any PyObject."},
    {"system", cpython_system, METH_VARARGS, "Execute a shell command."},
    {"arrayfunc", cpython_arrayfunc, METH_O, "Hello world for the numpy C API."},
    {"overwrite", cpython_overwrite, METH_O, "Overwrite a pointer."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyDoc_STRVAR(cpython_doc, "Docstring for module cpython.");

static struct PyModuleDef cpythonmodule = {
    PyModuleDef_HEAD_INIT,
    "cpython",          /* name of module */
    cpython_doc,        /* module documentation, may be NULL */
    -1,                 /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    CPythonMethods,
};

PyMODINIT_FUNC PyInit_cpython(void)
{
    import_array();
    return PyModule_Create(&cpythonmodule);
}
