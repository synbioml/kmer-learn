/* CPython binding for position-bounded LocalShuffler. */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "_common.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "_local.c"  /* single-TU build */

typedef struct {
    PyObject_HEAD
    Py_ssize_t max_shift;
    philox_stream_t rng;
    local_stats_t last_stats;
} LocalShufflerObject;

static int local_seed_from_object(PyObject *seed_obj, uint64_t *seed) {
    if (seed_obj != Py_None) {
        PyObject *seed_long = PyNumber_Long(seed_obj);
        if (!seed_long) return -1;
        *seed = (uint64_t)PyLong_AsUnsignedLongLongMask(seed_long);
        Py_DECREF(seed_long);
        return PyErr_Occurred() ? -1 : 0;
    }

    PyObject *os = PyImport_ImportModule("os");
    if (!os) return -1;
    PyObject *urandom = PyObject_GetAttrString(os, "urandom");
    Py_DECREF(os);
    if (!urandom) return -1;
    PyObject *bytes = PyObject_CallFunction(urandom, "i", 8);
    Py_DECREF(urandom);
    if (!bytes) return -1;

    char *buffer;
    Py_ssize_t length;
    if (PyBytes_AsStringAndSize(bytes, &buffer, &length) < 0) {
        Py_DECREF(bytes);
        return -1;
    }
    *seed = 0;
    for (int i = 0; i < 8; ++i) {
        *seed = (*seed << 8) | (uint8_t)buffer[i];
    }
    Py_DECREF(bytes);
    return 0;
}

static int LocalShuffler_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"max_shift", "seed", NULL};
    LocalShufflerObject *object = (LocalShufflerObject *)self;
    PyObject *shift_obj;
    PyObject *seed_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                     &shift_obj, &seed_obj)) return -1;
    if (PyBool_Check(shift_obj) || !PyLong_Check(shift_obj)) {
        PyErr_SetString(PyExc_TypeError, "max_shift must be an integer");
        return -1;
    }
    object->max_shift = PyLong_AsSsize_t(shift_obj);
    if (object->max_shift == -1 && PyErr_Occurred()) return -1;
    if (object->max_shift < 0) {
        PyErr_SetString(PyExc_ValueError, "max_shift must be non-negative");
        return -1;
    }

    uint64_t seed;
    if (local_seed_from_object(seed_obj, &seed) < 0) return -1;
    philox_init(&object->rng, seed);
    memset(&object->last_stats, 0, sizeof(object->last_stats));
    return 0;
}

static int local_parse_limit(PyObject *value, int *enabled, uint64_t *limit,
                             const char *name) {
    if (value == Py_None) {
        *enabled = 0;
        *limit = 0;
        return 0;
    }
    if (PyBool_Check(value) || !PyLong_Check(value)) {
        PyErr_Format(PyExc_TypeError, "%s must be an integer or None", name);
        return -1;
    }
    *limit = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred()) return -1;
    *enabled = 1;
    return 0;
}

static int local_run_permutation(LocalShufflerObject *object,
                                 Py_ssize_t length,
                                 PyObject *soft_obj, PyObject *hard_obj,
                                 Py_ssize_t *permutation) {
    int soft_enabled, hard_enabled;
    uint64_t soft_limit, hard_limit;
    if (local_parse_limit(soft_obj, &soft_enabled, &soft_limit,
                          "soft limit L") < 0) return -1;
    if (local_parse_limit(hard_obj, &hard_enabled, &hard_limit,
                          "hard limit H") < 0) return -1;

    int rc = local_permutation(length, object->max_shift,
                               soft_enabled, soft_limit,
                               hard_enabled, hard_limit,
                               &object->rng, permutation,
                               &object->last_stats);
    if (rc == LOCAL_OK) return 0;
    if (rc == LOCAL_ERR_MEMORY) PyErr_NoMemory();
    else if (rc == LOCAL_ERR_LENGTH) {
        PyErr_SetString(PyExc_OverflowError,
                        "sequence is too long for the native shuffler");
    } else if (rc == LOCAL_ERR_AUGMENT) {
        PyErr_SetString(PyExc_RuntimeError,
                        "no augmenting path despite identity matching");
    } else {
        PyErr_SetString(PyExc_RuntimeError, "local shuffle failed");
    }
    return -1;
}

static PyObject * LocalShuffler_permutation_bytes(PyObject *self,
                                                   PyObject *args) {
    LocalShufflerObject *object = (LocalShufflerObject *)self;
    Py_ssize_t length;
    PyObject *soft_obj, *hard_obj;
    if (!PyArg_ParseTuple(args, "nOO", &length, &soft_obj, &hard_obj)) {
        return NULL;
    }
    if (length < 0) {
        PyErr_SetString(PyExc_ValueError, "length must be non-negative");
        return NULL;
    }
    if ((size_t)length > PY_SSIZE_T_MAX / sizeof(Py_ssize_t)) {
        return PyErr_NoMemory();
    }

    PyObject *result = PyBytes_FromStringAndSize(
        NULL, length * (Py_ssize_t)sizeof(Py_ssize_t));
    if (!result) return NULL;
    Py_ssize_t *permutation = (Py_ssize_t *)PyBytes_AS_STRING(result);
    if (local_run_permutation(object, length, soft_obj, hard_obj,
                              permutation) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}

static PyObject * LocalShuffler_shuffle(PyObject *self, PyObject *args) {
    LocalShufflerObject *object = (LocalShufflerObject *)self;
    PyObject *sequence, *soft_obj, *hard_obj;
    if (!PyArg_ParseTuple(args, "OOO", &sequence, &soft_obj, &hard_obj)) {
        return NULL;
    }

    Py_ssize_t length;
    int is_bytes = PyBytes_Check(sequence);
    int is_unicode = PyUnicode_Check(sequence);
    if (is_bytes) length = PyBytes_GET_SIZE(sequence);
    else if (is_unicode) length = PyUnicode_GetLength(sequence);
    else {
        PyErr_SetString(PyExc_TypeError,
                        "sequence must be str, bytes, or a 1D NumPy array");
        return NULL;
    }

    Py_ssize_t *permutation = NULL;
    if (length > 0) {
        permutation = malloc((size_t)length * sizeof(*permutation));
        if (!permutation) return PyErr_NoMemory();
    }
    if (local_run_permutation(object, length, soft_obj, hard_obj,
                              permutation) < 0) {
        free(permutation);
        return NULL;
    }

    PyObject *result;
    if (is_bytes) {
        result = PyBytes_FromStringAndSize(NULL, length);
        if (result) {
            char *output = PyBytes_AS_STRING(result);
            const char *input = PyBytes_AS_STRING(sequence);
            for (Py_ssize_t slot = 0; slot < length; ++slot) {
                output[slot] = input[permutation[slot]];
            }
        }
    } else {
        Py_UCS4 maximum = PyUnicode_MAX_CHAR_VALUE(sequence);
        result = PyUnicode_New(length, maximum);
        if (result) {
            int output_kind = PyUnicode_KIND(result);
            void *output_data = PyUnicode_DATA(result);
            for (Py_ssize_t slot = 0; slot < length; ++slot) {
                Py_UCS4 value = PyUnicode_ReadChar(sequence, permutation[slot]);
                if (value == (Py_UCS4)-1 && PyErr_Occurred()) {
                    Py_DECREF(result);
                    result = NULL;
                    break;
                }
                PyUnicode_WRITE(output_kind, output_data, slot, value);
            }
        }
    }

    free(permutation);
    return result;
}

static PyObject * LocalShuffler_get_last_stats(PyObject *self,
                                                PyObject *Py_UNUSED(ignored)) {
    LocalShufflerObject *object = (LocalShufflerObject *)self;
    return Py_BuildValue(
        "{sK,sK,sK,sK,sK}",
        "turns", (unsigned long long)object->last_stats.turns,
        "collisions", (unsigned long long)object->last_stats.collisions,
        "soft_chains", (unsigned long long)object->last_stats.soft_chains,
        "fallbacks", (unsigned long long)object->last_stats.fallbacks,
        "max_collisions", (unsigned long long)object->last_stats.max_collisions);
}

static PyObject * LocalShuffler_getstate(PyObject *self,
                                          PyObject *Py_UNUSED(ignored)) {
    LocalShufflerObject *object = (LocalShufflerObject *)self;
    char state[44];
    memcpy(state + 0, object->rng.key, 8);
    memcpy(state + 8, object->rng.counter, 16);
    memcpy(state + 24, object->rng.buf, 16);
    memcpy(state + 40, &object->rng.buf_used, 4);
    return PyBytes_FromStringAndSize(state, sizeof(state));
}

static PyObject * LocalShuffler_setstate(PyObject *self, PyObject *arg) {
    LocalShufflerObject *object = (LocalShufflerObject *)self;
    char *state;
    Py_ssize_t length;
    if (PyBytes_AsStringAndSize(arg, &state, &length) < 0) return NULL;
    if (length != 44) {
        PyErr_SetString(PyExc_ValueError,
                        "state must contain exactly 44 bytes");
        return NULL;
    }
    memcpy(object->rng.key, state + 0, 8);
    memcpy(object->rng.counter, state + 8, 16);
    memcpy(object->rng.buf, state + 24, 16);
    memcpy(&object->rng.buf_used, state + 40, 4);
    if (object->rng.buf_used < 0 || object->rng.buf_used > 4) {
        PyErr_SetString(PyExc_ValueError, "invalid Philox buffer state");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject * LocalShuffler_repr(PyObject *self) {
    LocalShufflerObject *object = (LocalShufflerObject *)self;
    return PyUnicode_FromFormat("<_LocalShuffler max_shift=%zd>",
                                object->max_shift);
}

static PyMethodDef LocalShuffler_methods[] = {
    {"_shuffle", (PyCFunction)LocalShuffler_shuffle, METH_VARARGS,
     "_shuffle(sequence, L, H) -> str | bytes"},
    {"_permutation_bytes", (PyCFunction)LocalShuffler_permutation_bytes,
     METH_VARARGS, "_permutation_bytes(length, L, H) -> bytes"},
    {"_get_last_stats", (PyCFunction)LocalShuffler_get_last_stats,
     METH_NOARGS, "Return statistics from the last shuffle."},
    {"getstate", (PyCFunction)LocalShuffler_getstate, METH_NOARGS,
     "Return the complete Philox RNG state."},
    {"setstate", (PyCFunction)LocalShuffler_setstate, METH_O,
     "Restore a complete Philox RNG state."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject LocalShufflerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "kmer.perturb._native._local._LocalShuffler",
    .tp_basicsize = sizeof(LocalShufflerObject),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Native position-bounded collision-chain shuffler.",
    .tp_init = LocalShuffler_init,
    .tp_methods = LocalShuffler_methods,
    .tp_repr = LocalShuffler_repr,
    .tp_new = PyType_GenericNew,
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "kmer.perturb._native._local",
    "C-backed position-bounded local shuffler.", -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__local(void) {
    if (PyType_Ready(&LocalShufflerType) < 0) return NULL;
    PyObject *module = PyModule_Create(&moduledef);
    if (!module) return NULL;
    Py_INCREF(&LocalShufflerType);
    if (PyModule_AddObject(module, "_LocalShuffler",
                           (PyObject *)&LocalShufflerType) < 0) {
        Py_DECREF(&LocalShufflerType);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
