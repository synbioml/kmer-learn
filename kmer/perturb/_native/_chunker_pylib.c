/* _chunker_pylib.c — CPython bindings for the chunker. */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "_common.h"
#include <string.h>
#include <stdlib.h>

#include "_chunker.c"  /* single-TU build */

typedef struct {
    PyObject_HEAD
    int    min_sz;
    int    max_sz;
    int    algorithm;       /* 0=random, 1=backtrack */
    int    residual_mode;
    int    flip_strand;
    double flip_strand_prob;
    philox_stream_t rng;
} ChunkerObject;

static const char *_alg_str[] = {"random", "backtrack"};
static const char *_residual_str[] = {
    "end", "start", "random", "extend", "distribute"
};

static int _parse_str_enum(PyObject *obj, const char **valid, int n_valid,
                            int *out, const char *arg_name) {
    if (!PyUnicode_Check(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be a str", arg_name);
        return -1;
    }
    const char *s = PyUnicode_AsUTF8(obj);
    if (!s) return -1;
    for (int i = 0; i < n_valid; i++) {
        if (strcmp(s, valid[i]) == 0) { *out = i; return 0; }
    }
    /* Build a comma-separated list of valid values for the error message. */
    char buf[256];
    int pos = 0;
    buf[0] = '\0';
    for (int i = 0; i < n_valid && pos < (int)sizeof(buf) - 16; i++) {
        if (i > 0) pos += snprintf(buf + pos, sizeof(buf) - pos, ", ");
        pos += snprintf(buf + pos, sizeof(buf) - pos, "'%s'", valid[i]);
    }
    PyErr_Format(PyExc_ValueError, "%s must be one of { %s }; got '%s'",
                 arg_name, buf, s);
    return -1;
}

static int Chunker_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "min_size", "max_size", "algorithm", "residual_chunk",
        "flip_strand", "flip_strand_prob", "seed", NULL
    };
    ChunkerObject *o = (ChunkerObject *)self;
    o->min_sz = 2; o->max_sz = 4;
    o->algorithm = 0; o->residual_mode = RESIDUAL_DISTRIBUTE;
    o->flip_strand = 0; o->flip_strand_prob = 0.5;
    PyObject *alg_obj = NULL;
    PyObject *res_obj = NULL;
    PyObject *seed_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|OOpdO", kwlist,
                                      &o->min_sz, &o->max_sz,
                                      &alg_obj, &res_obj,
                                      &o->flip_strand,
                                      &o->flip_strand_prob,
                                      &seed_obj)) {
        return -1;
    }
    if (o->min_sz < 1 || o->max_sz < o->min_sz || o->max_sz > CHUNKER_MAX_SIZE) {
        PyErr_SetString(PyExc_ValueError,
                         "require 1 <= min_size <= max_size <= 65536");
        return -1;
    }
    if (alg_obj && alg_obj != Py_None) {
        const char *valid[] = {"random", "backtrack"};
        if (_parse_str_enum(alg_obj, valid, 2, &o->algorithm, "algorithm") < 0)
            return -1;
    }
    if (res_obj && res_obj != Py_None) {
        const char *valid[] = {"end", "start", "random", "extend", "distribute"};
        if (_parse_str_enum(res_obj, valid, 5, &o->residual_mode,
                             "residual_chunk") < 0)
            return -1;
    }
    if (o->flip_strand_prob < 0.0 || o->flip_strand_prob > 1.0) {
        PyErr_SetString(PyExc_ValueError, "flip_strand_prob must be in [0, 1]");
        return -1;
    }
    /* Init RNG. */
    uint64_t seed;
    if (seed_obj == Py_None) {
        PyObject *os = PyImport_ImportModule("os");
        if (!os) return -1;
        PyObject *urandom = PyObject_GetAttrString(os, "urandom");
        Py_DECREF(os);
        if (!urandom) return -1;
        PyObject *b8 = PyObject_CallFunction(urandom, "i", 8);
        Py_DECREF(urandom);
        if (!b8) return -1;
        char *buf; Py_ssize_t len;
        if (PyBytes_AsStringAndSize(b8, &buf, &len) < 0) { Py_DECREF(b8); return -1; }
        seed = 0;
        for (int i = 0; i < 8; i++) seed = (seed << 8) | (uint8_t)buf[i];
        Py_DECREF(b8);
    } else {
        PyObject *sl = PyNumber_Long(seed_obj);
        if (!sl) return -1;
        seed = (uint64_t)PyLong_AsUnsignedLongLongMask(sl);
        Py_DECREF(sl);
        if (PyErr_Occurred()) return -1;
    }
    philox_init(&o->rng, seed);
    return 0;
}

static int _coerce_seq(PyObject *obj, const char **out_s, Py_ssize_t *out_len,
                       int *out_is_bytes) {
    if (PyBytes_Check(obj)) {
        *out_s = PyBytes_AS_STRING(obj);
        *out_len = PyBytes_GET_SIZE(obj);
        *out_is_bytes = 1;
        return 0;
    }
    if (PyUnicode_Check(obj)) {
        Py_ssize_t len;
        const char *s = PyUnicode_AsUTF8AndSize(obj, &len);
        if (!s) return -1;
        *out_s = s; *out_len = len; *out_is_bytes = 0;
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "sequence must be str or bytes");
    return -1;
}

static PyObject * _build_output(const char *buf, Py_ssize_t len, int is_bytes) {
    if (is_bytes) return PyBytes_FromStringAndSize(buf, len);
    return PyUnicode_DecodeASCII(buf, len, "strict");
}

static PyObject * Chunker_chunk(PyObject *self, PyObject *arg) {
    ChunkerObject *o = (ChunkerObject *)self;
    const char *seq; Py_ssize_t seqlen; int is_bytes;
    if (_coerce_seq(arg, &seq, &seqlen, &is_bytes) < 0) return NULL;
    char *out = (char *)malloc(seqlen + 1);
    if (!out) return PyErr_NoMemory();
    int out_len = 0;
    int rc = chunk_one(seq, (int)seqlen, o->min_sz, o->max_sz,
                       o->algorithm, o->residual_mode,
                       o->flip_strand, o->flip_strand_prob,
                       &o->rng, out, &out_len);
    if (rc != CHUNKER_OK) {
        free(out);
        switch (rc) {
        case CHUNKER_ERR_BAD_RANGE:
            PyErr_SetString(PyExc_ValueError, "bad chunk size range"); break;
        case CHUNKER_ERR_MEMORY:
            PyErr_NoMemory(); break;
        case CHUNKER_ERR_BAD_ALG:
            PyErr_SetString(PyExc_ValueError, "bad algorithm"); break;
        case CHUNKER_ERR_BAD_RESIDUAL:
            PyErr_SetString(PyExc_ValueError, "bad residual mode"); break;
        case CHUNKER_ERR_NON_IUPAC:
            PyErr_SetString(PyExc_ValueError,
                             "flip_strand=True requires IUPAC characters"); break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "chunk failed");
        }
        return NULL;
    }
    PyObject *r = _build_output(out, out_len, is_bytes);
    free(out);
    return r;
}

static PyObject * Chunker_chunk_many(PyObject *self, PyObject *args,
                                      PyObject *kwds) {
    ChunkerObject *o = (ChunkerObject *)self;
    PyObject *seqs_obj;
    int n_jobs = 1;
    static char *kwlist[] = {"seqs", "n_jobs", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist,
                                      &seqs_obj, &n_jobs)) return NULL;
    PyObject *seqs_fast = PySequence_Fast(seqs_obj, "seqs must be a sequence");
    if (!seqs_fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seqs_fast);

    uint64_t master_seed = ((uint64_t)o->rng.key[1] << 32) | o->rng.key[0];
    const char **seq_bufs = malloc(n * sizeof(char *));
    Py_ssize_t *seq_lens = malloc(n * sizeof(Py_ssize_t));
    int *seq_is_bytes = malloc(n * sizeof(int));
    char **out_bufs = malloc(n * sizeof(char *));
    int *out_lens = malloc(n * sizeof(int));
    int *err_codes = malloc(n * sizeof(int));
    if (!seq_bufs || !seq_lens || !seq_is_bytes || !out_bufs || !out_lens || !err_codes) {
        free(seq_bufs); free(seq_lens); free(seq_is_bytes);
        free(out_bufs); free(out_lens); free(err_codes);
        Py_DECREF(seqs_fast);
        return PyErr_NoMemory();
    }
    int validation_failed = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seqs_fast, i);
        if (_coerce_seq(item, &seq_bufs[i], &seq_lens[i], &seq_is_bytes[i]) < 0) {
            validation_failed = 1; break;
        }
        out_bufs[i] = malloc(seq_lens[i] + 1);
        if (!out_bufs[i]) { PyErr_NoMemory(); validation_failed = 1; break; }
        out_lens[i] = 0;
        err_codes[i] = CHUNKER_OK;
    }
    if (validation_failed) {
        for (Py_ssize_t i = 0; i < n; i++) free(out_bufs[i]);
        free(seq_bufs); free(seq_lens); free(seq_is_bytes);
        free(out_bufs); free(out_lens); free(err_codes);
        Py_DECREF(seqs_fast);
        return NULL;
    }
    int threads = n_jobs; if (threads < 1) threads = 1;
    if (threads > (int)n) threads = (int)n;

    if (threads == 1) {
        /* Single-threaded: use the master RNG directly so that
         * chunk_many(seqs) == [chunk(s) for s in seqs] bit-for-bit. */
        for (Py_ssize_t i = 0; i < n; i++) {
            err_codes[i] = chunk_one(seq_bufs[i], (int)seq_lens[i],
                                      o->min_sz, o->max_sz,
                                      o->algorithm, o->residual_mode,
                                      o->flip_strand, o->flip_strand_prob,
                                      &o->rng, out_bufs[i], &out_lens[i]);
        }
    } else {
        /* Parallel: each task gets an independent key derived from
         * (master_seed, task_index). Output is reproducible for a given
         * n_jobs, but differs from the single-threaded stream. */
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (Py_ssize_t i = 0; i < n; i++) {
            philox_stream_t local_rng;
            uint32_t key[2];
            philox_derive_key(master_seed, (uint64_t)i, key);
            local_rng.key[0] = key[0]; local_rng.key[1] = key[1];
            local_rng.counter[0] = 0; local_rng.counter[1] = 0;
            local_rng.counter[2] = 0; local_rng.counter[3] = 0;
            local_rng.buf_used = 4;
            err_codes[i] = chunk_one(seq_bufs[i], (int)seq_lens[i],
                                      o->min_sz, o->max_sz,
                                      o->algorithm, o->residual_mode,
                                      o->flip_strand, o->flip_strand_prob,
                                      &local_rng, out_bufs[i], &out_lens[i]);
        }
        /* Advance master RNG by n calls to keep streams decorrelated. */
        for (Py_ssize_t i = 0; i < n; i++) (void)philox_u32(&o->rng);
    }

    PyObject *result = PyList_New(n);
    if (!result) {
        for (Py_ssize_t i = 0; i < n; i++) free(out_bufs[i]);
        free(seq_bufs); free(seq_lens); free(seq_is_bytes);
        free(out_bufs); free(out_lens); free(err_codes);
        Py_DECREF(seqs_fast);
        return NULL;
    }
    int have_error = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        if (err_codes[i] != CHUNKER_OK) {
            have_error = 1;
            if (err_codes[i] == CHUNKER_ERR_NON_IUPAC) {
                PyErr_Format(PyExc_ValueError,
                              "sequence %zd contains non-IUPAC characters", i);
            } else {
                PyErr_Format(PyExc_RuntimeError,
                              "chunk failed for sequence %zd", i);
            }
            break;
        }
        PyObject *out_obj = _build_output(out_bufs[i], out_lens[i], seq_is_bytes[i]);
        if (!out_obj) { have_error = 1; break; }
        PyList_SET_ITEM(result, i, out_obj);
    }
    for (Py_ssize_t i = 0; i < n; i++) free(out_bufs[i]);
    free(seq_bufs); free(seq_lens); free(seq_is_bytes);
    free(out_bufs); free(out_lens); free(err_codes);
    Py_DECREF(seqs_fast);
    if (have_error) { Py_DECREF(result); return NULL; }
    return result;
}

static PyObject * Chunker_getstate(PyObject *self, PyObject *Py_UNUSED(ign)) {
    ChunkerObject *o = (ChunkerObject *)self;
    char buf[24];
    memcpy(buf + 0,  &o->rng.key[0], 4);
    memcpy(buf + 4,  &o->rng.key[1], 4);
    memcpy(buf + 8,  &o->rng.counter[0], 4);
    memcpy(buf + 12, &o->rng.counter[1], 4);
    memcpy(buf + 16, &o->rng.counter[2], 4);
    memcpy(buf + 20, &o->rng.counter[3], 4);
    return PyBytes_FromStringAndSize(buf, 24);
}

static PyObject * Chunker_setstate(PyObject *self, PyObject *arg) {
    ChunkerObject *o = (ChunkerObject *)self;
    char *buf; Py_ssize_t len;
    if (PyBytes_AsStringAndSize(arg, &buf, &len) < 0) return NULL;
    if (len < 24) { PyErr_SetString(PyExc_ValueError, "state too short"); return NULL; }
    memcpy(&o->rng.key[0], buf + 0, 4);
    memcpy(&o->rng.key[1], buf + 4, 4);
    memcpy(&o->rng.counter[0], buf + 8, 4);
    memcpy(&o->rng.counter[1], buf + 12, 4);
    memcpy(&o->rng.counter[2], buf + 16, 4);
    memcpy(&o->rng.counter[3], buf + 20, 4);
    o->rng.buf_used = 4;
    Py_RETURN_NONE;
}

static PyObject * Chunker_repr(PyObject *self) {
    ChunkerObject *o = (ChunkerObject *)self;
    return PyUnicode_FromFormat("<Chunker min=%d max=%d alg='%s' residual='%s'>",
                                 o->min_sz, o->max_sz,
                                 _alg_str[o->algorithm],
                                 _residual_str[o->residual_mode]);
}

static PyMethodDef Chunker_methods[] = {
    {"chunk",      (PyCFunction)Chunker_chunk,      METH_O,
     "chunk(seq) -> str | bytes"},
    {"chunk_many", (PyCFunction)Chunker_chunk_many, METH_VARARGS | METH_KEYWORDS,
     "chunk_many(seqs, n_jobs=1) -> list"},
    {"getstate",   (PyCFunction)Chunker_getstate,   METH_NOARGS, "Get RNG state."},
    {"setstate",   (PyCFunction)Chunker_setstate,   METH_O, "Set RNG state."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject ChunkerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "kmer.perturb._native._chunker.Chunker",
    .tp_basicsize = sizeof(ChunkerObject),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Sequence chunker.",
    .tp_init = Chunker_init,
    .tp_methods = Chunker_methods,
    .tp_repr = Chunker_repr,
    .tp_new = PyType_GenericNew,
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "kmer.perturb._native._chunker",
    "C-backed sequence chunker.", -1, NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__chunker(void) {
    if (PyType_Ready(&ChunkerType) < 0) return NULL;
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    Py_INCREF(&ChunkerType);
    if (PyModule_AddObject(m, "Chunker", (PyObject *)&ChunkerType) < 0) {
        Py_DECREF(&ChunkerType); Py_DECREF(m); return NULL;
    }
    return m;
}
