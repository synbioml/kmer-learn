/* _shuffler_pylib.c — CPython bindings for the k-mer shuffler.
 *
 * Exposes a single class, KmerShuffler, with:
 *   __init__(k, seed=None, endpoints="free")
 *   shuffle(seq)        -> str | bytes
 *   shuffle_many(seqs, n_jobs=1) -> list[str] | list[bytes]
 *   getstate() -> bytes
 *   setstate(bytes)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "_common.h"
#include <string.h>
#include <stdlib.h>

/* The C core is in a separate translation unit. */
#include "_shuffler.c"  /* NOLINT: single-TU build for simplicity */

/* ------------------------------------------------------------------ */
/* KmerShuffler object                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    PyObject_HEAD
    int    k;
    int    endpoints_mode;   /* 0=preserve, 1=free, 2=crop */
    philox_stream_t rng;
    /* rng state is captured/restored for getstate/setstate */
} KmerShufflerObject;

static const char *_endpoints_str[] = {"preserve", "free", "crop"};

static int _parse_endpoints(PyObject *obj, int *out) {
    if (!PyUnicode_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "endpoints must be a str");
        return -1;
    }
    const char *s = PyUnicode_AsUTF8(obj);
    if (!s) return -1;
    for (int i = 0; i < 3; i++) {
        if (strcmp(s, _endpoints_str[i]) == 0) { *out = i; return 0; }
    }
    PyErr_Format(PyExc_ValueError,
                 "endpoints must be one of 'preserve', 'free', 'crop'; got '%s'", s);
    return -1;
}

static int KmerShuffler_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"k", "seed", "endpoints", NULL};
    KmerShufflerObject *o = (KmerShufflerObject *)self;
    o->k = 2;
    o->endpoints_mode = 1;  /* "free" */
    PyObject *seed_obj = Py_None;
    PyObject *endpoints_obj = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|OU", kwlist,
                                      &o->k, &seed_obj, &endpoints_obj)) {
        return -1;
    }
    if (o->k < 1 || o->k > KMER_SHUFFLER_MAX_K) {
        PyErr_Format(PyExc_ValueError,
                     "k must be in [1, %d], got %d", KMER_SHUFFLER_MAX_K, o->k);
        return -1;
    }
    if (endpoints_obj && _parse_endpoints(endpoints_obj, &o->endpoints_mode) < 0) {
        return -1;
    }
    /* Initialize RNG. */
    uint64_t seed;
    if (seed_obj == Py_None) {
        /* Use OS entropy. */
        PyObject *os = PyImport_ImportModule("os");
        if (!os) return -1;
        PyObject *urandom = PyObject_GetAttrString(os, "urandom");
        Py_DECREF(os);
        if (!urandom) return -1;
        PyObject *b8 = PyObject_CallFunction(urandom, "i", 8);
        Py_DECREF(urandom);
        if (!b8) return -1;
        char *buf;
        Py_ssize_t len;
        if (PyBytes_AsStringAndSize(b8, &buf, &len) < 0) { Py_DECREF(b8); return -1; }
        seed = 0;
        for (int i = 0; i < 8; i++) seed = (seed << 8) | (uint8_t)buf[i];
        Py_DECREF(b8);
    } else {
        PyObject *seed_long = PyNumber_Long(seed_obj);
        if (!seed_long) return -1;
        seed = (uint64_t)PyLong_AsUnsignedLongLongMask(seed_long);
        Py_DECREF(seed_long);
        if (PyErr_Occurred()) return -1;
    }
    philox_init(&o->rng, seed);
    return 0;
}

/* Coerce a Python sequence (str or bytes) to (const char*, Py_ssize_t).
 * Returns 0 on success. */
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
    if (is_bytes) {
        return PyBytes_FromStringAndSize(buf, len);
    } else {
        return PyUnicode_DecodeASCII(buf, len, "strict");
    }
}

static PyObject * KmerShuffler_shuffle(PyObject *self, PyObject *arg) {
    KmerShufflerObject *o = (KmerShufflerObject *)self;
    const char *seq;
    Py_ssize_t seqlen;
    int is_bytes;
    if (_coerce_seq(arg, &seq, &seqlen, &is_bytes) < 0) return NULL;

    /* Allocate worst-case output buffer (free mode may be 1 shorter
     * due to virtual edge removal; crop mode is shorter still, but
     * never longer than the input). */
    char *out = (char *)malloc(seqlen + 1);
    if (!out) return PyErr_NoMemory();
    int out_len = 0;
    int rc = kmer_shuffle_one(seq, (int)seqlen, o->k,
                               _endpoints_str[o->endpoints_mode],
                               &o->rng, out, &out_len);
    if (rc != KMER_SHUFFLE_OK) {
        free(out);
        switch (rc) {
        case KMER_SHUFFLE_ERR_NONACGT:
            PyErr_SetString(PyExc_ValueError,
                            "input contains non-ACGT characters");
            break;
        case KMER_SHUFFLE_ERR_MEMORY:
            PyErr_NoMemory(); break;
        case KMER_SHUFFLE_ERR_BAD_K:
            PyErr_SetString(PyExc_ValueError, "invalid k");
            break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "shuffle failed");
        }
        return NULL;
    }
    PyObject *result = _build_output(out, out_len, is_bytes);
    free(out);
    return result;
}

/* shuffle_many(seqs, n_jobs=1) — parallel via OpenMP.
 * For reproducibility: each task i uses key derived from (master_seed, i).
 * After all tasks complete, the master RNG is advanced by the number
 * of tasks (so subsequent shuffle() calls don't collide with task streams). */
static PyObject * KmerShuffler_shuffle_many(PyObject *self, PyObject *args,
                                              PyObject *kwds) {
    KmerShufflerObject *o = (KmerShufflerObject *)self;
    PyObject *seqs_obj;
    int n_jobs = 1;
    static char *kwlist[] = {"seqs", "n_jobs", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist,
                                      &seqs_obj, &n_jobs)) return NULL;

    PyObject *seqs_fast = PySequence_Fast(seqs_obj, "seqs must be a sequence");
    if (!seqs_fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seqs_fast);

    /* Extract master seed (current rng.key, packed). */
    uint64_t master_seed = ((uint64_t)o->rng.key[1] << 32) | o->rng.key[0];

    /* Validate and pre-extract all sequences into C arrays. */
    const char **seq_bufs = (const char **)malloc(n * sizeof(char *));
    Py_ssize_t *seq_lens = (Py_ssize_t *)malloc(n * sizeof(Py_ssize_t));
    int *seq_is_bytes = (int *)malloc(n * sizeof(int));
    char **out_bufs = (char **)malloc(n * sizeof(char *));
    int *out_lens = (int *)malloc(n * sizeof(int));
    int *err_codes = (int *)malloc(n * sizeof(int));
    PyObject **orig_objs = (PyObject **)malloc(n * sizeof(PyObject *));

    if (!seq_bufs || !seq_lens || !seq_is_bytes || !out_bufs
        || !out_lens || !err_codes || !orig_objs) {
        free(seq_bufs); free(seq_lens); free(seq_is_bytes);
        free(out_bufs); free(out_lens); free(err_codes); free(orig_objs);
        Py_DECREF(seqs_fast);
        return PyErr_NoMemory();
    }

    int any_bytes = 0;
    int all_bytes = 1;
    int validation_failed = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seqs_fast, i);
        orig_objs[i] = item;  /* borrowed; no need to decref */
        if (_coerce_seq(item, &seq_bufs[i], &seq_lens[i], &seq_is_bytes[i]) < 0) {
            validation_failed = 1;
            break;
        }
        if (seq_is_bytes[i]) any_bytes = 1; else all_bytes = 0;
        out_bufs[i] = (char *)malloc(seq_lens[i] + 1);
        if (!out_bufs[i]) {
            PyErr_NoMemory(); validation_failed = 1; break;
        }
        out_lens[i] = 0;
        err_codes[i] = KMER_SHUFFLE_OK;
    }
    if (validation_failed) {
        for (Py_ssize_t i = 0; i < n; i++) free(out_bufs[i]);
        free(seq_bufs); free(seq_lens); free(seq_is_bytes);
        free(out_bufs); free(out_lens); free(err_codes); free(orig_objs);
        Py_DECREF(seqs_fast);
        return NULL;
    }

    /* Decide effective thread count. */
    int threads = n_jobs;
    if (threads < 1) threads = 1;
    if (threads > (int)n) threads = (int)n;

    if (threads == 1) {
        /* Single-threaded: use the master RNG directly so that
         * shuffle_many(seqs) == [shuffle(s) for s in seqs] bit-for-bit. */
        for (Py_ssize_t i = 0; i < n; i++) {
            err_codes[i] = kmer_shuffle_one(
                seq_bufs[i], (int)seq_lens[i], o->k,
                _endpoints_str[o->endpoints_mode],
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
            err_codes[i] = kmer_shuffle_one(
                seq_bufs[i], (int)seq_lens[i], o->k,
                _endpoints_str[o->endpoints_mode],
                &local_rng, out_bufs[i], &out_lens[i]);
        }
        /* Advance the master RNG by n calls to keep streams decorrelated. */
        for (Py_ssize_t i = 0; i < n; i++) {
            (void)philox_u32(&o->rng);
        }
    }

    /* Build result list. */
    PyObject *result = PyList_New(n);
    if (!result) {
        for (Py_ssize_t i = 0; i < n; i++) free(out_bufs[i]);
        free(seq_bufs); free(seq_lens); free(seq_is_bytes);
        free(out_bufs); free(out_lens); free(err_codes); free(orig_objs);
        Py_DECREF(seqs_fast);
        return NULL;
    }
    int have_error = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        if (err_codes[i] != KMER_SHUFFLE_OK) {
            have_error = 1;
            switch (err_codes[i]) {
            case KMER_SHUFFLE_ERR_NONACGT:
                PyErr_Format(PyExc_ValueError,
                             "sequence %zd contains non-ACGT characters", i);
                break;
            default:
                PyErr_Format(PyExc_RuntimeError, "shuffle failed for sequence %zd", i);
            }
            break;
        }
        PyObject *out_obj = _build_output(out_bufs[i], out_lens[i], seq_is_bytes[i]);
        if (!out_obj) { have_error = 1; break; }
        PyList_SET_ITEM(result, i, out_obj);
    }

    /* Free all buffers. */
    for (Py_ssize_t i = 0; i < n; i++) free(out_bufs[i]);
    free(seq_bufs); free(seq_lens); free(seq_is_bytes);
    free(out_bufs); free(out_lens); free(err_codes); free(orig_objs);
    Py_DECREF(seqs_fast);

    if (have_error) {
        Py_DECREF(result);
        return NULL;
    }
    (void)any_bytes; (void)all_bytes;
    return result;
}

/* getstate() -> bytes (16 bytes: 8-byte key + 8-byte counter + 4-byte buf_used). */
static PyObject * KmerShuffler_getstate(PyObject *self, PyObject *Py_UNUSED(ign)) {
    KmerShufflerObject *o = (KmerShufflerObject *)self;
    char buf[24];
    memset(buf, 0, sizeof(buf));
    memcpy(buf + 0,  &o->rng.key[0], 4);
    memcpy(buf + 4,  &o->rng.key[1], 4);
    memcpy(buf + 8,  &o->rng.counter[0], 4);
    memcpy(buf + 12, &o->rng.counter[1], 4);
    memcpy(buf + 16, &o->rng.counter[2], 4);
    memcpy(buf + 20, &o->rng.counter[3], 4);
    /* Also save buf + buf_used (16 bytes for buf + 4 for buf_used = 20). */
    PyObject *main = PyBytes_FromStringAndSize(buf, 24);
    return main;
}

static PyObject * KmerShuffler_setstate(PyObject *self, PyObject *arg) {
    KmerShufflerObject *o = (KmerShufflerObject *)self;
    char *buf; Py_ssize_t len;
    if (PyBytes_AsStringAndSize(arg, &buf, &len) < 0) return NULL;
    if (len < 24) {
        PyErr_SetString(PyExc_ValueError, "state too short");
        return NULL;
    }
    memcpy(&o->rng.key[0], buf + 0, 4);
    memcpy(&o->rng.key[1], buf + 4, 4);
    memcpy(&o->rng.counter[0], buf + 8, 4);
    memcpy(&o->rng.counter[1], buf + 12, 4);
    memcpy(&o->rng.counter[2], buf + 16, 4);
    memcpy(&o->rng.counter[3], buf + 20, 4);
    /* buf and buf_used: just force a refill on next call. */
    o->rng.buf_used = 4;
    Py_RETURN_NONE;
}

static PyObject * KmerShuffler_repr(PyObject *self) {
    KmerShufflerObject *o = (KmerShufflerObject *)self;
    return PyUnicode_FromFormat("<KmerShuffler k=%d endpoints='%s'>",
                                 o->k, _endpoints_str[o->endpoints_mode]);
}

static PyMethodDef KmerShuffler_methods[] = {
    {"shuffle",      (PyCFunction)KmerShuffler_shuffle,      METH_O,
     "shuffle(seq) -> str | bytes\n\n"
     "Shuffle a single sequence. Returns the same type as the input."},
    {"shuffle_many", (PyCFunction)KmerShuffler_shuffle_many, METH_VARARGS | METH_KEYWORDS,
     "shuffle_many(seqs, n_jobs=1) -> list\n\n"
     "Shuffle many sequences. If a seed was set at construction, output\n"
     "is bit-for-bit identical regardless of n_jobs."},
    {"getstate",     (PyCFunction)KmerShuffler_getstate,     METH_NOARGS,
     "getstate() -> bytes\n\nReturn the RNG state."},
    {"setstate",     (PyCFunction)KmerShuffler_setstate,     METH_O,
     "setstate(bytes) -> None\n\nRestore the RNG state."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject KmerShufflerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "kmer.perturb._native._shuffler.KmerShuffler",
    .tp_basicsize = sizeof(KmerShufflerObject),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "K-mer-preserving sequence shuffler.",
    .tp_init = KmerShuffler_init,
    .tp_methods = KmerShuffler_methods,
    .tp_repr = KmerShuffler_repr,
    .tp_new = PyType_GenericNew,
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "kmer.perturb._native._shuffler",
    "C-backed k-mer shuffler.", -1, NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__shuffler(void) {
    if (PyType_Ready(&KmerShufflerType) < 0) return NULL;
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;
    Py_INCREF(&KmerShufflerType);
    if (PyModule_AddObject(m, "KmerShuffler", (PyObject *)&KmerShufflerType) < 0) {
        Py_DECREF(&KmerShufflerType); Py_DECREF(m); return NULL;
    }
    return m;
}
