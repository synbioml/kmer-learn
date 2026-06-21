/* _spectrum_pylib.c — CPython binding for the rolling-hash k-mer counter.
 *
 * Exposes one top-level function:
 *   _count_kmers(seqs, k, canonical_rc) -> (rows, cols, vals, n_features)
 *
 * The Python wrapper builds a scipy.sparse.csr_matrix from these.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "_common.h"
#include <string.h>
#include <stdlib.h>

#include "_spectrum.c"  /* single-TU build */

/* Coerce a Python sequence (str or bytes) to (const char*, Py_ssize_t). */
static int _coerce_seq(PyObject *obj, const char **out_s, Py_ssize_t *out_len) {
    if (PyBytes_Check(obj)) {
        *out_s = PyBytes_AS_STRING(obj);
        *out_len = PyBytes_GET_SIZE(obj);
        return 0;
    }
    if (PyUnicode_Check(obj)) {
        Py_ssize_t len;
        const char *s = PyUnicode_AsUTF8AndSize(obj, &len);
        if (!s) return -1;
        *out_s = s; *out_len = len;
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "sequence must be str or bytes");
    return -1;
}

static PyObject * _spectrum_count_kmers(PyObject *self, PyObject *args,
                                          PyObject *kwds) {
    static char *kwlist[] = {"seqs", "k", "canonical_rc", NULL};
    PyObject *seqs_obj;
    int k;
    int canonical_rc = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oi|p", kwlist,
                                      &seqs_obj, &k, &canonical_rc)) {
        return NULL;
    }

    PyObject *seqs_fast = PySequence_Fast(seqs_obj, "seqs must be a sequence");
    if (!seqs_fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seqs_fast);

    /* Extract all sequences into C arrays. */
    const char **seq_bufs = (const char **)malloc(n * sizeof(char *));
    int *seq_lens = (int *)malloc(n * sizeof(int));
    if (!seq_bufs || !seq_lens) {
        free(seq_bufs); free(seq_lens);
        Py_DECREF(seqs_fast);
        return PyErr_NoMemory();
    }
    int validation_failed = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seqs_fast, i);
        const char *s; Py_ssize_t len;
        if (_coerce_seq(item, &s, &len) < 0) {
            validation_failed = 1; break;
        }
        seq_bufs[i] = s;
        seq_lens[i] = (int)len;
    }
    if (validation_failed) {
        free(seq_bufs); free(seq_lens);
        Py_DECREF(seqs_fast);
        return NULL;
    }

    /* Count. */
    spectrum_output_t out;
    int rc = _spectrum_count_batch(seq_bufs, seq_lens, (int)n, k,
                                    canonical_rc, &out);
    free(seq_bufs); free(seq_lens);
    Py_DECREF(seqs_fast);
    if (rc != KMER_SPECTRUM_OK) {
        switch (rc) {
        case KMER_SPECTRUM_ERR_BAD_K:
            PyErr_Format(PyExc_ValueError,
                         "k must be in [1, 12], got %d (4^k exceeds 16M "
                         "feature names for k > 12)",
                         k);
            break;
        case KMER_SPECTRUM_ERR_NON_ACGT:
            PyErr_SetString(PyExc_ValueError,
                             "input contains non-ACGT characters");
            break;
        case KMER_SPECTRUM_ERR_MEMORY:
            PyErr_NoMemory(); break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "spectrum count failed");
        }
        if (rc == KMER_SPECTRUM_ERR_NON_ACGT || rc == KMER_SPECTRUM_ERR_MEMORY) {
            _spectrum_output_free(&out);
        }
        return NULL;
    }

    /* Build (rows, cols, vals, n_features) tuple. */
    PyObject *rows_list = PyList_New(out.nnz);
    PyObject *cols_list = PyList_New(out.nnz);
    PyObject *vals_list = PyList_New(out.nnz);
    if (!rows_list || !cols_list || !vals_list) {
        Py_XDECREF(rows_list); Py_XDECREF(cols_list); Py_XDECREF(vals_list);
        _spectrum_output_free(&out);
        return NULL;
    }
    for (int i = 0; i < out.nnz; i++) {
        PyList_SET_ITEM(rows_list, i, PyLong_FromLong(out.rows[i]));
        PyList_SET_ITEM(cols_list, i, PyLong_FromLong(out.cols[i]));
        PyList_SET_ITEM(vals_list, i, PyLong_FromLong(out.vals[i]));
    }
    PyObject *result = Py_BuildValue("OOOi",
                                      rows_list, cols_list, vals_list,
                                      out.n_features);
    Py_DECREF(rows_list); Py_DECREF(cols_list); Py_DECREF(vals_list);
    _spectrum_output_free(&out);
    return result;
}

static PyMethodDef _spectrum_methods[] = {
    {"_count_kmers", (PyCFunction)_spectrum_count_kmers,
     METH_VARARGS | METH_KEYWORDS,
     "_count_kmers(seqs, k, canonical_rc=False) -> "
     "(rows, cols, vals, n_features)\n\n"
     "Count k-mers in each sequence. Returns triplet lists suitable "
     "for building a scipy.sparse.csr_matrix."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef _spectrum_moduledef = {
    PyModuleDef_HEAD_INIT,
    "kmer.encoders._native._spectrum",
    "C-backed rolling-hash k-mer counter.",
    -1, _spectrum_methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__spectrum(void) {
    return PyModule_Create(&_spectrum_moduledef);
}
