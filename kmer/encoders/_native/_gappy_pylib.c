/* _gappy_pylib.c — CPython binding for the masked-hash gappy k-mer counter.
 *
 * Exposes:
 *   _count_gappy_kmers(seqs, masks, canonical_rc) -> (rows, cols, vals, n_features)
 *   _count_gappy_kmers_range(seqs, L, g_min, g_max, canonical_rc)
 *       -> (rows, cols, vals, n_features, mask_strings)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "_common.h"
#include <string.h>
#include <stdlib.h>

#include "_gappy.c"  /* single-TU build */

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

/* Convert the masks[] array in `out` to a list of mask strings. */
static PyObject * _masks_to_list(gappy_output_t *out) {
    PyObject *lst = PyList_New(out->n_masks);
    if (!lst) return NULL;
    for (int i = 0; i < out->n_masks; i++) {
        char buf[KMER_GAPPY_MAX_LEN + 1];
        gappy_mask_t *m = &out->masks[i];
        for (int j = 0; j < m->length; j++) buf[j] = '-';
        for (int j = 0; j < m->n_concrete; j++) buf[m->positions[j]] = '*';
        buf[m->length] = '\0';
        PyList_SET_ITEM(lst, i, PyUnicode_FromString(buf));
    }
    return lst;
}

static PyObject * _build_result(gappy_output_t *out, int include_masks) {
    PyObject *rows_list = PyList_New(out->nnz);
    PyObject *cols_list = PyList_New(out->nnz);
    PyObject *vals_list = PyList_New(out->nnz);
    if (!rows_list || !cols_list || !vals_list) {
        Py_XDECREF(rows_list); Py_XDECREF(cols_list); Py_XDECREF(vals_list);
        return NULL;
    }
    for (int i = 0; i < out->nnz; i++) {
        PyList_SET_ITEM(rows_list, i, PyLong_FromLong(out->rows[i]));
        PyList_SET_ITEM(cols_list, i, PyLong_FromLong(out->cols[i]));
        PyList_SET_ITEM(vals_list, i, PyLong_FromLong(out->vals[i]));
    }
    PyObject *result;
    if (include_masks) {
        PyObject *masks_list = _masks_to_list(out);
        if (!masks_list) {
            Py_DECREF(rows_list); Py_DECREF(cols_list); Py_DECREF(vals_list);
            return NULL;
        }
        result = Py_BuildValue("OOOiO",
                                rows_list, cols_list, vals_list,
                                out->n_features, masks_list);
        Py_DECREF(masks_list);
    } else {
        result = Py_BuildValue("OOOi",
                                rows_list, cols_list, vals_list,
                                out->n_features);
    }
    Py_DECREF(rows_list); Py_DECREF(cols_list); Py_DECREF(vals_list);
    return result;
}

static PyObject * _gappy_count_kmers(PyObject *self, PyObject *args,
                                      PyObject *kwds) {
    static char *kwlist[] = {"seqs", "masks", "canonical_rc", NULL};
    PyObject *seqs_obj;
    PyObject *masks_obj;
    int canonical_rc = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|p", kwlist,
                                      &seqs_obj, &masks_obj, &canonical_rc)) {
        return NULL;
    }

    PyObject *seqs_fast = PySequence_Fast(seqs_obj, "seqs must be a sequence");
    if (!seqs_fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seqs_fast);

    PyObject *masks_fast = PySequence_Fast(masks_obj, "masks must be a sequence");
    if (!masks_fast) { Py_DECREF(seqs_fast); return NULL; }
    Py_ssize_t n_masks = PySequence_Fast_GET_SIZE(masks_fast);
    if (n_masks > KMER_GAPPY_MAX_MASKS) {
        PyErr_Format(PyExc_ValueError, "too many masks (max %d)",
                     KMER_GAPPY_MAX_MASKS);
        Py_DECREF(seqs_fast); Py_DECREF(masks_fast);
        return NULL;
    }

    gappy_output_t out;
    int rc = _gappy_output_init(&out, (int)n, canonical_rc);
    if (rc != KMER_GAPPY_OK) {
        Py_DECREF(seqs_fast); Py_DECREF(masks_fast);
        return PyErr_NoMemory();
    }

    /* Parse masks. */
    for (Py_ssize_t i = 0; i < n_masks; i++) {
        PyObject *m_obj = PySequence_Fast_GET_ITEM(masks_fast, i);
        Py_ssize_t m_len;
        const char *m_str;
        if (_coerce_seq(m_obj, &m_str, &m_len) < 0) {
            _gappy_output_free(&out);
            Py_DECREF(seqs_fast); Py_DECREF(masks_fast);
            return NULL;
        }
        rc = _gappy_parse_mask(m_str, (int)m_len, &out.masks[out.n_masks]);
        if (rc != KMER_GAPPY_OK) {
            PyErr_Format(PyExc_ValueError,
                         "invalid mask at index %zd: '%.*s'",
                         i, (int)m_len, m_str);
            _gappy_output_free(&out);
            Py_DECREF(seqs_fast); Py_DECREF(masks_fast);
            return NULL;
        }
        out.n_masks++;
    }
    Py_DECREF(masks_fast);

    /* Extract sequences. */
    const char **seq_bufs = (const char **)malloc(n * sizeof(char *));
    int *seq_lens = (int *)malloc(n * sizeof(int));
    if (!seq_bufs || !seq_lens) {
        free(seq_bufs); free(seq_lens);
        _gappy_output_free(&out);
        Py_DECREF(seqs_fast);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seqs_fast, i);
        const char *s; Py_ssize_t len;
        if (_coerce_seq(item, &s, &len) < 0) {
            free(seq_bufs); free(seq_lens);
            _gappy_output_free(&out);
            Py_DECREF(seqs_fast);
            return NULL;
        }
        seq_bufs[i] = s; seq_lens[i] = (int)len;
    }
    Py_DECREF(seqs_fast);

    rc = _gappy_count_batch(seq_bufs, seq_lens, (int)n, canonical_rc, &out);
    free(seq_bufs); free(seq_lens);
    if (rc != KMER_GAPPY_OK) {
        switch (rc) {
        case KMER_GAPPY_ERR_NON_ACGT:
            PyErr_SetString(PyExc_ValueError,
                            "input contains non-ACGT characters");
            break;
        case KMER_GAPPY_ERR_MEMORY:
            PyErr_NoMemory(); break;
        default:
            PyErr_SetString(PyExc_RuntimeError, "gappy count failed");
        }
        _gappy_output_free(&out);
        return NULL;
    }

    PyObject *result = _build_result(&out, /*include_masks=*/0);
    _gappy_output_free(&out);
    return result;
}

static PyMethodDef _gappy_methods[] = {
    {"_count_gappy_kmers",
     (PyCFunction)_gappy_count_kmers, METH_VARARGS | METH_KEYWORDS,
     "_count_gappy_kmers(seqs, masks, canonical_rc=False) -> "
     "(rows, cols, vals, n_features)"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef _gappy_moduledef = {
    PyModuleDef_HEAD_INIT,
    "kmer.encoders._native._gappy",
    "C-backed masked-hash gappy k-mer counter.",
    -1, _gappy_methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__gappy(void) {
    return PyModule_Create(&_gappy_moduledef);
}
