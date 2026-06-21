/* _gkmkern_pylib.c
 *
 * CPython extension that exposes the gkm-kernel C core to Python.
 *
 * Exposed classes:
 *   - Kernel          : regular gkm kernel (uniform L-mer weights)
 *   - WeightedKernel  : weighted gkm kernel (positional weighting)
 *   - Sequence        : encoded nucleotide sequence (owned or borrowed)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "gkmkern.h"

/* ================================================================== */
/* Sequence object                                                    */
/* ================================================================== */

typedef struct {
    PyObject_HEAD
    gkm_sequence_t *seq;
    int borrowed;  /* if 1, the kernel owns it; if 0, we own it */
} SequenceObject;

static PyTypeObject SequenceType;

/* Forward declarations */
static int parse_padding(PyObject *pad_obj, int *pad_left, int *pad_right);

/* ================================================================== */
/* Kernel object (regular)                                            */
/* ================================================================== */

typedef struct {
    PyObject_HEAD
    gkm_kernel_t *kernel;
    PyObject **seq_wrappers;
    int        n_seq_wrappers;
    gkm_window_tree_cache_t *wcache;
    int wcache_window;
    int wcache_shift;
    int wcache_pad_left;
    int wcache_pad_right;
} KernelObject;

static PyTypeObject KernelType;

/* ================================================================== */
/* WeightedKernel object                                              */
/* ================================================================== */

typedef struct {
    PyObject_HEAD
    gkm_weighted_kernel_t *kernel;
    PyObject **seq_wrappers;
    int        n_seq_wrappers;
} WeightedKernelObject;

static PyTypeObject WeightedKernelType;

/* ================================================================== */
/* Shared helpers                                                     */
/* ================================================================== */

/* Accept either a SequenceObject or a raw string and produce a
 * (possibly transient) gkm_sequence_t*. */
static gkm_sequence_t *coerce_sequence_seq(void *kernel_base, PyObject *obj,
                                           int *is_owned)
{
    if (PyObject_TypeCheck(obj, &SequenceType)) {
        SequenceObject *s = (SequenceObject *)obj;
        *is_owned = 0;
        return s->seq;
    }
    if (PyUnicode_Check(obj)) {
        const char *s = PyUnicode_AsUTF8(obj);
        if (!s) return NULL;
        gkm_sequence_t *seq = gkm_sequence_create(kernel_base, s, NULL);
        *is_owned = 1;
        return seq;
    }
    PyErr_SetString(PyExc_TypeError,
        "expected a Sequence or a string");
    return NULL;
}

/* For KernelObject: use kernel->base as the base pointer. */
static gkm_sequence_t *coerce_sequence_kern(KernelObject *kobj, PyObject *obj,
                                            int *is_owned)
{
    return coerce_sequence_seq((void *)gkm_kernel_base(kobj->kernel), obj, is_owned);
}

/* For WeightedKernelObject: use kernel->base as the base pointer. */
static gkm_sequence_t *coerce_sequence_wkern(WeightedKernelObject *kobj,
                                             PyObject *obj, int *is_owned)
{
    return coerce_sequence_seq((void *)gkm_weighted_kernel_base(kobj->kernel), obj, is_owned);
}

/* Parse padding argument: int (symmetric) or (left, right) tuple. */
static int parse_padding(PyObject *pad_obj, int *pad_left, int *pad_right)
{
    *pad_left = 0;
    *pad_right = 0;
    if (pad_obj == NULL || pad_obj == Py_None) return 0;

    if (PyLong_Check(pad_obj)) {
        long v = PyLong_AsLong(pad_obj);
        if (PyErr_Occurred()) return -1;
        /* Negative padding is allowed (skips edge positions). */
        *pad_left = (int)v;
        *pad_right = (int)v;
        return 0;
    }
    if (PyTuple_Check(pad_obj) || PyList_Check(pad_obj)) {
        Py_ssize_t n = PySequence_Size(pad_obj);
        if (n != 2) {
            PyErr_SetString(PyExc_ValueError,
                "padding tuple/list must have 2 elements (left, right)");
            return -1;
        }
        PyObject *l = PySequence_GetItem(pad_obj, 0);
        PyObject *r = PySequence_GetItem(pad_obj, 1);
        long lv = PyLong_AsLong(l);
        long rv = PyLong_AsLong(r);
        Py_DECREF(l); Py_DECREF(r);
        if (PyErr_Occurred()) return -1;
        *pad_left = (int)lv;
        *pad_right = (int)rv;
        return 0;
    }
    PyErr_SetString(PyExc_TypeError,
        "padding must be an int or a (left, right) tuple");
    return -1;
}

/* Build an array of gkm_sequence_t* from a Python list/tuple of strings
 * or Sequence objects. Returns malloc'd array; caller must free.
 * Sets *owned to track which entries need freeing. */
static const gkm_sequence_t **build_seq_array(void *kernel_base,
                                              PyObject *seqs_obj,
                                              int *n_out, int **owned_out)
{
    if (!PyList_Check(seqs_obj) && !PyTuple_Check(seqs_obj)) {
        PyErr_SetString(PyExc_TypeError,
            "must be a list or tuple of strings or Sequence objects");
        return NULL;
    }
    Py_ssize_t n_py = PySequence_Size(seqs_obj);
    if (n_py < 0) return NULL;
    if (n_py == 0) {
        PyErr_SetString(PyExc_ValueError, "sequence list is empty");
        return NULL;
    }
    int n = (int)n_py;

    const gkm_sequence_t **seqs = (const gkm_sequence_t **)malloc(
        sizeof(gkm_sequence_t *) * (size_t)n);
    int *owned = (int *)calloc((size_t)n, sizeof(int));
    if (!seqs || !owned) {
        free(seqs); free(owned);
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(seqs_obj, i);
        if (!item) { goto fail; }
        if (PyObject_TypeCheck(item, &SequenceType)) {
            seqs[i] = ((SequenceObject *)item)->seq;
            owned[i] = 0;
            Py_DECREF(item);
        } else if (PyUnicode_Check(item)) {
            const char *s = PyUnicode_AsUTF8(item);
            if (!s) { Py_DECREF(item); goto fail; }
            gkm_sequence_t *gs = gkm_sequence_create(kernel_base, s, NULL);
            if (!gs) {
                Py_DECREF(item);
                PyErr_SetString(PyExc_ValueError,
                    "could not create sequence (too short for L, or contains non-ACGT characters)");
                goto fail;
            }
            seqs[i] = gs;
            owned[i] = 1;
            Py_DECREF(item);
        } else {
            Py_DECREF(item);
            PyErr_SetString(PyExc_TypeError,
                "elements must be str or Sequence");
            goto fail;
        }
    }

    *n_out = n;
    *owned_out = owned;
    return seqs;

fail:
    for (int i = 0; i < n; i++) {
        if (owned[i]) gkm_sequence_free((gkm_sequence_t *)seqs[i]);
    }
    free(seqs); free(owned);
    return NULL;
}

static void free_seq_array(const gkm_sequence_t **seqs, int *owned, int n)
{
    for (int i = 0; i < n; i++) {
        if (owned[i]) gkm_sequence_free((gkm_sequence_t *)seqs[i]);
    }
    free(seqs); free(owned);
}

/* ------------------------------------------------------------------ */
/* Flat-buffer → nested-Python-list conversion helpers               */
/* ------------------------------------------------------------------ */
/* These eliminate the repeated PyList_New, PyFloat_FromDouble, and
 * PyList_SET_ITEM boilerplate that appears in every method.        */

/* Convert a flat double array to a 2D nested Python list of shape
 * (n_rows, n_cols), row-major. */
static PyObject *flat_to_2d_list(const double *buf, int n_rows, int n_cols)
{
    PyObject *out = PyList_New(n_rows);
    if (!out) return NULL;
    for (int i = 0; i < n_rows; i++) {
        PyObject *row = PyList_New(n_cols);
        if (!row) { Py_DECREF(out); return NULL; }
        for (int j = 0; j < n_cols; j++) {
            PyList_SET_ITEM(row, j,
                PyFloat_FromDouble(buf[(size_t)i * n_cols + j]));
        }
        PyList_SET_ITEM(out, i, row);
    }
    return out;
}

/* Convert a flat double array to a 3D nested Python list of shape
 * (n_outer, n_rows, n_cols), row-major. */
static PyObject *flat_to_3d_list(const double *buf, int n_outer,
                                 int n_rows, int n_cols)
{
    PyObject *out = PyList_New(n_outer);
    if (!out) return NULL;
    for (int k = 0; k < n_outer; k++) {
        PyObject *matrix = PyList_New(n_rows);
        if (!matrix) { Py_DECREF(out); return NULL; }
        for (int i = 0; i < n_rows; i++) {
            PyObject *row = PyList_New(n_cols);
            if (!row) { Py_DECREF(matrix); Py_DECREF(out); return NULL; }
            for (int j = 0; j < n_cols; j++) {
                PyList_SET_ITEM(row, j,
                    PyFloat_FromDouble(
                        buf[(size_t)k * n_rows * n_cols
                           + (size_t)i * n_cols + j]));
            }
            PyList_SET_ITEM(matrix, i, row);
        }
        PyList_SET_ITEM(out, k, matrix);
    }
    return out;
}

/* ================================================================== */
/* Sequence type                                                      */
/* ================================================================== */

static int Sequence_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    KernelObject *kern = NULL;
    WeightedKernelObject *wkern = NULL;
    const char *seq_str;
    const char *sid_str = NULL;
    static char *kwlist[] = { "kernel", "seq", "sid", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Os|s", kwlist,
                                     &KernelType, &kern,
                                     &seq_str, &sid_str)) {
        PyErr_Clear();
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "Os|s", kwlist,
                                         &WeightedKernelType, &wkern,
                                         &seq_str, &sid_str))
            return -1;
    }
    void *base;
    if (kern) base = (void *)gkm_kernel_base(kern->kernel);
    else      base = (void *)gkm_weighted_kernel_base(wkern->kernel);

    SequenceObject *s = (SequenceObject *)self;
    s->seq = gkm_sequence_create(base, seq_str, sid_str);
    if (!s->seq) {
        PyErr_SetString(PyExc_ValueError,
            "gkm_sequence_create failed (sequence too short for L, or contains non-ACGT characters)");
        return -1;
    }
    s->borrowed = 0;
    return 0;
}

static void Sequence_dealloc(PyObject *self)
{
    SequenceObject *s = (SequenceObject *)self;
    if (s->seq && !s->borrowed) gkm_sequence_free(s->seq);
    Py_TYPE(self)->tp_free(self);
}

static PyObject *Sequence_length(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyLong_FromLong(gkm_sequence_length(((SequenceObject *)self)->seq));
}

static PyObject *Sequence_sqnorm(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyFloat_FromDouble(gkm_sequence_sqnorm(((SequenceObject *)self)->seq));
}

static PyObject *Sequence_sid(PyObject *self, PyObject *Py_UNUSED(arg))
{
    const char *sid = gkm_sequence_sid(((SequenceObject *)self)->seq);
    if (!sid) Py_RETURN_NONE;
    return PyUnicode_FromString(sid);
}

static PyMethodDef Sequence_methods[] = {
    {"length", Sequence_length, METH_NOARGS, "Return the sequence length."},
    {"sqnorm", Sequence_sqnorm, METH_NOARGS, "Return sqrt(K(self, self))."},
    {"sid",    Sequence_sid,    METH_NOARGS, "Return the sequence id (or None)."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject SequenceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gapped_kmer_kernel._gkmkern.Sequence",
    .tp_basicsize = sizeof(SequenceObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Encoded nucleotide sequence.",
    .tp_init = Sequence_init,
    .tp_dealloc = Sequence_dealloc,
    .tp_methods = Sequence_methods,
    .tp_new = PyType_GenericNew,
};

/* ================================================================== */
/* Shared: add_sequence, finalize, num_sequences, weight, etc.        */
/* ================================================================== */

/* Helper: add a sequence to a kernel (works for both types via base). */
static PyObject *common_add_sequence(void *kernel_base, PyObject *args,
                                     PyObject *kwds)
{
    const char *seq_str;
    const char *sid_str = NULL;
    static char *kwlist[] = { "seq", "sid", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|z", kwlist,
                                     &seq_str, &sid_str)) return NULL;

    gkm_sequence_t *s = gkm_sequence_create(kernel_base, seq_str, sid_str);
    if (!s) {
        PyErr_SetString(PyExc_ValueError,
            "sequence too short for L, or contains non-ACGT characters");
        return NULL;
    }
    int seqid = gkm_kernel_base_add_sequence(kernel_base, s);
    if (seqid < 0) {
        gkm_sequence_free(s);
        PyErr_SetString(PyExc_RuntimeError,
            "kernel already finalized; cannot add more sequences");
        return NULL;
    }
    return PyLong_FromLong(seqid);
}

/* ================================================================== */
/* Kernel: init / dealloc                                             */
/* ================================================================== */

static int Kernel_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    int L = 10, k = 6, d = 3;
    int weight_scheme = 0;
    int use_rc = 0;
    int nthreads = 1;
    static char *kwlist[] = { "L", "k", "d", "weight_scheme", "use_rc",
                              "nthreads", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiiii", kwlist,
                                     &L, &k, &d, &weight_scheme, &use_rc,
                                     &nthreads))
        return -1;
    gkm_parameter_t p;
    p.L = L; p.k = k; p.d = d;
    p.weight_scheme = (gkm_weight_scheme_t)weight_scheme;
    p.use_rc = use_rc ? 1 : 0;
    p.nthreads = nthreads;
    const char *err = gkm_parameter_check(&p);
    if (err) {
        PyErr_SetString(PyExc_ValueError, err);
        return -1;
    }
    KernelObject *kobj = (KernelObject *)self;
    kobj->kernel = gkm_kernel_create(&p);
    if (!kobj->kernel) {
        PyErr_SetString(PyExc_RuntimeError, "gkm_kernel_create failed");
        return -1;
    }
    kobj->seq_wrappers = NULL;
    kobj->n_seq_wrappers = 0;
    kobj->wcache = NULL;
    return 0;
}

static void Kernel_dealloc(PyObject *self)
{
    KernelObject *kobj = (KernelObject *)self;
    if (kobj->wcache) gkm_window_tree_cache_destroy(kobj->wcache);
    if (kobj->seq_wrappers) {
        for (int i = 0; i < kobj->n_seq_wrappers; i++)
            Py_XDECREF(kobj->seq_wrappers[i]);
        free(kobj->seq_wrappers);
    }
    if (kobj->kernel) gkm_kernel_destroy(kobj->kernel);
    Py_TYPE(self)->tp_free(self);
}

/* ================================================================== */
/* Kernel: methods                                                    */
/* ================================================================== */

static PyObject *Kernel_add_sequence(PyObject *self, PyObject *args, PyObject *kwds)
{
    KernelObject *kobj = (KernelObject *)self;
    return common_add_sequence((void *)gkm_kernel_base(kobj->kernel), args, kwds);
}

static PyObject *Kernel_finalize(PyObject *self, PyObject *Py_UNUSED(arg))
{
    KernelObject *kobj = (KernelObject *)self;
    gkm_kernel_finalize(kobj->kernel);
    Py_RETURN_NONE;
}

static PyObject *Kernel_num_sequences(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyLong_FromLong(gkm_kernel_base_num_sequences(
        gkm_kernel_base(((KernelObject *)self)->kernel)));
}

static PyObject *Kernel_weight(PyObject *self, PyObject *args)
{
    int m;
    if (!PyArg_ParseTuple(args, "i", &m)) return NULL;
    return PyFloat_FromDouble(gkm_kernel_base_weight(
        gkm_kernel_base(((KernelObject *)self)->kernel), m));
}

static PyObject *Kernel_weight_scheme(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyLong_FromLong(gkm_kernel_base_get_weight_scheme(
        gkm_kernel_base(((KernelObject *)self)->kernel)));
}

static PyObject *Kernel_use_rc(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyLong_FromLong(gkm_kernel_base_get_use_rc(
        gkm_kernel_base(((KernelObject *)self)->kernel)));
}

static PyObject *Kernel_get_L(PyObject *self, void *closure)
{
    int L, k, d;
    gkm_kernel_base_get_params(gkm_kernel_base(((KernelObject *)self)->kernel), &L, &k, &d);
    return PyLong_FromLong(L);
}
static PyObject *Kernel_get_k(PyObject *self, void *closure)
{
    int L, k, d;
    gkm_kernel_base_get_params(gkm_kernel_base(((KernelObject *)self)->kernel), &L, &k, &d);
    return PyLong_FromLong(k);
}
static PyObject *Kernel_get_d(PyObject *self, void *closure)
{
    int L, k, d;
    gkm_kernel_base_get_params(gkm_kernel_base(((KernelObject *)self)->kernel), &L, &k, &d);
    return PyLong_FromLong(d);
}

static PyGetSetDef Kernel_getset[] = {
    {"L", Kernel_get_L, NULL, "Full word length L.", NULL},
    {"k", Kernel_get_k, NULL, "Number of non-gap positions k.", NULL},
    {"d", Kernel_get_d, NULL, "Max mismatches d.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

/* Kernel.eval_batch(queries) -> list[list[float]] */
static PyObject *Kernel_eval_batch(PyObject *self, PyObject *args)
{
    PyObject *queries_obj;
    if (!PyArg_ParseTuple(args, "O", &queries_obj)) return NULL;
    KernelObject *kobj = (KernelObject *)self;

    int n_q;
    int *q_owned;
    const gkm_sequence_t **qs = build_seq_array((void *)gkm_kernel_base(kobj->kernel),
                                                 queries_obj, &n_q, &q_owned);
    if (!qs) return NULL;

    int n_ref = gkm_kernel_base_num_sequences(gkm_kernel_base(kobj->kernel));
    double *res = (double *)malloc(sizeof(double) * (size_t)n_q * (size_t)(n_ref ? n_ref : 1));
    if (!res) { PyErr_NoMemory(); free_seq_array(qs, q_owned, n_q); return NULL; }

    gkm_kernel_eval_batch(kobj->kernel, qs, n_q, res);

    PyObject *out = flat_to_2d_list(res, n_q, n_ref);

    free(res);
    free_seq_array(qs, q_owned, n_q);
    return out;
}

/* Kernel.sliding_query(seq, window, shift, padding=0) -> list[list[float]] */
static PyObject *Kernel_sliding_query(PyObject *self, PyObject *args,
                                                    PyObject *kwds)
{
    PyObject *q;
    int window, shift;
    PyObject *pad_obj = NULL;
    static char *kwlist[] = { "seq", "window", "shift", "padding", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oii|O", kwlist,
                                     &q, &window, &shift, &pad_obj)) return NULL;
    KernelObject *kobj = (KernelObject *)self;

    int pad_left, pad_right;
    if (parse_padding(pad_obj, &pad_left, &pad_right) < 0) return NULL;

    int owned = 0;
    gkm_sequence_t *qs = coerce_sequence_kern(kobj, q, &owned);
    if (!qs) return NULL;

    gkm_window_iter_t *it = gkm_window_iter_create(kobj->kernel, qs,
                                                    window, shift,
                                                    pad_left, pad_right);
    if (!it) {
        if (owned) gkm_sequence_free(qs);
        PyErr_SetString(PyExc_ValueError,
            "could not create window iterator (check window/shift/padding)");
        return NULL;
    }
    int n_windows = gkm_window_iter_count(it);
    int n_refs = gkm_kernel_base_num_sequences(gkm_kernel_base(kobj->kernel));

    PyObject *out = PyList_New(n_windows);
    if (!out) {
        gkm_window_iter_destroy(it);
        if (owned) gkm_sequence_free(qs);
        return NULL;
    }
    double *res = (double *)malloc(sizeof(double) * (size_t)(n_refs ? n_refs : 1));
    for (int i = 0; i < n_windows; i++) {
        if (!gkm_window_iter_next(it, res)) {
            free(res);
            gkm_window_iter_destroy(it);
            if (owned) gkm_sequence_free(qs);
            Py_DECREF(out);
            PyErr_SetString(PyExc_RuntimeError, "window iter ended early");
            return NULL;
        }
        PyObject *row = PyList_New(n_refs);
        for (int j = 0; j < n_refs; j++) {
            PyList_SET_ITEM(row, j, PyFloat_FromDouble(res[j]));
        }
        PyList_SET_ITEM(out, i, row);
    }
    free(res);
    gkm_window_iter_destroy(it);
    if (owned) gkm_sequence_free(qs);
    return out;
}

/* Kernel.self_window_eval(seqs, window, shift, padding=0) -> 3D list (n_windows, n, n) */
static PyObject *Kernel_self_window_eval(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *seqs_obj;
    int window, shift;
    PyObject *pad_obj = NULL;
    static char *kwlist[] = { "seqs", "window", "shift", "padding", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oii|O", kwlist,
                                     &seqs_obj, &window, &shift, &pad_obj)) return NULL;
    KernelObject *kobj = (KernelObject *)self;

    int pad_left, pad_right;
    if (parse_padding(pad_obj, &pad_left, &pad_right) < 0) return NULL;

    int n;
    int *owned;
    const gkm_sequence_t **seqs = build_seq_array((void *)gkm_kernel_base(kobj->kernel),
                                                   seqs_obj, &n, &owned);
    if (!seqs) return NULL;

    int seqlen = gkm_sequence_length(seqs[0]);
    if (seqlen < window) {
        PyErr_SetString(PyExc_ValueError, "sequences shorter than window");
        free_seq_array(seqs, owned, n);
        return NULL;
    }

    /* Count valid windows. */
    int n_windows = gkm_compute_n_windows(seqlen, window, shift,
                                      pad_left, pad_right,
                                      gkm_kernel_base_get_L(gkm_kernel_base(kobj->kernel)));
    if (n_windows <= 0) {
        PyErr_SetString(PyExc_ValueError, "no valid windows with given padding");
        free_seq_array(seqs, owned, n);
        return NULL;
    }

    double *out_buf = (double *)malloc(
        sizeof(double) * (size_t)n_windows * (size_t)n * (size_t)n);
    if (!out_buf) { PyErr_NoMemory(); free_seq_array(seqs, owned, n); return NULL; }

    int rc = gkm_kernel_window_kernel(kobj->kernel, seqs, n, window, shift,
                                      pad_left, pad_right, out_buf);
    if (rc != 0) {
        free(out_buf);
        free_seq_array(seqs, owned, n);
        PyErr_SetString(PyExc_RuntimeError, "gkm_kernel_window_kernel failed");
        return NULL;
    }

    PyObject *result = flat_to_3d_list(out_buf, n_windows, n, n);

    free(out_buf);
    free_seq_array(seqs, owned, n);
    return result;
}

/* Kernel.cross_window_eval(queries, refs, window, shift, padding=0) -> 3D list */
static PyObject *Kernel_cross_window_eval(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *queries_obj, *refs_obj;
    int window, shift;
    PyObject *pad_obj = NULL;
    static char *kwlist[] = { "queries", "refs", "window", "shift", "padding", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOii|O", kwlist,
                                     &queries_obj, &refs_obj, &window, &shift,
                                     &pad_obj)) return NULL;
    KernelObject *kobj = (KernelObject *)self;

    int pad_left, pad_right;
    if (parse_padding(pad_obj, &pad_left, &pad_right) < 0) return NULL;

    int n_q, n_ref;
    int *q_owned, *r_owned;
    const gkm_sequence_t **qs = build_seq_array((void *)gkm_kernel_base(kobj->kernel),
                                                 queries_obj, &n_q, &q_owned);
    if (!qs) return NULL;
    const gkm_sequence_t **rs = build_seq_array((void *)gkm_kernel_base(kobj->kernel),
                                                 refs_obj, &n_ref, &r_owned);
    if (!rs) { free_seq_array(qs, q_owned, n_q); return NULL; }

    int seqlen = gkm_sequence_length(qs[0]);
    int n_windows = gkm_compute_n_windows(seqlen, window, shift,
                                      pad_left, pad_right,
                                      gkm_kernel_base_get_L(gkm_kernel_base(kobj->kernel)));
    if (n_windows <= 0) {
        PyErr_SetString(PyExc_ValueError, "no valid windows with given padding");
        free_seq_array(qs, q_owned, n_q);
        free_seq_array(rs, r_owned, n_ref);
        return NULL;
    }

    double *out_buf = (double *)malloc(
        sizeof(double) * (size_t)n_windows * (size_t)n_q * (size_t)n_ref);
    if (!out_buf) {
        PyErr_NoMemory();
        free_seq_array(qs, q_owned, n_q);
        free_seq_array(rs, r_owned, n_ref);
        return NULL;
    }

    /* Use cached window trees if available (same window/shift/padding). */
    if (kobj->wcache && kobj->wcache_window == window &&
        kobj->wcache_shift == shift &&
        kobj->wcache_pad_left == pad_left &&
        kobj->wcache_pad_right == pad_right) {
        int rc = gkm_window_tree_cache_query(kobj->kernel, kobj->wcache,
                                             qs, n_q, out_buf);
        if (rc != 0) {
            free(out_buf);
            PyErr_SetString(PyExc_RuntimeError, "window_tree_cache_query failed");
            goto cwk_fail;
        }
    } else {
        if (kobj->wcache) {
            gkm_window_tree_cache_destroy(kobj->wcache);
            kobj->wcache = NULL;
        }
        kobj->wcache = gkm_window_tree_cache_build(kobj->kernel, rs, n_ref,
                                                    window, shift,
                                                    pad_left, pad_right);
        if (!kobj->wcache) {
            free(out_buf);
            PyErr_SetString(PyExc_RuntimeError, "window_tree_cache_build failed");
            goto cwk_fail;
        }
        kobj->wcache_window = window;
        kobj->wcache_shift = shift;
        kobj->wcache_pad_left = pad_left;
        kobj->wcache_pad_right = pad_right;
        int rc = gkm_window_tree_cache_query(kobj->kernel, kobj->wcache,
                                             qs, n_q, out_buf);
        if (rc != 0) {
            free(out_buf);
            PyErr_SetString(PyExc_RuntimeError, "window_tree_cache_query failed");
            goto cwk_fail;
        }
    }

    /* Convert to nested list. */
    {
    PyObject *result = flat_to_3d_list(out_buf, n_windows, n_q, n_ref);
    free(out_buf);
    free_seq_array(qs, q_owned, n_q);
    free_seq_array(rs, r_owned, n_ref);
    return result;
    }

cwk_fail:
    free_seq_array(qs, q_owned, n_q);
    free_seq_array(rs, r_owned, n_ref);
    return NULL;
}

/* ================================================================== */
/* WeightedKernel: init / dealloc                                     */
/* ================================================================== */

static int WeightedKernel_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    int L = 10, k = 6, d = 3;
    int weight_scheme = 0;
    int use_rc = 0;
    int pos_kernel = 0;  /* GKM_POS_TRIANGULAR */
    double weight_gamma = 1.0;
    double weight_sigma = 50.0;
    double weight_peak = 50.0;
    int nthreads = 1;
    static char *kwlist[] = { "L", "k", "d", "weight_scheme", "use_rc",
                              "pos_kernel", "weight_gamma", "weight_sigma",
                              "weight_peak", "nthreads", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiiiidddi", kwlist,
                                     &L, &k, &d, &weight_scheme, &use_rc,
                                     &pos_kernel, &weight_gamma, &weight_sigma,
                                     &weight_peak, &nthreads))
        return -1;

    gkm_weighted_params_t p;
    p.base.L = L; p.base.k = k; p.base.d = d;
    p.base.weight_scheme = (gkm_weight_scheme_t)weight_scheme;
    p.base.use_rc = use_rc ? 1 : 0;
    p.base.nthreads = nthreads;
    p.pos_kernel = (gkm_pos_kernel_t)pos_kernel;
    p.weight_gamma = weight_gamma;
    p.weight_sigma = weight_sigma;
    p.weight_peak = weight_peak;

    const char *err = gkm_parameter_check(&p.base);
    if (err) {
        PyErr_SetString(PyExc_ValueError, err);
        return -1;
    }

    WeightedKernelObject *kobj = (WeightedKernelObject *)self;
    kobj->kernel = gkm_weighted_kernel_create(&p);
    if (!kobj->kernel) {
        PyErr_SetString(PyExc_RuntimeError, "gkm_weighted_kernel_create failed");
        return -1;
    }
    kobj->seq_wrappers = NULL;
    kobj->n_seq_wrappers = 0;
    return 0;
}

static void WeightedKernel_dealloc(PyObject *self)
{
    WeightedKernelObject *kobj = (WeightedKernelObject *)self;
    if (kobj->seq_wrappers) {
        for (int i = 0; i < kobj->n_seq_wrappers; i++)
            Py_XDECREF(kobj->seq_wrappers[i]);
        free(kobj->seq_wrappers);
    }
    if (kobj->kernel) gkm_weighted_kernel_destroy(kobj->kernel);
    Py_TYPE(self)->tp_free(self);
}

/* ================================================================== */
/* WeightedKernel: methods                                            */
/* ================================================================== */

static PyObject *WeightedKernel_add_sequence(PyObject *self, PyObject *args, PyObject *kwds)
{
    WeightedKernelObject *kobj = (WeightedKernelObject *)self;
    return common_add_sequence((void *)gkm_weighted_kernel_base(kobj->kernel), args, kwds);
}

static PyObject *WeightedKernel_finalize(PyObject *self, PyObject *Py_UNUSED(arg))
{
    WeightedKernelObject *kobj = (WeightedKernelObject *)self;
    gkm_kernel_base_finalize(gkm_weighted_kernel_base(kobj->kernel));
    Py_RETURN_NONE;
}

static PyObject *WeightedKernel_num_sequences(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyLong_FromLong(gkm_kernel_base_num_sequences(
        gkm_weighted_kernel_base(((WeightedKernelObject *)self)->kernel)));
}

static PyObject *WeightedKernel_weight(PyObject *self, PyObject *args)
{
    int m;
    if (!PyArg_ParseTuple(args, "i", &m)) return NULL;
    return PyFloat_FromDouble(gkm_kernel_base_weight(
        gkm_weighted_kernel_base(((WeightedKernelObject *)self)->kernel), m));
}

static PyObject *WeightedKernel_weight_scheme(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyLong_FromLong(gkm_kernel_base_get_weight_scheme(
        gkm_weighted_kernel_base(((WeightedKernelObject *)self)->kernel)));
}

static PyObject *WeightedKernel_use_rc(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyLong_FromLong(gkm_kernel_base_get_use_rc(
        gkm_weighted_kernel_base(((WeightedKernelObject *)self)->kernel)));
}

static PyObject *WeightedKernel_pos_kernel(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyLong_FromLong((int)gkm_weighted_kernel_pos_kernel(
        ((WeightedKernelObject *)self)->kernel));
}

static PyObject *WeightedKernel_gamma(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyFloat_FromDouble(gkm_weighted_kernel_gamma(
        ((WeightedKernelObject *)self)->kernel));
}

static PyObject *WeightedKernel_sigma(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyFloat_FromDouble(gkm_weighted_kernel_sigma(
        ((WeightedKernelObject *)self)->kernel));
}

static PyObject *WeightedKernel_peak(PyObject *self, PyObject *Py_UNUSED(arg))
{
    return PyFloat_FromDouble(gkm_weighted_kernel_peak(
        ((WeightedKernelObject *)self)->kernel));
}

static PyObject *WeightedKernel_get_L(PyObject *self, void *closure)
{
    int L, k, d;
    gkm_kernel_base_get_params(gkm_weighted_kernel_base(((WeightedKernelObject *)self)->kernel),
                               &L, &k, &d);
    return PyLong_FromLong(L);
}
static PyObject *WeightedKernel_get_k(PyObject *self, void *closure)
{
    int L, k, d;
    gkm_kernel_base_get_params(gkm_weighted_kernel_base(((WeightedKernelObject *)self)->kernel),
                               &L, &k, &d);
    return PyLong_FromLong(k);
}
static PyObject *WeightedKernel_get_d(PyObject *self, void *closure)
{
    int L, k, d;
    gkm_kernel_base_get_params(gkm_weighted_kernel_base(((WeightedKernelObject *)self)->kernel),
                               &L, &k, &d);
    return PyLong_FromLong(d);
}

static PyGetSetDef WeightedKernel_getset[] = {
    {"L", WeightedKernel_get_L, NULL, "Full word length L.", NULL},
    {"k", WeightedKernel_get_k, NULL, "Number of non-gap positions k.", NULL},
    {"d", WeightedKernel_get_d, NULL, "Max mismatches d.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

/* Helper: resolve the `offsets` argument to a list of center positions
 * (in base-pair coordinates). Returns a malloc'd double array; caller frees.
 * `seqlen` is needed to compute the midpoint.
 * `n_centers_out` receives the count.
 * Returns NULL on error. */
static double *resolve_offsets(PyObject *offsets_obj, int seqlen, int L,
                               int *n_centers_out)
{
    int nkmers = seqlen - L + 1;
    /* The sequence midpoint: for even seqlen, this is half-integer
     * (e.g. 49.5 for seqlen=100); for odd, integer (e.g. 49 for seqlen=99).
     * This is (seqlen - 1) / 2.0 in base-pair coordinates. */
    double midpoint = (seqlen - 1) / 2.0;

    if (offsets_obj == NULL || offsets_obj == Py_None) {
        /* Single center: the midpoint. */
        double *centers = (double *)malloc(sizeof(double) * 1);
        centers[0] = midpoint;
        *n_centers_out = 1;
        return centers;
    }

    /* Int or float: single offset. */
    if (PyLong_Check(offsets_obj) || PyFloat_Check(offsets_obj)) {
        double off = PyFloat_AsDouble(offsets_obj);
        if (PyErr_Occurred()) return NULL;
        double *centers = (double *)malloc(sizeof(double) * 1);
        centers[0] = midpoint + off;
        *n_centers_out = 1;
        return centers;
    }

    /* Slice: expand to range. */
    if (PySlice_Check(offsets_obj)) {
        Py_ssize_t start, stop, step, slicelen;
        if (PySlice_GetIndicesEx(offsets_obj, nkmers, &start, &stop, &step, &slicelen) < 0)
            return NULL;
        double *centers = (double *)malloc(sizeof(double) * (size_t)slicelen);
        for (int i = 0; i < slicelen; i++) {
            /* Each L-mer offset j has midpoint j + (L-1)/2.0. */
            int j = (int)(start + i * step);
            centers[i] = j + (L - 1) / 2.0;
        }
        *n_centers_out = (int)slicelen;
        return centers;
    }

    /* List or tuple of int/float. */
    if (PyList_Check(offsets_obj) || PyTuple_Check(offsets_obj)) {
        Py_ssize_t n = PySequence_Size(offsets_obj);
        if (n < 0) return NULL;
        if (n == 0) {
            PyErr_SetString(PyExc_ValueError, "offsets list is empty");
            return NULL;
        }
        double *centers = (double *)malloc(sizeof(double) * (size_t)n);
        for (int i = 0; i < n; i++) {
            PyObject *item = PySequence_GetItem(offsets_obj, i);
            double off = PyFloat_AsDouble(item);
            Py_DECREF(item);
            if (PyErr_Occurred()) { free(centers); return NULL; }
            centers[i] = midpoint + off;
        }
        *n_centers_out = (int)n;
        return centers;
    }

    PyErr_SetString(PyExc_TypeError,
        "offsets must be None, int, float, slice, or list/tuple of int/float");
    return NULL;
}

/* WeightedKernel.self_eval(seqs, offsets=None) -> 2D or 3D list */
static PyObject *WeightedKernel_self_eval(PyObject *self, PyObject *args,
                                                    PyObject *kwds)
{
    PyObject *seqs_obj;
    PyObject *offsets_obj = NULL;
    static char *kwlist[] = { "seqs", "offsets", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                     &seqs_obj, &offsets_obj)) return NULL;
    WeightedKernelObject *kobj = (WeightedKernelObject *)self;

    int n;
    int *owned;
    const gkm_sequence_t **seqs = build_seq_array((void *)gkm_weighted_kernel_base(kobj->kernel),
                                                   seqs_obj, &n, &owned);
    if (!seqs) return NULL;

    int seqlen = gkm_sequence_length(seqs[0]);
    int L = gkm_kernel_base_get_L(gkm_weighted_kernel_base(kobj->kernel));
    int n_centers;
    double *centers = resolve_offsets(offsets_obj, seqlen, L, &n_centers);
    if (!centers) { free_seq_array(seqs, owned, n); return NULL; }

    double *out_buf = (double *)malloc(
        sizeof(double) * (size_t)n_centers * (size_t)n * (size_t)n);
    if (!out_buf) {
        PyErr_NoMemory();
        free(centers);
        free_seq_array(seqs, owned, n);
        return NULL;
    }

    int rc = gkm_weighted_kernel_eval_self(kobj->kernel, seqs, n,
                                           centers, n_centers, out_buf);
    if (rc != 0) {
        free(out_buf);
        free(centers);
        free_seq_array(seqs, owned, n);
        PyErr_SetString(PyExc_RuntimeError, "gkm_weighted_kernel_eval_self failed");
        return NULL;
    }

    /* If n_centers == 1, return 2D (n, n). Otherwise 3D (n_centers, n, n). */
    PyObject *result;
    if (n_centers == 1) {
        result = flat_to_2d_list(out_buf, n, n);
    } else {
        result = flat_to_3d_list(out_buf, n_centers, n, n);
    }

    free(out_buf);
    free(centers);
    free_seq_array(seqs, owned, n);
    return result;
}

/* WeightedKernel.cross_eval(queries, refs, offsets=None) -> 2D or 3D list */
static PyObject *WeightedKernel_cross_eval(PyObject *self, PyObject *args,
                                                     PyObject *kwds)
{
    PyObject *queries_obj, *refs_obj;
    PyObject *offsets_obj = NULL;
    static char *kwlist[] = { "queries", "refs", "offsets", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O", kwlist,
                                     &queries_obj, &refs_obj, &offsets_obj)) return NULL;
    WeightedKernelObject *kobj = (WeightedKernelObject *)self;

    int n_q, n_ref;
    int *q_owned, *r_owned;
    const gkm_sequence_t **qs = build_seq_array((void *)gkm_weighted_kernel_base(kobj->kernel),
                                                 queries_obj, &n_q, &q_owned);
    if (!qs) return NULL;
    const gkm_sequence_t **rs = build_seq_array((void *)gkm_weighted_kernel_base(kobj->kernel),
                                                 refs_obj, &n_ref, &r_owned);
    if (!rs) { free_seq_array(qs, q_owned, n_q); return NULL; }

    int seqlen = gkm_sequence_length(qs[0]);
    int L = gkm_kernel_base_get_L(gkm_weighted_kernel_base(kobj->kernel));
    int n_centers;
    double *centers = resolve_offsets(offsets_obj, seqlen, L, &n_centers);
    if (!centers) { free_seq_array(qs, q_owned, n_q); free_seq_array(rs, r_owned, n_ref); return NULL; }

    double *out_buf = (double *)malloc(
        sizeof(double) * (size_t)n_centers * (size_t)n_q * (size_t)n_ref);
    if (!out_buf) {
        PyErr_NoMemory();
        free(centers);
        free_seq_array(qs, q_owned, n_q);
        free_seq_array(rs, r_owned, n_ref);
        return NULL;
    }

    int rc = gkm_weighted_kernel_eval_cross(kobj->kernel, qs, n_q, rs, n_ref,
                                            centers, n_centers, out_buf);
    if (rc != 0) {
        free(out_buf);
        free(centers);
        free_seq_array(qs, q_owned, n_q);
        free_seq_array(rs, r_owned, n_ref);
        PyErr_SetString(PyExc_RuntimeError, "gkm_weighted_kernel_eval_cross failed");
        return NULL;
    }

    PyObject *result;
    if (n_centers == 1) {
        result = flat_to_2d_list(out_buf, n_q, n_ref);
    } else {
        result = flat_to_3d_list(out_buf, n_centers, n_q, n_ref);
    }

    free(out_buf);
    free(centers);
    free_seq_array(qs, q_owned, n_q);
    free_seq_array(rs, r_owned, n_ref);
    return result;
}

/* WeightedKernel.window_kernel(seqs, window, shift, padding=0) -> 3D list */
static PyObject *WeightedKernel_self_window_eval(PyObject *self, PyObject *args,
                                              PyObject *kwds)
{
    PyObject *seqs_obj;
    int window, shift;
    PyObject *pad_obj = NULL;
    static char *kwlist[] = { "seqs", "window", "shift", "padding", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oii|O", kwlist,
                                     &seqs_obj, &window, &shift, &pad_obj)) return NULL;
    WeightedKernelObject *kobj = (WeightedKernelObject *)self;

    int pad_left, pad_right;
    if (parse_padding(pad_obj, &pad_left, &pad_right) < 0) return NULL;

    int n;
    int *owned;
    const gkm_sequence_t **seqs = build_seq_array((void *)gkm_weighted_kernel_base(kobj->kernel),
                                                   seqs_obj, &n, &owned);
    if (!seqs) return NULL;

    int seqlen = gkm_sequence_length(seqs[0]);
    int n_windows = gkm_compute_n_windows(seqlen, window, shift,
                                      pad_left, pad_right,
                                      gkm_kernel_base_get_L(gkm_weighted_kernel_base(kobj->kernel)));
    if (n_windows <= 0) {
        PyErr_SetString(PyExc_ValueError, "no valid windows with given padding");
        free_seq_array(seqs, owned, n);
        return NULL;
    }

    double *out_buf = (double *)malloc(
        sizeof(double) * (size_t)n_windows * (size_t)n * (size_t)n);
    if (!out_buf) { PyErr_NoMemory(); free_seq_array(seqs, owned, n); return NULL; }

    int rc = gkm_weighted_kernel_window_kernel(kobj->kernel, seqs, n,
                                               window, shift,
                                               pad_left, pad_right, out_buf);
    if (rc != 0) {
        free(out_buf);
        free_seq_array(seqs, owned, n);
        PyErr_SetString(PyExc_RuntimeError, "weighted window_kernel failed");
        return NULL;
    }

    PyObject *result = flat_to_3d_list(out_buf, n_windows, n, n);

    free(out_buf);
    free_seq_array(seqs, owned, n);
    return result;
}

/* WeightedKernel.sliding_query(seq, window, shift, padding=0) -> list[list[float]] */
static PyObject *WeightedKernel_sliding_query(PyObject *self, PyObject *args,
                                              PyObject *kwds)
{
    PyObject *q;
    int window, shift;
    PyObject *pad_obj = NULL;
    static char *kwlist[] = { "seq", "window", "shift", "padding", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oii|O", kwlist,
                                     &q, &window, &shift, &pad_obj)) return NULL;
    WeightedKernelObject *kobj = (WeightedKernelObject *)self;

    int pad_left, pad_right;
    if (parse_padding(pad_obj, &pad_left, &pad_right) < 0) return NULL;

    int owned = 0;
    gkm_sequence_t *qs = coerce_sequence_wkern(kobj, q, &owned);
    if (!qs) return NULL;

    int n_ref = gkm_kernel_base_num_sequences(gkm_weighted_kernel_base(kobj->kernel));
    /* Pre-allocate a generous buffer. With negative padding, more windows
     * may be produced, so add headroom proportional to |pad|. */
    int seqlen_q = gkm_sequence_length(qs);
    int max_windows = 2 + (seqlen_q + abs(pad_left) + abs(pad_right) - window + shift) / shift;
    if (max_windows < 1) max_windows = 1;
    double *out_buf = (double *)malloc(
        sizeof(double) * (size_t)max_windows * (size_t)(n_ref ? n_ref : 1));
    if (!out_buf) { PyErr_NoMemory(); if (owned) gkm_sequence_free(qs); return NULL; }

    int n_windows = 0;
    int rc = gkm_weighted_kernel_sliding_query(kobj->kernel, qs,
                                               window, shift,
                                               pad_left, pad_right,
                                               out_buf, &n_windows);
    if (rc != 0) {
        free(out_buf);
        if (owned) gkm_sequence_free(qs);
        PyErr_SetString(PyExc_RuntimeError, "weighted sliding_query failed");
        return NULL;
    }

    PyObject *result = flat_to_2d_list(out_buf, n_windows, n_ref);

    free(out_buf);
    if (owned) gkm_sequence_free(qs);
    return result;
}

/* ================================================================== */
/* Method tables and type definitions                                 */
/* ================================================================== */

static PyMethodDef Kernel_methods[] = {
    {"add_sequence", (PyCFunction)Kernel_add_sequence,
     METH_VARARGS | METH_KEYWORDS, "add_sequence(seq, sid=None) -> int."},
    {"finalize", Kernel_finalize, METH_NOARGS, "finalize() -> None."},
    {"num_sequences", Kernel_num_sequences, METH_NOARGS, "num_sequences() -> int."},
    {"weight", Kernel_weight, METH_VARARGS, "weight(m) -> float."},
    {"weight_scheme", Kernel_weight_scheme, METH_NOARGS,
     "weight_scheme() -> int (0=full, 1=truncated, 2=estimated_full)."},
    {"use_rc", Kernel_use_rc, METH_NOARGS, "use_rc() -> int (0 or 1)."},
    {"eval_batch", Kernel_eval_batch, METH_VARARGS,
     "eval_batch(queries) -> list[list[float]]."},
    {"sliding_query", (PyCFunction)Kernel_sliding_query,
     METH_VARARGS | METH_KEYWORDS,
     "sliding_query(seq, window, shift, padding=0) -> list[list[float]]."},
    {"self_window_eval", (PyCFunction)Kernel_self_window_eval,
     METH_VARARGS | METH_KEYWORDS,
     "self_window_eval(seqs, window, shift, padding=0) -> 3D list."},
    {"cross_window_eval", (PyCFunction)Kernel_cross_window_eval,
     METH_VARARGS | METH_KEYWORDS,
     "cross_window_eval(queries, refs, window, shift, padding=0) -> 3D list."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject KernelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gapped_kmer_kernel._gkmkern.Kernel",
    .tp_basicsize = sizeof(KernelObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Regular gapped k-mer kernel (uniform L-mer weights).",
    .tp_init = Kernel_init,
    .tp_dealloc = Kernel_dealloc,
    .tp_methods = Kernel_methods,
    .tp_getset = Kernel_getset,
    .tp_new = PyType_GenericNew,
};

static PyMethodDef WeightedKernel_methods[] = {
    {"add_sequence", (PyCFunction)WeightedKernel_add_sequence,
     METH_VARARGS | METH_KEYWORDS, "add_sequence(seq, sid=None) -> int."},
    {"finalize", WeightedKernel_finalize, METH_NOARGS, "finalize() -> None."},
    {"num_sequences", WeightedKernel_num_sequences, METH_NOARGS, "num_sequences() -> int."},
    {"weight", WeightedKernel_weight, METH_VARARGS, "weight(m) -> float."},
    {"weight_scheme", WeightedKernel_weight_scheme, METH_NOARGS,
     "weight_scheme() -> int."},
    {"use_rc", WeightedKernel_use_rc, METH_NOARGS, "use_rc() -> int."},
    {"pos_kernel", WeightedKernel_pos_kernel, METH_NOARGS, "pos_kernel() -> int."},
    {"gamma", WeightedKernel_gamma, METH_NOARGS, "gamma() -> float."},
    {"sigma", WeightedKernel_sigma, METH_NOARGS, "sigma() -> float."},
    {"peak", WeightedKernel_peak, METH_NOARGS, "peak() -> float."},
    {"self_eval", (PyCFunction)WeightedKernel_self_eval,
     METH_VARARGS | METH_KEYWORDS,
     "self_eval(seqs, offsets=None) -> 2D or 3D list."},
    {"cross_eval", (PyCFunction)WeightedKernel_cross_eval,
     METH_VARARGS | METH_KEYWORDS,
     "cross_eval(queries, refs, offsets=None) -> 2D or 3D list."},
    {"self_window_eval", (PyCFunction)WeightedKernel_self_window_eval,
     METH_VARARGS | METH_KEYWORDS,
     "window_kernel(seqs, window, shift, padding=0) -> 3D list."},
    {"sliding_query", (PyCFunction)WeightedKernel_sliding_query,
     METH_VARARGS | METH_KEYWORDS,
     "sliding_query(seq, window, shift, padding=0) -> list[list[float]]."},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject WeightedKernelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gapped_kmer_kernel._gkmkern.WeightedKernel",
    .tp_basicsize = sizeof(WeightedKernelObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Weighted gapped k-mer kernel (positional weighting).",
    .tp_init = WeightedKernel_init,
    .tp_dealloc = WeightedKernel_dealloc,
    .tp_methods = WeightedKernel_methods,
    .tp_getset = WeightedKernel_getset,
    .tp_new = PyType_GenericNew,
};

/* ================================================================== */
/* Module                                                             */
/* ================================================================== */

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "gapped_kmer_kernel._gkmkern",
    .m_doc = "C core for the gapped k-mer kernel (regular + weighted).",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__gkmkern(void)
{
    if (PyType_Ready(&SequenceType) < 0) return NULL;
    if (PyType_Ready(&KernelType) < 0) return NULL;
    if (PyType_Ready(&WeightedKernelType) < 0) return NULL;

    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    Py_INCREF(&SequenceType);
    if (PyModule_AddObject(m, "Sequence", (PyObject *)&SequenceType) < 0) {
        Py_DECREF(&SequenceType); Py_DECREF(m); return NULL;
    }
    Py_INCREF(&KernelType);
    if (PyModule_AddObject(m, "Kernel", (PyObject *)&KernelType) < 0) {
        Py_DECREF(&KernelType); Py_DECREF(m); return NULL;
    }
    Py_INCREF(&WeightedKernelType);
    if (PyModule_AddObject(m, "WeightedKernel", (PyObject *)&WeightedKernelType) < 0) {
        Py_DECREF(&WeightedKernelType); Py_DECREF(m); return NULL;
    }

    /* Expose constants for the pos_kernel enum. */
    PyModule_AddIntConstant(m, "POS_TRIANGULAR",   (int)GKM_POS_TRIANGULAR);
    PyModule_AddIntConstant(m, "POS_EPANECHNIKOV", (int)GKM_POS_EPANECHNIKOV);
    PyModule_AddIntConstant(m, "POS_GAUSSIAN",     (int)GKM_POS_GAUSSIAN);
    PyModule_AddIntConstant(m, "POS_LAPLACIAN",    (int)GKM_POS_LAPLACIAN);
    PyModule_AddIntConstant(m, "POS_CAUCHY",       (int)GKM_POS_CAUCHY);

    /* Weight scheme constants. */
    PyModule_AddIntConstant(m, "WEIGHT_FULL",           (int)GKM_WEIGHT_FULL);
    PyModule_AddIntConstant(m, "WEIGHT_TRUNCATED",      (int)GKM_WEIGHT_TRUNCATED);
    PyModule_AddIntConstant(m, "WEIGHT_ESTIMATED_FULL", (int)GKM_WEIGHT_ESTIMATED_FULL);

    return m;
}
