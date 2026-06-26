/* _gkm_hnsw.cpp — HNSW index for GKM kernels and alignment metrics.
 *
 * Uses vendored hnswlib with custom Space subclasses.
 * gkmkern.c is included directly (compiled as C++ with C linkage via
 * extern "C"), avoiding dual-compilation and dlopen issues.
 *
 * Similarities (GKM kernel, NW/SW scores) are converted to distances:
 *   GKM:    dist = 1.0 - K(a,b)
 *   NW/SW:  dist = max_score - score
 *   Lev/Ham: already distances
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

// Hnswlib
#include "hnswlib.h"
#include "hnswalg.h"

// Include gkmkern.h and gkmkern.c directly, compiled as C++ with C linkage.
// This avoids dual-compilation issues — one copy of the code in this .so.
extern "C" {
#include "gkmkern.h"
}
extern "C" {
#include "gkmkern.c"
}

// ===================================================================
// Global state for distance computation
// ===================================================================

struct IndexState {
    int metric_type; // 0=gkm, 1=lev, 2=ham, 3=nw, 4=sw

    // GKM
    gkm_kernel_t *gkm_kernel;
    std::vector<gkm_sequence_t*> gkm_seqs;
    std::vector<double> gkm_profile;
    int gkm_current; // -1=none, -2=query, >=0=insert

    // String (Levenshtein/Hamming)
    std::vector<std::string> str_seqs;
    std::string str_query;

    // Alignment (NW/SW)
    std::vector<std::string> align_seqs;
    std::string align_query;
    int align_type; // 0=NW, 1=SW
    int al_match, al_mismatch, al_gap;
};
static IndexState *g_state = nullptr;

// ===================================================================
// Distance function (called by hnswlib during graph traversal)
// ===================================================================

static float dist_func(const void *p1, const void *p2, const void *) {
    int i1 = *(const int*)p1, i2 = *(const int*)p2;
    IndexState *st = g_state;

    if (st->metric_type == 0) { // GKM
        if (st->gkm_current == -2 && (i1 == -1 || i2 == -1)) {
            int ref = (i1 == -1) ? i2 : i1;
            if (ref >= 0 && ref < (int)st->gkm_profile.size())
                return (float)(1.0 - st->gkm_profile[ref]);
            return 1.0f;
        }
        if (st->gkm_current >= 0) {
            if (i1 == st->gkm_current && i2 >= 0 && i2 < (int)st->gkm_profile.size())
                return (float)(1.0 - st->gkm_profile[i2]);
            if (i2 == st->gkm_current && i1 >= 0 && i1 < (int)st->gkm_profile.size())
                return (float)(1.0 - st->gkm_profile[i1]);
        }
        // Fallback: compute directly (rare — only during neighbor pruning)
        if (i1 >= 0 && i2 >= 0 && i1 < (int)st->gkm_seqs.size() && i2 < (int)st->gkm_seqs.size()) {
            gkm_kernel_eval_all(st->gkm_kernel, st->gkm_seqs[i1], st->gkm_profile.data());
            return (float)(1.0 - st->gkm_profile[i2]);
        }
        return 1.0f;
    }

    // String / alignment metrics
    const std::string *s1, *s2;
    if (st->metric_type <= 2) {
        s1 = (i1 == -1) ? &st->str_query : &st->str_seqs[i1];
        s2 = (i2 == -1) ? &st->str_query : &st->str_seqs[i2];
    } else {
        s1 = (i1 == -1) ? &st->align_query : &st->align_seqs[i1];
        s2 = (i2 == -1) ? &st->align_query : &st->align_seqs[i2];
    }

    if (st->metric_type == 2) { // Hamming
        if (s1->size() != s2->size()) return (float)std::max(s1->size(), s2->size());
        int d = 0;
        for (size_t i = 0; i < s1->size(); i++) if ((*s1)[i] != (*s2)[i]) d++;
        return (float)d;
    }
    if (st->metric_type == 1) { // Levenshtein
        int la = s1->size(), lb = s2->size();
        if (la == 0) return (float)lb; if (lb == 0) return (float)la;
        if (la > lb) { std::swap(s1, s2); std::swap(la, lb); }
        std::vector<int> prev(lb+1), curr(lb+1);
        for (int j = 0; j <= lb; j++) prev[j] = j;
        for (int i = 1; i <= la; i++) {
            curr[0] = i;
            for (int j = 1; j <= lb; j++) {
                int c = ((*s1)[i-1] == (*s2)[j-1]) ? 0 : 1;
                curr[j] = std::min({prev[j]+1, curr[j-1]+1, prev[j-1]+c});
            }
            std::swap(prev, curr);
        }
        return (float)prev[lb];
    }

    // NW or SW
    int la = s1->size(), lb = s2->size();
    int g = st->al_gap;
    std::vector<int> prev(lb+1), curr(lb+1);
    int best = 0;
    for (int j = 0; j <= lb; j++) prev[j] = (st->align_type == 0) ? j * g : 0;
    for (int i = 1; i <= la; i++) {
        curr[0] = (st->align_type == 0) ? i * g : 0;
        for (int j = 1; j <= lb; j++) {
            int sub = ((*s1)[i-1] == (*s2)[j-1]) ? st->al_match : st->al_mismatch;
            int v = prev[j-1] + sub;
            if (prev[j] + g > v) v = prev[j] + g;
            if (curr[j-1] + g > v) v = curr[j-1] + g;
            if (st->align_type == 1 && v < 0) v = 0;
            curr[j] = v;
            if (v > best) best = v;
        }
        std::swap(prev, curr);
    }
    int score = (st->align_type == 0) ? prev[lb] : best;
    int max_score = std::max(la, lb) * st->al_match;
    return (float)(max_score - score);
}

// ===================================================================
// hnswlib Space subclass
// ===================================================================

class SeqSpace : public hnswlib::SpaceInterface<float> {
public:
    typedef hnswlib::DISTFUNC<float> dist_func_t_;
    dist_func_t_ df_;
    SeqSpace() { df_ = (dist_func_t_)dist_func; }
    size_t get_data_size() override { return sizeof(int); }
    dist_func_t_ get_dist_func() override { return df_; }
    void *get_dist_func_param() override { return nullptr; }
};

// ===================================================================
// CPython type
// ===================================================================

typedef struct {
    PyObject_HEAD
    hnswlib::HierarchicalNSW<float> *index;
    SeqSpace *space;
    IndexState state;
    int n_seqs;
} HNSWObj;

static int HNSW_init(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"seqs","metric","M","ef_construction","ef_search",
                             "gkm_L","gkm_k","gkm_d","gkm_use_rc",
                             "match","mismatch","gap",NULL};
    PyObject *seqs_obj;
    const char *ms = "levenshtein";
    int M=16, efc=200, efs=50;
    int gL=10, gk=6, gd=3, gurc=0;
    int m_m=1, m_mm=-1, m_g=-1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|siiiiiiiiii", kwlist,
                                     &seqs_obj, &ms, &M, &efc, &efs,
                                     &gL, &gk, &gd, &gurc, &m_m, &m_mm, &m_g))
        return -1;

    HNSWObj *o = (HNSWObj*)self;
    o->index = nullptr;
    o->space = nullptr;
    memset(&o->state, 0, sizeof(IndexState));

    if (!strcmp(ms, "gkm")) o->state.metric_type = 0;
    else if (!strcmp(ms, "levenshtein")) o->state.metric_type = 1;
    else if (!strcmp(ms, "hamming")) o->state.metric_type = 2;
    else if (!strcmp(ms, "nw")) o->state.metric_type = 3;
    else if (!strcmp(ms, "sw")) o->state.metric_type = 4;
    else { PyErr_SetString(PyExc_ValueError, "unknown metric"); return -1; }

    // Extract sequences
    PyObject *sf = PySequence_Fast(seqs_obj, "seqs must be a sequence");
    if (!sf) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(sf);
    o->n_seqs = n;

    std::vector<std::string> seqs;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(sf, i);
        const char *s; Py_ssize_t len;
        if (PyBytes_Check(item)) { s = PyBytes_AS_STRING(item); len = PyBytes_GET_SIZE(item); }
        else if (PyUnicode_Check(item)) { s = PyUnicode_AsUTF8AndSize(item, &len); if (!s) { Py_DECREF(sf); return -1; } }
        else { PyErr_SetString(PyExc_TypeError, "seqs must be str/bytes"); Py_DECREF(sf); return -1; }
        seqs.push_back(std::string(s, len));
    }
    Py_DECREF(sf);

    g_state = &o->state;
    o->state.gkm_current = -1;

    // Initialize metric-specific state
    if (o->state.metric_type == 0) {
        // GKM kernel
        gkm_parameter_t param;
        param.L = gL; param.k = gk; param.d = gd;
        param.weight_scheme = GKM_WEIGHT_TRUNCATED;
        param.use_rc = gurc ? 1 : 0;
        param.nthreads = 1;

        o->state.gkm_kernel = gkm_kernel_create(&param);
        if (!o->state.gkm_kernel) {
            PyErr_SetString(PyExc_RuntimeError, "GKM kernel creation failed");
            return -1;
        }

        o->state.gkm_seqs.resize(n);
        o->state.gkm_profile.resize(n);

        void *base = gkm_kernel_base(o->state.gkm_kernel);
        for (Py_ssize_t i = 0; i < n; i++) {
            char sid[32]; snprintf(sid, 32, "s%zd", i);
                o->state.gkm_seqs[i] = gkm_sequence_create(base, seqs[i].c_str(), sid);
            if (!o->state.gkm_seqs[i]) {
                PyErr_Format(PyExc_RuntimeError, "GKM seq create failed for %zd (len=%zu, L=%d)",
                             i, seqs[i].size(), gL);
                return -1;
            }
                gkm_kernel_base_add_sequence(base, o->state.gkm_seqs[i]);
        }
        gkm_kernel_finalize(o->state.gkm_kernel);
    } else if (o->state.metric_type <= 2) {
        o->state.str_seqs = seqs;
    } else {
        o->state.align_seqs = seqs;
        o->state.align_type = (o->state.metric_type == 4) ? 1 : 0;
        o->state.al_match = m_m;
        o->state.al_mismatch = m_mm;
        o->state.al_gap = m_g;
    }

    // Create HNSW index
    o->space = new SeqSpace();
    o->index = new hnswlib::HierarchicalNSW<float>(o->space, n, M, efc);
    o->index->ef_ = efs;

    // Insert all sequences
    try {
        for (Py_ssize_t i = 0; i < n; i++) {
            int idx = (int)i;
            if (o->state.metric_type == 0) {
                    gkm_kernel_eval_all(o->state.gkm_kernel, o->state.gkm_seqs[i],
                                   o->state.gkm_profile.data());
                o->state.gkm_current = (int)i;
            }
                o->index->addPoint(&idx, (size_t)i);
        }
    } catch (const std::exception& e) {
        PyErr_Format(PyExc_RuntimeError, "HNSW insert failed at seq %d: %s",
                     o->state.gkm_current, e.what());
        return -1;
    }
    o->state.gkm_current = -1;

    return 0;
}

static void HNSW_dealloc(PyObject *self) {
    HNSWObj *o = (HNSWObj*)self;
    if (o->index) delete o->index;
    if (o->space) delete o->space;
    if (o->state.metric_type == 0 && o->state.gkm_kernel) {
        // gkm_kernel_destroy frees all sequences internally via gkm_kernel_base_destroy
        gkm_kernel_destroy(o->state.gkm_kernel);
    }
    Py_TYPE(self)->tp_free(self);
}

static PyObject *HNSW_query(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"query", "k", NULL};
    PyObject *qo; int k = 10;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &qo, &k))
        return NULL;

    HNSWObj *o = (HNSWObj*)self;
    if (!o->index) { PyErr_SetString(PyExc_RuntimeError, "no index"); return NULL; }

    // Handle single seq or list
    PyObject *qf;
    if (PyBytes_Check(qo) || PyUnicode_Check(qo)) {
        PyObject *l = PyList_New(1); Py_INCREF(qo); PyList_SET_ITEM(l, 0, qo); qf = l;
    } else {
        qf = PySequence_Fast(qo, "query must be seq or list"); if (!qf) return NULL;
    }

    Py_ssize_t nq = PySequence_Fast_GET_SIZE(qf);
    PyObject *results = PyList_New(nq);

    for (Py_ssize_t q = 0; q < nq; q++) {
        PyObject *item = PySequence_Fast_GET_ITEM(qf, q);
        const char *s; Py_ssize_t len;
        if (PyBytes_Check(item)) { s = PyBytes_AS_STRING(item); len = PyBytes_GET_SIZE(item); }
        else if (PyUnicode_Check(item)) { s = PyUnicode_AsUTF8AndSize(item, &len); if (!s) return NULL; }
        else { PyErr_SetString(PyExc_TypeError, "query must be str/bytes"); return NULL; }

        std::string qseq(s, len);

        // Set up query state
        if (o->state.metric_type == 0) {
            void *base = gkm_kernel_base(o->state.gkm_kernel);
            gkm_sequence_t *qs = gkm_sequence_create(base, qseq.c_str(), "query");
            if (!qs) {
                PyErr_Format(PyExc_RuntimeError, "GKM query seq create failed (len=%zu)", qseq.size());
                Py_DECREF(qf);
                return NULL;
            }
            gkm_kernel_eval_all(o->state.gkm_kernel, qs, o->state.gkm_profile.data());
            gkm_sequence_free(qs);
            o->state.gkm_current = -2;
        } else if (o->state.metric_type <= 2) {
            o->state.str_query = qseq;
        } else {
            o->state.align_query = qseq;
        }

        // Search (use -1 as query index)
        int qi = -1;
        std::vector<std::pair<float, size_t>> res;
        try {
            res = o->index->searchKnnCloserFirst(&qi, k);
        } catch (const std::exception& e) {
            PyErr_Format(PyExc_RuntimeError, "HNSW query failed: %s", e.what());
            Py_DECREF(qf);
            return NULL;
        }

        PyObject *il = PyList_New(res.size());
        PyObject *dl = PyList_New(res.size());
        for (size_t i = 0; i < res.size(); i++) {
            PyList_SET_ITEM(il, i, PyLong_FromSize_t(res[i].second));
            PyList_SET_ITEM(dl, i, PyFloat_FromDouble(res[i].first));
        }
        PyList_SET_ITEM(results, q, Py_BuildValue("(OO)", il, dl));
    }

    o->state.gkm_current = -1;
    Py_DECREF(qf);
    return results;
}

static PyObject *HNSW_set_ef(PyObject *self, PyObject *args) {
    int ef; if (!PyArg_ParseTuple(args, "i", &ef)) return NULL;
    ((HNSWObj*)self)->index->ef_ = ef; Py_RETURN_NONE;
}
static PyObject *HNSW_size(PyObject *self, PyObject*) {
    return PyLong_FromLong(((HNSWObj*)self)->n_seqs);
}

static PyMethodDef HNSW_methods[] = {
    {"query", (PyCFunction)HNSW_query, METH_VARARGS | METH_KEYWORDS, "query(q, k=10)"},
    {"set_ef_search", HNSW_set_ef, METH_VARARGS, "set_ef_search(ef)"},
    {"get_size", HNSW_size, METH_NOARGS, "get_size()"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject HNSWType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "kmer.ann._native._hnsw.HNSWIndex",
    .tp_basicsize = sizeof(HNSWObj),
    .tp_dealloc = (destructor)HNSW_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "HNSW index for sequence kernels and distances.",
    .tp_methods = HNSW_methods,
    .tp_init = (initproc)HNSW_init,
    .tp_new = PyType_GenericNew,
};

static PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT, "kmer.ann._native._hnsw",
    "HNSW ANN index with custom sequence metrics.", -1, NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__hnsw(void) {
    if (PyType_Ready(&HNSWType) < 0) return NULL;
    PyObject *m = PyModule_Create(&moddef);
    if (!m) return NULL;
    Py_INCREF(&HNSWType);
    if (PyModule_AddObject(m, "HNSWIndex", (PyObject *)&HNSWType) < 0) {
        Py_DECREF(&HNSWType); Py_DECREF(m); return NULL;
    }
    return m;
}
