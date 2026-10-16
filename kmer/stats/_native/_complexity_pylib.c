/* Native suffix-automaton distinct-substring counter. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int len;
    int link;
} sam_state_t;

static int alphabet_index(const char *alphabet, int alphabet_size, char value) {
    int i;
    for (i = 0; i < alphabet_size; i++) {
        if (alphabet[i] == value) return i;
    }
    return -1;
}

static int count_one(const char *sequence, Py_ssize_t length,
                     int L_min, int L_max,
                     const char *alphabet, int alphabet_size,
                     uint64_t *result) {
    Py_ssize_t max_states;
    sam_state_t *states = NULL;
    int *transitions = NULL;
    int size = 1;
    int last = 0;
    Py_ssize_t position;
    uint64_t total = 0;

    if (length > (INT_MAX - 1) / 2) return -2;
    max_states = length > 0 ? 2 * length : 1;
    states = (sam_state_t *)calloc((size_t)max_states, sizeof(sam_state_t));
    transitions = (int *)malloc(
        (size_t)max_states * (size_t)alphabet_size * sizeof(int)
    );
    if (!states || !transitions) {
        free(states); free(transitions); return -1;
    }
    for (position = 0; position < max_states * alphabet_size; position++) {
        transitions[position] = -1;
    }
    states[0].len = 0;
    states[0].link = -1;

    for (position = 0; position < length; position++) {
        int symbol = alphabet_index(alphabet, alphabet_size, sequence[position]);
        int current, p, q, clone;
        if (symbol < 0) {
            free(states); free(transitions); return -3;
        }
        current = size++;
        states[current].len = states[last].len + 1;
        states[current].link = 0;
        p = last;
        while (p != -1 && transitions[p * alphabet_size + symbol] == -1) {
            transitions[p * alphabet_size + symbol] = current;
            p = states[p].link;
        }
        if (p == -1) {
            states[current].link = 0;
        } else {
            q = transitions[p * alphabet_size + symbol];
            if (states[p].len + 1 == states[q].len) {
                states[current].link = q;
            } else {
                clone = size++;
                states[clone].len = states[p].len + 1;
                states[clone].link = states[q].link;
                memcpy(
                    &transitions[clone * alphabet_size],
                    &transitions[q * alphabet_size],
                    (size_t)alphabet_size * sizeof(int)
                );
                while (p != -1 && transitions[p * alphabet_size + symbol] == q) {
                    transitions[p * alphabet_size + symbol] = clone;
                    p = states[p].link;
                }
                states[q].link = clone;
                states[current].link = clone;
            }
        }
        last = current;
    }

    if (L_max < 0 || L_max > (int)length) L_max = (int)length;
    for (int state = 1; state < size; state++) {
        int lower = states[states[state].link].len + 1;
        int upper = states[state].len;
        if (lower < L_min) lower = L_min;
        if (upper > L_max) upper = L_max;
        if (upper >= lower) total += (uint64_t)(upper - lower + 1);
    }
    free(states);
    free(transitions);
    *result = total;
    return 0;
}

static PyObject *count_distinct_substrings(PyObject *self, PyObject *args,
                                            PyObject *kwargs) {
    static char *kwlist[] = {"seqs", "L_min", "L_max", "alphabet", NULL};
    PyObject *seqs_obj;
    PyObject *seqs_fast;
    const char *alphabet = "ACGT";
    Py_ssize_t alphabet_size;
    int L_min = 1;
    int L_max = -1;
    PyObject *output;
    Py_ssize_t n, i;
    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|iiz", kwlist,
            &seqs_obj, &L_min, &L_max, &alphabet)) {
        return NULL;
    }
    if (!alphabet) alphabet = "ACGT";
    alphabet_size = (Py_ssize_t)strlen(alphabet);
    if (L_min < 1) {
        PyErr_SetString(PyExc_ValueError, "L_min must be at least 1");
        return NULL;
    }
    if (L_max != -1 && L_max < L_min) {
        PyErr_SetString(PyExc_ValueError, "L_max must be >= L_min or None");
        return NULL;
    }
    if (alphabet_size < 1 || alphabet_size > 32) {
        PyErr_SetString(PyExc_ValueError, "alphabet must contain 1 to 32 ASCII symbols");
        return NULL;
    }
    for (i = 0; i < alphabet_size; i++) {
        for (Py_ssize_t j = i + 1; j < alphabet_size; j++) {
            if (alphabet[i] == alphabet[j]) {
                PyErr_SetString(PyExc_ValueError, "alphabet symbols must be unique");
                return NULL;
            }
        }
    }

    seqs_fast = PySequence_Fast(seqs_obj, "seqs must be a sequence of str");
    if (!seqs_fast) return NULL;
    n = PySequence_Fast_GET_SIZE(seqs_fast);
    output = PyList_New(n);
    if (!output) {
        Py_DECREF(seqs_fast); return NULL;
    }
    for (i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seqs_fast, i);
        const char *sequence;
        Py_ssize_t length;
        uint64_t count;
        int status;
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "every sequence must be a str");
            Py_DECREF(output); Py_DECREF(seqs_fast); return NULL;
        }
        sequence = PyUnicode_AsUTF8AndSize(item, &length);
        if (!sequence) {
            Py_DECREF(output); Py_DECREF(seqs_fast); return NULL;
        }
        status = count_one(sequence, length, L_min, L_max,
                           alphabet, (int)alphabet_size, &count);
        if (status != 0) {
            if (status == -1) PyErr_NoMemory();
            else if (status == -2) PyErr_SetString(PyExc_OverflowError, "sequence is too long");
            else PyErr_SetString(PyExc_ValueError, "input contains characters outside alphabet");
            Py_DECREF(output); Py_DECREF(seqs_fast); return NULL;
        }
        PyList_SET_ITEM(output, i, PyLong_FromUnsignedLongLong(count));
    }
    Py_DECREF(seqs_fast);
    return output;
}

static PyMethodDef methods[] = {
    {"_count_distinct_substrings", (PyCFunction)count_distinct_substrings,
     METH_VARARGS | METH_KEYWORDS,
     "Count distinct substrings in a length range with a suffix automaton."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "kmer.stats._native._complexity",
    "Native sequence-complexity primitives.",
    -1, methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__complexity(void) {
    return PyModule_Create(&module);
}
