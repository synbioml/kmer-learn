/* _spectrum.c — C core for rolling-hash k-mer counting.
 *
 * For each sequence, compute the packed 2-bit code of every k-mer via
 * rolling hash, optionally collapsing to the reverse-complement
 * canonical form. Emit (row, col, val) triplets for a sparse CSR matrix.
 *
 * Supports k in [1, 15] (4^15 entries × 4 bytes = 4 GB cap).
 *
 * Designed to be #included by _spectrum_pylib.c (single-TU build).
 */

#include "_common.h"   /* pulls in kmer/_common.h via -I path */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define KMER_SPECTRUM_MAX_K 12

#define KMER_SPECTRUM_OK              0
#define KMER_SPECTRUM_ERR_BAD_K      -1
#define KMER_SPECTRUM_ERR_NON_ACGT   -2
#define KMER_SPECTRUM_ERR_MEMORY     -3

typedef struct {
    int n_seqs;
    int k;
    int canonical_rc;
    int n_features;          /* output columns (compact if canonical_rc) */
    int n_features_raw;      /* 4^k (internal, for counts array) */
    int *canonical_map;      /* [n_features_raw]: canonical code -> compact col, or -1 */
    int    *rows;            /* [nnz_cap] */
    int    *cols;            /* [nnz_cap] */
    int32_t *vals;           /* [nnz_cap] */
    int    nnz;
    int    nnz_cap;
} spectrum_output_t;

static int _spectrum_output_init(spectrum_output_t *out, int n_seqs, int k,
                                  int canonical_rc) {
    out->n_seqs = n_seqs;
    out->k = k;
    out->canonical_rc = canonical_rc;
    out->n_features_raw = 1;
    for (int i = 0; i < k; i++) out->n_features_raw *= 4;
    out->n_features = out->n_features_raw;
    out->canonical_map = NULL;
    out->nnz = 0;
    out->nnz_cap = 1024;
    out->rows = (int *)malloc(sizeof(int) * out->nnz_cap);
    out->cols = (int *)malloc(sizeof(int) * out->nnz_cap);
    out->vals = (int32_t *)malloc(sizeof(int32_t) * out->nnz_cap);
    if (!out->rows || !out->cols || !out->vals) {
        free(out->rows); free(out->cols); free(out->vals);
        return KMER_SPECTRUM_ERR_MEMORY;
    }
    return KMER_SPECTRUM_OK;
}

static void _spectrum_output_free(spectrum_output_t *out) {
    free(out->rows); out->rows = NULL;
    free(out->cols); out->cols = NULL;
    free(out->vals); out->vals = NULL;
    free(out->canonical_map); out->canonical_map = NULL;
    out->nnz = 0; out->nnz_cap = 0;
}

static int _spectrum_emit(spectrum_output_t *out, int row, int col, int32_t val) {
    if (out->nnz >= out->nnz_cap) {
        int new_cap = out->nnz_cap * 2;
        int *new_rows = (int *)realloc(out->rows, sizeof(int) * new_cap);
        if (!new_rows) return KMER_SPECTRUM_ERR_MEMORY;
        out->rows = new_rows;
        int *new_cols = (int *)realloc(out->cols, sizeof(int) * new_cap);
        if (!new_cols) return KMER_SPECTRUM_ERR_MEMORY;
        out->cols = new_cols;
        int32_t *new_vals = (int32_t *)realloc(out->vals, sizeof(int32_t) * new_cap);
        if (!new_vals) return KMER_SPECTRUM_ERR_MEMORY;
        out->vals = new_vals;
        out->nnz_cap = new_cap;
    }
    out->rows[out->nnz] = row;
    out->cols[out->nnz] = col;
    out->vals[out->nnz] = val;
    out->nnz++;
    return KMER_SPECTRUM_OK;
}

/* Count k-mers in one sequence; emit triplets into `out`. */
static int _spectrum_count_one(const char *seq, int seqlen, int row,
                                int k, int canonical_rc,
                                spectrum_output_t *out) {
    if (seqlen < k) return KMER_SPECTRUM_OK;

    int n_features_raw = out->n_features_raw;
    int32_t *counts = (int32_t *)calloc(n_features_raw, sizeof(int32_t));
    if (!counts) return KMER_SPECTRUM_ERR_MEMORY;

    /* Validate strict ACGT (uppercase). */
    for (int i = 0; i < seqlen; i++) {
        char c = seq[i];
        if (c != 'A' && c != 'C' && c != 'G' && c != 'T') {
            free(counts);
            return KMER_SPECTRUM_ERR_NON_ACGT;
        }
    }

    /* Rolling hash over the forward strand only. With canonical_rc,
     * each k-mer is mapped to min(code, rc(code)) so that a k-mer
     * and its reverse complement share the same column. This does
     * NOT double counts -- the total is always len-k+1. */
    uint32_t mask = (k == 16) ? 0xFFFFFFFFu : ((1u << (2 * k)) - 1);
    uint32_t code = 0;
    for (int i = 0; i < seqlen; i++) {
        code = ((code << 2) | KMER_NUC_LUT[(uint8_t)seq[i]]) & mask;
        if (i >= k - 1) {
            uint32_t c = code;
            if (canonical_rc) {
                c = kmer_pack_canonical(c, k);
            }
            counts[c]++;
        }
    }

    /* Flush non-zero entries. With canonical_rc, map canonical codes
     * to compact column indices; skip non-canonical codes (always 0). */
    for (int c = 0; c < n_features_raw; c++) {
        if (counts[c] > 0) {
            int col = out->canonical_map ? out->canonical_map[c] : c;
            if (col >= 0) {
                int rc = _spectrum_emit(out, row, col, counts[c]);
                if (rc != KMER_SPECTRUM_OK) {
                    free(counts);
                    return rc;
                }
            }
        }
    }
    free(counts);
    return KMER_SPECTRUM_OK;
}

/* Count k-mers for a batch of sequences. */
static int _spectrum_count_batch(const char **seqs, const int *seq_lens,
                                  int n_seqs, int k, int canonical_rc,
                                  spectrum_output_t *out) {
    if (k < 1 || k > KMER_SPECTRUM_MAX_K) return KMER_SPECTRUM_ERR_BAD_K;
    int rc = _spectrum_output_init(out, n_seqs, k, canonical_rc);
    if (rc != KMER_SPECTRUM_OK) return rc;

    /* If canonical_rc, build the canonical->compact mapping so only
     * canonical codes get output columns. Non-canonical codes map to -1. */
    if (canonical_rc) {
        out->canonical_map = (int *)malloc(sizeof(int) * out->n_features_raw);
        if (!out->canonical_map) return KMER_SPECTRUM_ERR_MEMORY;
        int n_canonical = 0;
        for (int c = 0; c < out->n_features_raw; c++) {
            uint32_t crc = kmer_pack_rc((uint32_t)c, k);
            if ((uint32_t)c <= crc) {
                out->canonical_map[c] = n_canonical++;
            } else {
                out->canonical_map[c] = -1;
            }
        }
        out->n_features = n_canonical;
    }

    for (int i = 0; i < n_seqs; i++) {
        rc = _spectrum_count_one(seqs[i], seq_lens[i], i, k, canonical_rc, out);
        if (rc != KMER_SPECTRUM_OK) {
            _spectrum_output_free(out);
            return rc;
        }
    }
    return KMER_SPECTRUM_OK;
}
