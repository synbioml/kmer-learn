/* _mismatch.c — C core for mismatch k-mer counting.
 *
 * For each k-mer in the sequence, increments counts for ALL k-mers
 * within Hamming distance m (including the exact match). This is the
 * "mismatch kernel" feature space of Leslie, Eskin, Noble (2004).
 *
 * Implementation: for each k-mer window, generate all neighbors by
 * choosing 0..m positions to mutate and assigning each a non-matching
 * base. Each neighbor's packed code is computed incrementally.
 *
 * Supports k in [1, 12], m in [0, k].
 */

#include "_common.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define KMER_MISMATCH_MAX_K 12

#define KMER_MISMATCH_OK              0
#define KMER_MISMATCH_ERR_BAD_K      -1
#define KMER_MISMATCH_ERR_BAD_M      -2
#define KMER_MISMATCH_ERR_NON_ACGT   -3
#define KMER_MISMATCH_ERR_MEMORY     -4

typedef struct {
    int n_seqs;
    int k;
    int m;
    int canonical_rc;
    int n_features;          /* output columns (compact if canonical_rc) */
    int n_features_raw;      /* 4^k (internal) */
    int *canonical_map;      /* [n_features_raw]: canonical code -> compact col, or -1 */
    int    *rows;
    int    *cols;
    int32_t *vals;
    int    nnz;
    int    nnz_cap;
} mismatch_output_t;

static int _mismatch_output_init(mismatch_output_t *out, int n_seqs, int k,
                                  int m, int canonical_rc) {
    out->n_seqs = n_seqs;
    out->k = k;
    out->m = m;
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
        return KMER_MISMATCH_ERR_MEMORY;
    }
    return KMER_MISMATCH_OK;
}

static void _mismatch_output_free(mismatch_output_t *out) {
    free(out->rows); out->rows = NULL;
    free(out->cols); out->cols = NULL;
    free(out->vals); out->vals = NULL;
    free(out->canonical_map); out->canonical_map = NULL;
    out->nnz = 0; out->nnz_cap = 0;
}

static int _mismatch_emit(mismatch_output_t *out, int row, int col, int32_t val) {
    if (out->nnz >= out->nnz_cap) {
        int new_cap = out->nnz_cap * 2;
        int *nr = (int *)realloc(out->rows, sizeof(int) * new_cap);
        if (!nr) return KMER_MISMATCH_ERR_MEMORY;
        out->rows = nr;
        int *nc = (int *)realloc(out->cols, sizeof(int) * new_cap);
        if (!nc) return KMER_MISMATCH_ERR_MEMORY;
        out->cols = nc;
        int32_t *nv = (int32_t *)realloc(out->vals, sizeof(int32_t) * new_cap);
        if (!nv) return KMER_MISMATCH_ERR_MEMORY;
        out->vals = nv;
        out->nnz_cap = new_cap;
    }
    out->rows[out->nnz] = row;
    out->cols[out->nnz] = col;
    out->vals[out->nnz] = val;
    out->nnz++;
    return KMER_MISMATCH_OK;
}

/* Recursively generate all neighbors of a k-mer within m mismatches.
 *
 * Parameters:
 *   code     - packed k-mer code (MSB-first, 2 bits per base)
 *   k        - k-mer length
 *   pos      - current position being considered (0..k-1)
 *   remaining - number of mismatches still allowed
 *   counts   - dense count array of size 4^k
 */
static void _mismatch_generate_neighbors(uint32_t code, int k, int pos,
                                          int remaining, int32_t *counts) {
    /* Always count the current code (it's a neighbor of itself with 0 mismatches). */
    /* But only count at the leaf (pos == k) to avoid double-counting. */
    if (pos == k) {
        counts[code]++;
        return;
    }
    /* Option 1: don't mutate position pos. */
    _mismatch_generate_neighbors(code, k, pos + 1, remaining, counts);
    /* Option 2: mutate position pos (if remaining > 0). */
    if (remaining > 0) {
        uint32_t shift = 2 * (k - 1 - pos);
        uint32_t mask = 0b11u << shift;
        uint32_t base = (code >> shift) & 0b11u;
        uint32_t code_without = code & ~mask;
        for (uint32_t b = 0; b < 4; b++) {
            if (b == base) continue;  /* skip the original base */
            uint32_t new_code = code_without | (b << shift);
            _mismatch_generate_neighbors(new_code, k, pos + 1, remaining - 1, counts);
        }
    }
}

static int _mismatch_count_one(const char *seq, int seqlen, int row,
                                int k, int m, int canonical_rc,
                                mismatch_output_t *out) {
    if (seqlen < k) return KMER_MISMATCH_OK;

    int n_features_raw = out->n_features_raw;
    int32_t *counts = (int32_t *)calloc(n_features_raw, sizeof(int32_t));
    if (!counts) return KMER_MISMATCH_ERR_MEMORY;

    /* Validate strict ACGT (uppercase). */
    for (int i = 0; i < seqlen; i++) {
        char c = seq[i];
        if (c != 'A' && c != 'C' && c != 'G' && c != 'T') {
            free(counts);
            return KMER_MISMATCH_ERR_NON_ACGT;
        }
    }

    /* For each k-mer window on the forward strand, generate all neighbors
     * and increment counts. Canonical RC collapsing happens during the
     * flush step (below), not by scanning the RC strand. */
    uint32_t mask = (k == 16) ? 0xFFFFFFFFu : ((1u << (2 * k)) - 1);
    uint32_t code = 0;
    for (int i = 0; i < seqlen; i++) {
        code = ((code << 2) | KMER_NUC_LUT[(uint8_t)seq[i]]) & mask;
        if (i >= k - 1) {
            _mismatch_generate_neighbors(code, k, 0, m, counts);
        }
    }

    /* Flush non-zero entries. With canonical_rc, map to compact columns. */
    for (int c = 0; c < n_features_raw; c++) {
        if (counts[c] > 0) {
            int col = out->canonical_map ? out->canonical_map[c] : c;
            if (col >= 0) {
                int rc = _mismatch_emit(out, row, col, counts[c]);
                if (rc != KMER_MISMATCH_OK) {
                    free(counts);
                    return rc;
                }
            }
        }
    }
    free(counts);
    return KMER_MISMATCH_OK;
}

static int _mismatch_count_batch(const char **seqs, const int *seq_lens,
                                  int n_seqs, int k, int m, int canonical_rc,
                                  mismatch_output_t *out) {
    if (k < 1 || k > KMER_MISMATCH_MAX_K) return KMER_MISMATCH_ERR_BAD_K;
    if (m < 0 || m > k) return KMER_MISMATCH_ERR_BAD_M;
    int rc = _mismatch_output_init(out, n_seqs, k, m, canonical_rc);
    if (rc != KMER_MISMATCH_OK) return rc;

    if (canonical_rc) {
        out->canonical_map = (int *)malloc(sizeof(int) * out->n_features_raw);
        if (!out->canonical_map) return KMER_MISMATCH_ERR_MEMORY;
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
        rc = _mismatch_count_one(seqs[i], seq_lens[i], i, k, m, canonical_rc, out);
        if (rc != KMER_MISMATCH_OK) {
            _mismatch_output_free(out);
            return rc;
        }
    }
    return KMER_MISMATCH_OK;
}
