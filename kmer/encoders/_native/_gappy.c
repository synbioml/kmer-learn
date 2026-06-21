/* _gappy.c — C core for masked-hash gappy k-mer counting.
 *
 * Supports two modes:
 *   1. Explicit mask(s): e.g. masks=["*--*", "*---*"]
 *      `*` = non-gap (concrete position), `-` = gap (don't-care)
 *   2. Gap range: L (total length), g_min..g_max (number of gaps)
 *      → all C(L, g) masks for g in [g_min, g_max] are generated.
 *
 * For each (mask, position in sequence), pack the non-gap positions
 * into a code, use it as a hash key. Output one feature column per
 * (mask, pattern) pair.
 *
 * Designed to be #included by _gappy_pylib.c (single-TU build).
 */

#include "_common.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define KMER_GAPPY_MAX_K 15
#define KMER_GAPPY_MAX_LEN 32
#define KMER_GAPPY_MAX_MASKS 1024

#define KMER_GAPPY_OK              0
#define KMER_GAPPY_ERR_BAD_K      -1
#define KMER_GAPPY_ERR_BAD_MASK   -2
#define KMER_GAPPY_ERR_NON_ACGT   -3
#define KMER_GAPPY_ERR_MEMORY     -4
#define KMER_GAPPY_ERR_TOO_MANY   -5

typedef struct {
    int length;             /* total mask length L */
    int n_concrete;         /* number of non-gap positions (k) */
    int is_symmetric;       /* 1 if mask == RC(mask), 0 otherwise */
    int positions[KMER_GAPPY_MAX_LEN];  /* concrete positions, ascending */
} gappy_mask_t;

/* Hash map: pattern code -> feature column index.
 * Simple open-addressing with linear probing. */
typedef struct {
    int32_t *keys;          /* [cap] pattern code, or -1 if empty */
    int     *vals;          /* [cap] feature column index */
    int      cap;           /* power of 2 */
    int      mask;          /* cap - 1 */
    int      n_entries;     /* number of distinct patterns seen */
} gappy_hash_t;

static int _gappy_hash_init(gappy_hash_t *h, int expected) {
    int cap = 16;
    while (cap < expected * 2) cap <<= 1;
    h->cap = cap;
    h->mask = cap - 1;
    h->keys = (int32_t *)malloc(sizeof(int32_t) * cap);
    h->vals = (int *)malloc(sizeof(int) * cap);
    if (!h->keys || !h->vals) {
        free(h->keys); free(h->vals);
        return KMER_GAPPY_ERR_MEMORY;
    }
    for (int i = 0; i < cap; i++) h->keys[i] = -1;
    h->n_entries = 0;
    return KMER_GAPPY_OK;
}

static void _gappy_hash_free(gappy_hash_t *h) {
    free(h->keys); h->keys = NULL;
    free(h->vals); h->vals = NULL;
    h->cap = h->mask = h->n_entries = 0;
}

/* Lookup or insert. Returns the feature column index. */
static int _gappy_hash_upsert(gappy_hash_t *h, uint32_t key, int *col_out) {
    /* Mix the key for distribution. */
    uint32_t hh = key;
    hh ^= hh >> 16;
    hh *= 0x7feb352du;
    hh ^= hh >> 15;
    hh *= 0x846ca68bu;
    hh ^= hh >> 16;
    int idx = (int)(hh & h->mask);
    while (h->keys[idx] != -1) {
        if ((uint32_t)h->keys[idx] == key) {
            *col_out = h->vals[idx];
            return KMER_GAPPY_OK;
        }
        idx = (idx + 1) & h->mask;
    }
    /* Insert. */
    int new_col = h->n_entries++;
    h->keys[idx] = (int32_t)key;
    h->vals[idx] = new_col;
    *col_out = new_col;
    return KMER_GAPPY_OK;
}

typedef struct {
    int n_seqs;
    int n_masks;
    int n_features;
    int canonical_rc;
    int *rows;
    int *cols;
    int32_t *vals;
    int nnz;
    int nnz_cap;
    gappy_mask_t masks[KMER_GAPPY_MAX_MASKS];
    /* Per-mask hash table for pattern -> feature column mapping. */
    gappy_hash_t pattern_hashes[KMER_GAPPY_MAX_MASKS];
    /* Per-mask base column offset: features of mask i start at base[i]. */
    int base[KMER_GAPPY_MAX_MASKS + 1];
} gappy_output_t;

static int _gappy_output_init(gappy_output_t *out, int n_seqs,
                               int canonical_rc) {
    out->n_seqs = n_seqs;
    out->n_masks = 0;
    out->n_features = 0;
    out->canonical_rc = canonical_rc;
    out->nnz = 0;
    out->nnz_cap = 1024;
    out->rows = (int *)malloc(sizeof(int) * out->nnz_cap);
    out->cols = (int *)malloc(sizeof(int) * out->nnz_cap);
    out->vals = (int32_t *)malloc(sizeof(int32_t) * out->nnz_cap);
    if (!out->rows || !out->cols || !out->vals) {
        free(out->rows); free(out->cols); free(out->vals);
        return KMER_GAPPY_ERR_MEMORY;
    }
    return KMER_GAPPY_OK;
}

static void _gappy_output_free(gappy_output_t *out) {
    free(out->rows); out->rows = NULL;
    free(out->cols); out->cols = NULL;
    free(out->vals); out->vals = NULL;
    out->nnz = 0; out->nnz_cap = 0;
    for (int i = 0; i < out->n_masks; i++) {
        _gappy_hash_free(&out->pattern_hashes[i]);
    }
    out->n_masks = 0;
}

static int _gappy_emit(gappy_output_t *out, int row, int col, int32_t val) {
    if (out->nnz >= out->nnz_cap) {
        int new_cap = out->nnz_cap * 2;
        int *nr = (int *)realloc(out->rows, sizeof(int) * new_cap);
        if (!nr) return KMER_GAPPY_ERR_MEMORY;
        out->rows = nr;
        int *nc = (int *)realloc(out->cols, sizeof(int) * new_cap);
        if (!nc) return KMER_GAPPY_ERR_MEMORY;
        out->cols = nc;
        int32_t *nv = (int32_t *)realloc(out->vals, sizeof(int32_t) * new_cap);
        if (!nv) return KMER_GAPPY_ERR_MEMORY;
        out->vals = nv;
        out->nnz_cap = new_cap;
    }
    out->rows[out->nnz] = row;
    out->cols[out->nnz] = col;
    out->vals[out->nnz] = val;
    out->nnz++;
    return KMER_GAPPY_OK;
}

/* Parse a mask string like "*--*" into a gappy_mask_t. */
static int _gappy_parse_mask(const char *mask_str, int mask_len,
                              gappy_mask_t *m) {
    if (mask_len < 1 || mask_len > KMER_GAPPY_MAX_LEN) {
        return KMER_GAPPY_ERR_BAD_MASK;
    }
    m->length = mask_len;
    m->n_concrete = 0;
    for (int i = 0; i < mask_len; i++) {
        if (mask_str[i] == '*') {
            m->positions[m->n_concrete++] = i;
        } else if (mask_str[i] != '-') {
            return KMER_GAPPY_ERR_BAD_MASK;
        }
    }
    if (m->n_concrete < 1 || m->n_concrete > KMER_GAPPY_MAX_K) {
        return KMER_GAPPY_ERR_BAD_MASK;
    }
    /* Check symmetry: mask == RC(mask) iff for every concrete position p,
     * position (L-1-p) is also concrete. */
    m->is_symmetric = 1;
    for (int i = 0; i < m->n_concrete; i++) {
        int p = m->positions[i];
        int rc_p = m->length - 1 - p;
        /* Check if rc_p is also a concrete position. */
        int found = 0;
        for (int j = 0; j < m->n_concrete; j++) {
            if (m->positions[j] == rc_p) { found = 1; break; }
        }
        if (!found) { m->is_symmetric = 0; break; }
    }
    return KMER_GAPPY_OK;
}

/* Generate all C(L, g) masks for g in [g_min, g_max]. Appends to out->masks[]. */
static int _gappy_generate_masks(int L, int g_min, int g_max,
                                  gappy_output_t *out) {
    if (L < 2 || L > KMER_GAPPY_MAX_LEN) return KMER_GAPPY_ERR_BAD_MASK;
    /* Positions 0 and L-1 must be concrete (no leading/trailing gaps).
     * Gaps are chosen from interior positions 1..L-2.
     * Max gaps = L-2. Number of masks with g gaps = C(L-2, g). */
    int interior = L - 2;
    if (g_min < 0 || g_max > interior || g_min > g_max)
        return KMER_GAPPY_ERR_BAD_MASK;
    for (int g = g_min; g <= g_max; g++) {
        int k = L - g;
        if (k < 1 || k > KMER_GAPPY_MAX_K) continue;
        /* Enumerate combinations of g gap positions from {1, 2, ..., L-2}. */
        int *c = (int *)malloc(sizeof(int) * (g > 0 ? g : 1));
        if (!c) return KMER_GAPPY_ERR_MEMORY;
        for (int i = 0; i < g; i++) c[i] = i + 1;  /* start at 1, not 0 */
        while (1) {
            if (out->n_masks >= KMER_GAPPY_MAX_MASKS) {
                free(c);
                return KMER_GAPPY_ERR_TOO_MANY;
            }
            gappy_mask_t *m = &out->masks[out->n_masks];
            m->length = L;
            m->n_concrete = 0;
            int gap_idx = 0;
            for (int i = 0; i < L; i++) {
                if (gap_idx < g && c[gap_idx] == i) {
                    gap_idx++;
                } else {
                    m->positions[m->n_concrete++] = i;
                }
            }
            /* Check symmetry. */
            m->is_symmetric = 1;
            for (int i = 0; i < m->n_concrete; i++) {
                int rc_p = L - 1 - m->positions[i];
                int found = 0;
                for (int j = 0; j < m->n_concrete; j++) {
                    if (m->positions[j] == rc_p) { found = 1; break; }
                }
                if (!found) { m->is_symmetric = 0; break; }
            }
            out->n_masks++;
            if (g == 0) break;
            /* Next combination from {1, ..., L-2}. */
            int i = g - 1;
            while (i >= 0 && c[i] == interior - g + i + 1) i--;
            if (i < 0) break;
            c[i]++;
            for (int j = i + 1; j < g; j++) c[j] = c[j - 1] + 1;
        }
        free(c);
    }
    return KMER_GAPPY_OK;
}

/* Reverse-complement a mask: reverse the string and swap * with * (unchanged),
 * - with -. Position mapping is also reversed. */
static void _gappy_mask_rc(const gappy_mask_t *m, gappy_mask_t *out) {
    out->length = m->length;
    out->n_concrete = m->n_concrete;
    for (int i = 0; i < m->n_concrete; i++) {
        /* Position p in original becomes position (L - 1 - p) in RC. */
        out->positions[m->n_concrete - 1 - i] = m->length - 1 - m->positions[i];
    }
}

/* Compute the packed code for a mask applied at position `pos` in `seq`. */
static inline uint32_t _gappy_pack(const char *seq, int pos,
                                    const gappy_mask_t *m) {
    uint32_t code = 0;
    for (int i = 0; i < m->n_concrete; i++) {
        char c = seq[pos + m->positions[i]];
        code = (code << 2) | KMER_NUC_LUT[(uint8_t)c];
    }
    return code;
}

/* Compute the RC of a packed code, given the mask's n_concrete (= k). */
static inline uint32_t _gappy_pack_rc(uint32_t code, int k) {
    return kmer_pack_rc(code, k);
}

/* Count gappy k-mers in one sequence. */
static int _gappy_count_one(const char *seq, int seqlen, int row,
                             gappy_output_t *out) {
    /* Validate strict ACGT (uppercase). */
    for (int i = 0; i < seqlen; i++) {
        char c = seq[i];
        if (c != 'A' && c != 'C' && c != 'G' && c != 'T') {
            return KMER_GAPPY_ERR_NON_ACGT;
        }
    }

    /* Per-sequence dense count scratch — one per mask.
     * Each mask has at most 4^k entries. We allocate lazily. */
    int32_t **counts = (int32_t **)calloc(out->n_masks, sizeof(int32_t *));
    if (!counts) return KMER_GAPPY_ERR_MEMORY;
    int any_alloc_failed = 0;
    for (int m_idx = 0; m_idx < out->n_masks; m_idx++) {
        int k = out->masks[m_idx].n_concrete;
        int sz = 1;
        for (int i = 0; i < k; i++) sz *= 4;
        counts[m_idx] = (int32_t *)calloc(sz, sizeof(int32_t));
        if (!counts[m_idx]) { any_alloc_failed = 1; break; }
    }
    if (any_alloc_failed) {
        for (int i = 0; i < out->n_masks; i++) free(counts[i]);
        free(counts);
        return KMER_GAPPY_ERR_MEMORY;
    }

    /* Walk each position, for each mask. */
    for (int m_idx = 0; m_idx < out->n_masks; m_idx++) {
        gappy_mask_t *m = &out->masks[m_idx];
        if (seqlen < m->length) continue;
        for (int pos = 0; pos <= seqlen - m->length; pos++) {
            uint32_t code = _gappy_pack(seq, pos, m);
            /* Only collapse RC for symmetric masks (mask == RC(mask)).
             * For asymmetric masks, RC collapsing is handled at the
             * mask level (canonical masks only). */
            if (out->canonical_rc && m->is_symmetric) {
                uint32_t rc = _gappy_pack_rc(code, m->n_concrete);
                if (rc < code) code = rc;
            }
            counts[m_idx][code]++;
        }
    }

    /* Flush non-zero entries to triplets. */
    for (int m_idx = 0; m_idx < out->n_masks; m_idx++) {
        gappy_mask_t *m = &out->masks[m_idx];
        int k = m->n_concrete;
        int sz = 1;
        for (int i = 0; i < k; i++) sz *= 4;
        for (int c = 0; c < sz; c++) {
            if (counts[m_idx][c] > 0) {
                /* The feature column is base[m_idx] + hash_lookup(c). */
                int col;
                int rc = _gappy_hash_upsert(&out->pattern_hashes[m_idx],
                                              (uint32_t)c, &col);
                if (rc != KMER_GAPPY_OK) {
                    for (int i = 0; i < out->n_masks; i++) free(counts[i]);
                    free(counts);
                    return rc;
                }
                int global_col = out->base[m_idx] + col;
                rc = _gappy_emit(out, row, global_col, counts[m_idx][c]);
                if (rc != KMER_GAPPY_OK) {
                    for (int i = 0; i < out->n_masks; i++) free(counts[i]);
                    free(counts);
                    return rc;
                }
            }
        }
    }

    for (int i = 0; i < out->n_masks; i++) free(counts[i]);
    free(counts);
    return KMER_GAPPY_OK;
}

/* Count gappy k-mers for a batch of sequences.
 *
 * Caller must have:
 *   1. Called _gappy_output_init(out, ...) to allocate triplet arrays.
 *   2. Populated out->masks[] and out->n_masks.
 *
 * This function:
 *   1. Pre-populates per-mask hash tables with ALL possible patterns
 *      (in canonical code order), so column indices are deterministic
 *      across calls.
 *   2. Computes base offsets.
 *   3. Walks each sequence.
 */
static int _gappy_count_batch(const char **seqs, const int *seq_lens,
                               int n_seqs, int canonical_rc,
                               gappy_output_t *out) {
    /* Initialize per-mask hash tables, pre-populate with all patterns,
     * and compute base offsets. */
    out->base[0] = 0;
    for (int i = 0; i < out->n_masks; i++) {
        int k = out->masks[i].n_concrete;
        int max_patterns = 1;
        for (int j = 0; j < k; j++) max_patterns *= 4;
        int rc = _gappy_hash_init(&out->pattern_hashes[i], max_patterns);
        if (rc != KMER_GAPPY_OK) {
            for (int j = 0; j < i; j++) _gappy_hash_free(&out->pattern_hashes[j]);
            return rc;
        }
        /* Pre-populate hash table. With canonical_rc, only canonical
         * codes get columns — but only for symmetric masks (where
         * RC collapsing is applied at the gappy k-mer level).
         * For asymmetric masks, all codes get columns. */
        int n_cols_for_mask = 0;
        int do_collapse = canonical_rc && out->masks[i].is_symmetric;
        for (int c = 0; c < max_patterns; c++) {
            int include = 1;
            if (do_collapse) {
                uint32_t rc_code = kmer_pack_rc((uint32_t)c, k);
                include = ((uint32_t)c <= rc_code);
            }
            if (include) {
                int col_dummy;
                _gappy_hash_upsert(&out->pattern_hashes[i], (uint32_t)c, &col_dummy);
                n_cols_for_mask++;
            }
        }
        out->base[i + 1] = out->base[i] + n_cols_for_mask;
    }
    out->n_features = out->base[out->n_masks];

    for (int i = 0; i < n_seqs; i++) {
        int rc = _gappy_count_one(seqs[i], seq_lens[i], i, out);
        if (rc != KMER_GAPPY_OK) {
            return rc;
        }
    }
    return KMER_GAPPY_OK;
}
