/* _chunker.c — C core for sequence chunking.
 *
 * Algorithm:
 *   Phase 1: decide chunk sizes (greedy 'random' or DP-based 'backtrack').
 *            If a residual (too-short tail) remains, apply one of:
 *              'end', 'start', 'random', 'extend', 'distribute'.
 *   Phase 2: slice the sequence into chunks of the chosen sizes,
 *            optionally reverse-complement each chunk (IUPAC), shuffle
 *            the chunk order, and concatenate back.
 *
 * Output length always equals input length.
 */

#include "_common.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CHUNKER_MAX_SIZE 65536

/* ----- DP for backtrack mode -----
 *
 * dp[i] = number of ways to decompose length i into sizes from [min, max].
 * dp[0] = 1; dp[i] = sum dp[i - s] for s in [min, min(i, max)].
 *
 * To sample uniformly: start at L, pick s with prob dp[L-s]/dp[L].
 */
static int dp_count_ways(int L, int min_sz, int max_sz, int64_t *dp) {
    /* dp must be of size L+1. Returns 0 on success, -1 on overflow. */
    dp[0] = 1;
    for (int i = 1; i <= L; i++) {
        int64_t sum = 0;
        int lo = (i >= max_sz) ? (i - max_sz) : 0;
        int hi = i - min_sz;
        for (int j = lo; j <= hi; j++) {
            if (j < 0) continue;
            sum += dp[j];
            if (sum < 0) return -1;  /* overflow */
        }
        dp[i] = sum;
    }
    return 0;
}

/* Sample a decomposition of L into sizes from [min, max] using dp[].
 * Returns the number of chunks, fills sizes[] (caller-allocated, size >= L/min + 1). */
static int dp_sample(int L, int min_sz, int max_sz, int64_t *dp,
                     philox_stream_t *rng, int *sizes) {
    int n = 0;
    int remaining = L;
    while (remaining > 0) {
        int lo = min_sz;
        int hi = (remaining < max_sz) ? remaining : max_sz;
        /* For each candidate size s in [lo, hi], weight = dp[remaining - s]. */
        int64_t total = 0;
        for (int s = lo; s <= hi; s++) {
            int prev = remaining - s;
            if (prev < 0) continue;
            total += dp[prev];
        }
        if (total == 0) {
            /* Shouldn't happen if dp[L] > 0, but fall back. */
            sizes[n++] = remaining;
            break;
        }
        int64_t pick = (int64_t)(philox_double(rng) * (double)total);
        if (pick >= total) pick = total - 1;
        int chosen = -1;
        int64_t cum = 0;
        for (int s = lo; s <= hi; s++) {
            int prev = remaining - s;
            if (prev < 0) continue;
            cum += dp[prev];
            if (pick < cum) { chosen = s; break; }
        }
        if (chosen < 0) chosen = hi;
        sizes[n++] = chosen;
        remaining -= chosen;
    }
    return n;
}

/* ----- Greedy random -----
 *
 * Pick next size uniformly from [min, min(max, remaining)].
 * If remaining < min, leave it as a residual (caller handles).
 */
static int greedy_random(int L, int min_sz, int max_sz,
                         philox_stream_t *rng, int *sizes,
                         int *residual_out) {
    int n = 0;
    int remaining = L;
    *residual_out = 0;
    while (remaining >= max_sz) {
        int s = (int)philox_below(rng, (uint32_t)(max_sz - min_sz + 1)) + min_sz;
        sizes[n++] = s;
        remaining -= s;
    }
    if (remaining > 0) {
        if (remaining >= min_sz) {
            /* Fits as a final in-range chunk. */
            sizes[n++] = remaining;
        } else {
            /* Residual — too short. */
            *residual_out = remaining;
        }
    }
    return n;
}

/* ----- Residual handling -----
 *
 * Given the current sizes[] (n of them), distribute/extend/etc. a residual
 * of size r. Modifies sizes[] in place; may add chunks (extend/distribute)
 * or mark a position as "residual". Returns the new n.
 */

#define RESIDUAL_END        0
#define RESIDUAL_START      1
#define RESIDUAL_RANDOM     2
#define RESIDUAL_EXTEND     3
#define RESIDUAL_DISTRIBUTE 4

static int apply_residual(int *sizes, int n, int residual, int mode,
                          philox_stream_t *rng) {
    if (residual <= 0) return n;
    switch (mode) {
    case RESIDUAL_END:
        sizes[n++] = residual;
        return n;
    case RESIDUAL_START:
        /* Shift everything right by one. */
        memmove(sizes + 1, sizes, n * sizeof(int));
        sizes[0] = residual;
        return n + 1;
    case RESIDUAL_RANDOM:
        /* Insert at a random position. */
        {
            int pos = (int)philox_below(rng, (uint32_t)(n + 1));
            memmove(sizes + pos + 1, sizes + pos, (n - pos) * sizeof(int));
            sizes[pos] = residual;
            return n + 1;
        }
    case RESIDUAL_EXTEND:
        /* Pick a random existing chunk, extend it. */
        {
            int pos = (int)philox_below(rng, (uint32_t)n);
            sizes[pos] += residual;
            return n;
        }
    case RESIDUAL_DISTRIBUTE:
        /* For each of `residual` units, pick a random chunk (with
         * replacement) and extend it by 1. */
        {
            for (int i = 0; i < residual; i++) {
                int pos = (int)philox_below(rng, (uint32_t)n);
                sizes[pos]++;
            }
            return n;
        }
    default:
        sizes[n++] = residual;
        return n;
    }
}

/* ----- Reverse complement (IUPAC) ----- */

static int rc_chunk_inplace(char *chunk, int len) {
    /* Reverse-complement in place. Returns 0 on success, -1 if any
     * character is not IUPAC. */
    for (int i = 0, j = len - 1; i <= j; i++, j--) {
        uint8_t a = (uint8_t)chunk[i];
        uint8_t b = (uint8_t)chunk[j];
        uint8_t ra = IUPAC_RC[a];
        uint8_t rb = IUPAC_RC[b];
        if ((ra == 0 && a != 0) || (rb == 0 && b != 0)) return -1;
        chunk[i] = (char)rb;
        chunk[j] = (char)ra;
    }
    return 0;
}

/* ----- Public entry point ----- */

#define CHUNKER_OK                0
#define CHUNKER_ERR_BAD_RANGE    -1
#define CHUNKER_ERR_MEMORY       -2
#define CHUNKER_ERR_BAD_ALG      -3
#define CHUNKER_ERR_BAD_RESIDUAL -4
#define CHUNKER_ERR_NON_IUPAC    -5

static int chunk_one(const char *seq, int seqlen,
                     int min_sz, int max_sz,
                     int algorithm,        /* 0=random, 1=backtrack */
                     int residual_mode,    /* RESIDUAL_* */
                     int flip_strand,      /* 0 or 1 */
                     double flip_strand_prob,
                     philox_stream_t *rng,
                     char *out, int *out_len) {
    if (min_sz < 1 || max_sz < min_sz || max_sz > CHUNKER_MAX_SIZE) {
        return CHUNKER_ERR_BAD_RANGE;
    }
    /* Trivial edge cases. */
    if (seqlen == 0) { *out_len = 0; return CHUNKER_OK; }
    if (seqlen < min_sz) {
        /* Can't form a single valid chunk — just copy input. */
        memcpy(out, seq, seqlen);
        *out_len = seqlen;
        return CHUNKER_OK;
    }

    /* Phase 1: decide sizes. */
    int sizes_buf[CHUNKER_MAX_SIZE + 1];
    int n = 0;
    int residual = 0;

    if (algorithm == 0) {
        /* greedy random */
        n = greedy_random(seqlen, min_sz, max_sz, rng, sizes_buf, &residual);
        if (residual > 0) {
            n = apply_residual(sizes_buf, n, residual, residual_mode, rng);
        }
    } else if (algorithm == 1) {
        /* backtrack */
        int64_t *dp = (int64_t *)malloc((seqlen + 1) * sizeof(int64_t));
        if (!dp) return CHUNKER_ERR_MEMORY;
        if (dp_count_ways(seqlen, min_sz, max_sz, dp) < 0) {
            free(dp); return CHUNKER_ERR_MEMORY;
        }
        if (dp[seqlen] > 0) {
            n = dp_sample(seqlen, min_sz, max_sz, dp, rng, sizes_buf);
        } else {
            /* No exact decomposition. Find smallest residual r such that
             * (seqlen - r) has a decomposition. Apply residual_mode. */
            int best_r = -1;
            for (int r = 1; r < min_sz; r++) {
                if (seqlen - r >= 0 && dp[seqlen - r] > 0) {
                    best_r = r; break;
                }
            }
            if (best_r < 0) {
                /* Truly impossible (shouldn't happen for seqlen >= min_sz). */
                free(dp);
                n = greedy_random(seqlen, min_sz, max_sz, rng, sizes_buf, &residual);
                if (residual > 0) {
                    n = apply_residual(sizes_buf, n, residual, residual_mode, rng);
                }
            } else {
                n = dp_sample(seqlen - best_r, min_sz, max_sz, dp, rng, sizes_buf);
                n = apply_residual(sizes_buf, n, best_r, residual_mode, rng);
            }
        }
        free(dp);
    } else {
        return CHUNKER_ERR_BAD_ALG;
    }

    /* Phase 2: slice, optionally RC, shuffle, concatenate. */
    /* Allocate chunk pointer array. */
    char **chunks = (char **)malloc(n * sizeof(char *));
    int *chunk_lens = (int *)malloc(n * sizeof(int));
    if (!chunks || !chunk_lens) {
        free(chunks); free(chunk_lens);
        return CHUNKER_ERR_MEMORY;
    }
    int pos = 0;
    for (int i = 0; i < n; i++) {
        int sz = sizes_buf[i];
        /* Allocate fresh buffer for this chunk (we may RC in place). */
        chunks[i] = (char *)malloc(sz);
        if (!chunks[i]) {
            for (int j = 0; j < i; j++) free(chunks[j]);
            free(chunks); free(chunk_lens);
            return CHUNKER_ERR_MEMORY;
        }
        memcpy(chunks[i], seq + pos, sz);
        chunk_lens[i] = sz;
        pos += sz;
    }
    /* pos should equal seqlen. */
    if (pos != seqlen) {
        /* Shouldn't happen, but be safe. */
        for (int i = 0; i < n; i++) free(chunks[i]);
        free(chunks); free(chunk_lens);
        return CHUNKER_ERR_BAD_ALG;
    }

    /* Optional RC per chunk. */
    if (flip_strand) {
        for (int i = 0; i < n; i++) {
            double p = philox_double(rng);
            if (p < flip_strand_prob) {
                if (rc_chunk_inplace(chunks[i], chunk_lens[i]) < 0) {
                    for (int j = 0; j < n; j++) free(chunks[j]);
                    free(chunks); free(chunk_lens);
                    return CHUNKER_ERR_NON_IUPAC;
                }
            }
        }
    }

    /* Fisher-Yates shuffle of chunk order. */
    for (int i = n - 1; i > 0; i--) {
        int j = (int)philox_below(rng, (uint32_t)(i + 1));
        char *t = chunks[i]; chunks[i] = chunks[j]; chunks[j] = t;
        int tl = chunk_lens[i]; chunk_lens[i] = chunk_lens[j]; chunk_lens[j] = tl;
    }

    /* Concatenate into out. */
    int out_pos = 0;
    for (int i = 0; i < n; i++) {
        memcpy(out + out_pos, chunks[i], chunk_lens[i]);
        out_pos += chunk_lens[i];
        free(chunks[i]);
    }
    free(chunks); free(chunk_lens);

    *out_len = out_pos;
    return CHUNKER_OK;
}
