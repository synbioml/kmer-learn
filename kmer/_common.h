/* kmer/_common.h — shared utilities across all C extensions.
 *
 * This header is the single source of truth for:
 *   - ASCII nucleotide encoding tables (NUC_LUT, IUPAC_RC)
 *   - Philox4x32-10 counter-based RNG (Salmon et al. 2011)
 *   - splitmix64 for key derivation
 *
 * Every C extension in kmer includes this via the -I path. Submodule-
 * specific headers (e.g. perturb/_native/_shuffler.c) may add their
 * own utilities, but the basics live here.
 */
#ifndef KMER_COMMON_H_INCLUDED
#define KMER_COMMON_H_INCLUDED

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * ASCII nucleotide encoding
 * ====================================================================== */

/* A=0, C=1, G=2, T=3, everything else = 0xFF (invalid for strict ACGT).
 *
 * Indexed by uint8_t cast of the ASCII char. Note that this table gives
 * 0 for both 'A' and any unknown char — callers that need to distinguish
 * should check the ASCII value explicitly.
 */
static const uint8_t KMER_NUC_LUT[256] = {
    ['A'] = 0, ['a'] = 0,
    ['C'] = 1, ['c'] = 1,
    ['G'] = 2, ['g'] = 2,
    ['T'] = 3, ['t'] = 3,
};

/* IUPAC reverse-complement table (uppercase only). Unmapped chars
 * are 0; callers must check for 0 to detect invalid input. */
static const uint8_t KMER_IUPAC_RC[256] = {
    ['A'] = 'T',
    ['C'] = 'G',
    ['G'] = 'C',
    ['T'] = 'A',
    ['R'] = 'Y',
    ['Y'] = 'R',
    ['S'] = 'S',
    ['W'] = 'W',
    ['K'] = 'M',
    ['M'] = 'K',
    ['B'] = 'V',
    ['V'] = 'B',
    ['D'] = 'H',
    ['H'] = 'D',
    ['N'] = 'N',
};

/* Returns 1 if s is uppercase ACGT, 0 otherwise. */
static inline int kmer_is_strict_acgt(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (c != 'A' && c != 'C' && c != 'G' && c != 'T') return 0;
    }
    return 1;
}

/* Returns 1 if s is uppercase IUPAC, 0 otherwise. */
static inline int kmer_is_iupac(const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (KMER_IUPAC_RC[(uint8_t)s[i]] == 0) return 0;
    }
    return 1;
}

/* Pack a k-mer (k <= 12) into a uint32_t, 2 bits per base, MSB-first.
 * Caller must ensure strict ACGT (use kmer_is_strict_acgt first). */
static inline uint32_t kmer_pack(const char *s, int k) {
    uint32_t p = 0;
    for (int i = 0; i < k; i++) {
        p = (p << 2) | KMER_NUC_LUT[(uint8_t)s[i]];
    }
    return p;
}

/* Reverse-complement a packed k-mer (k <= 12). */
static inline uint32_t kmer_pack_rc(uint32_t code, int k) {
    static const uint8_t COMP[4] = {3, 2, 1, 0};  /* A<->T, C<->G */
    uint32_t out = 0;
    for (int i = 0; i < k; i++) {
        out = (out << 2) | COMP[code & 0b11];
        code >>= 2;
    }
    return out;
}

/* Canonical packed k-mer: min(code, rc(code)). Used for canonical_rc
 * collapsing in encoders. */
static inline uint32_t kmer_pack_canonical(uint32_t code, int k) {
    uint32_t rc = kmer_pack_rc(code, k);
    return code < rc ? code : rc;
}

/* ======================================================================
 * Philox4x32-10 — counter-based RNG
 *
 * State = (key[2], counter[2]). Each call produces 4 uint32s derived
 * from (key, counter), then increments the counter. Independent streams
 * are obtained by varying the key (e.g. one key per parallel task).
 *
 * Reference: Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3"
 * SC '11. Passes TestU01 BigCrush at 10 rounds.
 * ====================================================================== */

static const uint32_t KMER_PHILOX_M0 = 0xD2511F53u;
static const uint32_t KMER_PHILOX_M1 = 0xCD9E8D57u;
static const uint32_t KMER_PHILOX_W0 = 0x9E3779B9u;
static const uint32_t KMER_PHILOX_W1 = 0xBB67AE85u;

static inline uint32_t _kmer_philox_mulhi(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}

static inline void _kmer_philox_round(uint32_t ctr[4], uint32_t key[2]) {
    uint32_t hi0 = _kmer_philox_mulhi(KMER_PHILOX_M0, ctr[0]);
    uint32_t lo0 = KMER_PHILOX_M0 * ctr[0];
    uint32_t hi1 = _kmer_philox_mulhi(KMER_PHILOX_M1, ctr[2]);
    uint32_t lo1 = KMER_PHILOX_M1 * ctr[2];
    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
}

static inline void _kmer_philox_bumpkey(uint32_t key[2]) {
    key[0] += KMER_PHILOX_W0;
    key[1] += KMER_PHILOX_W1;
}

/* Generate 4 uint32s from (counter, key). 10 rounds per Salmon et al. */
static inline void kmer_philox4x32_10(uint32_t counter_in[4], uint32_t key_in[2],
                                       uint32_t out[4]) {
    uint32_t ctr[4] = {counter_in[0], counter_in[1], counter_in[2], counter_in[3]};
    uint32_t key[2] = {key_in[0], key_in[1]};
    for (int i = 0; i < 10; i++) {
        _kmer_philox_round(ctr, key);
        if (i < 9) _kmer_philox_bumpkey(key);
    }
    out[0] = ctr[0]; out[1] = ctr[1]; out[2] = ctr[2]; out[3] = ctr[3];
}

/* RNG stream — 64-bit key, 128-bit counter, 4-entry buffer. */
typedef struct {
    uint32_t key[2];
    uint32_t counter[4];
    uint32_t buf[4];
    int buf_used;
} kmer_philox_stream_t;

static inline void kmer_philox_init(kmer_philox_stream_t *s, uint64_t seed) {
    s->key[0] = (uint32_t)(seed & 0xFFFFFFFFu);
    s->key[1] = (uint32_t)(seed >> 32);
    s->counter[0] = 0; s->counter[1] = 0;
    s->counter[2] = 0; s->counter[3] = 0;
    s->buf_used = 4;
}

static inline void _kmer_philox_refill(kmer_philox_stream_t *s) {
    kmer_philox4x32_10(s->counter, s->key, s->buf);
    if (++s->counter[0] == 0) {
        if (++s->counter[1] == 0) {
            if (++s->counter[2] == 0) {
                ++s->counter[3];
            }
        }
    }
    s->buf_used = 0;
}

static inline uint32_t kmer_philox_u32(kmer_philox_stream_t *s) {
    if (s->buf_used >= 4) _kmer_philox_refill(s);
    return s->buf[s->buf_used++];
}

/* Uniform uint in [0, n). Lemire debiasing. */
static inline uint32_t kmer_philox_below(kmer_philox_stream_t *s, uint32_t n) {
    uint64_t m = (uint64_t)kmer_philox_u32(s) * (uint64_t)n;
    uint32_t l = (uint32_t)m;
    if (l < n) {
        uint32_t t = (0u - n) % n;
        while (l < t) {
            m = (uint64_t)kmer_philox_u32(s) * (uint64_t)n;
            l = (uint32_t)m;
        }
    }
    return (uint32_t)(m >> 32);
}

/* Uniform double in [0, 1). 53 bits of precision. */
static inline double kmer_philox_double(kmer_philox_stream_t *s) {
    uint32_t hi = kmer_philox_u32(s);
    uint32_t lo = kmer_philox_u32(s);
    uint64_t bits = ((uint64_t)hi << 21) | (lo >> 11);
    return (double)bits * (1.0 / (1ULL << 53));
}

/* splitmix64 — for deriving independent per-task keys. */
static inline uint64_t kmer_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

static inline void kmer_philox_derive_key(uint64_t master_seed, uint64_t task_index,
                                           uint32_t key_out[2]) {
    uint64_t k = kmer_splitmix64(master_seed ^ (task_index * 0x9E3779B97F4A7C15ULL));
    key_out[0] = (uint32_t)(k & 0xFFFFFFFFu);
    key_out[1] = (uint32_t)(k >> 32);
}

#ifdef __cplusplus
}
#endif

#endif  /* KMER_COMMON_H_INCLUDED */
