/* gkmkern.h
 *
 * C core for the gkmSVM / LS-GKM gapped k-mer kernel family, with two
 * kernel types:
 *
 *   gkm_kernel_t          — regular kernel (uniform L-mer weights)
 *   gkm_weighted_kernel_t — weighted kernel (positional weighting via
 *                            a configurable spatial kernel)
 *
 * Both share a common base struct (gkm_kernel_base_t) that holds the
 * mismatch weights, sqnorm lookup table, and reference sequences.
 *
 * Positional weighting supports 5 spatial kernel types (triangular,
 * epanechnikov, gaussian, laplacian, cauchy) applied to the L-mer
 * midpoint distance from a user-specified center.
 *
 * The kernel semantics mirror Dongwon Lee's libgkm.c (the gkmSVM /
 * lsgkm lineage). See gkmkern.c for implementation details.
 */

#ifndef GKM_KERN_H_INCLUDED
#define GKM_KERN_H_INCLUDED

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */
#define GKM_MAX_ALPHABET_SIZE       4   /* ACGT only; do NOT change */
#define GKM_MAX_ALPHABET_SIZE_SQ    16
#define GKM_MAX_MM                  12
#define GKM_MAX_SEQ_LENGTH          2048
#define GKM_MMCNT_LOOKUPTAB_WIDTH   8

/* ------------------------------------------------------------------ */
/* Enums                                                              */
/* ------------------------------------------------------------------ */

/* Per-mismatch weight scheme (how w[m] is computed from L, k, d). */
typedef enum {
    GKM_WEIGHT_FULL           = 0,  /* C(L-m, K); kernel_type=0 (GKM) */
    GKM_WEIGHT_TRUNCATED      = 1,  /* estimated + truncation; kernel_type=2 */
    GKM_WEIGHT_ESTIMATED_FULL = 2,  /* estimated, no truncation; kernel_type=1 */
} gkm_weight_scheme_t;

/* Positional (spatial) kernel type for weighted_kernel.
 *
 * Applied to the L-mer midpoint distance d from the user-specified
 * center. The result is multiplied by weight_peak to get the
 * positional weight, then used in the kernel evaluation.
 *
 * Order is fixed: triangular, epanechnikov, gaussian, laplacian, cauchy.
 */
typedef enum {
    GKM_POS_TRIANGULAR    = 0,  /* max(0, 1 - |d|/sigma) */
    GKM_POS_EPANECHNIKOV  = 1,  /* max(0, 1 - (d/sigma)^2) */
    GKM_POS_GAUSSIAN      = 2,  /* exp(-d^2 / (2*sigma^2)) */
    GKM_POS_LAPLACIAN     = 3,  /* exp(-gamma * |d|) */
    GKM_POS_CAUCHY        = 4,  /* 1 / (1 + (d/sigma)^2) */
} gkm_pos_kernel_t;

/* ------------------------------------------------------------------ */
/* Parameters                                                         */
/* ------------------------------------------------------------------ */

/* Basic kernel parameters (shared by both kernel types). */
typedef struct gkm_parameter_t {
    int L;          /* full word length (gapped), 2 <= L <= 12 */
    int k;          /* number of non-gap positions, k <= L */
    int d;          /* max mismatches, d <= L - k */
    gkm_weight_scheme_t weight_scheme;
    int use_rc;     /* if 1, RC strand is also indexed */
    int nthreads;   /* OpenMP thread count for parallel eval */
} gkm_parameter_t;

/* ------------------------------------------------------------------ */
/* Opaque handles                                                     */
/* ------------------------------------------------------------------ */
typedef struct gkm_sequence_s      gkm_sequence_t;
typedef struct gkm_kernel_s        gkm_kernel_t;
typedef struct gkm_weighted_kernel_s gkm_weighted_kernel_t;
typedef struct gkm_window_iter_s   gkm_window_iter_t;
typedef struct gkm_window_tree_cache gkm_window_tree_cache_t;

/* ------------------------------------------------------------------ */
/* Sequence encoding (shared by both kernel types)                    */
/* ------------------------------------------------------------------ */

/* Encode a nucleotide string. The kernel pointer is required because
 * encoding depends on L (for kmer-id pre-compute) and the kernel's
 * internal lookup tables. Either a gkm_kernel_t* or a
 * gkm_weighted_kernel_t* may be passed; the function reads only the
 * shared base fields. */
gkm_sequence_t *gkm_sequence_create(const void *kernel_base,
                                    const char *seq,
                                    const char *sid /* may be NULL */);

void gkm_sequence_free(gkm_sequence_t *seq);

int     gkm_sequence_length(const gkm_sequence_t *seq);
const char *gkm_sequence_sid(const gkm_sequence_t *seq);
double  gkm_sequence_sqnorm(const gkm_sequence_t *seq);

/* ------------------------------------------------------------------ */
/* Common base operations (work on either kernel type via base ptr)   */
/* ------------------------------------------------------------------ */

/* Validate parameters; returns NULL on success, error string otherwise. */
const char *gkm_parameter_check(const gkm_parameter_t *param);

/* Compute the number of valid windows for a given configuration.
 * Returns 0 if no valid windows fit. */
int gkm_compute_n_windows(int seqlen, int window, int shift,
                          int pad_left, int pad_right, int L);

/* Per-mismatch weight w[m] for m in [0, d]. */
double gkm_kernel_base_weight(const void *kernel_base, int m);

/* Return L, k, d via out-params. */
void gkm_kernel_base_get_params(const void *kernel_base, int *L, int *k, int *d);

/* Convenience: return L (frequently needed by the CPython extension). */
int gkm_kernel_base_get_L(const void *kernel_base);

/* Return weight_scheme and use_rc. */
int gkm_kernel_base_get_weight_scheme(const void *kernel_base);
int gkm_kernel_base_get_use_rc(const void *kernel_base);

/* Add a sequence to the kernel's reference set. The kernel takes
 * ownership of the gkm_sequence_t pointer. Returns seqid or -1. */
int gkm_kernel_base_add_sequence(void *kernel_base, gkm_sequence_t *seq);

/* Finalize the reference set (build prob_tree for regular kernel;
 * no-op for weighted kernel, which builds transient trees per call). */
void gkm_kernel_base_finalize(void *kernel_base);

int gkm_kernel_base_num_sequences(const void *kernel_base);
const gkm_sequence_t *gkm_kernel_base_get_sequence(const void *kernel_base, int seqid);

/* ------------------------------------------------------------------ */
/* Regular kernel (uniform L-mer weights)                             */
/* ------------------------------------------------------------------ */

gkm_kernel_t *gkm_kernel_create(const gkm_parameter_t *param);
void gkm_kernel_destroy(gkm_kernel_t *kernel);

/* Finalize: build prob_tree from the reference sequences. Must be
 * called after all add_sequence calls and before any eval. */
void gkm_kernel_finalize(gkm_kernel_t *kernel);

/* Accessor: get the base pointer (for use by the CPython extension). */
void *gkm_kernel_base(gkm_kernel_t *kernel);
void *gkm_kernel_base_const(const gkm_kernel_t *kernel);

/* K(a, b_seqid) for query a against all references. res must have
 * room for n_seqs doubles. */
void gkm_kernel_eval_all(const gkm_kernel_t *kernel,
                         const gkm_sequence_t *a, double *res);

/* Batch: K(query_i, ref_j) for n_q queries vs n_ref references.
 * OpenMP parallelised across queries. res must have n_q * n_ref doubles. */
void gkm_kernel_eval_batch(const gkm_kernel_t *kernel,
                           const gkm_sequence_t **queries, int n_q,
                           double *res);

/* ------------------------------------------------------------------ */
/* Regular kernel: windowed evaluation                                */
/* ------------------------------------------------------------------ */

gkm_window_iter_t *gkm_window_iter_create(const gkm_kernel_t *kernel,
                                          const gkm_sequence_t *seq,
                                          int window, int shift,
                                          int pad_left, int pad_right);
void gkm_window_iter_destroy(gkm_window_iter_t *it);
int  gkm_window_iter_count(const gkm_window_iter_t *it);
int  gkm_window_iter_next(gkm_window_iter_t *it, double *res);
void gkm_window_iter_reset(gkm_window_iter_t *it);

/* Within-window pairwise: n equal-length seqs vs themselves per window.
 * Output: out[w * n * n + i * n + j] = K(seq_i_w, seq_j_w).
 * n_windows * n * n doubles. */
int gkm_kernel_window_kernel(const gkm_kernel_t *kernel,
                             const gkm_sequence_t **seqs, int n,
                             int window, int shift,
                             int pad_left, int pad_right,
                             double *out);

/* Cross within-window: m queries vs n refs per window.
 * Output: out[w * n_q * n_ref + i * n_ref + j] = K(q_i_w, r_j_w). */
int gkm_kernel_cross_window_kernel(const gkm_kernel_t *kernel,
                                   const gkm_sequence_t **queries, int n_q,
                                   const gkm_sequence_t **refs, int n_ref,
                                   int window, int shift,
                                   int pad_left, int pad_right,
                                   double *out);

/* Cached cross-window queries (reuses per-window trees across calls). */
gkm_window_tree_cache_t *gkm_window_tree_cache_build(
    const gkm_kernel_t *kernel,
    const gkm_sequence_t **refs, int n_ref,
    int window, int shift,
    int pad_left, int pad_right);
int gkm_window_tree_cache_query(
    const gkm_kernel_t *kernel,
    const gkm_window_tree_cache_t *cache,
    const gkm_sequence_t **queries, int n_q,
    double *out);
void gkm_window_tree_cache_destroy(gkm_window_tree_cache_t *cache);

/* ------------------------------------------------------------------ */
/* Weighted kernel (positional weighting via spatial kernel)          */
/* ------------------------------------------------------------------ */

typedef struct gkm_weighted_params_t {
    gkm_parameter_t base;       /* L, k, d, weight_scheme, use_rc, nthreads */
    gkm_pos_kernel_t pos_kernel;/* triangular, epanechnikov, gaussian, laplacian, cauchy */
    double weight_gamma;        /* laplacian only: decay rate */
    double weight_sigma;        /* triangular/epanechnikov/gaussian/cauchy: scale */
    double weight_peak;         /* peak positional weight at the center */
} gkm_weighted_params_t;

gkm_weighted_kernel_t *gkm_weighted_kernel_create(const gkm_weighted_params_t *params);
void gkm_weighted_kernel_destroy(gkm_weighted_kernel_t *kernel);

/* Accessor: get the base pointer. */
void *gkm_weighted_kernel_base(gkm_weighted_kernel_t *kernel);
void *gkm_weighted_kernel_base_const(const gkm_weighted_kernel_t *kernel);

/* Return the spatial kernel type and params. */
gkm_pos_kernel_t gkm_weighted_kernel_pos_kernel(const gkm_weighted_kernel_t *kernel);
double gkm_weighted_kernel_gamma(const gkm_weighted_kernel_t *kernel);
double gkm_weighted_kernel_sigma(const gkm_weighted_kernel_t *kernel);
double gkm_weighted_kernel_peak(const gkm_weighted_kernel_t *kernel);

/* Compute the positional weight for a single distance d.
 * Inline-friendly; used by both sqnorm and DFS paths. */
static inline double gkm_pos_weight(double d, gkm_pos_kernel_t kind,
                                    double gamma, double sigma)
{
    switch (kind) {
        case GKM_POS_TRIANGULAR: {
            double ad = d < 0 ? -d : d;
            return ad >= sigma ? 0.0 : 1.0 - ad / sigma;
        }
        case GKM_POS_EPANECHNIKOV: {
            double ad = d < 0 ? -d : d;
            return ad >= sigma ? 0.0 : 1.0 - (ad * ad) / (sigma * sigma);
        }
        case GKM_POS_GAUSSIAN: {
            return exp(-(d * d) / (2.0 * sigma * sigma));
        }
        case GKM_POS_LAPLACIAN: {
            double ad = d < 0 ? -d : d;
            return exp(-gamma * ad);
        }
        case GKM_POS_CAUCHY: {
            return 1.0 / (1.0 + (d * d) / (sigma * sigma));
        }
        default:
            return 1.0;
    }
}

/* Self weighted kernel: n sequences vs themselves at n_centers centers.
 * centers[i] is a float position (in base pairs, 0-based, can be
 * half-integer for even-length sequences).
 *
 * Output: out[c * n * n + i * n + j] = K_w(seq_i, seq_j) at center c.
 * n_centers * n * n doubles. */
int gkm_weighted_kernel_eval_self(const gkm_weighted_kernel_t *kernel,
                                  const gkm_sequence_t **seqs, int n,
                                  const double *centers, int n_centers,
                                  double *out);

/* Cross weighted kernel: n_q queries vs n_ref refs at n_centers centers.
 * Output: out[c * n_q * n_ref + i * n_ref + j]. */
int gkm_weighted_kernel_eval_cross(const gkm_weighted_kernel_t *kernel,
                                   const gkm_sequence_t **queries, int n_q,
                                   const gkm_sequence_t **refs, int n_ref,
                                   const double *centers, int n_centers,
                                   double *out);

/* Windowed weighted kernel: n equal-length seqs, each window evaluated
 * with positional weights centered on the window's midpoint.
 * Output: out[w * n * n + i * n + j]. */
int gkm_weighted_kernel_window_kernel(const gkm_weighted_kernel_t *kernel,
                                      const gkm_sequence_t **seqs, int n,
                                      int window, int shift,
                                      int pad_left, int pad_right,
                                      double *out);

/* Sliding window query: scan a long query sequence with windows; for
 * each window, evaluate the weighted kernel against all references
 * (window-midpoint centering). Each window produces n_ref values.
 * Output: out[w * n_ref + j]. */
int gkm_weighted_kernel_sliding_query(const gkm_weighted_kernel_t *kernel,
                                      const gkm_sequence_t *query,
                                      int window, int shift,
                                      int pad_left, int pad_right,
                                      double *out, int *n_windows_out);

#ifdef __cplusplus
}
#endif

#endif  /* GKM_KERN_H_INCLUDED */
