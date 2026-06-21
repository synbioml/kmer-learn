/* gkmkern.c
 *
 * C implementation of the gkmSVM / LS-GKM gapped k-mer kernel family,
 * with two kernel types sharing a common base:
 *
 *   gkm_kernel_t          — regular kernel (uniform L-mer weights)
 *   gkm_weighted_kernel_t — weighted kernel (positional weighting via
 *                            a configurable spatial kernel)
 *
 * The regular kernel matches libgkm kernel_type 0/1/2 (full /
 * estimated_full / truncated). The weighted kernel adds positional
 * weighting (analogous to libgkm kernel_type 3/4/5 wgkm), but with
 * five spatial kernel types instead of just the Laplacian.
 *
 * See gkmkern.h for the public API and gkmkern.c (this file) for
 * implementation details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "gkmkern.h"

/* ================================================================== */
/* Internal limits and helpers                                        */
/* ================================================================== */

#define LEAF_COUNT(a) (1 << (2 * (a)))   /* 4^L: only valid for A=4   */
#define NODE_COUNT(a) (((1 << (2 * (a))) - 1) / (GKM_MAX_ALPHABET_SIZE - 1))
#define GKM_MAX_NUM   999999

/* ================================================================== */
/* Internal data structures                                           */
/* ================================================================== */

typedef struct {
    uint8_t *bid;     /* pointer into the query sequence's encoded buf */
    double   wt;      /* per-L-mer query weight (1.0 for uniform path) */
    int      mmcnt;   /* current mismatch count for this open L-mer    */
} base_mismatch_count_t;

typedef struct {
    int seqid;
    double wt;        /* accumulated positional weight for this seqid */
} kmer_leaf_data_t;

/* Offset-based leaf entry: used by weighted kernel to avoid rebuilding
 * the tree per centre. The tree stores (seqid, offset) for every L-mer
 * occurrence; the weight is computed on the fly during the DFS. */
typedef struct {
    int seqid;
    int offset;       /* L-mer offset in the sequence (0-based) */
} kmer_leaf_offset_t;

typedef struct {
    int                  count;
    int                  capacity;
    kmer_leaf_data_t    *data;       /* used by the regular tree */
    kmer_leaf_offset_t  *odata;      /* used by the offset-based tree */
    int                  use_offsets; /* 1 = odata, 0 = data */
} kmer_leaf_t;

typedef struct {
    int         L;
    int         k;
    int         d;
    int         node_count;
    int         leaf_count;
    int        *node;
    kmer_leaf_t *leaf;
} kmer_tree_t;

struct gkm_sequence_s {
    char    *sid;
    int      seqid;
    int      seqlen;
    uint8_t *seq;
    uint8_t *seq_rc;
    double  *wt;          /* positional weights (double, not uint8_t) */
    double  *wt_rc;
    int     *kmerids;
    int     *kmerids_rc;
    int      nkmers;
    double   sqnorm;
};

/* Shared base struct — embedded as the first member of both
 * gkm_kernel_t and gkm_weighted_kernel_t. */
typedef struct {
    gkm_parameter_t param;
    double          weights[GKM_MAX_MM + 1];
    gkm_sequence_t **seqs;
    int             n_seqs;
    int             cap_seqs;
    int             finalized;

    /* mismatch-count lookup table (for sqnorm) */
    uint8_t *mmcnt_tab;
    int      mmcnt_mask;
    int      mmcnt_nlookups;
} gkm_kernel_base_t;

struct gkm_kernel_s {
    gkm_kernel_base_t base;
    kmer_tree_t      *prob_tree;
};

struct gkm_weighted_kernel_s {
    gkm_kernel_base_t base;
    gkm_pos_kernel_t  pos_kernel;
    double            weight_gamma;
    double            weight_sigma;
    double            weight_peak;
};

/* ================================================================== */
/* Combinations                                                       */
/* ================================================================== */

static double dCombinations(int n, int r)
{
    if (r < 0) return 0;
    if (n < 0) return dCombinations(r - n - 1, r) * ((r % 2 == 0) ? 1 : -1);
    if (n < r) return 0;
    if ((n == 0) && (r == 0)) return 1.0;

    int i, j;
    double *nn = (double *)malloc(sizeof(double) * (size_t)(r + 1));
    double *no = (double *)malloc(sizeof(double) * (size_t)(r + 1));
    for (i = 0; i <= r; i++) nn[i] = no[i] = 0;
    nn[0] = no[0] = 1;

    for (i = 1; i <= n; i++) {
        double *h = no; no = nn; nn = h;
        for (j = 1; j <= r; j++) {
            nn[j] = no[j] + no[j - 1];
        }
    }
    double res = nn[r];
    free(nn);
    free(no);
    return res;
}

/* ================================================================== */
/* Weight calculation (operates on gkm_kernel_base_t)                 */
/* ================================================================== */

static void calc_full_weights(gkm_kernel_base_t *base)
{
    int L = base->param.L;
    int K = base->param.k;
    int i;
    for (i = 0; i <= L; i++) {
        if ((L - i) >= K) {
            base->weights[i] = dCombinations(L - i, K);
        } else {
            base->weights[i] = 0.0;
        }
    }
}

static void calc_estimated_weights(gkm_kernel_base_t *base, int truncated)
{
    int b = GKM_MAX_ALPHABET_SIZE;
    int L = base->param.L;
    int K = base->param.k;
    double *res = base->weights;
    double **wL  = (double **)malloc(sizeof(double *) * (size_t)(K + 1));
    double **wLp = (double **)malloc(sizeof(double *) * (size_t)(K + 1));
    double *wm      = (double *)malloc(sizeof(double) * (size_t)(K + 1));
    double *kernel  = (double *)malloc(sizeof(double) * (size_t)(L + 1));
    double *kernelTr = (double *)malloc(sizeof(double) * (size_t)(L + 1));
    double **hv;
    int i, j;
    int iL, iK, jM;
    int m;

    for (i = 0; i <= K; i++) {
        wL[i]  = (double *)malloc(sizeof(double) * (size_t)(K + 1));
        wLp[i] = (double *)malloc(sizeof(double) * (size_t)(K + 1));
        for (j = 0; j <= K; j++) {
            wL[i][j] = wLp[i][j] = 1.0;
        }
    }

    for (iL = 1; iL <= L; iL++) {
        for (iK = 1; iK <= K; iK++) {
            wL[iK][0] = wLp[iK][0] + (b - 1) * wLp[iK - 1][0];
            for (jM = 1; jM <= iK; jM++) {
                wL[iK][jM] = (wL[iK - 1][jM - 1] * (iK - iL)) / iK;
            }
        }
        hv = wLp; wLp = wL; wL = hv;
    }

    double nnorm = dCombinations(L, K) * pow((double)b, 1.0 * L);
    for (i = 0; i <= K; i++) {
        wm[i] = wLp[K][i] / nnorm;
    }

    for (m = 0; m <= L; m++) {
        int ub = (m < K) ? m : K;
        kernel[m] = 0;
        for (i = 0; i <= ub; i++) {
            kernel[m] += wm[i] * dCombinations(L - m, K - i) * dCombinations(m, i);
        }
    }

    int hn = 1;
    for (i = 0; i <= L; i++) {
        if (kernel[i] < 1e-50) hn = 0;
        kernelTr[i] = (truncated && !hn) ? 0.0 : kernel[i];
    }

    for (m = 0; m <= L; m++) {
        int m1, m2, t;
        double w = 0;
        for (m1 = 0; m1 <= L; m1++) {
            for (m2 = 0; m2 <= L; m2++) {
                for (t = 0; t <= L; t++) {
                    int r = m1 + m2 - 2 * t - L + m;
                    if ((t <= m) && ((m1 - t) <= (L - m)) &&
                        (r <= (m1 - t)) && (r >= 0)) {
                        double cc = dCombinations(m, t) *
                                    dCombinations(L - m, m1 - t) *
                                    dCombinations(m1 - t, r) *
                                    pow((double)(b - 1), 1.0 * t) *
                                    pow((double)(b - 2), 1.0 * r);
                        w += cc * kernelTr[m1] * kernelTr[m2];
                    }
                }
            }
        }
        res[L - m] = w;
    }

    for (i = 0; i <= K; i++) {
        free(wL[i]);
        free(wLp[i]);
    }
    free(wL);
    free(wLp);
    free(wm);
    free(kernel);
    free(kernelTr);
}

/* ================================================================== */
/* Kmer tree                                                          */
/* ================================================================== */

static void kmer_tree_init(kmer_tree_t *tree, int L, int k, int d)
{
    int i;
    tree->L = L;
    tree->k = k;
    tree->d = d;
    tree->node_count = NODE_COUNT(L);
    tree->node = (int *)calloc((size_t)tree->node_count, sizeof(int));
    for (i = 0; i < tree->node_count; i++) {
        tree->node[i] = GKM_MAX_NUM;
    }
    tree->leaf_count = LEAF_COUNT(L);
    tree->leaf = (kmer_leaf_t *)calloc((size_t)tree->leaf_count,
                                       sizeof(kmer_leaf_t));
}

static void kmer_tree_destroy(kmer_tree_t *tree)
{
    if (!tree) return;
    if (tree->node) free(tree->node);
    if (tree->leaf) {
        int i;
        for (i = 0; i < tree->leaf_count; i++) {
            if (tree->leaf[i].data)  free(tree->leaf[i].data);
            if (tree->leaf[i].odata) free(tree->leaf[i].odata);
        }
        free(tree->leaf);
    }
    free(tree);
}

/* Insert all L-mers of `d` (both forward and RC strand) into the tree
 * under seqid `seqid`. The tree stores, per leaf, a small stack of
 * (seqid, accumulated weight) pairs. */
static void kmer_tree_add_sequence(const kmer_tree_t *tree,
                                   int seqid,
                                   const gkm_sequence_t *d,
                                   int use_rc)
{
    int i, j, k;
    int n_strands = use_rc ? 2 : 1;
    uint8_t *seqs[2] = { d->seq, d->seq_rc };
    double  *wts[2]  = { d->wt,  d->wt_rc  };

    for (k = 0; k < n_strands; k++) {
        uint8_t *seq = seqs[k];
        double  *wt  = wts[k];
        for (j = 0; j < d->nkmers; j++) {
            int node_index = 0;
            int found = 0;
            for (i = 0; i < tree->L; i++) {
                if (tree->node[node_index] > seqid) {
                    tree->node[node_index] = seqid;
                }
                node_index = (node_index * GKM_MAX_ALPHABET_SIZE) + seq[i + j];
            }
            kmer_leaf_t *leaf = tree->leaf + (node_index - tree->node_count);
            if (leaf->capacity == 0) {
                leaf->count = 0;
                leaf->capacity = 1;
                leaf->data = (kmer_leaf_data_t *)malloc(
                    sizeof(kmer_leaf_data_t) * 1);
            } else if (leaf->count == leaf->capacity) {
                int new_cap = leaf->capacity * 2;
                kmer_leaf_data_t *newdata = (kmer_leaf_data_t *)malloc(
                    sizeof(kmer_leaf_data_t) * (size_t)new_cap);
                for (i = 0; i < leaf->count; i++) {
                    newdata[i].seqid = leaf->data[i].seqid;
                    newdata[i].wt    = leaf->data[i].wt;
                }
                free(leaf->data);
                leaf->capacity = new_cap;
                leaf->data = newdata;
            }
            for (i = 0; i < leaf->count; i++) {
                if (leaf->data[i].seqid == seqid) {
                    leaf->data[i].wt += wt[j];
                    found = 1;
                    break;
                }
            }
            if (!found) {
                leaf->data[leaf->count].seqid = seqid;
                leaf->data[leaf->count].wt    = wt[j];
                leaf->count++;
            }
        }
    }
}

/* Like kmer_tree_add_sequence, but stores (seqid, offset) instead of
 * (seqid, wt). Used by the weighted kernel to build the tree once and
 * reweight on the fly per centre. */
static void kmer_tree_add_sequence_offsets(const kmer_tree_t *tree,
                                           int seqid,
                                           const gkm_sequence_t *d,
                                           int use_rc)
{
    int i, j, k;
    int n_strands = use_rc ? 2 : 1;
    uint8_t *seqs[2] = { d->seq, d->seq_rc };

    for (k = 0; k < n_strands; k++) {
        uint8_t *seq = seqs[k];
        for (j = 0; j < d->nkmers; j++) {
            int node_index = 0;
            for (i = 0; i < tree->L; i++) {
                if (tree->node[node_index] > seqid) {
                    tree->node[node_index] = seqid;
                }
                node_index = (node_index * GKM_MAX_ALPHABET_SIZE) + seq[i + j];
            }
            kmer_leaf_t *leaf = tree->leaf + (node_index - tree->node_count);
            leaf->use_offsets = 1;
            if (leaf->capacity == 0) {
                leaf->count = 0;
                leaf->capacity = 1;
                leaf->odata = (kmer_leaf_offset_t *)malloc(
                    sizeof(kmer_leaf_offset_t) * 1);
            } else if (leaf->count == leaf->capacity) {
                int new_cap = leaf->capacity * 2;
                kmer_leaf_offset_t *newdata = (kmer_leaf_offset_t *)malloc(
                    sizeof(kmer_leaf_offset_t) * (size_t)new_cap);
                for (i = 0; i < leaf->count; i++) {
                    newdata[i].seqid  = leaf->odata[i].seqid;
                    newdata[i].offset = leaf->odata[i].offset;
                }
                free(leaf->odata);
                leaf->capacity = new_cap;
                leaf->odata = newdata;
            }
            int stored_offset = (k == 1) ? (d->nkmers - 1 - j) : j;
            leaf->odata[leaf->count].seqid  = seqid;
            leaf->odata[leaf->count].offset = stored_offset;
            leaf->count++;
        }
    }
}

/* Regular DFS: accumulate mmprofile[mmcnt][seqid] += wt * qwt for
 * every (seqid, wt) at each reached leaf, restricted to seqid < last_seqid. */
static void kmer_tree_dfs(const kmer_tree_t *tree,
                          int last_seqid,
                          int depth,
                          int curr_node_index,
                          const base_mismatch_count_t *curr_matching,
                          int curr_n_matching,
                          double **mmprof)
{
    int i, j;
    int bid;
    const int d = tree->d;

    if (depth == tree->L - 1) {
        kmer_leaf_t *leaf = tree->leaf +
            (curr_node_index * GKM_MAX_ALPHABET_SIZE) - tree->node_count;
        for (bid = 1; bid <= GKM_MAX_ALPHABET_SIZE; bid++) {
            leaf++;
            if (leaf->count > 0) {
                for (j = 0; j < curr_n_matching; j++) {
                    uint8_t currbase       = *curr_matching[j].bid;
                    int     currbase_mmcnt = curr_matching[j].mmcnt;
                    double  currbase_wt    = curr_matching[j].wt;
                    if (currbase == bid) {
                        int leaf_cnt = leaf->count;
                        const kmer_leaf_data_t *data = leaf->data;
                        double *mmprof_mmcnt = mmprof[currbase_mmcnt];
                        for (i = 0; i < leaf_cnt; i++) {
                            if (data[i].seqid < last_seqid) {
                                mmprof_mmcnt[data[i].seqid] += data[i].wt * currbase_wt;
                            }
                        }
                    } else if (currbase_mmcnt < d) {
                        int leaf_cnt = leaf->count;
                        const kmer_leaf_data_t *data = leaf->data;
                        double *mmprof_mmcnt = mmprof[currbase_mmcnt + 1];
                        for (i = 0; i < leaf_cnt; i++) {
                            if (data[i].seqid < last_seqid) {
                                mmprof_mmcnt[data[i].seqid] += data[i].wt * currbase_wt;
                            }
                        }
                    }
                }
            }
        }
    } else {
        int daughter_node_index = curr_node_index * GKM_MAX_ALPHABET_SIZE;
        for (bid = 1; bid <= GKM_MAX_ALPHABET_SIZE; bid++) {
            daughter_node_index++;
            if (tree->node[daughter_node_index] < last_seqid) {
                base_mismatch_count_t next_matching[GKM_MAX_SEQ_LENGTH];
                int next_n_matching = 0;
                for (j = 0; j < curr_n_matching; j++) {
                    uint8_t *currbase_ptr = curr_matching[j].bid;
                    int      currbase_mmcnt = curr_matching[j].mmcnt;
                    double   currbase_wt    = curr_matching[j].wt;
                    if (*currbase_ptr == bid) {
                        next_matching[next_n_matching].bid    = currbase_ptr + 1;
                        next_matching[next_n_matching].wt     = currbase_wt;
                        next_matching[next_n_matching].mmcnt  = currbase_mmcnt;
                        next_n_matching++;
                    } else if (currbase_mmcnt < d) {
                        next_matching[next_n_matching].bid    = currbase_ptr + 1;
                        next_matching[next_n_matching].wt     = currbase_wt;
                        next_matching[next_n_matching].mmcnt  = currbase_mmcnt + 1;
                        next_n_matching++;
                    }
                }
                if (next_n_matching > 0) {
                    kmer_tree_dfs(tree, last_seqid, depth + 1,
                                  daughter_node_index,
                                  next_matching, next_n_matching,
                                  mmprof);
                }
            }
        }
    }
}

/* Multi-centre weighted DFS: accumulates into n_centers separate
 * mmprofile arrays simultaneously. The trie traversal and mismatch-count
 * computation happen ONCE; only the weight multiplication is done per
 * centre.
 *
 * `mmprof_multi` is a 3D array of shape [n_centers][d+1][n], laid out as
 * mmprof_multi[c * (d+1) * n + m * n + seqid].
 *
 * `centers` is an array of n_centers center positions (in base pairs,
 * 0-based, can be half-integer). The center is the *sequence* position
 * (not L-mer offset); the L-mer at offset j has midpoint j + (L-1)/2.0,
 * so the distance from center is (j + (L-1)/2.0) - center.
 *
 * `query_seq` is a pointer to the start of the query's encoded forward
 * strand, used to compute the query L-mer offset (= bid - query_seq). */
static void kmer_tree_dfs_weighted_multi(
    const kmer_tree_t *tree,
    int last_seqid,
    int depth,
    int curr_node_index,
    const base_mismatch_count_t *curr_matching,
    int curr_n_matching,
    double *mmprof_multi,           /* [n_centers][d+1][n] */
    int n_centers,
    const double *centers,
    const uint8_t *query_seq,
    gkm_pos_kernel_t pos_kernel,
    double gamma, double sigma, double peak,
    int n                       /* number of sequences */
)
{
    int i, j, c;
    int bid;
    const int d = tree->d;
    int L = tree->L;
    double lmer_midpoint_offset = (L - 1) / 2.0;

    if (depth == L - 1) {
        kmer_leaf_t *leaf = tree->leaf +
            (curr_node_index * GKM_MAX_ALPHABET_SIZE) - tree->node_count;
        for (bid = 1; bid <= GKM_MAX_ALPHABET_SIZE; bid++) {
            leaf++;
            if (leaf->count > 0) {
                const kmer_leaf_offset_t *odata = leaf->odata;
                int leaf_cnt = leaf->count;
                for (j = 0; j < curr_n_matching; j++) {
                    uint8_t currbase       = *curr_matching[j].bid;
                    int     currbase_mmcnt = curr_matching[j].mmcnt;
                    /* Query L-mer offset: at depth == L-1, bid points
                     * to the last base of the L-mer, so the start
                     * offset is (bid - query_seq) - (L-1). */
                    int q_offset = (int)(curr_matching[j].bid - query_seq) - (L - 1);
                    if (q_offset < 0) q_offset = 0;
                    double q_midpoint = q_offset + lmer_midpoint_offset;

                    int mmcnt_idx;
                    if (currbase == bid) {
                        mmcnt_idx = currbase_mmcnt;
                    } else if (currbase_mmcnt < d) {
                        mmcnt_idx = currbase_mmcnt + 1;
                    } else {
                        continue;
                    }

                    for (i = 0; i < leaf_cnt; i++) {
                        if (odata[i].seqid >= last_seqid) continue;
                        double t_midpoint = odata[i].offset + lmer_midpoint_offset;
                        for (c = 0; c < n_centers; c++) {
                            double center = centers[c];
                            double q_dist = q_midpoint - center;
                            double t_dist = t_midpoint - center;
                            double q_w = peak * gkm_pos_weight(q_dist, pos_kernel, gamma, sigma);
                            double t_w = peak * gkm_pos_weight(t_dist, pos_kernel, gamma, sigma);
                            int idx = c * (d + 1) * n + mmcnt_idx * n + odata[i].seqid;
                            mmprof_multi[idx] += (q_w * t_w);
                        }
                    }
                }
            }
        }
    } else {
        int daughter_node_index = curr_node_index * GKM_MAX_ALPHABET_SIZE;
        for (bid = 1; bid <= GKM_MAX_ALPHABET_SIZE; bid++) {
            daughter_node_index++;
            if (tree->node[daughter_node_index] < last_seqid) {
                base_mismatch_count_t next_matching[GKM_MAX_SEQ_LENGTH];
                int next_n_matching = 0;
                for (j = 0; j < curr_n_matching; j++) {
                    uint8_t *currbase_ptr = curr_matching[j].bid;
                    int      currbase_mmcnt = curr_matching[j].mmcnt;
                    if (*currbase_ptr == bid) {
                        next_matching[next_n_matching].bid    = currbase_ptr + 1;
                        next_matching[next_n_matching].mmcnt  = currbase_mmcnt;
                        next_n_matching++;
                    } else if (currbase_mmcnt < d) {
                        next_matching[next_n_matching].bid    = currbase_ptr + 1;
                        next_matching[next_n_matching].mmcnt  = currbase_mmcnt + 1;
                        next_n_matching++;
                    }
                }
                if (next_n_matching > 0) {
                    kmer_tree_dfs_weighted_multi(
                        tree, last_seqid, depth + 1,
                        daughter_node_index,
                        next_matching, next_n_matching,
                        mmprof_multi, n_centers, centers,
                        query_seq, pos_kernel, gamma, sigma, peak, n);
                }
            }
        }
    }
}

/* ================================================================== */
/* Sqnorm via two-bit XOR lookup table                                */
/* ================================================================== */

static void build_mmcnt_lookuptable(gkm_kernel_base_t *base)
{
    int i, j;
    int mask = 3;
    unsigned int tablesize = (1u << (GKM_MMCNT_LOOKUPTAB_WIDTH * 2));
    base->mmcnt_tab = (uint8_t *)malloc(sizeof(uint8_t) * tablesize);
    for (i = 0; i < (int)tablesize; i++) {
        int xor_word = i;
        base->mmcnt_tab[i] = 0;
        for (j = 0; j < GKM_MMCNT_LOOKUPTAB_WIDTH; j++) {
            if ((xor_word & mask) != 0) {
                base->mmcnt_tab[i]++;
            }
            xor_word >>= 2;
        }
    }
    base->mmcnt_mask = 0;
    for (i = 0; i < GKM_MMCNT_LOOKUPTAB_WIDTH; i++) {
        base->mmcnt_mask = ((base->mmcnt_mask << 2) | 3);
    }
    if (base->param.L <= GKM_MMCNT_LOOKUPTAB_WIDTH) {
        base->mmcnt_nlookups = 1;
    } else if (base->param.L <= GKM_MMCNT_LOOKUPTAB_WIDTH * 2) {
        base->mmcnt_nlookups = 2;
    } else {
        base->mmcnt_nlookups = 2;
    }
}

static int sequence_to_twobitids(const gkm_kernel_base_t *base,
                                 const gkm_sequence_t *d,
                                 int *twobitids)
{
    int i, j;
    uint8_t *seqs[2] = { d->seq, d->seq_rc };
    int nids = d->nkmers * 2;
    int mask = 3;
    for (i = 0; i < base->param.L - 1; i++) {
        mask = ((mask << 2) | 3);
    }
    int cnt = 0;
    for (j = 0; j < 2; j++) {
        int twobitid = 0;
        uint8_t *s = seqs[j];
        for (i = 0; i < base->param.L - 1; i++) {
            twobitid = ((twobitid << 2) | (s[i] - 1));
        }
        for (i = base->param.L - 1; i < d->seqlen; i++) {
            twobitid = (((twobitid << 2) & mask) | (s[i] - 1));
            twobitids[cnt] = (twobitid & base->mmcnt_mask);
            if (base->mmcnt_nlookups == 2) {
                twobitids[nids + cnt] =
                    ((twobitid >> (2 * GKM_MMCNT_LOOKUPTAB_WIDTH)) &
                     base->mmcnt_mask);
            }
            cnt++;
        }
    }
    return cnt;
}

static double compute_sqnorm(const gkm_kernel_base_t *base,
                             const gkm_sequence_t *da)
{
    int i, j, k;
    int twobitids[GKM_MAX_SEQ_LENGTH * 2 * 2];
    double wt[GKM_MAX_SEQ_LENGTH * 2];
    int nkmerids = da->nkmers;
    int d = base->param.d;
    double mmprofile[GKM_MAX_MM];
    int use_rc = base->param.use_rc;

    for (k = 0; k <= d; k++) mmprofile[k] = 0;

    if (use_rc) {
        int nids = sequence_to_twobitids(base, da, twobitids);
        for (i = 0; i < nkmerids; i++) wt[i] = da->wt[i];
        for (i = 0; i < nkmerids; i++) wt[nkmerids + i] = da->wt_rc[i];

        for (i = 0; i < nkmerids; i++) {
            int id0 = twobitids[i];
            int id1 = twobitids[nids + i];
            double wt_i = wt[i];
            for (j = 0; j < nids; j++) {
                int mmcnt = base->mmcnt_tab[id0 ^ twobitids[j]];
                if ((mmcnt <= d) && (base->mmcnt_nlookups == 2)) {
                    mmcnt += base->mmcnt_tab[id1 ^ twobitids[nids + j]];
                }
                if (mmcnt <= d) {
                    mmprofile[mmcnt] += (wt_i * wt[j]);
                }
            }
        }
    } else {
        int nids = sequence_to_twobitids(base, da, twobitids);
        for (i = 0; i < nkmerids; i++) wt[i] = da->wt[i];

        for (i = 0; i < nkmerids; i++) {
            int id0 = twobitids[i];
            int id1 = twobitids[nids + i];
            double wt_i = wt[i];
            for (j = 0; j < nkmerids; j++) {
                int mmcnt = base->mmcnt_tab[id0 ^ twobitids[j]];
                if ((mmcnt <= d) && (base->mmcnt_nlookups == 2)) {
                    mmcnt += base->mmcnt_tab[id1 ^ twobitids[nids + j]];
                }
                if (mmcnt <= d) {
                    mmprofile[mmcnt] += (wt_i * wt[j]);
                }
            }
        }
        (void)nids;
    }

    double sum = 0;
    for (k = 0; k <= d; k++) {
        sum += base->weights[k] * mmprofile[k];
    }
    if (sum < 0) sum = 0;
    return sqrt(sum);
}

/* ================================================================== */
/* Sequence encoding                                                  */
/* ================================================================== */

gkm_sequence_t *gkm_sequence_create(const void *kernel_base,
                                    const char *seq,
                                    const char *sid)
{
    const gkm_kernel_base_t *base = (const gkm_kernel_base_t *)kernel_base;
    int i, j, k;
    int L = base->param.L;
    int seqlen = (int)strlen(seq);
    if (seqlen < L) return NULL;
    if (seqlen > GKM_MAX_SEQ_LENGTH - 1) seqlen = GKM_MAX_SEQ_LENGTH - 1;

    gkm_sequence_t *d = (gkm_sequence_t *)calloc(1, sizeof(gkm_sequence_t));
    if (sid) {
        d->sid = (char *)malloc(strlen(sid) + 1);
        strcpy(d->sid, sid);
    }
    d->seqlen = seqlen;
    d->seq    = (uint8_t *)malloc((size_t)seqlen);
    d->seq_rc = (uint8_t *)malloc((size_t)seqlen);

    for (i = 0; i < seqlen; i++) {
        switch (toupper((unsigned char)seq[i])) {
            case 'A': d->seq[i] = 1; break;
            case 'C': d->seq[i] = 2; break;
            case 'G': d->seq[i] = 3; break;
            case 'T': d->seq[i] = 4; break;
            default:
                if (d->sid) free(d->sid);
                free(d->seq); free(d->seq_rc);
                free(d);
                return NULL;
        }
    }
    for (i = 0; i < seqlen; i++) {
        switch (d->seq[seqlen - i - 1]) {
            case 1: d->seq_rc[i] = 4; break;
            case 2: d->seq_rc[i] = 3; break;
            case 3: d->seq_rc[i] = 2; break;
            case 4: d->seq_rc[i] = 1; break;
            default: d->seq_rc[i] = 1; break;
        }
    }

    d->nkmers  = seqlen - L + 1;
    d->kmerids    = (int *)malloc(sizeof(int) * (size_t)d->nkmers);
    d->kmerids_rc = (int *)malloc(sizeof(int) * (size_t)d->nkmers);
    d->wt    = (double *)malloc(sizeof(double) * (size_t)d->nkmers);
    d->wt_rc = (double *)malloc(sizeof(double) * (size_t)d->nkmers);

    {
        int total_node_count = NODE_COUNT(L);
        uint8_t *seqs[2]  = { d->seq, d->seq_rc };
        int     *kids[2]  = { d->kmerids, d->kmerids_rc };
        for (k = 0; k < 2; k++) {
            uint8_t *s = seqs[k];
            int     *kid = kids[k];
            for (j = 0; j < d->nkmers; j++) {
                int node_index = 0;
                for (i = 0; i < L; i++) {
                    node_index = (node_index * GKM_MAX_ALPHABET_SIZE) + s[i + j];
                }
                kid[j] = node_index - total_node_count;
            }
        }
    }

    /* Uniform weights (1.0) by default. The weighted kernel overrides
     * these by writing positional weights into a transient view before
     * calling compute_sqnorm. */
    for (i = 0; i < d->nkmers; i++) {
        d->wt[i]       = 1.0;
        d->wt_rc[i]    = 1.0;
    }

    d->sqnorm = compute_sqnorm(base, d);
    return d;
}

void gkm_sequence_free(gkm_sequence_t *seq)
{
    if (!seq) return;
    if (seq->sid)        free(seq->sid);
    if (seq->seq)        free(seq->seq);
    if (seq->seq_rc)     free(seq->seq_rc);
    if (seq->wt)         free(seq->wt);
    if (seq->wt_rc)      free(seq->wt_rc);
    if (seq->kmerids)    free(seq->kmerids);
    if (seq->kmerids_rc) free(seq->kmerids_rc);
    free(seq);
}

int     gkm_sequence_length(const gkm_sequence_t *seq) { return seq->seqlen; }
const char *gkm_sequence_sid(const gkm_sequence_t *seq) { return seq->sid; }
double  gkm_sequence_sqnorm(const gkm_sequence_t *seq) { return seq->sqnorm; }

/* ================================================================== */
/* Common base operations                                             */
/* ================================================================== */

const char *gkm_parameter_check(const gkm_parameter_t *param)
{
    static char buf[128];
    if (param->L < 2)              return "L < 2";
    if (param->L > 12)             return "L > 12";
    if (param->k > param->L)       return "k > L";
    if (param->k < 1)              return "k < 1";
    if (param->d > (param->L - param->k)) return "d > L - k";
    if (param->d < 0) {
        snprintf(buf, sizeof(buf), "d < 0");
        return buf;
    }
    return NULL;
}

double gkm_kernel_base_weight(const void *kernel_base, int m)
{
    const gkm_kernel_base_t *base = (const gkm_kernel_base_t *)kernel_base;
    if (m < 0 || m > base->param.d) return 0.0;
    return base->weights[m];
}

void gkm_kernel_base_get_params(const void *kernel_base, int *L, int *k, int *d)
{
    const gkm_kernel_base_t *base = (const gkm_kernel_base_t *)kernel_base;
    if (L) *L = base->param.L;
    if (k) *k = base->param.k;
    if (d) *d = base->param.d;
}

int gkm_kernel_base_get_L(const void *kernel_base)
{
    return ((const gkm_kernel_base_t *)kernel_base)->param.L;
}

int gkm_kernel_base_get_weight_scheme(const void *kernel_base)
{
    return (int)((const gkm_kernel_base_t *)kernel_base)->param.weight_scheme;
}

int gkm_kernel_base_get_use_rc(const void *kernel_base)
{
    return ((const gkm_kernel_base_t *)kernel_base)->param.use_rc;
}

int gkm_kernel_base_add_sequence(void *kernel_base, gkm_sequence_t *seq)
{
    gkm_kernel_base_t *base = (gkm_kernel_base_t *)kernel_base;
    if (base->finalized) return -1;
    if (base->n_seqs == base->cap_seqs) {
        int new_cap = base->cap_seqs == 0 ? 16 : base->cap_seqs * 2;
        base->seqs = (gkm_sequence_t **)realloc(
            base->seqs, sizeof(gkm_sequence_t *) * (size_t)new_cap);
        base->cap_seqs = new_cap;
    }
    int seqid = base->n_seqs++;
    seq->seqid = seqid;
    base->seqs[seqid] = seq;
    return seqid;
}

void gkm_kernel_base_finalize(void *kernel_base)
{
    gkm_kernel_base_t *base = (gkm_kernel_base_t *)kernel_base;
    if (base->finalized) return;
    /* For the regular kernel, finalize() builds prob_tree. The weighted
     * kernel doesn't use prob_tree (it builds transient trees per call),
     * so finalize() is a no-op for it — it just sets the finalized flag. */
    base->finalized = 1;
}

int gkm_kernel_base_num_sequences(const void *kernel_base)
{
    return ((const gkm_kernel_base_t *)kernel_base)->n_seqs;
}

const gkm_sequence_t *gkm_kernel_base_get_sequence(const void *kernel_base, int seqid)
{
    const gkm_kernel_base_t *base = (const gkm_kernel_base_t *)kernel_base;
    if (seqid < 0 || seqid >= base->n_seqs) return NULL;
    return base->seqs[seqid];
}

/* Helper to init the base struct (called by both kernel constructors). */
static void gkm_kernel_base_init(gkm_kernel_base_t *base,
                                 const gkm_parameter_t *param)
{
    base->param = *param;
    switch (param->weight_scheme) {
        case GKM_WEIGHT_TRUNCATED:
            calc_estimated_weights(base, 1);
            break;
        case GKM_WEIGHT_ESTIMATED_FULL:
            calc_estimated_weights(base, 0);
            break;
        case GKM_WEIGHT_FULL:
        default:
            calc_full_weights(base);
            break;
    }
    build_mmcnt_lookuptable(base);
    base->seqs = NULL;
    base->n_seqs = 0;
    base->cap_seqs = 0;
    base->finalized = 0;
}

static void gkm_kernel_base_destroy(gkm_kernel_base_t *base)
{
    if (base->seqs) {
        int i;
        for (i = 0; i < base->n_seqs; i++) {
            if (base->seqs[i]) gkm_sequence_free(base->seqs[i]);
        }
        free(base->seqs);
    }
    if (base->mmcnt_tab) free(base->mmcnt_tab);
}

/* ================================================================== */
/* Regular kernel: create / destroy                                   */
/* ================================================================== */

gkm_kernel_t *gkm_kernel_create(const gkm_parameter_t *param)
{
    const char *err = gkm_parameter_check(param);
    if (err) return NULL;
    gkm_kernel_t *k = (gkm_kernel_t *)calloc(1, sizeof(gkm_kernel_t));
    gkm_kernel_base_init(&k->base, param);
    k->prob_tree = NULL;
    return k;
}

void gkm_kernel_destroy(gkm_kernel_t *kernel)
{
    if (!kernel) return;
    if (kernel->prob_tree) kmer_tree_destroy(kernel->prob_tree);
    gkm_kernel_base_destroy(&kernel->base);
    free(kernel);
}

void *gkm_kernel_base(gkm_kernel_t *kernel)
{
    return (void *)&kernel->base;
}

void *gkm_kernel_base_const(const gkm_kernel_t *kernel)
{
    return (void *)&kernel->base;
}

/* ================================================================== */
/* Regular kernel: finalize (builds prob_tree)                        */
/* ================================================================== */

/* Finalize for the regular kernel: build prob_tree. */
void gkm_kernel_finalize(gkm_kernel_t *kernel)
{
    int i;
    if (kernel->base.finalized) return;
    kernel->prob_tree = (kmer_tree_t *)malloc(sizeof(kmer_tree_t));
    kmer_tree_init(kernel->prob_tree,
                   kernel->base.param.L, kernel->base.param.k, kernel->base.param.d);
    for (i = 0; i < kernel->base.n_seqs; i++) {
        kmer_tree_add_sequence(kernel->prob_tree, i, kernel->base.seqs[i],
                               kernel->base.param.use_rc);
    }
    kernel->base.finalized = 1;
}

/* ================================================================== */
/* Regular kernel: evaluation                                         */
/* ================================================================== */

static void eval_against_tree(const gkm_kernel_base_t *base,
                              const kmer_tree_t *tree,
                              const gkm_sequence_t *a,
                              int n_refs,
                              double *res,
                              double **mmprofile,
                              base_mismatch_count_t *matching_bases)
{
    int i, j, k;
    int d = base->param.d;
    int num_matching = a->nkmers;

    for (i = 0; i < num_matching; i++) {
        matching_bases[i].bid   = a->seq + i;
        matching_bases[i].wt = 1.0;

        matching_bases[i].mmcnt = 0;
    }
    for (k = 0; k <= d; k++) {
        for (j = 0; j < n_refs; j++) mmprofile[k][j] = 0;
    }

    kmer_tree_dfs(tree, n_refs, 0, 0,
                  matching_bases, num_matching, mmprofile);

    for (j = 0; j < n_refs; j++) {
        double sum = 0;
        for (k = 0; k <= d; k++) {
            sum += base->weights[k] * mmprofile[k][j];
        }
        res[j] = sum;
    }
}

void gkm_kernel_eval_all(const gkm_kernel_t *kernel,
                         const gkm_sequence_t *a, double *res)
{
    int k;
    if (!kernel->prob_tree) {
        for (k = 0; k < kernel->base.n_seqs; k++) res[k] = 0.0;
        return;
    }
    int d = kernel->base.param.d;
    int n = kernel->base.n_seqs;
    double **mmprofile = (double **)malloc(sizeof(double *) * (size_t)(d + 1));
    for (k = 0; k <= d; k++) {
        mmprofile[k] = (double *)calloc((size_t)n, sizeof(double));
    }
    base_mismatch_count_t *matching_bases =
        (base_mismatch_count_t *)malloc(
            sizeof(base_mismatch_count_t) * (size_t)a->nkmers);
    if (!mmprofile || !matching_bases) {
        for (k = 0; k <= d; k++) free(mmprofile[k]);
        free(mmprofile); free(matching_bases);
        for (k = 0; k < n; k++) res[k] = 0.0;
        return;
    }

    eval_against_tree(&kernel->base, kernel->prob_tree, a, n, res,
                      mmprofile, matching_bases);

    double a_sqnorm = a->sqnorm;
    for (k = 0; k < n; k++) {
        double b_sqnorm = kernel->base.seqs[k]->sqnorm;
        if (a_sqnorm > 0 && b_sqnorm > 0) {
            res[k] /= (a_sqnorm * b_sqnorm);
        } else {
            res[k] = 0.0;
        }
    }

    for (k = 0; k <= d; k++) free(mmprofile[k]);
    free(mmprofile);
    free(matching_bases);
}

void gkm_kernel_eval_batch(const gkm_kernel_t *kernel,
                           const gkm_sequence_t **queries, int n_q,
                           double *res)
{
    int n_ref = kernel->base.n_seqs;
    if (!kernel->prob_tree) {
        for (int i = 0; i < n_q * n_ref; i++) res[i] = 0.0;
        return;
    }
    int nthreads = kernel->base.param.nthreads;
    if (nthreads < 1) nthreads = 1;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
    for (int i = 0; i < n_q; i++) {
        double *row = res + (size_t)i * n_ref;
        gkm_kernel_eval_all(kernel, queries[i], row);
    }
}

/* ================================================================== */
/* Padding helpers                                                    */
/*                                                                    */
/* Padding allows windows to overhang from the sequence:             */
/*   pad_left  = bases the first window extends left of position 0   */
/*   pad_right = bases the last window extends right of seqlen        */
/*                                                                    */
/* When a window overhangs, its effective length is reduced by the    */
/* overhang amount (we don't actually pad the sequence with dummy     */
/* bases; we just start the window later / end it earlier).           */
/*                                                                    */
/* Window w starts at position: w * shift - pad_left                  */
/*   (w=0 starts at -pad_left, i.e. the first `pad_left` bases are    */
/*    "outside" the sequence; the window's first valid base is at 0,  */
/*    so its effective length is window - pad_left)                   */
/* Window w ends at position: w * shift - pad_left + window           */
/*   (if this exceeds seqlen, the window's effective length is        */
/*    window - (end - seqlen) = window - pad_right)                   */
/*                                                                    */
/* A window must have effective length >= L to have at least one      */
/* L-mer. If the effective length < L, that window is skipped.        */
/* ================================================================== */

/* Compute the number of windows for a given configuration.
 * Returns 0 if no valid windows fit. */
int gkm_compute_n_windows(int seqlen, int window, int shift,
                          int pad_left, int pad_right, int L)
{
    /* First window starts at -pad_left; its effective length is
     * window - pad_left. Subsequent windows start at w*shift - pad_left.
     * A window is valid if its effective length >= L.
     *
     * We iterate until the window's start position is >= seqlen - L + 1
     * (no L-mer fits). */
    if (pad_left > 0 && window - pad_left < L) return 0;
    if (pad_right > 0 && window - pad_right < L) return 0;
    /* combined check above */
    int n_windows = 0;
    int w = 0;
    while (1) {
        int start = w * shift - pad_left;
        int end = start + window;
        int eff_start = start < 0 ? 0 : start;
        int eff_end = end > seqlen ? seqlen : end;
        int eff_len = eff_end - eff_start;
        if (eff_len < L) {
            /* If start is already past seqlen - L, no more windows. */
            if (start >= seqlen - L + 1) break;
            /* Otherwise, this window is too short but a later one might
             * still fit. Skip it. */
            w++;
            continue;
        }
        n_windows++;
        /* Stop once we've covered the last L-mer. */
        if (start >= seqlen - L - pad_right) break;
        w++;
        if (w > 1000000) break;  /* safety */
    }
    return n_windows;
}

/* ================================================================== */
/* Regular kernel: windowed iterator                                  */
/* ================================================================== */

struct gkm_window_iter_s {
    const gkm_kernel_t   *kernel;
    const gkm_sequence_t *seq;
    int  window;
    int  shift;
    int  pad_left;
    int  pad_right;
    int  cur_w;        /* current window index */
    int  n_windows;

    double                **mmprofile;
    base_mismatch_count_t *matching_bases;
    int                   n_refs;
    int                   d;
};

gkm_window_iter_t *gkm_window_iter_create(const gkm_kernel_t *kernel,
                                          const gkm_sequence_t *seq,
                                          int window, int shift,
                                          int pad_left, int pad_right)
{
    if (!kernel || !seq) return NULL;
    if (window < kernel->base.param.L) return NULL;
    if (shift < 1) return NULL;
    if (pad_left > 0 && window - pad_left < kernel->base.param.L) return NULL;
    if (pad_right > 0 && window - pad_right < kernel->base.param.L) return NULL;
    /* combined check above */

    gkm_window_iter_t *it = (gkm_window_iter_t *)calloc(1, sizeof(*it));
    it->kernel = kernel;
    it->seq    = seq;
    it->window = window;
    it->shift  = shift;
    it->pad_left = pad_left;
    it->pad_right = pad_right;
    it->cur_w = 0;
    it->n_windows = gkm_compute_n_windows(seq->seqlen, window, shift,
                                      pad_left, pad_right, kernel->base.param.L);
    it->n_refs = kernel->base.n_seqs;
    it->d      = kernel->base.param.d;

    int k;
    it->mmprofile = (double **)malloc(sizeof(double *) * (size_t)(it->d + 1));
    for (k = 0; k <= it->d; k++) {
        it->mmprofile[k] = (double *)calloc((size_t)it->n_refs, sizeof(double));
    }
    it->matching_bases = (base_mismatch_count_t *)malloc(
        sizeof(base_mismatch_count_t) * (size_t)(window - kernel->base.param.L + 1));
    return it;
}

void gkm_window_iter_destroy(gkm_window_iter_t *it)
{
    if (!it) return;
    if (it->mmprofile) {
        int k;
        for (k = 0; k <= it->d; k++) free(it->mmprofile[k]);
        free(it->mmprofile);
    }
    if (it->matching_bases) free(it->matching_bases);
    free(it);
}

int gkm_window_iter_count(const gkm_window_iter_t *it)
{
    return it->n_windows;
}

void gkm_window_iter_reset(gkm_window_iter_t *it)
{
    it->cur_w = 0;
}

int gkm_window_iter_next(gkm_window_iter_t *it, double *res)
{
    if (!it->kernel->prob_tree) return 0;
    const gkm_kernel_t *kernel = it->kernel;
    int L = kernel->base.param.L;
    int seqlen = it->seq->seqlen;

    /* Find the next valid window. */
    int w = it->cur_w;
    int start, end, eff_start, eff_end, eff_len;
    while (1) {
        if (w >= it->n_windows) return 0;
        start = w * it->shift - it->pad_left;
        end = start + it->window;
        eff_start = start < 0 ? 0 : start;
        eff_end = end > seqlen ? seqlen : end;
        eff_len = eff_end - eff_start;
        if (eff_len >= L) break;
        w++;
    }
    it->cur_w = w + 1;

    int nkmers = eff_len - L + 1;
    int rc_start = seqlen - eff_start - eff_len;
    int k, j;

    /* Build a transient gkm_sequence_t view for the window subsequence. */
    gkm_sequence_t vw;
    memset(&vw, 0, sizeof(vw));
    vw.seqlen = eff_len;
    vw.seq    = it->seq->seq + eff_start;
    vw.seq_rc = it->seq->seq_rc + rc_start;
    vw.nkmers = nkmers;
    vw.wt     = it->seq->wt + eff_start;
    vw.wt_rc  = it->seq->wt_rc + rc_start;
    vw.sqnorm = compute_sqnorm(&kernel->base, &vw);

    int num_matching = nkmers;
    for (j = 0; j < num_matching; j++) {
        it->matching_bases[j].bid   = vw.seq + j;
        it->matching_bases[j].wt = 1.0;

        it->matching_bases[j].mmcnt = 0;
    }
    for (k = 0; k <= it->d; k++) {
        for (j = 0; j < it->n_refs; j++) it->mmprofile[k][j] = 0;
    }
    kmer_tree_dfs(kernel->prob_tree, it->n_refs, 0, 0,
                  it->matching_bases, num_matching, it->mmprofile);

    double w_sqnorm = vw.sqnorm;
    for (j = 0; j < it->n_refs; j++) {
        double sum = 0;
        for (k = 0; k <= it->d; k++) {
            sum += kernel->base.weights[k] * it->mmprofile[k][j];
        }
        double b_sqnorm = kernel->base.seqs[j]->sqnorm;
        if (w_sqnorm > 0 && b_sqnorm > 0) {
            res[j] = sum / (w_sqnorm * b_sqnorm);
        } else {
            res[j] = 0.0;
        }
    }

    return 1;
}

/* ================================================================== */
/* Regular kernel: within-window pairwise                             */
/* ================================================================== */

static void make_window_view(const gkm_kernel_base_t *base,
                             const gkm_sequence_t *seq,
                             int window_start, int window_len,
                             gkm_sequence_t *out_view)
{
    int seqlen = seq->seqlen;
    int rc_start = seqlen - window_start - window_len;
    memset(out_view, 0, sizeof(*out_view));
    out_view->seqlen = window_len;
    out_view->seq    = seq->seq + window_start;
    out_view->seq_rc = seq->seq_rc + rc_start;
    out_view->nkmers = window_len - base->param.L + 1;
    out_view->wt     = seq->wt + window_start;
    out_view->wt_rc  = seq->wt_rc + rc_start;
    out_view->sqnorm = compute_sqnorm(base, out_view);
}

/* Helper: iterate valid windows and call a callback for each.
 * The callback receives:
 *   w_idx     — 0-based index of the valid window
 *   start     — raw window start (may be negative if padding > 0)
 *   eff_start — effective start (clipped to [0, seqlen])
 *   eff_len   — effective length (may be < window for edge windows)
 * The raw `start` is passed so callbacks can compute the original-window
 * midpoint (for approach-B centering in the weighted kernel). */
typedef void (*window_callback_t)(int w_idx, int start, int eff_start,
                                  int eff_len, void *user_data);

static int iter_windows(int seqlen, int window, int shift,
                        int pad_left, int pad_right, int L,
                        window_callback_t cb, void *user_data)
{
    if (pad_left > 0 && window - pad_left < L) return -1;
    if (pad_right > 0 && window - pad_right < L) return -1;
    /* combined check above */
    int w = 0;
    int n_valid = 0;
    while (1) {
        int start = w * shift - pad_left;
        int end = start + window;
        int eff_start = start < 0 ? 0 : start;
        int eff_end = end > seqlen ? seqlen : end;
        int eff_len = eff_end - eff_start;
        if (eff_len >= L) {
            cb(n_valid, start, eff_start, eff_len, user_data);
            n_valid++;
            if (start >= seqlen - L - pad_right) break;
        } else {
            if (start >= seqlen - L + 1) break;
        }
        w++;
        if (w > 1000000) break;
    }
    return n_valid;
}

typedef struct {
    const gkm_kernel_base_t *base;
    const gkm_sequence_t **seqs;
    int n;
    int window;
    int shift;
    int pad_left;
    int pad_right;
    double *out;
    double *sqnorm_cache;  /* [n_valid * n] */
    int n_valid;
} window_kernel_state_t;

/* First pass: compute sqnorms for all (window, sequence) pairs. */
static void cb_sqnorms(int w_idx, int start, int eff_start, int eff_len, void *user_data)
{
    window_kernel_state_t *st = (window_kernel_state_t *)user_data;
    for (int i = 0; i < st->n; i++) {
        gkm_sequence_t view;
        make_window_view(st->base, st->seqs[i], eff_start, eff_len, &view);
        st->sqnorm_cache[w_idx * st->n + i] = view.sqnorm;
    }
}

/* Second pass: build tree per window, query n times. */
static void cb_eval(int w_idx, int start, int eff_start, int eff_len, void *user_data)
{
    window_kernel_state_t *st = (window_kernel_state_t *)user_data;
    const gkm_kernel_base_t *base = st->base;
    int L = base->param.L;
    int d = base->param.d;
    int n = st->n;

    kmer_tree_t *tree = (kmer_tree_t *)malloc(sizeof(kmer_tree_t));
    kmer_tree_init(tree, L, base->param.k, d);
    for (int i = 0; i < n; i++) {
        gkm_sequence_t view;
        make_window_view(base, st->seqs[i], eff_start, eff_len, &view);
        kmer_tree_add_sequence(tree, i, &view, base->param.use_rc);
    }

    double **mmprofile = (double **)malloc(sizeof(double *) * (size_t)(d + 1));
    for (int k = 0; k <= d; k++) {
        mmprofile[k] = (double *)malloc(sizeof(double) * (size_t)n);
    }
    base_mismatch_count_t *matching_bases =
        (base_mismatch_count_t *)malloc(
            sizeof(base_mismatch_count_t) * (size_t)(eff_len - L + 1));

    for (int i = 0; i < n; i++) {
        gkm_sequence_t view;
        make_window_view(base, st->seqs[i], eff_start, eff_len, &view);
        int num_matching = view.nkmers;
        for (int j = 0; j < num_matching; j++) {
            matching_bases[j].bid   = view.seq + j;
            matching_bases[j].wt = 1.0;

            matching_bases[j].mmcnt = 0;
        }
        for (int k = 0; k <= d; k++) {
            for (int j = 0; j < n; j++) mmprofile[k][j] = 0;
        }
        kmer_tree_dfs(tree, n, 0, 0,
                      matching_bases, num_matching, mmprofile);

        double i_sqnorm = st->sqnorm_cache[w_idx * n + i];
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k <= d; k++) {
                sum += base->weights[k] * mmprofile[k][j];
            }
            double j_sqnorm = st->sqnorm_cache[w_idx * n + j];
            if (i_sqnorm > 0 && j_sqnorm > 0) {
                st->out[(size_t)w_idx * (size_t)n * (size_t)n
                        + (size_t)i * (size_t)n
                        + (size_t)j] = sum / (i_sqnorm * j_sqnorm);
            } else {
                st->out[(size_t)w_idx * (size_t)n * (size_t)n
                        + (size_t)i * (size_t)n
                        + (size_t)j] = 0.0;
            }
        }
    }

    for (int k = 0; k <= d; k++) free(mmprofile[k]);
    free(mmprofile);
    free(matching_bases);
    kmer_tree_destroy(tree);
}

int gkm_kernel_window_kernel(const gkm_kernel_t *kernel,
                             const gkm_sequence_t **seqs, int n,
                             int window, int shift,
                             int pad_left, int pad_right,
                             double *out)
{
    if (!kernel || !seqs || !out) return -1;
    if (n <= 0) return -1;
    if (window < kernel->base.param.L) return -1;
    if (shift < 1) return -1;
    if (pad_left > 0 && window - pad_left < kernel->base.param.L) return -1;
    if (pad_right > 0 && window - pad_right < kernel->base.param.L) return -1;
    /* combined check above */

    int seqlen = seqs[0]->seqlen;
    for (int i = 1; i < n; i++) {
        if (seqs[i]->seqlen != seqlen) return -1;
    }

    /* Count valid windows. */
    int n_valid = gkm_compute_n_windows(seqlen, window, shift,
                                    pad_left, pad_right, kernel->base.param.L);
    if (n_valid <= 0) return -1;

    window_kernel_state_t st;
    st.base = &kernel->base;
    st.seqs = seqs;
    st.n = n;
    st.window = window;
    st.shift = shift;
    st.pad_left = pad_left;
    st.pad_right = pad_right;
    st.out = out;
    st.sqnorm_cache = (double *)malloc(
        sizeof(double) * (size_t)n_valid * (size_t)n);
    st.n_valid = n_valid;

    if (!st.sqnorm_cache) return -1;

    /* First pass: sqnorms. */
    iter_windows(seqlen, window, shift, pad_left, pad_right,
                 kernel->base.param.L, cb_sqnorms, &st);
    /* Second pass: evaluate. */
    iter_windows(seqlen, window, shift, pad_left, pad_right,
                 kernel->base.param.L, cb_eval, &st);

    free(st.sqnorm_cache);
    return 0;
}

/* ================================================================== */
/* Regular kernel: cross within-window                                */
/* ================================================================== */

typedef struct {
    const gkm_kernel_base_t *base;
    const gkm_sequence_t **queries;
    int n_q;
    const gkm_sequence_t **refs;
    int n_ref;
    int window;
    int shift;
    int pad_left;
    int pad_right;
    double *out;
    double *ref_sqnorms;
    double *qry_sqnorms;
} cross_window_state_t;

static void cb_cross_sqnorms(int w_idx, int start, int eff_start, int eff_len, void *user_data)
{
    cross_window_state_t *st = (cross_window_state_t *)user_data;
    for (int i = 0; i < st->n_ref; i++) {
        gkm_sequence_t view;
        make_window_view(st->base, st->refs[i], eff_start, eff_len, &view);
        st->ref_sqnorms[w_idx * st->n_ref + i] = view.sqnorm;
    }
    for (int i = 0; i < st->n_q; i++) {
        gkm_sequence_t view;
        make_window_view(st->base, st->queries[i], eff_start, eff_len, &view);
        st->qry_sqnorms[w_idx * st->n_q + i] = view.sqnorm;
    }
}

static void cb_cross_eval(int w_idx, int start, int eff_start, int eff_len, void *user_data)
{
    cross_window_state_t *st = (cross_window_state_t *)user_data;
    const gkm_kernel_base_t *base = st->base;
    int L = base->param.L;
    int d = base->param.d;
    int n_ref = st->n_ref;
    int n_q = st->n_q;

    kmer_tree_t *tree = (kmer_tree_t *)malloc(sizeof(kmer_tree_t));
    kmer_tree_init(tree, L, base->param.k, d);
    for (int j = 0; j < n_ref; j++) {
        gkm_sequence_t view;
        make_window_view(base, st->refs[j], eff_start, eff_len, &view);
        kmer_tree_add_sequence(tree, j, &view, base->param.use_rc);
    }

    double **mmprofile = (double **)malloc(sizeof(double *) * (size_t)(d + 1));
    for (int k = 0; k <= d; k++) {
        mmprofile[k] = (double *)malloc(sizeof(double) * (size_t)n_ref);
    }
    base_mismatch_count_t *matching_bases =
        (base_mismatch_count_t *)malloc(
            sizeof(base_mismatch_count_t) * (size_t)(eff_len - L + 1));

    for (int i = 0; i < n_q; i++) {
        gkm_sequence_t view;
        make_window_view(base, st->queries[i], eff_start, eff_len, &view);
        int num_matching = view.nkmers;
        for (int j = 0; j < num_matching; j++) {
            matching_bases[j].bid   = view.seq + j;
            matching_bases[j].wt = 1.0;

            matching_bases[j].mmcnt = 0;
        }
        for (int k = 0; k <= d; k++)
            for (int j = 0; j < n_ref; j++) mmprofile[k][j] = 0;

        kmer_tree_dfs(tree, n_ref, 0, 0,
                      matching_bases, num_matching, mmprofile);

        double i_sqnorm = st->qry_sqnorms[w_idx * n_q + i];
        for (int j = 0; j < n_ref; j++) {
            double sum = 0;
            for (int k = 0; k <= d; k++)
                sum += base->weights[k] * mmprofile[k][j];
            double j_sqnorm = st->ref_sqnorms[w_idx * n_ref + j];
            if (i_sqnorm > 0 && j_sqnorm > 0) {
                st->out[(size_t)w_idx * (size_t)n_q * (size_t)n_ref
                        + (size_t)i * (size_t)n_ref + (size_t)j] =
                    sum / (i_sqnorm * j_sqnorm);
            } else {
                st->out[(size_t)w_idx * (size_t)n_q * (size_t)n_ref
                        + (size_t)i * (size_t)n_ref + (size_t)j] = 0.0;
            }
        }
    }

    for (int k = 0; k <= d; k++) free(mmprofile[k]);
    free(mmprofile);
    free(matching_bases);
    kmer_tree_destroy(tree);
}

int gkm_kernel_cross_window_kernel(const gkm_kernel_t *kernel,
                                   const gkm_sequence_t **queries, int n_q,
                                   const gkm_sequence_t **refs, int n_ref,
                                   int window, int shift,
                                   int pad_left, int pad_right,
                                   double *out)
{
    if (!kernel || !queries || !refs || !out) return -1;
    if (n_q <= 0 || n_ref <= 0) return -1;
    if (window < kernel->base.param.L) return -1;
    if (shift < 1) return -1;
    if (pad_left > 0 && window - pad_left < kernel->base.param.L) return -1;
    if (pad_right > 0 && window - pad_right < kernel->base.param.L) return -1;
    /* combined check above */

    int seqlen = queries[0]->seqlen;
    for (int i = 1; i < n_q; i++)
        if (queries[i]->seqlen != seqlen) return -1;
    for (int i = 0; i < n_ref; i++)
        if (refs[i]->seqlen != seqlen) return -1;

    int n_valid = gkm_compute_n_windows(seqlen, window, shift,
                                    pad_left, pad_right, kernel->base.param.L);
    if (n_valid <= 0) return -1;

    cross_window_state_t st;
    st.base = &kernel->base;
    st.queries = queries; st.n_q = n_q;
    st.refs = refs; st.n_ref = n_ref;
    st.window = window; st.shift = shift;
    st.pad_left = pad_left; st.pad_right = pad_right;
    st.out = out;
    st.ref_sqnorms = (double *)malloc(sizeof(double) * (size_t)n_valid * (size_t)n_ref);
    st.qry_sqnorms = (double *)malloc(sizeof(double) * (size_t)n_valid * (size_t)n_q);
    if (!st.ref_sqnorms || !st.qry_sqnorms) {
        free(st.ref_sqnorms); free(st.qry_sqnorms);
        return -1;
    }

    iter_windows(seqlen, window, shift, pad_left, pad_right,
                 kernel->base.param.L, cb_cross_sqnorms, &st);
    iter_windows(seqlen, window, shift, pad_left, pad_right,
                 kernel->base.param.L, cb_cross_eval, &st);

    free(st.ref_sqnorms); free(st.qry_sqnorms);
    return 0;
}

/* ================================================================== */
/* Regular kernel: window tree cache                                  */
/* ================================================================== */

struct gkm_window_tree_cache {
    int n_valid;
    int n_ref;
    int window;
    int shift;
    int pad_left;
    int pad_right;
    kmer_tree_t **trees;
    double *ref_sqnorms;
    int L, d;
};

typedef struct {
    const gkm_kernel_base_t *base;
    const gkm_sequence_t **refs;
    int n_ref;
    int window;
    int shift;
    int pad_left;
    int pad_right;
    gkm_window_tree_cache_t *cache;
} cache_build_state_t;

static void cb_cache_build(int w_idx, int start, int eff_start, int eff_len, void *user_data)
{
    cache_build_state_t *st = (cache_build_state_t *)user_data;
    const gkm_kernel_base_t *base = st->base;
    int L = base->param.L;
    int d = base->param.d;

    st->cache->trees[w_idx] = (kmer_tree_t *)malloc(sizeof(kmer_tree_t));
    kmer_tree_init(st->cache->trees[w_idx], L, base->param.k, d);
    for (int j = 0; j < st->n_ref; j++) {
        gkm_sequence_t view;
        make_window_view(base, st->refs[j], eff_start, eff_len, &view);
        st->cache->ref_sqnorms[w_idx * st->n_ref + j] = view.sqnorm;
        kmer_tree_add_sequence(st->cache->trees[w_idx], j, &view, base->param.use_rc);
    }
}

gkm_window_tree_cache_t *gkm_window_tree_cache_build(
    const gkm_kernel_t *kernel,
    const gkm_sequence_t **refs, int n_ref,
    int window, int shift,
    int pad_left, int pad_right)
{
    if (!kernel || !refs || n_ref <= 0) return NULL;
    if (window < kernel->base.param.L || shift < 1) return NULL;
    if (pad_left > 0 && window - pad_left < kernel->base.param.L) return NULL;
    if (pad_right > 0 && window - pad_right < kernel->base.param.L) return NULL;
    /* combined check above */

    int seqlen = refs[0]->seqlen;
    for (int i = 1; i < n_ref; i++)
        if (refs[i]->seqlen != seqlen) return NULL;

    int n_valid = gkm_compute_n_windows(seqlen, window, shift,
                                    pad_left, pad_right, kernel->base.param.L);
    if (n_valid <= 0) return NULL;

    gkm_window_tree_cache_t *cache = (gkm_window_tree_cache_t *)calloc(1, sizeof(*cache));
    cache->n_valid = n_valid;
    cache->n_ref = n_ref;
    cache->window = window;
    cache->shift = shift;
    cache->pad_left = pad_left;
    cache->pad_right = pad_right;
    cache->L = kernel->base.param.L;
    cache->d = kernel->base.param.d;
    cache->trees = (kmer_tree_t **)calloc(n_valid, sizeof(kmer_tree_t *));
    cache->ref_sqnorms = (double *)malloc(sizeof(double) * (size_t)n_valid * (size_t)n_ref);
    if (!cache->trees || !cache->ref_sqnorms) {
        free(cache->trees); free(cache->ref_sqnorms); free(cache);
        return NULL;
    }

    cache_build_state_t st;
    st.base = &kernel->base;
    st.refs = refs; st.n_ref = n_ref;
    st.window = window; st.shift = shift;
    st.pad_left = pad_left; st.pad_right = pad_right;
    st.cache = cache;

    iter_windows(seqlen, window, shift, pad_left, pad_right,
                 kernel->base.param.L, cb_cache_build, &st);
    return cache;
}

typedef struct {
    const gkm_kernel_base_t *base;
    const gkm_sequence_t **queries;
    int n_q;
    int n_ref;
    int L, d;
    double *out;
    const gkm_window_tree_cache_t *cache;
} cache_query_state_t;

static void cb_cache_query(int w_idx, int start, int eff_start, int eff_len, void *user_data)
{
    cache_query_state_t *st = (cache_query_state_t *)user_data;
    const gkm_kernel_base_t *base = st->base;
    int L = st->L;
    int d = st->d;
    int n_ref = st->n_ref;
    int n_q = st->n_q;

    double **mmprofile = (double **)malloc(sizeof(double *) * (size_t)(d + 1));
    for (int k = 0; k <= d; k++)
        mmprofile[k] = (double *)malloc(sizeof(double) * (size_t)n_ref);
    base_mismatch_count_t *matching_bases =
        (base_mismatch_count_t *)malloc(
            sizeof(base_mismatch_count_t) * (size_t)(eff_len - L + 1));

    for (int i = 0; i < n_q; i++) {
        gkm_sequence_t view;
        make_window_view(base, st->queries[i], eff_start, eff_len, &view);
        int num_matching = view.nkmers;
        for (int j = 0; j < num_matching; j++) {
            matching_bases[j].bid   = view.seq + j;
            matching_bases[j].wt = 1.0;

            matching_bases[j].mmcnt = 0;
        }
        for (int k = 0; k <= d; k++)
            for (int j = 0; j < n_ref; j++) mmprofile[k][j] = 0;

        kmer_tree_dfs(st->cache->trees[w_idx], n_ref, 0, 0,
                      matching_bases, num_matching, mmprofile);

        double i_sqnorm = view.sqnorm;
        for (int j = 0; j < n_ref; j++) {
            double sum = 0;
            for (int k = 0; k <= d; k++)
                sum += base->weights[k] * mmprofile[k][j];
            double j_sqnorm = st->cache->ref_sqnorms[w_idx * n_ref + j];
            if (i_sqnorm > 0 && j_sqnorm > 0) {
                st->out[(size_t)w_idx * (size_t)n_q * (size_t)n_ref
                        + (size_t)i * (size_t)n_ref + (size_t)j] =
                    sum / (i_sqnorm * j_sqnorm);
            } else {
                st->out[(size_t)w_idx * (size_t)n_q * (size_t)n_ref
                        + (size_t)i * (size_t)n_ref + (size_t)j] = 0.0;
            }
        }
    }

    for (int k = 0; k <= d; k++) free(mmprofile[k]);
    free(mmprofile); free(matching_bases);
}

int gkm_window_tree_cache_query(
    const gkm_kernel_t *kernel,
    const gkm_window_tree_cache_t *cache,
    const gkm_sequence_t **queries, int n_q,
    double *out)
{
    if (!kernel || !cache || !queries || !out) return -1;
    if (n_q <= 0) return -1;

    /* Determine seqlen from the first query. */
    int seqlen = queries[0]->seqlen;

    cache_query_state_t st;
    st.base = &kernel->base;
    st.queries = queries; st.n_q = n_q;
    st.n_ref = cache->n_ref;
    st.L = cache->L; st.d = cache->d;
    st.out = out;
    st.cache = cache;

    iter_windows(seqlen, cache->window, cache->shift,
                 cache->pad_left, cache->pad_right,
                 cache->L, cb_cache_query, &st);
    return 0;
}

void gkm_window_tree_cache_destroy(gkm_window_tree_cache_t *cache)
{
    if (!cache) return;
    if (cache->trees) {
        for (int w = 0; w < cache->n_valid; w++) {
            if (cache->trees[w]) kmer_tree_destroy(cache->trees[w]);
        }
        free(cache->trees);
    }
    free(cache->ref_sqnorms);
    free(cache);
}

/* ================================================================== */
/* Weighted kernel: create / destroy / accessors                      */
/* ================================================================== */

gkm_weighted_kernel_t *gkm_weighted_kernel_create(const gkm_weighted_params_t *params)
{
    const char *err = gkm_parameter_check(&params->base);
    if (err) return NULL;
    if (params->weight_gamma < 0) return NULL;
    if (params->weight_sigma <= 0 &&
        params->pos_kernel != GKM_POS_LAPLACIAN) return NULL;
    if (params->weight_peak <= 0) return NULL;

    gkm_weighted_kernel_t *k = (gkm_weighted_kernel_t *)calloc(1, sizeof(*k));
    gkm_kernel_base_init(&k->base, &params->base);
    k->pos_kernel = params->pos_kernel;
    k->weight_gamma = params->weight_gamma;
    k->weight_sigma = params->weight_sigma;
    k->weight_peak = params->weight_peak;
    return k;
}

void gkm_weighted_kernel_destroy(gkm_weighted_kernel_t *kernel)
{
    if (!kernel) return;
    gkm_kernel_base_destroy(&kernel->base);
    free(kernel);
}

void *gkm_weighted_kernel_base(gkm_weighted_kernel_t *kernel)
{
    return (void *)&kernel->base;
}

void *gkm_weighted_kernel_base_const(const gkm_weighted_kernel_t *kernel)
{
    return (void *)&kernel->base;
}

gkm_pos_kernel_t gkm_weighted_kernel_pos_kernel(const gkm_weighted_kernel_t *kernel)
{
    return kernel->pos_kernel;
}

double gkm_weighted_kernel_gamma(const gkm_weighted_kernel_t *kernel)
{
    return kernel->weight_gamma;
}

double gkm_weighted_kernel_sigma(const gkm_weighted_kernel_t *kernel)
{
    return kernel->weight_sigma;
}

double gkm_weighted_kernel_peak(const gkm_weighted_kernel_t *kernel)
{
    return kernel->weight_peak;
}

/* ================================================================== */
/* Weighted kernel: positional weight buffer computation              */
/* ================================================================== */

/* Compute positional weights for a sequence at a given center.
 * Writes wt[j] and wt_rc[j] for j in [0, nkmers-1].
 * wt_rc is mirrored: wt_rc[j] = wt[nkmers-1-j]. */
static void compute_pos_weights_for_seq(int nkmers, int L, double center,
                                        gkm_pos_kernel_t pos_kernel,
                                        double gamma, double sigma, double peak,
                                        double *wt, double *wt_rc)
{
    double lmer_midpoint_offset = (L - 1) / 2.0;
    for (int j = 0; j < nkmers; j++) {
        double midpoint = j + lmer_midpoint_offset;
        double dist = midpoint - center;
        wt[j] = peak * gkm_pos_weight(dist, pos_kernel, gamma, sigma);
    }
    for (int j = 0; j < nkmers; j++) {
        wt_rc[j] = wt[nkmers - 1 - j];
    }
}

/* ================================================================== */
/* Weighted kernel: self evaluation (multi-center)                    */
/* ================================================================== */

int gkm_weighted_kernel_eval_self(const gkm_weighted_kernel_t *kernel,
                                  const gkm_sequence_t **seqs, int n,
                                  const double *centers, int n_centers,
                                  double *out)
{
    if (!kernel || !seqs || !out || !centers) return -1;
    if (n <= 0) return -1;
    if (n_centers <= 0) return -1;

    int L = kernel->base.param.L;
    int d = kernel->base.param.d;
    int use_rc = kernel->base.param.use_rc;
    int seqlen = seqs[0]->seqlen;
    int nkmers = seqlen - L + 1;
    if (nkmers < 1) return -1;

    /* Build the offset-based tree ONCE. */
    kmer_tree_t *tree = (kmer_tree_t *)malloc(sizeof(kmer_tree_t));
    if (!tree) return -1;
    kmer_tree_init(tree, L, kernel->base.param.k, d);
    for (int i = 0; i < n; i++) {
        kmer_tree_add_sequence_offsets(tree, i, seqs[i], use_rc);
    }

    /* Per-sequence weight buffers for sqnorm computation. */
    double **wt_f = (double **)malloc(sizeof(double *) * (size_t)n);
    double **wt_r = (double **)malloc(sizeof(double *) * (size_t)n);
    gkm_sequence_t *views = (gkm_sequence_t *)malloc(
        sizeof(gkm_sequence_t) * (size_t)n);
    if (!wt_f || !wt_r || !views) {
        free(wt_f); free(wt_r); free(views);
        kmer_tree_destroy(tree);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        wt_f[i] = (double *)malloc(sizeof(double) * (size_t)nkmers);
        wt_r[i] = (double *)malloc(sizeof(double) * (size_t)nkmers);
        if (!wt_f[i] || !wt_r[i]) {
            for (int j = 0; j <= i; j++) { free(wt_f[j]); free(wt_r[j]); }
            free(wt_f); free(wt_r); free(views);
            kmer_tree_destroy(tree);
            return -1;
        }
    }

    /* sqnorm cache: sqnorm_cache[c * n + i]. */
    double *sqnorm_cache = (double *)malloc(
        sizeof(double) * (size_t)n_centers * (size_t)n);
    if (!sqnorm_cache) {
        for (int i = 0; i < n; i++) { free(wt_f[i]); free(wt_r[i]); }
        free(wt_f); free(wt_r); free(views);
        kmer_tree_destroy(tree);
        return -1;
    }

    /* Compute all sqnorms (per centre, per sequence). */
    for (int c = 0; c < n_centers; c++) {
        double center = centers[c];
        for (int i = 0; i < n; i++) {
            compute_pos_weights_for_seq(nkmers, L, center,
                                        kernel->pos_kernel,
                                        kernel->weight_gamma,
                                        kernel->weight_sigma,
                                        kernel->weight_peak,
                                        wt_f[i], wt_r[i]);
            memset(&views[i], 0, sizeof(views[i]));
            views[i].seqlen = seqlen;
            views[i].seq    = seqs[i]->seq;
            views[i].seq_rc = seqs[i]->seq_rc;
            views[i].nkmers = nkmers;
            views[i].wt     = wt_f[i];
            views[i].wt_rc  = wt_r[i];
            views[i].sqnorm = compute_sqnorm(&kernel->base, &views[i]);
            sqnorm_cache[c * n + i] = views[i].sqnorm;
        }
    }

    /* Multi-centre mmprofile: [n_centers][d+1][n], flat. */
    size_t mmprof_size = (size_t)n_centers * (size_t)(d + 1) * (size_t)n;
    double *mmprof_multi = (double *)calloc(mmprof_size, sizeof(double));
    if (!mmprof_multi) {
        free(sqnorm_cache);
        for (int i = 0; i < n; i++) { free(wt_f[i]); free(wt_r[i]); }
        free(wt_f); free(wt_r); free(views);
        kmer_tree_destroy(tree);
        return -1;
    }

    base_mismatch_count_t *matching_bases =
        (base_mismatch_count_t *)malloc(
            sizeof(base_mismatch_count_t) * (size_t)nkmers);
    if (!matching_bases) {
        free(mmprof_multi); free(sqnorm_cache);
        for (int i = 0; i < n; i++) { free(wt_f[i]); free(wt_r[i]); }
        free(wt_f); free(wt_r); free(views);
        kmer_tree_destroy(tree);
        return -1;
    }

    /* Main loop: for each query sequence i, run ONE multi-centre DFS. */
    for (int i = 0; i < n; i++) {
        int num_matching = seqs[i]->nkmers;
        for (int j = 0; j < num_matching; j++) {
            matching_bases[j].bid   = seqs[i]->seq + j;
            matching_bases[j].wt = 1.0;

            matching_bases[j].mmcnt = 0;
        }
        size_t row_size = (size_t)(d + 1) * (size_t)n;
        for (int c = 0; c < n_centers; c++) {
            memset(mmprof_multi + (size_t)c * row_size, 0,
                   row_size * sizeof(double));
        }

        kmer_tree_dfs_weighted_multi(
            tree, n, 0, 0,
            matching_bases, num_matching,
            mmprof_multi, n_centers, centers,
            seqs[i]->seq, kernel->pos_kernel,
            kernel->weight_gamma, kernel->weight_sigma, kernel->weight_peak, n);

        for (int c = 0; c < n_centers; c++) {
            double i_sqnorm = sqnorm_cache[c * n + i];
            double *mmprof_c = mmprof_multi + (size_t)c * row_size;
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k <= d; k++) {
                    sum += kernel->base.weights[k] * mmprof_c[k * n + j];
                }
                double j_sqnorm = sqnorm_cache[c * n + j];
                if (i_sqnorm > 0 && j_sqnorm > 0) {
                    out[(size_t)c * (size_t)n * (size_t)n
                        + (size_t)i * (size_t)n
                        + (size_t)j] = sum / (i_sqnorm * j_sqnorm);
                } else {
                    out[(size_t)c * (size_t)n * (size_t)n
                        + (size_t)i * (size_t)n
                        + (size_t)j] = 0.0;
                }
            }
        }
    }

    free(mmprof_multi);
    free(sqnorm_cache);
    for (int i = 0; i < n; i++) { free(wt_f[i]); free(wt_r[i]); }
    free(wt_f); free(wt_r); free(views);
    free(matching_bases);
    kmer_tree_destroy(tree);
    return 0;
}

/* ================================================================== */
/* Weighted kernel: cross evaluation                                  */
/* ================================================================== */

int gkm_weighted_kernel_eval_cross(const gkm_weighted_kernel_t *kernel,
                                   const gkm_sequence_t **queries, int n_q,
                                   const gkm_sequence_t **refs, int n_ref,
                                   const double *centers, int n_centers,
                                   double *out)
{
    if (!kernel || !queries || !refs || !out || !centers) return -1;
    if (n_q <= 0 || n_ref <= 0 || n_centers <= 0) return -1;

    int L = kernel->base.param.L;
    int d = kernel->base.param.d;
    int use_rc = kernel->base.param.use_rc;
    int seqlen = queries[0]->seqlen;
    if (seqlen < L) return -1;
    int nkmers = seqlen - L + 1;
    if (nkmers < 1) return -1;

    for (int i = 1; i < n_q; i++)
        if (queries[i]->seqlen != seqlen) return -1;
    for (int j = 0; j < n_ref; j++)
        if (refs[j]->seqlen != seqlen) return -1;

    /* Build the offset-based tree ONCE with refs only. */
    kmer_tree_t *tree = (kmer_tree_t *)malloc(sizeof(kmer_tree_t));
    if (!tree) return -1;
    kmer_tree_init(tree, L, kernel->base.param.k, d);
    for (int j = 0; j < n_ref; j++) {
        kmer_tree_add_sequence_offsets(tree, j, refs[j], use_rc);
    }

    double **wt_qf = (double **)malloc(sizeof(double *) * (size_t)n_q);
    double **wt_qr = (double **)malloc(sizeof(double *) * (size_t)n_q);
    double **wt_rf = (double **)malloc(sizeof(double *) * (size_t)n_ref);
    double **wt_rr = (double **)malloc(sizeof(double *) * (size_t)n_ref);
    gkm_sequence_t *views_q = (gkm_sequence_t *)malloc(sizeof(gkm_sequence_t) * (size_t)n_q);
    gkm_sequence_t *views_r = (gkm_sequence_t *)malloc(sizeof(gkm_sequence_t) * (size_t)n_ref);
    if (!wt_qf || !wt_qr || !wt_rf || !wt_rr || !views_q || !views_r) {
        free(wt_qf); free(wt_qr); free(wt_rf); free(wt_rr);
        free(views_q); free(views_r);
        kmer_tree_destroy(tree);
        return -1;
    }
    for (int i = 0; i < n_q; i++) {
        wt_qf[i] = (double *)malloc(sizeof(double) * (size_t)nkmers);
        wt_qr[i] = (double *)malloc(sizeof(double) * (size_t)nkmers);
        if (!wt_qf[i] || !wt_qr[i]) {
            for (int j = 0; j <= i; j++) { free(wt_qf[j]); free(wt_qr[j]); }
            free(wt_qf); free(wt_qr); free(wt_rf); free(wt_rr);
            free(views_q); free(views_r);
            kmer_tree_destroy(tree);
            return -1;
        }
    }
    for (int j = 0; j < n_ref; j++) {
        wt_rf[j] = (double *)malloc(sizeof(double) * (size_t)nkmers);
        wt_rr[j] = (double *)malloc(sizeof(double) * (size_t)nkmers);
        if (!wt_rf[j] || !wt_rr[j]) {
            for (int k = 0; k <= j; k++) { free(wt_rf[k]); free(wt_rr[k]); }
            for (int i = 0; i < n_q; i++) { free(wt_qf[i]); free(wt_qr[i]); }
            free(wt_qf); free(wt_qr); free(wt_rf); free(wt_rr);
            free(views_q); free(views_r);
            kmer_tree_destroy(tree);
            return -1;
        }
    }

    double *query_sqnorms = (double *)malloc(sizeof(double) * (size_t)n_centers * (size_t)n_q);
    double *ref_sqnorms = (double *)malloc(sizeof(double) * (size_t)n_centers * (size_t)n_ref);
    if (!query_sqnorms || !ref_sqnorms) {
        free(query_sqnorms); free(ref_sqnorms);
        for (int i = 0; i < n_q; i++) { free(wt_qf[i]); free(wt_qr[i]); }
        for (int j = 0; j < n_ref; j++) { free(wt_rf[j]); free(wt_rr[j]); }
        free(wt_qf); free(wt_qr); free(wt_rf); free(wt_rr);
        free(views_q); free(views_r);
        kmer_tree_destroy(tree);
        return -1;
    }

    for (int c = 0; c < n_centers; c++) {
        double center = centers[c];
        for (int i = 0; i < n_q; i++) {
            compute_pos_weights_for_seq(nkmers, L, center,
                                        kernel->pos_kernel,
                                        kernel->weight_gamma,
                                        kernel->weight_sigma,
                                        kernel->weight_peak,
                                        wt_qf[i], wt_qr[i]);
            memset(&views_q[i], 0, sizeof(views_q[i]));
            views_q[i].seqlen = seqlen;
            views_q[i].seq    = queries[i]->seq;
            views_q[i].seq_rc = queries[i]->seq_rc;
            views_q[i].nkmers = nkmers;
            views_q[i].wt     = wt_qf[i];
            views_q[i].wt_rc  = wt_qr[i];
            views_q[i].sqnorm = compute_sqnorm(&kernel->base, &views_q[i]);
            query_sqnorms[c * n_q + i] = views_q[i].sqnorm;
        }
        for (int j = 0; j < n_ref; j++) {
            compute_pos_weights_for_seq(nkmers, L, center,
                                        kernel->pos_kernel,
                                        kernel->weight_gamma,
                                        kernel->weight_sigma,
                                        kernel->weight_peak,
                                        wt_rf[j], wt_rr[j]);
            memset(&views_r[j], 0, sizeof(views_r[j]));
            views_r[j].seqlen = seqlen;
            views_r[j].seq    = refs[j]->seq;
            views_r[j].seq_rc = refs[j]->seq_rc;
            views_r[j].nkmers = nkmers;
            views_r[j].wt     = wt_rf[j];
            views_r[j].wt_rc  = wt_rr[j];
            views_r[j].sqnorm = compute_sqnorm(&kernel->base, &views_r[j]);
            ref_sqnorms[c * n_ref + j] = views_r[j].sqnorm;
        }
    }

    size_t mmprof_size = (size_t)n_centers * (size_t)(d + 1) * (size_t)n_ref;
    double *mmprof_multi = (double *)calloc(mmprof_size, sizeof(double));
    if (!mmprof_multi) {
        free(query_sqnorms); free(ref_sqnorms);
        for (int i = 0; i < n_q; i++) { free(wt_qf[i]); free(wt_qr[i]); }
        for (int j = 0; j < n_ref; j++) { free(wt_rf[j]); free(wt_rr[j]); }
        free(wt_qf); free(wt_qr); free(wt_rf); free(wt_rr);
        free(views_q); free(views_r);
        kmer_tree_destroy(tree);
        return -1;
    }

    base_mismatch_count_t *matching_bases =
        (base_mismatch_count_t *)malloc(
            sizeof(base_mismatch_count_t) * (size_t)nkmers);
    if (!matching_bases) {
        free(mmprof_multi);
        free(query_sqnorms); free(ref_sqnorms);
        for (int i = 0; i < n_q; i++) { free(wt_qf[i]); free(wt_qr[i]); }
        for (int j = 0; j < n_ref; j++) { free(wt_rf[j]); free(wt_rr[j]); }
        free(wt_qf); free(wt_qr); free(wt_rf); free(wt_rr);
        free(views_q); free(views_r);
        kmer_tree_destroy(tree);
        return -1;
    }

    size_t row_size = (size_t)(d + 1) * (size_t)n_ref;
    for (int i = 0; i < n_q; i++) {
        int num_matching = queries[i]->nkmers;
        for (int j = 0; j < num_matching; j++) {
            matching_bases[j].bid   = queries[i]->seq + j;
            matching_bases[j].wt = 1.0;

            matching_bases[j].mmcnt = 0;
        }
        for (int c = 0; c < n_centers; c++) {
            memset(mmprof_multi + (size_t)c * row_size, 0,
                   row_size * sizeof(double));
        }

        kmer_tree_dfs_weighted_multi(
            tree, n_ref, 0, 0,
            matching_bases, num_matching,
            mmprof_multi, n_centers, centers,
            queries[i]->seq, kernel->pos_kernel,
            kernel->weight_gamma, kernel->weight_sigma, kernel->weight_peak, n_ref);

        for (int c = 0; c < n_centers; c++) {
            double i_sqnorm = query_sqnorms[c * n_q + i];
            double *mmprof_c = mmprof_multi + (size_t)c * row_size;
            for (int j = 0; j < n_ref; j++) {
                double sum = 0;
                for (int k = 0; k <= d; k++) {
                    sum += kernel->base.weights[k] * mmprof_c[k * n_ref + j];
                }
                double j_sqnorm = ref_sqnorms[c * n_ref + j];
                if (i_sqnorm > 0 && j_sqnorm > 0) {
                    out[(size_t)c * (size_t)n_q * (size_t)n_ref
                        + (size_t)i * (size_t)n_ref
                        + (size_t)j] = sum / (i_sqnorm * j_sqnorm);
                } else {
                    out[(size_t)c * (size_t)n_q * (size_t)n_ref
                        + (size_t)i * (size_t)n_ref
                        + (size_t)j] = 0.0;
                }
            }
        }
    }

    free(mmprof_multi);
    free(query_sqnorms); free(ref_sqnorms);
    for (int i = 0; i < n_q; i++) { free(wt_qf[i]); free(wt_qr[i]); }
    for (int j = 0; j < n_ref; j++) { free(wt_rf[j]); free(wt_rr[j]); }
    free(wt_qf); free(wt_qr); free(wt_rf); free(wt_rr);
    free(views_q); free(views_r);
    free(matching_bases);
    kmer_tree_destroy(tree);
    return 0;
}

/* ================================================================== */
/* Weighted kernel: windowed evaluation (window-midpoint centering)   */
/* ================================================================== */

typedef struct {
    const gkm_weighted_kernel_t *kernel;
    const gkm_sequence_t **seqs;
    int n;
    int window;
    int shift;
    int pad_left;
    int pad_right;
    double *out;
    double *sqnorm_cache;  /* [n_valid * n] */
    int n_valid;
} weighted_window_state_t;

static void cb_wsqnorms(int w_idx, int start, int eff_start, int eff_len, void *user_data)
{
    weighted_window_state_t *st = (weighted_window_state_t *)user_data;
    const gkm_weighted_kernel_t *kernel = st->kernel;
    int L = kernel->base.param.L;
    int nkmers_win = eff_len - L + 1;
    double center = start + (st->window - 1) / 2.0;  /* approach B: original window midpoint */

    double *wt_f = (double *)malloc(sizeof(double) * (size_t)nkmers_win);
    double *wt_r = (double *)malloc(sizeof(double) * (size_t)nkmers_win);
    if (!wt_f || !wt_r) { free(wt_f); free(wt_r); return; }

    for (int i = 0; i < st->n; i++) {
        compute_pos_weights_for_seq(nkmers_win, L, center,
                                    kernel->pos_kernel,
                                    kernel->weight_gamma,
                                    kernel->weight_sigma,
                                    kernel->weight_peak,
                                    wt_f, wt_r);
        gkm_sequence_t view;
        int seqlen = st->seqs[i]->seqlen;
        int rc_start = seqlen - eff_start - eff_len;
        memset(&view, 0, sizeof(view));
        view.seqlen = eff_len;
        view.seq    = st->seqs[i]->seq + eff_start;
        view.seq_rc = st->seqs[i]->seq_rc + rc_start;
        view.nkmers = nkmers_win;
        view.wt     = wt_f;
        view.wt_rc  = wt_r;
        view.sqnorm = compute_sqnorm(&kernel->base, &view);
        st->sqnorm_cache[w_idx * st->n + i] = view.sqnorm;
    }

    free(wt_f); free(wt_r);
}

static void cb_weval(int w_idx, int start, int eff_start, int eff_len, void *user_data)
{
    weighted_window_state_t *st = (weighted_window_state_t *)user_data;
    const gkm_weighted_kernel_t *kernel = st->kernel;
    int L = kernel->base.param.L;
    int d = kernel->base.param.d;
    int n = st->n;
    int nkmers_win = eff_len - L + 1;
    double center = start + (st->window - 1) / 2.0;  /* approach B */

    /* Build offset tree with all n windows. */
    kmer_tree_t *tree = (kmer_tree_t *)malloc(sizeof(kmer_tree_t));
    kmer_tree_init(tree, L, kernel->base.param.k, d);
    double *wt_f = (double *)malloc(sizeof(double) * (size_t)nkmers_win);
    double *wt_r = (double *)malloc(sizeof(double) * (size_t)nkmers_win);
    if (!wt_f || !wt_r) { free(wt_f); free(wt_r); kmer_tree_destroy(tree); return; }

    for (int i = 0; i < n; i++) {
        compute_pos_weights_for_seq(nkmers_win, L, center,
                                    kernel->pos_kernel,
                                    kernel->weight_gamma,
                                    kernel->weight_sigma,
                                    kernel->weight_peak,
                                    wt_f, wt_r);
        int seqlen = st->seqs[i]->seqlen;
        int rc_start = seqlen - eff_start - eff_len;
        gkm_sequence_t view;
        memset(&view, 0, sizeof(view));
        view.seqlen = eff_len;
        view.seq    = st->seqs[i]->seq + eff_start;
        view.seq_rc = st->seqs[i]->seq_rc + rc_start;
        view.nkmers = nkmers_win;
        view.wt     = wt_f;
        view.wt_rc  = wt_r;
        kmer_tree_add_sequence(tree, i, &view, kernel->base.param.use_rc);
    }

    /* Now query n times. The tree stores (seqid, weight) pairs, so the
     * regular DFS works (not the offset DFS). */
    double **mmprofile = (double **)malloc(sizeof(double *) * (size_t)(d + 1));
    for (int k = 0; k <= d; k++) {
        mmprofile[k] = (double *)malloc(sizeof(double) * (size_t)n);
    }
    base_mismatch_count_t *matching_bases =
        (base_mismatch_count_t *)malloc(
            sizeof(base_mismatch_count_t) * (size_t)nkmers_win);

    for (int i = 0; i < n; i++) {
        int num_matching = nkmers_win;
        for (int j = 0; j < num_matching; j++) {
            matching_bases[j].bid   = st->seqs[i]->seq + eff_start + j;
            matching_bases[j].wt = wt_f[j];

            matching_bases[j].mmcnt = 0;
        }
        for (int k = 0; k <= d; k++) {
            for (int j = 0; j < n; j++) mmprofile[k][j] = 0;
        }
        kmer_tree_dfs(tree, n, 0, 0,
                      matching_bases, num_matching, mmprofile);

        double i_sqnorm = st->sqnorm_cache[w_idx * n + i];
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k <= d; k++) {
                sum += kernel->base.weights[k] * (double)mmprofile[k][j];
            }
            double j_sqnorm = st->sqnorm_cache[w_idx * n + j];
            if (i_sqnorm > 0 && j_sqnorm > 0) {
                st->out[(size_t)w_idx * (size_t)n * (size_t)n
                        + (size_t)i * (size_t)n
                        + (size_t)j] = sum / (i_sqnorm * j_sqnorm);
            } else {
                st->out[(size_t)w_idx * (size_t)n * (size_t)n
                        + (size_t)i * (size_t)n
                        + (size_t)j] = 0.0;
            }
        }
    }

    for (int k = 0; k <= d; k++) free(mmprofile[k]);
    free(mmprofile);
    free(matching_bases);
    free(wt_f); free(wt_r);
    kmer_tree_destroy(tree);
}

int gkm_weighted_kernel_window_kernel(const gkm_weighted_kernel_t *kernel,
                                      const gkm_sequence_t **seqs, int n,
                                      int window, int shift,
                                      int pad_left, int pad_right,
                                      double *out)
{
    if (!kernel || !seqs || !out) return -1;
    if (n <= 0) return -1;
    if (window < kernel->base.param.L) return -1;
    if (shift < 1) return -1;
    if (pad_left > 0 && window - pad_left < kernel->base.param.L) return -1;
    if (pad_right > 0 && window - pad_right < kernel->base.param.L) return -1;
    /* combined check above */

    int seqlen = seqs[0]->seqlen;
    for (int i = 1; i < n; i++)
        if (seqs[i]->seqlen != seqlen) return -1;

    int n_valid = gkm_compute_n_windows(seqlen, window, shift,
                                    pad_left, pad_right, kernel->base.param.L);
    if (n_valid <= 0) return -1;

    weighted_window_state_t st;
    st.kernel = kernel;
    st.seqs = seqs; st.n = n;
    st.window = window; st.shift = shift;
    st.pad_left = pad_left; st.pad_right = pad_right;
    st.out = out;
    st.sqnorm_cache = (double *)malloc(sizeof(double) * (size_t)n_valid * (size_t)n);
    st.n_valid = n_valid;
    if (!st.sqnorm_cache) return -1;

    iter_windows(seqlen, window, shift, pad_left, pad_right,
                 kernel->base.param.L, cb_wsqnorms, &st);
    iter_windows(seqlen, window, shift, pad_left, pad_right,
                 kernel->base.param.L, cb_weval, &st);

    free(st.sqnorm_cache);
    return 0;
}

/* ================================================================== */
/* Weighted kernel: sliding window query (window-midpoint centering)  */
/* ================================================================== */

int gkm_weighted_kernel_sliding_query(const gkm_weighted_kernel_t *kernel,
                                      const gkm_sequence_t *query,
                                      int window, int shift,
                                      int pad_left, int pad_right,
                                      double *out, int *n_windows_out)
{
    if (!kernel || !query || !out) return -1;
    if (window < kernel->base.param.L) return -1;
    if (shift < 1) return -1;
    if (pad_left > 0 && window - pad_left < kernel->base.param.L) return -1;
    if (pad_right > 0 && window - pad_right < kernel->base.param.L) return -1;
    /* combined check above */
    if (!kernel->base.finalized) return -1;

    int L = kernel->base.param.L;
    int d = kernel->base.param.d;
    int n_ref = kernel->base.n_seqs;
    int seqlen = query->seqlen;

    int n_valid = gkm_compute_n_windows(seqlen, window, shift,
                                    pad_left, pad_right, L);
    if (n_valid <= 0) {
        if (n_windows_out) *n_windows_out = 0;
        return 0;
    }

    /* For each window, build a transient offset tree with the n_ref
     * references (using window-midpoint positional weights), then run
     * the multi-centre DFS with a single center (the window midpoint).
     * This is O(n_ref) per window.
     *
     * With negative padding, effective window size can exceed `window`,
     * so allocate wt buffers based on the max possible nkmers. */
    int max_eff_window = window - (pad_left < 0 ? pad_left : 0) - (pad_right < 0 ? pad_right : 0);
    int max_nkmers_win = max_eff_window - L + 1;
    double *wt_f = (double *)malloc(sizeof(double) * (size_t)max_nkmers_win);
    double *wt_r = (double *)malloc(sizeof(double) * (size_t)max_nkmers_win);
    if (!wt_f || !wt_r) { free(wt_f); free(wt_r); return -1; }

    /* sqnorm cache for refs (recomputed per window because the weights
     * depend on the window midpoint, which is the same for all refs at
     * a given window). */
    double *ref_sqnorms = (double *)malloc(sizeof(double) * (size_t)n_ref);

    int w_idx = 0;
    int w = 0;
    while (1) {
        int start = w * shift - pad_left;
        int end = start + window;
        int eff_start = start < 0 ? 0 : start;
        int eff_end = end > seqlen ? seqlen : end;
        int eff_len = eff_end - eff_start;
        if (eff_len >= L) {
            int nkmers_win = eff_len - L + 1;
            double center = start + (window - 1) / 2.0;  /* approach B */

            /* Compute positional weights for this window. */
            compute_pos_weights_for_seq(nkmers_win, L, center,
                                        kernel->pos_kernel,
                                        kernel->weight_gamma,
                                        kernel->weight_sigma,
                                        kernel->weight_peak,
                                        wt_f, wt_r);

            /* Build offset tree with refs (using window-midpoint weights). */
            kmer_tree_t *tree = (kmer_tree_t *)malloc(sizeof(kmer_tree_t));
            kmer_tree_init(tree, L, kernel->base.param.k, d);
            for (int j = 0; j < n_ref; j++) {
                int ref_seqlen = kernel->base.seqs[j]->seqlen;
                int rc_start = ref_seqlen - eff_start - eff_len;
                gkm_sequence_t view;
                memset(&view, 0, sizeof(view));
                view.seqlen = eff_len;
                view.seq    = kernel->base.seqs[j]->seq + eff_start;
                view.seq_rc = kernel->base.seqs[j]->seq_rc + rc_start;
                view.nkmers = nkmers_win;
                view.wt     = wt_f;
                view.wt_rc  = wt_r;
                view.sqnorm = compute_sqnorm(&kernel->base, &view);
                ref_sqnorms[j] = view.sqnorm;
                kmer_tree_add_sequence(tree, j, &view, kernel->base.param.use_rc);
            }

            /* Query: build a view for the query window. */
            int q_seqlen = query->seqlen;
            int q_rc_start = q_seqlen - eff_start - eff_len;
            gkm_sequence_t qview;
            memset(&qview, 0, sizeof(qview));
            qview.seqlen = eff_len;
            qview.seq    = query->seq + eff_start;
            qview.seq_rc = query->seq_rc + q_rc_start;
            qview.nkmers = nkmers_win;
            qview.wt     = wt_f;
            qview.wt_rc  = wt_r;
            qview.sqnorm = compute_sqnorm(&kernel->base, &qview);

            /* Run regular DFS (the tree stores (seqid, weight) pairs,
             * not offsets, because we baked the window-midpoint weights
             * in at tree-build time). */
            double **mmprofile = (double **)malloc(sizeof(double *) * (size_t)(d + 1));
            for (int k = 0; k <= d; k++) {
                mmprofile[k] = (double *)calloc((size_t)n_ref, sizeof(double));
            }
            base_mismatch_count_t *matching_bases =
                (base_mismatch_count_t *)malloc(
                    sizeof(base_mismatch_count_t) * (size_t)nkmers_win);
            for (int j = 0; j < nkmers_win; j++) {
                matching_bases[j].bid   = qview.seq + j;
                matching_bases[j].wt = qview.wt[j];

                matching_bases[j].mmcnt = 0;
            }

            kmer_tree_dfs(tree, n_ref, 0, 0,
                          matching_bases, nkmers_win, mmprofile);

            double q_sqnorm = qview.sqnorm;
            for (int j = 0; j < n_ref; j++) {
                double sum = 0;
                for (int k = 0; k <= d; k++) {
                    sum += kernel->base.weights[k] * (double)mmprofile[k][j];
                }
                double r_sqnorm = ref_sqnorms[j];
                if (q_sqnorm > 0 && r_sqnorm > 0) {
                    out[(size_t)w_idx * (size_t)n_ref + j] = sum / (q_sqnorm * r_sqnorm);
                } else {
                    out[(size_t)w_idx * (size_t)n_ref + j] = 0.0;
                }
            }

            for (int k = 0; k <= d; k++) free(mmprofile[k]);
            free(mmprofile);
            free(matching_bases);
            kmer_tree_destroy(tree);

            w_idx++;
            if (start >= seqlen - L - pad_right) break;
        } else {
            if (start >= seqlen - L + 1) break;
        }
        w++;
        if (w > 1000000) break;
    }

    free(wt_f); free(wt_r);
    free(ref_sqnorms);
    if (n_windows_out) *n_windows_out = w_idx;
    return 0;
}
