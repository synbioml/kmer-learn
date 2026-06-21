/* _shuffler.c — C core for k-mer-preserving sequence shuffling.
 *
 * Algorithm: build the De Bruijn multigraph of the input sequence
 * (nodes = (k-1)-mers, edges = k-mers, each occurrence is a distinct
 * edge), then find a random Eulerian path via Hierholzer's algorithm
 * with random edge selection.
 *
 * Three endpoint modes:
 *   "preserve"  — start from the original odd-degree vertex; both
 *                 endpoints match the input. Exact k-mer composition.
 *   "free"      — close the graph into a circuit (add an edge from the
 *                 last node to the first), find an Eulerian circuit,
 *                 break it at a random edge. Endpoints can change;
 *                 k-mer composition is exact + 1 wrap-around k-mer.
 *   "crop"      — drop the first and last (k-1)-mer, shuffle the
 *                 interior as a circuit. Output length = len - 2(k-1).
 *
 * RNG: Philox4x32-10 (see _common.h). State lives on the Python object;
 * it advances on every edge choice, so successive shuffle() calls with
 * the same Shuffler produce independent outputs. For parallel
 * shuffle_many(), each task gets an independent key derived from the
 * master seed + task index — output is bit-for-bit identical regardless
 * of n_jobs.
 */

#include "_common.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ======================================================================
 * De Bruijn graph representation
 *
 * We use a simple open-addressing hash table to map (k-1)-mer -> node id.
 * For k <= 12, (k-1) <= 11, so a (k-1)-mer fits in 22 bits — we pack it
 * into a uint32_t (2 bits per base, MSB-first) for fast hashing.
 *
 * Adjacency is CSR: edges grouped by source node, each edge = (dst, used).
 * A workspace "available edges" list per node is maintained for O(1)
 * random selection.
 * ====================================================================== */

#define KMER_SHUFFLER_MAX_K 12

typedef struct {
    uint32_t packed;        /* (k-1)-mer packed 2 bits/base */
    int      out_deg;
    int      in_deg;
    int      edge_offset;   /* CSR offset into edge_dst[] */
    int      avail_count;   /* number of unused out-edges */
} dbn_node_t;

typedef struct {
    int n_nodes;
    int n_edges;
    int k;
    dbn_node_t *nodes;
    int        *edge_dst;       /* [n_edges] destination node id */
    uint8_t    *edge_used;      /* [n_edges] 0/1 */
    int        *avail_list;     /* [n_edges] workspace: list of unused
                                 * edge indices per node, contiguous
                                 * starting at nodes[i].edge_offset */
    /* Hash table: maps packed (k-1)-mer -> node id.
     * Open addressing, linear probing. */
    int        *hash_keys;      /* [hash_size] packed (k-1)-mer or -1 */
    int        *hash_vals;      /* [hash_size] node id */
    int         hash_size;      /* power of 2 */
    int         hash_mask;      /* hash_size - 1 */
    /* Odd-degree vertex tracking */
    int odd_vertices[4];        /* at most 2 in a real De Bruijn graph,
                                 * but we allow slack */
    int n_odd;
    /* First and last node ids (set by build) */
    int first_node;
    int last_node;
} dbn_graph_t;

/* ----- (k-1)-mer packing ----- */

static inline uint32_t pack_kmer(const char *s, int k) {
    /* Pack k <= 12 bases into a uint32 (2 bits each, MSB-first). */
    uint32_t p = 0;
    for (int i = 0; i < k; i++) {
        uint8_t c = (uint8_t)s[i];
        uint8_t v = NUC_LUT[c];
        /* NUC_LUT returns 0 for both 'A' and unknown chars. Caller must
         * have already validated strict ACGT, so we trust the value. */
        p = (p << 2) | v;
    }
    return p;
}

/* ----- Hash table ----- */

static void hash_init(dbn_graph_t *g, int min_entries) {
    int size = 16;
    while (size < min_entries * 2) size <<= 1;
    g->hash_size = size;
    g->hash_mask = size - 1;
    g->hash_keys = (int *)malloc(sizeof(int) * size);
    g->hash_vals = (int *)malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++) g->hash_keys[i] = -1;
}

static void hash_free(dbn_graph_t *g) {
    free(g->hash_keys); g->hash_keys = NULL;
    free(g->hash_vals); g->hash_vals = NULL;
}

static int hash_lookup(dbn_graph_t *g, uint32_t key, int create) {
    /* Mix the key for better distribution. */
    uint32_t h = key;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    int idx = (int)(h & g->hash_mask);
    while (g->hash_keys[idx] != -1) {
        if ((uint32_t)g->hash_keys[idx] == key) return g->hash_vals[idx];
        idx = (idx + 1) & g->hash_mask;
    }
    if (create) {
        int new_id = g->n_nodes++;
        g->hash_keys[idx] = (int)key;
        g->hash_vals[idx] = new_id;
        return new_id;
    }
    return -1;
}

/* ----- Graph build ----- */

/* Returns 0 on success, -1 on allocation failure, -2 on non-ACGT char. */
static int dbn_build(dbn_graph_t *g, const char *seq, int seqlen, int k) {
    memset(g, 0, sizeof(*g));
    g->k = k;
    if (seqlen < k) {
        /* Edge case: no k-mers. Single node (the whole sequence, if
         * k-1 <= seqlen) and no edges. We'll just return the input. */
        g->n_nodes = 0;
        g->n_edges = 0;
        return 0;
    }
    int n_edges = seqlen - k + 1;
    int n_nodes_max = n_edges + 1;

    /* Validate strict ACGT (uppercase). */
    for (int i = 0; i < seqlen; i++) {
        char c = seq[i];
        if (c != 'A' && c != 'C' && c != 'G' && c != 'T') {
            return -2;
        }
    }

    hash_init(g, n_nodes_max);
    g->nodes = (dbn_node_t *)calloc(n_nodes_max, sizeof(dbn_node_t));
    g->edge_dst = (int *)malloc(sizeof(int) * n_edges);
    g->edge_used = (uint8_t *)calloc(n_edges, sizeof(uint8_t));
    g->avail_list = (int *)malloc(sizeof(int) * n_edges);
    if (!g->nodes || !g->edge_dst || !g->edge_used || !g->avail_list) {
        return -1;
    }

    /* First pass: register nodes (each (k-1)-mer). */
    int prev_node = -1;
    for (int i = 0; i + k - 1 < seqlen; i++) {
        uint32_t p = pack_kmer(seq + i, k - 1);
        int n = hash_lookup(g, p, 1);
        (void)n;
    }
    /* Second pass: add edges. */
    /* We need to know node ids for both endpoints of each k-mer. */
    int edge_idx = 0;
    for (int i = 0; i + k <= seqlen; i++) {
        uint32_t p_src = pack_kmer(seq + i, k - 1);
        uint32_t p_dst = pack_kmer(seq + i + 1, k - 1);
        int src = hash_lookup(g, p_src, 0);
        int dst = hash_lookup(g, p_dst, 0);
        /* src must exist from first pass; dst may or may not, but for
         * internal k-mers it does. The very last destination might be
         * a new node — register it. */
        if (dst < 0) dst = hash_lookup(g, p_dst, 1);
        g->edge_dst[edge_idx] = dst;
        g->nodes[src].out_deg++;
        g->nodes[dst].in_deg++;
        edge_idx++;
        (void)prev_node;
    }
    g->n_edges = edge_idx;

    /* Set first/last node. */
    if (seqlen >= k - 1) {
        uint32_t p_first = pack_kmer(seq, k - 1);
        uint32_t p_last = pack_kmer(seq + seqlen - (k - 1), k - 1);
        g->first_node = hash_lookup(g, p_first, 0);
        g->last_node = hash_lookup(g, p_last, 0);
    } else {
        g->first_node = g->last_node = -1;
    }

    /* Build CSR: sort edges by source node. */
    /* Count out-degree per node (already done), compute offsets. */
    int offset = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        g->nodes[i].edge_offset = offset;
        offset += g->nodes[i].out_deg;
        g->nodes[i].avail_count = g->nodes[i].out_deg;
    }
    /* Place edges into avail_list. We need a temporary per-node cursor. */
    int *cursor = (int *)calloc(g->n_nodes, sizeof(int));
    if (!cursor) return -1;
    for (int i = 0; i + k <= seqlen; i++) {
        uint32_t p_src = pack_kmer(seq + i, k - 1);
        int src = hash_lookup(g, p_src, 0);
        int off = g->nodes[src].edge_offset + cursor[src];
        g->avail_list[off] = i;  /* edge id = original position; we'll
                                  * look up dst from edge_dst[] */
        cursor[src]++;
    }
    free(cursor);

    /* edge_dst is indexed by edge id (0..n_edges-1) in the order they
     * were added (i.e., input position). avail_list[off] gives the
     * edge id; we look up edge_dst[edge_id] for the destination. */

    /* Find odd-degree vertices: out_deg - in_deg = ±1. */
    g->n_odd = 0;
    for (int i = 0; i < g->n_nodes && g->n_odd < 4; i++) {
        int diff = g->nodes[i].out_deg - g->nodes[i].in_deg;
        if (diff == 1) {
            g->odd_vertices[g->n_odd++] = i;
        }
    }
    /* Also collect -1 odd (the end vertex) for completeness. */
    for (int i = 0; i < g->n_nodes && g->n_odd < 4; i++) {
        int diff = g->nodes[i].out_deg - g->nodes[i].in_deg;
        if (diff == -1) {
            g->odd_vertices[g->n_odd++] = i;
        }
    }
    return 0;
}

static void dbn_free(dbn_graph_t *g) {
    free(g->nodes); g->nodes = NULL;
    free(g->edge_dst); g->edge_dst = NULL;
    free(g->edge_used); g->edge_used = NULL;
    free(g->avail_list); g->avail_list = NULL;
    hash_free(g);
}

/* Reset edge_used and avail_list to "all unused" — used when we want
 * to run Hierholzer multiple times on the same graph. */
static void dbn_reset_edges(dbn_graph_t *g) {
    memset(g->edge_used, 0, g->n_edges);
    for (int i = 0; i < g->n_nodes; i++) {
        g->nodes[i].avail_count = g->nodes[i].out_deg;
    }
}

/* Take a random unused out-edge from node v. Returns the edge index
 * (into edge_dst[]) or -1 if no unused edges remain. */
static int take_random_edge(dbn_graph_t *g, int v, philox_stream_t *rng) {
    if (g->nodes[v].avail_count == 0) return -1;
    int off = g->nodes[v].edge_offset;
    int n = g->nodes[v].avail_count;
    /* Pick a random slot in [0, n), swap it to the end, decrement. */
    int pick = (int)philox_below(rng, (uint32_t)n);
    int edge_id = g->avail_list[off + pick];
    /* Swap with last available. */
    g->avail_list[off + pick] = g->avail_list[off + n - 1];
    g->avail_list[off + n - 1] = edge_id;
    g->nodes[v].avail_count--;
    g->edge_used[edge_id] = 1;
    return edge_id;
}

/* ======================================================================
 * Hierholzer's algorithm — randomised
 *
 * Iterative implementation with an explicit stack. Produces an Eulerian
 * path/circuit starting from `start_v`. Records the edge id taken at
 * each step alongside the node id, so the caller can decode the output
 * without re-selecting edges (which could differ for parallel edges).
 *
 * Output:
 *   path[]     — array of node ids, length path_len
 *   edges[]    — array of edge ids, length path_len - 1
 *                (edges[i] is the edge from path[i] to path[i+1])
 *
 * Caller must free both arrays.
 * ====================================================================== */

static int hierholzer(dbn_graph_t *g, int start_v, philox_stream_t *rng,
                       int **path_out, int **edges_out, int *path_len_out) {
    int max_path = g->n_edges + 1;
    /* Stack of (node, edge_id_taken_to_get_here). edge_id = -1 for start. */
    int *stack_node = (int *)malloc(sizeof(int) * max_path);
    int *stack_edge = (int *)malloc(sizeof(int) * max_path);
    int *path = (int *)malloc(sizeof(int) * max_path);
    int *edges = (int *)malloc(sizeof(int) * max_path);  /* over-allocated */
    if (!stack_node || !stack_edge || !path || !edges) {
        free(stack_node); free(stack_edge); free(path); free(edges);
        return -1;
    }
    int sp = 0, plen = 0, elen = 0;
    stack_node[sp] = start_v;
    stack_edge[sp] = -1;
    sp++;
    while (sp > 0) {
        int v = stack_node[sp - 1];
        int e = take_random_edge(g, v, rng);
        if (e < 0) {
            /* No unused edges from v — pop to path. Record the edge that
             * brought us here (if any). */
            path[plen++] = v;
            if (stack_edge[sp - 1] >= 0) {
                edges[elen++] = stack_edge[sp - 1];
            }
            sp--;
        } else {
            stack_node[sp] = g->edge_dst[e];
            stack_edge[sp] = e;
            sp++;
        }
    }
    /* path and edges are in reverse order. Reverse both. */
    for (int i = 0; i < plen / 2; i++) {
        int t = path[i]; path[i] = path[plen - 1 - i]; path[plen - 1 - i] = t;
    }
    for (int i = 0; i < elen / 2; i++) {
        int t = edges[i]; edges[i] = edges[elen - 1 - i]; edges[elen - 1 - i] = t;
    }
    free(stack_node);
    free(stack_edge);
    *path_out = path;
    *edges_out = edges;
    *path_len_out = plen;
    return 0;
}

/* ======================================================================
 * Public entry point: shuffle one sequence.
 *
 * Parameters:
 *   seq       — input sequence (strict ACGT, validated here)
 *   seqlen    — length
 *   k         — k-mer size (1..12)
 *   endpoints — "preserve" | "free" | "crop"
 *   rng       — Philox stream (advanced by this call)
 *   out       — output buffer, size >= seqlen (caller allocates)
 *   out_len   — actual output length (set by this function)
 *
 * Returns 0 on success, negative on error.
 * ====================================================================== */

#define KMER_SHUFFLE_OK             0
#define KMER_SHUFFLE_ERR_NONACGT   -1
#define KMER_SHUFFLE_ERR_MEMORY    -2
#define KMER_SHUFFLE_ERR_BAD_K     -3
#define KMER_SHUFFLE_ERR_INTERNAL  -4

static int kmer_shuffle_one(const char *seq, int seqlen, int k,
                             const char *endpoints,
                             philox_stream_t *rng,
                             char *out, int *out_len) {
    if (k < 1 || k > KMER_SHUFFLER_MAX_K) return KMER_SHUFFLE_ERR_BAD_K;

    /* Edge case: sequence shorter than k → return as-is. */
    if (seqlen < k) {
        memcpy(out, seq, seqlen);
        *out_len = seqlen;
        return KMER_SHUFFLE_OK;
    }

    /* Build the De Bruijn graph. */
    dbn_graph_t g;
    int rc = dbn_build(&g, seq, seqlen, k);
    if (rc == -2) { dbn_free(&g); return KMER_SHUFFLE_ERR_NONACGT; }
    if (rc != 0) { dbn_free(&g); return KMER_SHUFFLE_ERR_MEMORY; }

    int start_v;
    int *path = NULL;
    int *edges = NULL;
    int path_len = 0;
    int free_mode_circuit = 0;
    int virtual_edge_id = -1;  /* only set in free mode */

    if (endpoints[0] == 'p') {
        /* "preserve" — start from the original odd-degree vertex. */
        if (g.n_odd > 0) {
            start_v = g.first_node;
            for (int i = 0; i < g.n_odd; i++) {
                int v = g.odd_vertices[i];
                if (g.nodes[v].out_deg - g.nodes[v].in_deg == 1) {
                    start_v = v; break;
                }
            }
        } else {
            start_v = g.first_node;
        }
        if (hierholzer(&g, start_v, rng, &path, &edges, &path_len) < 0) {
            dbn_free(&g); return KMER_SHUFFLE_ERR_MEMORY;
        }
    } else if (endpoints[0] == 'f') {
        /* "free" — add a wrap-around edge from last_node to first_node,
         * find Eulerian circuit, break at the virtual edge. */
        if (g.last_node >= 0 && g.first_node >= 0 && g.n_edges > 0) {
            /* Augment graph: add the virtual edge. */
            int *new_avail = (int *)realloc(g.avail_list,
                                             sizeof(int) * (g.n_edges + 1));
            if (!new_avail) { dbn_free(&g); return KMER_SHUFFLE_ERR_MEMORY; }
            g.avail_list = new_avail;
            int *new_dst = (int *)realloc(g.edge_dst,
                                           sizeof(int) * (g.n_edges + 1));
            if (!new_dst) { dbn_free(&g); return KMER_SHUFFLE_ERR_MEMORY; }
            g.edge_dst = new_dst;
            g.edge_dst[g.n_edges] = g.first_node;
            uint8_t *new_used = (uint8_t *)realloc(g.edge_used,
                                                    (g.n_edges + 1));
            if (!new_used) { dbn_free(&g); return KMER_SHUFFLE_ERR_MEMORY; }
            g.edge_used = new_used;
            g.edge_used[g.n_edges] = 0;
            /* Bump degrees. */
            g.nodes[g.last_node].out_deg++;
            g.nodes[g.first_node].in_deg++;
            /* Rebuild CSR offsets and avail_list. */
            int offset = 0;
            for (int i = 0; i < g.n_nodes; i++) {
                g.nodes[i].edge_offset = offset;
                offset += g.nodes[i].out_deg;
                g.nodes[i].avail_count = g.nodes[i].out_deg;
            }
            int *cursor = (int *)calloc(g.n_nodes, sizeof(int));
            if (!cursor) { dbn_free(&g); return KMER_SHUFFLE_ERR_MEMORY; }
            for (int i = 0; i + k <= seqlen; i++) {
                uint32_t p_src = pack_kmer(seq + i, k - 1);
                int src = hash_lookup(&g, p_src, 0);
                int off = g.nodes[src].edge_offset + cursor[src]++;
                g.avail_list[off] = i;
            }
            g.avail_list[g.nodes[g.last_node].edge_offset
                         + cursor[g.last_node]++] = g.n_edges;
            free(cursor);
            virtual_edge_id = g.n_edges;
            g.n_edges += 1;
            /* Pick a random start vertex (any with out_deg > 0). */
            int tries = 0;
            do {
                start_v = (int)philox_below(rng, (uint32_t)g.n_nodes);
                tries++;
            } while (g.nodes[start_v].out_deg == 0 && tries < 100);
            if (g.nodes[start_v].out_deg == 0) start_v = g.first_node;
            if (hierholzer(&g, start_v, rng, &path, &edges, &path_len) < 0) {
                dbn_free(&g); return KMER_SHUFFLE_ERR_MEMORY;
            }
            free_mode_circuit = 1;
        } else {
            memcpy(out, seq, seqlen);
            *out_len = seqlen;
            dbn_free(&g);
            return KMER_SHUFFLE_OK;
        }
    } else if (endpoints[0] == 'c') {
        /* "crop" — drop first and last (k-1)-mer, shuffle interior as
         * a circuit. The interior's De Bruijn graph may have odd-degree
         * vertices (the first and last (k-1)-mer), so we add a virtual
         * wrap edge from last to first to balance the graph, then find
         * an Eulerian circuit and break it at the virtual edge — same
         * approach as free mode but on the interior. */
        int drop = k - 1;
        if (seqlen < 2 * drop + 1) {
            memcpy(out, seq, seqlen);
            *out_len = seqlen;
            dbn_free(&g);
            return KMER_SHUFFLE_OK;
        }
        int interior_len = seqlen - 2 * drop;
        dbn_graph_t g2;
        rc = dbn_build(&g2, seq + drop, interior_len, k);
        if (rc != 0) { dbn_free(&g); dbn_free(&g2); return KMER_SHUFFLE_ERR_MEMORY; }
        /* If g2 has edges, augment with a virtual wrap edge. */
        if (g2.n_edges > 0 && g2.first_node >= 0 && g2.last_node >= 0) {
            /* Re-alloc to fit one extra edge. */
            int *new_avail = (int *)realloc(g2.avail_list,
                                             sizeof(int) * (g2.n_edges + 1));
            if (!new_avail) { dbn_free(&g); dbn_free(&g2); return KMER_SHUFFLE_ERR_MEMORY; }
            g2.avail_list = new_avail;
            int *new_dst = (int *)realloc(g2.edge_dst,
                                           sizeof(int) * (g2.n_edges + 1));
            if (!new_dst) { dbn_free(&g); dbn_free(&g2); return KMER_SHUFFLE_ERR_MEMORY; }
            g2.edge_dst = new_dst;
            g2.edge_dst[g2.n_edges] = g2.first_node;
            uint8_t *new_used = (uint8_t *)realloc(g2.edge_used,
                                                    (g2.n_edges + 1));
            if (!new_used) { dbn_free(&g); dbn_free(&g2); return KMER_SHUFFLE_ERR_MEMORY; }
            g2.edge_used = new_used;
            g2.edge_used[g2.n_edges] = 0;
            g2.nodes[g2.last_node].out_deg++;
            g2.nodes[g2.first_node].in_deg++;
            /* Rebuild CSR offsets + avail_list. */
            int offset = 0;
            for (int i = 0; i < g2.n_nodes; i++) {
                g2.nodes[i].edge_offset = offset;
                offset += g2.nodes[i].out_deg;
                g2.nodes[i].avail_count = g2.nodes[i].out_deg;
            }
            int *cursor = (int *)calloc(g2.n_nodes, sizeof(int));
            if (!cursor) { dbn_free(&g); dbn_free(&g2); return KMER_SHUFFLE_ERR_MEMORY; }
            for (int i = 0; i + k <= interior_len; i++) {
                uint32_t p_src = pack_kmer(seq + drop + i, k - 1);
                int src = hash_lookup(&g2, p_src, 0);
                int off = g2.nodes[src].edge_offset + cursor[src]++;
                g2.avail_list[off] = i;
            }
            g2.avail_list[g2.nodes[g2.last_node].edge_offset
                          + cursor[g2.last_node]++] = g2.n_edges;
            free(cursor);
            int crop_virtual_edge_id = g2.n_edges;
            g2.n_edges += 1;
            /* Pick a random start vertex. */
            int tries = 0;
            do {
                start_v = (int)philox_below(rng, (uint32_t)g2.n_nodes);
                tries++;
            } while (g2.nodes[start_v].out_deg == 0 && tries < 100);
            if (g2.nodes[start_v].out_deg == 0) start_v = g2.first_node;
            if (hierholzer(&g2, start_v, rng, &path, &edges, &path_len) < 0) {
                dbn_free(&g); dbn_free(&g2);
                return KMER_SHUFFLE_ERR_MEMORY;
            }
            /* Find virtual edge in path; rotate to start at its destination. */
            int virtual_idx = -1;
            for (int i = 0; i + 1 < path_len; i++) {
                if (edges[i] == crop_virtual_edge_id) {
                    virtual_idx = i; break;
                }
            }
            if (virtual_idx < 0) {
                free(path); free(edges);
                dbn_free(&g); dbn_free(&g2);
                return KMER_SHUFFLE_ERR_INTERNAL;
            }
            int new_start = path[virtual_idx + 1];
            /* Find new_start's (k-1)-mer. */
            int found_new = 0;
            char new_first[KMER_SHUFFLER_MAX_K + 1];
            for (int i = 0; i + k - 1 <= interior_len && !found_new; i++) {
                uint32_t p = pack_kmer(seq + drop + i, k - 1);
                if (hash_lookup(&g2, p, 0) == new_start) {
                    memcpy(new_first, seq + drop + i, k - 1);
                    found_new = 1;
                }
            }
            if (!found_new) {
                free(path); free(edges);
                dbn_free(&g); dbn_free(&g2);
                return KMER_SHUFFLE_ERR_INTERNAL;
            }
            int out_pos = 0;
            memcpy(out + out_pos, new_first, k - 1);
            out_pos += k - 1;
            /* Emit edges after the virtual, then before it. */
            for (int i = virtual_idx + 1; i + 1 < path_len; i++) {
                out[out_pos++] = seq[drop + edges[i] + k - 1];
            }
            for (int i = 0; i < virtual_idx; i++) {
                out[out_pos++] = seq[drop + edges[i] + k - 1];
            }
            *out_len = out_pos;
            free(path); free(edges);
            dbn_free(&g); dbn_free(&g2);
            return KMER_SHUFFLE_OK;
        } else {
            /* No edges in interior — return empty. */
            *out_len = 0;
            dbn_free(&g); dbn_free(&g2);
            return KMER_SHUFFLE_OK;
        }
    } else {
        dbn_free(&g);
        return KMER_SHUFFLE_ERR_INTERNAL;
    }

    if (!path || !edges) {
        free(path); free(edges);
        dbn_free(&g);
        return KMER_SHUFFLE_ERR_MEMORY;
    }

    /* Decode preserve / free modes.
     * First node contributes (k-1) chars (its (k-1)-mer); each edge
     * contributes 1 char (the last char of its k-mer). For free mode,
     * the virtual edge is skipped (it has no associated k-mer char). */
    int found = 0;
    char first_kmer[KMER_SHUFFLER_MAX_K + 1];
    for (int i = 0; i + k - 1 <= seqlen && !found; i++) {
        uint32_t p = pack_kmer(seq + i, k - 1);
        if (hash_lookup(&g, p, 0) == path[0]) {
            memcpy(first_kmer, seq + i, k - 1);
            found = 1;
        }
    }
    if (!found) {
        free(path); free(edges);
        dbn_free(&g);
        return KMER_SHUFFLE_ERR_INTERNAL;
    }

    int out_pos = 0;

    if (free_mode_circuit) {
        /* Find the virtual edge in the path. The output is the path
         * rotated so it starts at the virtual edge's destination. */
        int virtual_idx = -1;
        for (int i = 0; i + 1 < path_len; i++) {
            if (edges[i] == virtual_edge_id) {
                virtual_idx = i;
                break;
            }
        }
        if (virtual_idx < 0) {
            /* Shouldn't happen — Hierholzer must traverse all edges. */
            free(path); free(edges);
            dbn_free(&g);
            return KMER_SHUFFLE_ERR_INTERNAL;
        }
        /* The new start node is path[virtual_idx + 1]. */
        int new_start_node = path[virtual_idx + 1];
        /* Find new_start_node's (k-1)-mer. */
        int found_new = 0;
        char new_first_kmer[KMER_SHUFFLER_MAX_K + 1];
        for (int i = 0; i + k - 1 <= seqlen && !found_new; i++) {
            uint32_t p = pack_kmer(seq + i, k - 1);
            if (hash_lookup(&g, p, 0) == new_start_node) {
                memcpy(new_first_kmer, seq + i, k - 1);
                found_new = 1;
            }
        }
        if (!found_new) {
            free(path); free(edges);
            dbn_free(&g);
            return KMER_SHUFFLE_ERR_INTERNAL;
        }
        memcpy(out + out_pos, new_first_kmer, k - 1);
        out_pos += k - 1;
        /* Emit last chars of edges in rotated order:
         *   edges[virtual_idx+1 .. path_len-2]  (the "after virtual" half)
         *   edges[0 .. virtual_idx-1]            (the "before virtual" half)
         * Skip edges[virtual_idx] (the virtual edge itself). */
        for (int i = virtual_idx + 1; i + 1 < path_len; i++) {
            out[out_pos++] = seq[edges[i] + k - 1];
        }
        for (int i = 0; i < virtual_idx; i++) {
            out[out_pos++] = seq[edges[i] + k - 1];
        }
    } else {
        /* Preserve mode: emit first node's (k-1)-mer, then last chars
         * of all edges in path order. */
        memcpy(out + out_pos, first_kmer, k - 1);
        out_pos += k - 1;
        for (int i = 0; i + 1 < path_len; i++) {
            out[out_pos++] = seq[edges[i] + k - 1];
        }
    }

    (void)found;  /* silence unused warning in free-mode branch */
    *out_len = out_pos;
    free(path); free(edges);
    dbn_free(&g);
    return KMER_SHUFFLE_OK;
}
