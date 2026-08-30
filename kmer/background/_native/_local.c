/* Position-bounded collision-chain permutation core. */

#include "_common.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LOCAL_OK 0
#define LOCAL_ERR_MEMORY -1
#define LOCAL_ERR_LENGTH -2
#define LOCAL_ERR_AUGMENT -3

typedef struct {
    uint64_t turns;
    uint64_t collisions;
    uint64_t soft_chains;
    uint64_t fallbacks;
    uint64_t max_collisions;
} local_stats_t;

static Py_ssize_t local_lower(Py_ssize_t item, Py_ssize_t shift) {
    return item > shift ? item - shift : 0;
}

static Py_ssize_t local_upper(Py_ssize_t item, Py_ssize_t shift,
                              Py_ssize_t length) {
    Py_ssize_t remaining = length - 1 - item;
    return item + (remaining > shift ? shift : remaining);
}

static Py_ssize_t local_random_slot(Py_ssize_t lower, Py_ssize_t upper,
                                    philox_stream_t *rng) {
    uint32_t width = (uint32_t)(upper - lower + 1);
    return lower + (Py_ssize_t)philox_below(rng, width);
}

static Py_ssize_t local_random_free_slot(const Py_ssize_t *owner,
                                         Py_ssize_t lower,
                                         Py_ssize_t upper,
                                         philox_stream_t *rng) {
    uint32_t free_count = 0;
    for (Py_ssize_t slot = lower; slot <= upper; ++slot) {
        if (owner[slot] < 0) ++free_count;
    }
    if (free_count == 0) return -1;

    uint32_t target = philox_below(rng, free_count);
    for (Py_ssize_t slot = lower; slot <= upper; ++slot) {
        if (owner[slot] < 0 && target-- == 0) return slot;
    }
    return -1;
}

/* Complete the current partial matching with a randomized BFS augmenting path. */
static int local_augment(Py_ssize_t unmatched, Py_ssize_t length,
                         Py_ssize_t shift, Py_ssize_t *owner,
                         Py_ssize_t *assigned, philox_stream_t *rng) {
    Py_ssize_t *queue = malloc((size_t)length * sizeof(*queue));
    Py_ssize_t *parent_item = malloc((size_t)length * sizeof(*parent_item));
    Py_ssize_t *parent_slot = malloc((size_t)length * sizeof(*parent_slot));
    uint8_t *seen = calloc((size_t)length, sizeof(*seen));
    if (!queue || !parent_item || !parent_slot || !seen) {
        free(queue); free(parent_item); free(parent_slot); free(seen);
        return LOCAL_ERR_MEMORY;
    }

    Py_ssize_t head = 0, tail = 0;
    queue[tail++] = unmatched;
    seen[unmatched] = 1;
    parent_item[unmatched] = -1;

    while (head < tail) {
        Py_ssize_t item = queue[head++];
        Py_ssize_t lower = local_lower(item, shift);
        Py_ssize_t upper = local_upper(item, shift, length);
        uint32_t width = (uint32_t)(upper - lower + 1);
        uint32_t offset = philox_below(rng, width);

        for (uint32_t step = 0; step < width; ++step) {
            Py_ssize_t slot = lower + (Py_ssize_t)((offset + step) % width);
            Py_ssize_t occupant = owner[slot];
            if (occupant < 0) {
                Py_ssize_t path_item = item;
                Py_ssize_t path_slot = slot;
                while (1) {
                    owner[path_slot] = path_item;
                    assigned[path_item] = path_slot;
                    if (path_item == unmatched) break;
                    path_slot = parent_slot[path_item];
                    path_item = parent_item[path_item];
                }
                free(queue); free(parent_item); free(parent_slot); free(seen);
                return LOCAL_OK;
            }
            if (!seen[occupant]) {
                seen[occupant] = 1;
                parent_item[occupant] = item;
                parent_slot[occupant] = slot;
                queue[tail++] = occupant;
            }
        }
    }

    free(queue); free(parent_item); free(parent_slot); free(seen);
    return LOCAL_ERR_AUGMENT;
}

/* out[slot] = original item index. */
static int local_permutation(Py_ssize_t length, Py_ssize_t max_shift,
                             int soft_enabled, uint64_t soft_limit,
                             int hard_enabled, uint64_t hard_limit,
                             philox_stream_t *rng, Py_ssize_t *out,
                             local_stats_t *stats) {
    memset(stats, 0, sizeof(*stats));
    if (length < 0 || (uint64_t)length > UINT32_MAX) return LOCAL_ERR_LENGTH;
    if (length == 0) return LOCAL_OK;
    if (max_shift > length - 1) max_shift = length - 1;

    Py_ssize_t *assigned = malloc((size_t)length * sizeof(*assigned));
    Py_ssize_t *order = malloc((size_t)length * sizeof(*order));
    if (!assigned || !order) {
        free(assigned); free(order);
        return LOCAL_ERR_MEMORY;
    }
    for (Py_ssize_t i = 0; i < length; ++i) {
        out[i] = -1;
        assigned[i] = -1;
        order[i] = i;
    }

    for (Py_ssize_t i = length - 1; i > 0; --i) {
        Py_ssize_t j = (Py_ssize_t)philox_below(rng, (uint32_t)(i + 1));
        Py_ssize_t tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }

    for (Py_ssize_t insertion = 0; insertion < length; ++insertion) {
        Py_ssize_t current = order[insertion];
        uint64_t collisions = 0;
        int soft_recorded = 0;

        while (1) {
            if (hard_enabled && collisions >= hard_limit) {
                int rc = local_augment(current, length, max_shift,
                                       out, assigned, rng);
                if (rc != LOCAL_OK) {
                    free(assigned); free(order);
                    return rc;
                }
                ++stats->fallbacks;
                break;
            }

            Py_ssize_t lower = local_lower(current, max_shift);
            Py_ssize_t upper = local_upper(current, max_shift, length);
            Py_ssize_t destination = -1;
            if (soft_enabled && collisions >= soft_limit) {
                if (!soft_recorded) {
                    ++stats->soft_chains;
                    soft_recorded = 1;
                }
                destination = local_random_free_slot(out, lower, upper, rng);
            }
            if (destination < 0) {
                destination = local_random_slot(lower, upper, rng);
            }

            ++stats->turns;
            Py_ssize_t displaced = out[destination];
            out[destination] = current;
            assigned[current] = destination;
            if (displaced < 0) break;

            assigned[displaced] = -1;
            current = displaced;
            ++collisions;
            ++stats->collisions;
        }
        if (collisions > stats->max_collisions) {
            stats->max_collisions = collisions;
        }
    }

    free(assigned); free(order);
    return LOCAL_OK;
}
