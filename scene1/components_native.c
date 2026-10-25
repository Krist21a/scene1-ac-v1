#include <stdint.h>

static int64_t root(int64_t *parent, int64_t i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

/* Preserve the reference algorithm's edge order, roots and component order. */
void scene1_union_pairs(const int64_t *pairs, int64_t n_pairs,
                       int64_t *parent, int64_t n_points) {
    for (int64_t i = 0; i < n_pairs; ++i) {
        int64_t a = root(parent, pairs[2*i]);
        int64_t b = root(parent, pairs[2*i+1]);
        if (a != b) parent[b] = a;
    }
    for (int64_t i = 0; i < n_points; ++i) parent[i] = root(parent, i);
}
