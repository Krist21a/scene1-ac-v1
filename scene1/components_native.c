#include <stdint.h>
#include <math.h>

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

/* Float32 CHW -> uint8 HWC, with the reference's truncate-after-multiply rule.
   Any unusual value requests the original Python validation/conversion path. */
int scene1_rgb_chw(const float *input, int64_t pixels, uint8_t *output) {
    for (int64_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            float value = input[c*pixels+i];
            if (!(value >= 0.0f && value <= 1.0f)) return 1;
            output[3*i+c] = (uint8_t)(value*255.0f);
        }
    }
    return 0;
}

int64_t scene1_cloud(const double *depth, const uint8_t *hsv,
                    int64_t height, int64_t width, int64_t stride,
                    const double *k, const double *m,
                    double *points, uint8_t *colors) {
    int64_t count = 0;
    for (int64_t v = 0; v < height; v += stride) {
        for (int64_t u = 0; u < width; u += stride) {
            int64_t pixel = v*width+u;
            double z = depth[pixel];
            if (!(z > 0.1 && z < 2.5)) continue;
            double x = ((double)u-k[2])*z/k[0];
            double y = ((double)v-k[5])*z/k[4];
            double px = (x*m[0]+y*m[1]+z*m[2])+m[3];
            double py = (x*m[4]+y*m[5]+z*m[6])+m[7];
            double pz = (x*m[8]+y*m[9]+z*m[10])+m[11];
            if (!(isfinite(px) && isfinite(py) && isfinite(pz))) continue;
            points[3*count] = px;
            points[3*count+1] = py;
            points[3*count+2] = pz;
            for (int c = 0; c < 3; ++c) colors[3*count+c] = hsv[3*pixel+c];
            ++count;
        }
    }
    return count;
}

/* Organized projection: preserve every sample, including invalid depths. */
void scene1_project_grid(const double *depth, int64_t height, int64_t width,
                         int64_t stride, const double *k, const double *m,
                         double *points) {
    int64_t i = 0;
    for (int64_t v = 0; v < height; v += stride) {
        for (int64_t u = 0; u < width; u += stride) {
            double z = depth[v*width+u];
            double x = ((double)u-k[2])*z/k[0];
            double y = ((double)v-k[5])*z/k[4];
            for (int r = 0; r < 3; ++r)
                points[3*i+r] = (x*m[4*r]+y*m[4*r+1]+z*m[4*r+2])+m[4*r+3];
            ++i;
        }
    }
}
