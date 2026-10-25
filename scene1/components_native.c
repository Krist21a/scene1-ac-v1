#include <stdint.h>
#include <math.h>

/* Same 15 separating axes as clearance.py. Boundary/invalid inputs request
   the original NumPy path, including its exact-contact scalar fallback. */
int scene1_boxes_overlap(const double *a, const double *ra, const double *ha,
                         const double *b, const double *rb, const double *hb) {
    double axes[15][3], delta[3];
    int separated = 0, ambiguous = 0;
    for (int i=0; i<3; ++i) {
        if (!isfinite(a[i]) || !isfinite(b[i]) || !isfinite(ha[i]) || !isfinite(hb[i])) return -1;
        delta[i] = b[i]-a[i];
        for (int j=0; j<3; ++j) {
            if (!isfinite(ra[3*i+j]) || !isfinite(rb[3*i+j])) return -1;
            axes[j][i] = ra[3*i+j]; axes[3+j][i] = rb[3*i+j];
        }
    }
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
        int n=6+3*i+j;
        for (int k=0; k<3; ++k)
            axes[n][k]=axes[i][(k+1)%3]*axes[3+j][(k+2)%3]-axes[i][(k+2)%3]*axes[3+j][(k+1)%3];
    }
    for (int n=0; n<15; ++n) {
        double norm=0., distance=0., radius_a=0., radius_b=0.;
        for (int k=0; k<3; ++k) norm+=axes[n][k]*axes[n][k];
        double length=sqrt(norm);
        if (fabs(length-1e-8)<=1e-20) return -1;
        if (length<1e-8) continue;
        for (int j=0; j<3; ++j) {
            double pa=0., pb=0.;
            distance+=axes[n][j]*delta[j];
            for (int k=0; k<3; ++k) {
                pa+=axes[n][k]*ra[3*k+j]; pb+=axes[n][k]*rb[3*k+j];
            }
            radius_a+=fabs(pa)*ha[j]; radius_b+=fabs(pb)*hb[j];
        }
        double gap=fabs(distance)-(radius_a+radius_b);
        if (!isfinite(gap) || fabs(gap)<=1e-12) ambiguous=1;
        if (gap>0.) separated=1;
    }
    return ambiguous ? -1 : !separated;
}

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

void scene1_roof_mask(const double *depth, const uint8_t *hsv, int64_t height, int64_t width,
                     const double *k, const double *m, const double *center,
                     const double *axes, const double *size, double top,
                     double *local, uint8_t *roof, uint8_t *lower) {
    for (int64_t v = 0; v < height; ++v) {
        for (int64_t u = 0; u < width; ++u) {
            int64_t i = v*width+u;
            double z = depth[i];
            double x = ((double)u-k[2])*z/k[0];
            double y = ((double)v-k[5])*z/k[4];
            double px = (x*m[0]+y*m[1]+z*m[2])+m[3];
            double py = (x*m[4]+y*m[5]+z*m[6])+m[7];
            double pz = (x*m[8]+y*m[9]+z*m[10])+m[11];
            double a = (px-center[0])*axes[0]+(py-center[1])*axes[2];
            double b = (px-center[0])*axes[1]+(py-center[1])*axes[3];
            local[2*i] = a; local[2*i+1] = b;
            int valid = z > .1 && isfinite(px) && isfinite(py) && isfinite(pz);
            int green = hsv[3*i] >= 40 && hsv[3*i] <= 85 && hsv[3*i+1] >= 100;
            roof[i] = valid && fabs(a) < size[0]/2.+.025 && fabs(b) < size[1]/2.+.025
                      && fabs(pz-top) < .010 && hsv[3*i+2] >= 90 && !green;
            lower[i] = valid && pz < top-.015;
        }
    }
}
