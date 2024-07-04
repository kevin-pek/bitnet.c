#ifndef RMSNORM_H
#define RMSNORM_H

#include <math.h>
#include "../utils/matrix.h"

#define EPS 1e-5f

/// @brief Compute 1 / RMS using an array of given length.
static inline float rmsnorm_inv(const float* x, int len) {
    float ss = 0.0f;
    for (int i = 0; i < len; i++)
        ss += x[i] * x[i];
    ss /= len;
    ss = 1.0f / (sqrtf(ss) + EPS);
    return ss;
}


/// @brief Apply RMSNorm layer with no bias term to a single input vector x.
/// @param y output array
/// @param x input array
/// @param g scaling factors for each element in the vector
/// @param dim dimensionality of the input vector
void rmsnorm_fwd(float* y, const float* x, const float* g, int dim) {
    float rms_inv = rmsnorm_inv(x, dim);
    for (int i = 0; i < dim; i++)
        y[i] = g[i] * rms_inv * x[i];
}


/// @brief Compute gradients for scaling factors and inputs to RMSNorm layer.
/// @param dg  gradient of scaling factors of RMSNorm layer
/// @param dx  gradient of inputs of RMSNorm layer
/// @param dy  gradient of outputs of RMSNorm layer
/// @param x   input array that was passed during forward pass
/// @param g   scaling weights for RMSNorm
/// @param dim dimensionality of RMSNorm layer
void rmsnorm_bkwd(float* dg, float* dx,
                  const float* dy, const float* x, const float* g,
                  int dim) {
    float rms_inv = rmsnorm_inv(x, dim); // 1 / RMS(x)
    for (int i = 0; i < dim; i++) {
        dg[i] = dy[i] * x[i] * rms_inv; // dL / dg_i
        for (int j = 0; j < dim; j++) { // dL / dx_i
            if (i == j)
                dx[i] += dy[j] * g[j] * rms_inv * (1 - rms_inv * x[i] * x[i] / dim);
            else
                dx[i] -= g[j] * x[i] * x[j] * rms_inv * rms_inv * rms_inv / dim;
        }
    }
}

#endif
