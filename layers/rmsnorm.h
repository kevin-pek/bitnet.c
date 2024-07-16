#ifndef RMSNORM_H
#define RMSNORM_H

#include <math.h>
#include "../utils/matrix.h"

#define RMSNORM_EPS 1e-5f


/// @brief Compute 1 / RMS using an array of given length.
static inline float rmsnorm_inv(const float* x, int len) {
    float ss = 0.0f;
    for (int i = 0; i < len; i++)
        ss += x[i] * x[i];
    ss /= len;
    ss = 1.0f / (sqrtf(ss) + RMSNORM_EPS);
    return ss;
}


/// @brief Apply RMSNorm layer with no bias term to a single input vector x.
/// @param y output array
/// @param x input array
/// @param g scaling factors for each element in the vector
/// @param dim dimensionality of the input vector
/// @param batch_size
void rmsnorm_fwd(float* y, const float* x, const float* g, int dim, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        const float* x_b = x + b * dim;
        const float* g_b = g + b * dim;
        float* y_b = y + b * dim;
        float rms_inv = rmsnorm_inv(x_b, dim);
        for (int i = 0; i < dim; i++)
            y_b[i] = g_b[i] * rms_inv * x_b[i];
    }
}


/// @brief Compute gradients for scaling factors and inputs to RMSNorm layer.
/// @param dg  gradient of scaling factors of RMSNorm layer
/// @param dx  gradient of inputs of RMSNorm layer
/// @param dy  gradient of outputs of RMSNorm layer
/// @param x   input array that was passed during forward pass
/// @param g   scaling weights for RMSNorm
/// @param dim dimensionality of RMSNorm layer
/// @param batch_size
void rmsnorm_bkwd(float* dg, float* dx,
                  const float* dy, const float* x, const float* g,
                  int dim, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        const float* x_b = x + b * dim;
        const float* g_b = g + b * dim;
        const float* dy_b = dy + b * dim;
        float* dx_b = dx + b * dim;
        float* dg_b = dg + b * dim;
        float rms_inv = rmsnorm_inv(x_b, dim); // 1 / RMS(x)
        for (int i = 0; i < dim; i++) {
            dg_b[i] = dy_b[i] * x_b[i] * rms_inv; // dL / dg_i
            for (int j = 0; j < dim; j++) { // dL / dx_i
                if (i == j)
                    dx_b[i] += dy_b[j] * g_b[j] * rms_inv * (1 - rms_inv * x_b[i] * x_b[i] / dim);
                else
                    dx_b[i] -= g_b[j] * x_b[i] * x_b[j] * rms_inv * rms_inv * rms_inv / dim;
            }
        }
    }
}

#endif
