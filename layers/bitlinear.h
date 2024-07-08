#ifndef BITLINEAR_H
#define BITLINEAR_H

#include <float.h>
#include <math.h>
#include <string.h>
#include "activation.h"
#include "rmsnorm.h"
#include "../utils/matrix.h"

/// @brief Return max absolute value in given float array.
static inline float max_abs(const float* arr, int len) {
    float max_val = 0.0f;
    for (int i = 0; i < len; ++i) {
        float abs_val = fabsf(arr[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val;
}


/// @brief Return mean absolute value in given float array.
static inline float mean_abs(const float* w, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++)
        sum += w[i];
    return sum;
}


/// @brief Clip integer value to int8 min and max values.
static inline int8_t clip(int x) {
    x = (x > INT8_MAX) ? INT8_MAX : x;
    x = (x < INT8_MIN) ? INT8_MIN : x;
    return (int8_t) x;
}


/// @brief Quantize activations to 8-bit precision as used in the BitNet paper.
/// @param xq   array of 8-bit quantized activations
/// @param x    array of input activations
/// @param dim  dimensionality of each input vectors
/// @returns scale factor for input activations
static inline float activation_quant(int8_t* xq, const float* x, int dim) {
    // compute scale = Qb / gamma
    float scale = (float) INT8_MAX / fmaxf(max_abs(x, dim), FLT_MIN);
    for (int i = 0; i < dim; i++)
        xq[i] = clip((int) roundf(x[i] * scale)); // clip round to nearest integer
    return scale;
}


/// @brief Dequantize activations from 8-bit precision to floating point weights.
static inline void activation_dequant(float* y, int8_t* xq, float beta, float scale, int len) {
    for (int i = 0; i < len; i++)
        y[i] = beta * (float) xq[i] / scale;
}


/// @brief Centralize weights in give matrix to be 0-mean before sign binarization.
///        We only need 1 len argument here since values are computed for the
///        entire matrix.
/// @param wq  array of quantized weights
/// @param w   array of matrix weights
/// @param len number of entries in weight matrix
static inline void weight_quant(uint8_t* wq, const float* w, int len) {
    float e = mean(w, len);
    for (int i = 0; i < len; i++)
        if ((w[i] - e) < 0) wq[i / 8] |= (1 << (i % 8)); // bit is 1 for negative weights
}


/// @brief Forward pass with floating point weights. This applies quantization
///        to the floating point weights w before doing matrix multiplication.
/// @param y       pointer to output matrix
/// @param rms     pointer to store RMSNorm layer output
/// @param x       pointer to input matrix
/// @param w       pointer to weights matrix
/// @param g       scaling factor for rmsnorm layer
/// @param in_dim  dimensionality of input vectors, cols of weight matrix
/// @param out_dim dimensionality of output vectors, rows of weight matrix
void bitlinear_fwd(float* y, float* rms,
                   const float* x, const float* w, const float* g,
                   int in_dim, int out_dim) {
    uint8_t* wq = (uint8_t*) malloc(sizeof(uint8_t) * (in_dim + 7) / 8);
    memset(wq, 0, (in_dim + 7) / 8); // initialize weights to 0
    int8_t* xq = (int8_t*) malloc(sizeof(int8_t) * in_dim);
    int8_t* yq = (int8_t*) malloc(sizeof(int8_t) * out_dim);

    rmsnorm_fwd(rms, x, g, in_dim);
    float beta = mean_abs(w, in_dim * out_dim);
    float scale = activation_quant(xq, rms, in_dim); // scale = Qb / gamma
    weight_quant(wq, w, in_dim * out_dim);
    bitmatmul_fwd(yq, xq, wq, out_dim, in_dim, 1);
    activation_dequant(y, yq, beta, scale, out_dim);

    free(yq);
    free(wq);
    free(xq);
}


/// @brief Compute gradients for BitLinear layer and its RMSNorm weights.
/// @param dx      gradient of input
/// @param dw      gradient of bit matrix weigths
/// @param dg      gradient of rmsnorm scaling factors
/// @param dy      gradient of loss wrt output of this layer
/// @param x       input to bitlinear layer
/// @param w       unquantized weights in bit matrix
/// @param g       scaling factors for rmsnorm
/// @param y_rms   output of rmsnorm layer
/// @param in_dim  dimensionality of input vectors, cols of weight matrix
/// @param out_dim dimensionality of output vectors, rows of weight matrix
void bitlinear_bkwd(float* dx, float* dw, float* dg,
                    const float* dy, const float* x, const float* w, const float* g, const float* y_rms,
                    int in_dim, int out_dim) {
    matmul_bkwd(dw, dx, dy, w, y_rms, out_dim, in_dim, 1);
    rmsnorm_bkwd(dg, dx, dx, x, g, in_dim);
}

#endif
