#ifndef BITLINEAR_H
#define BITLINEAR_H

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "activation.h"
#include "rmsnorm.h"
#include "../utils/matrix.h"

// Store outputs of inputs to layers for gradient computation.
typedef struct {
    float* x;     // input to bitlinear layer
    float* y_rms; // output of rmsnorm, input to bit matrix multiplication
    float* w;     // weights are stored in row major order and full precision for training
} bitlinear_mem_t;

typedef struct {
    float* dw;
    float* dg;
} bitlinear_grad_t;

typedef struct {
    float* g;    // scaling factors for rmsnorm
    uint8_t* wq; // 1-bit quantized weights for bit matrix
    int8_t* xq;  // 8 bit quantized input activations
    int8_t* yq;  // 8 bit quantized outputs
} bitlinear_t;


// Initialize values for training parameters for matrix.
void bitlinear_init(bitlinear_t* bitlin, bitlinear_mem_t* mem, size_t in_dim, size_t out_dim) {
    mat_init_kaiming(bitlin->g, in_dim);
    mat_init_kaiming(mem->w, in_dim * out_dim);
}


// Assign memory regions for intermediate values and gradients in the bitlinear layer.
void bitlinear_train_init(bitlinear_mem_t* mem, bitlinear_grad_t* grads, float* arr,
                          size_t in_dim, size_t out_dim, size_t batch_size) {
    float* arr_ptr = arr;
    mem->x = arr_ptr;

    arr_ptr += in_dim * batch_size;
    mem->y_rms = arr_ptr;

    arr_ptr += in_dim * batch_size;
    mem->w = arr_ptr;

    arr_ptr += in_dim * out_dim;
    grads->dw = arr_ptr;

    arr_ptr += in_dim * out_dim;
    grads->dg = arr_ptr;
}


/**
 * @brief Allocate memory for a single bitlinear layer and initialize them to 0.
 *
 * @param bitlin pointer to bitlinear parameters
 * @param in_dim input dimension
 * @param out_dim output dimension
 * @param batch_size batch_size
 * @return 0 if successful, non-zero if error occurred.
 */
int bitlinear_alloc(bitlinear_t* bitlin, size_t in_dim, size_t out_dim, size_t batch_size) {
    int exit_code = 0;

    size_t w_params = in_dim * out_dim * batch_size;
    bitlin->wq = (uint8_t*) calloc(w_params, sizeof(uint8_t));
    if (bitlin->wq == NULL) { exit_code = 1; goto cleanup; }

    size_t in_params = in_dim * batch_size;
    bitlin->xq = (int8_t*) calloc(in_params, sizeof(int8_t));
    if (bitlin->xq == NULL) { exit_code = 2; goto cleanup; }

    size_t out_params = out_dim * batch_size;
    bitlin->yq = (int8_t*) calloc(out_params, sizeof(int8_t));
    if (bitlin->yq == NULL) { exit_code = 3; goto cleanup; }

    size_t scaling_params = in_dim * batch_size;
    bitlin->g = (float*) calloc(scaling_params, sizeof(float));
    if (bitlin->g == NULL) { exit_code = 4; goto cleanup; }

cleanup:
    if (exit_code != 0) {
        fprintf(stderr, "Failed to allocate memory for bitlinear! Exit code: %d\n", exit_code);
        if (bitlin->wq) free(bitlin->wq);
        if (bitlin->xq) free(bitlin->xq);
        if (bitlin->yq) free(bitlin->yq);
        if (bitlin->g) free(bitlin->g);
    }
    return exit_code;
}


// Return max absolute value in given float array.
static inline float max_abs(const float* arr, int len) {
    float max_val = 0.0f;
    for (int i = 0; i < len; ++i) {
        float abs_val = fabsf(arr[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val;
}


// Return mean absolute value in given float array.
static inline float mean_abs(const float* w, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++)
        sum += w[i];
    return sum;
}


// Clip integer value to int8 min and max values.
static inline int8_t clip(int x) {
    x = (x > INT8_MAX) ? INT8_MAX : x;
    x = (x < INT8_MIN) ? INT8_MIN : x;
    return (int8_t) x;
}


/**
 * @brief Quantize activations to 8-bit precision as used in the BitNet paper.
 *
 * @param xq   array of 8-bit quantized activations
 * @param x    array of input activations
 * @param dim  dimensionality of each input vectors
 * @returns scale factor for input activations
 */
static inline float activation_quant(int8_t* xq, const float* x, int dim) {
    // compute scale = Qb / gamma
    float scale = (float) INT8_MAX / fmaxf(max_abs(x, dim), FLT_MIN);
    for (int i = 0; i < dim; i++)
        xq[i] = clip((int) roundf(x[i] * scale)); // clip round to nearest integer
    return scale;
}


// Dequantize activations from 8-bit precision to floating point weights.
static inline void activation_dequant(float* y, int8_t* xq, float beta, float scale, int len) {
    for (int i = 0; i < len; i++)
        y[i] = beta * (float) xq[i] / scale;
}


/**
 * @brief Centralize weights in give matrix to be 0-mean before sign binarization.
 *        We only need 1 len argument here since values are computed for the
 *        entire matrix.
 *
 * @param wq  array of quantized weights
 * @param w   array of matrix weights
 * @param len number of entries in weight matrix
 */
static inline void weight_quant(uint8_t* wq, const float* w, int len) {
    float e = mean(w, len);
    for (int i = 0; i < len; i++)
        if ((w[i] - e) < 0) wq[i / 8] |= (1 << (i % 8)); // bit is 1 for negative weights
}


/**
 * @brief Forward pass with floating point weights. This applies quantization
 *        to the floating point weights w before doing matrix multiplication.
 *
 * @param y        pointer to output matrix
 * @param rms      pointer to store RMSNorm layer output
 * @param x        pointer to input matrix
 * @param w        pointer to weights matrix
 * @param g        scaling factor for rmsnorm layer
 * @param in_dim   dimensionality of input vectors, cols of weight matrix
 * @param out_dim  dimensionality of output vectors, rows of weight matrix
 * @param batch_size
 */
void bitlinear_fwd(float* y, float* rms,
                   const float* x, const float* w, const float* g,
                   uint8_t* wq, int8_t* yq, int8_t* xq,
                   size_t in_dim, size_t out_dim, size_t batch_size) {
#ifdef DEBUG
    printf("Bitlinear inputs:\n");
    print_mat(x, batch_size, in_dim);
#endif
    rmsnorm_fwd(rms, x, g, in_dim, batch_size);

    for (size_t b = 0; b < batch_size; b++) {
        float* rms_b = rms + b * in_dim;
        int8_t* xq_b = xq + b * in_dim;
        int8_t* yq_b = yq + b * out_dim;
        float* y_b = y + b * out_dim;

        float beta = mean_abs(w, in_dim * out_dim);
        float scale = activation_quant(xq_b, rms_b, in_dim); // scale = Qb / gamma
        weight_quant(wq, w, in_dim * out_dim);
        bitmatmul_fwd(yq_b, xq_b, wq, out_dim, in_dim, 1);
        activation_dequant(y_b, yq_b, beta, scale, out_dim);
    }
#ifdef DEBUG
    printf("Bitlinear outputs:\n");
    print_mat(y, batch_size, out_dim);
#endif
}


/**
 * @brief Compute gradients for BitLinear layer and its RMSNorm weights.
 *
 * @param dx      gradient of input
 * @param dw      gradient of bit matrix weights
 * @param dg      gradient of rmsnorm scaling factors
 * @param dy      gradient of loss wrt output of this layer
 * @param x       input to bitlinear layer
 * @param w       unquantized weights in bit matrix
 * @param g       scaling factors for rmsnorm
 * @param y_rms   output of rmsnorm layer
 * @param in_dim  dimensionality of input vectors, cols of weight matrix
 * @param out_dim dimensionality of output vectors, rows of weight matrix
 * @param batch_size
 */
void bitlinear_bkwd(float* dx, float* dw, float* dg,
                    const float* dy, const float* x, const float* w, const float* g, const float* y_rms,
                    size_t in_dim, size_t out_dim, size_t batch_size) {
    matmul_bkwd(dw, dx, dy, w, y_rms, out_dim, in_dim, batch_size);
    rmsnorm_bkwd(dg, dx, dx, x, g, in_dim, batch_size);
}


// Save trained scaling factors and 1-bit weights to file pointer for inference.
void bitlinear_save_weights(bitlinear_t* params, bitlinear_mem_t* mem, size_t in_dim, size_t out_dim, FILE* fp) {
    fwrite(params->g, sizeof(float), in_dim, fp);
    size_t len = out_dim * ((in_dim + 7) / 8);
    weight_quant(params->wq, mem->w, in_dim * out_dim);
    fwrite(params->wq, 1, len, fp);
}


// Load trained weights from file for inference.
void bitlinear_load_weights(bitlinear_t* params, size_t in_dim, size_t out_dim, FILE* fp) {
    fread(params->g, sizeof(float), in_dim, fp);
    fread(params->wq, 1, out_dim * ((in_dim + 7) / 8), fp);
}

#endif
