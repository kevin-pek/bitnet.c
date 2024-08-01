#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "logging.h"

#define PI 3.1415927f // max precision of π for 32 bit floating point numbers


// Compute mean of a input vector of given length.
static inline float mean(const float* x, size_t len) {
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++)
        sum += x[i];
    return sum / len;
}


// Return a random float (0, 1)
static inline float rand_float() {
    return (float) rand() / ((float) RAND_MAX + 1.0f);
}


// Sample from standard normal distribution N(0,1)
// µ + εσ = 0 + z * 1
static inline float std_norm() {
    return sqrtf(-2.0f * logf(rand_float())) * cosf(2.0f * PI * rand_float());
}


/**
 * @brief Matrix multiplication y = Wx, with W (n, m), x (m, b)
 * 
 * @param y              output matrix, column major
 * @param x              input matrix, column major
 * @param w              weight matrix, row major
 * @param out_dim    (n) num rows in weight matrix
 * @param in_dim     (m) num cols in weight matrix/rows in activation x
 * @param batch_size (b) num cols in matrix x
 */
void matmul_fwd(float* y, const float* x, const float* w, size_t n, size_t m, size_t b) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < b; j++) {
            float* y_ij = y + i * b + j;
            *y_ij = 0.0f;
            for (size_t k = 0; k < m; k++) // y_ij = sum(w_ik * x_kj), k in [1, m]
                *y_ij += w[i * m + k] * x[k * b + j];
        }
    }
}


/**
 * @brief Matrix multiplication using 8-bit quantized activations and 1-bit
 *        quantized weight matrices.  y = Wx, where W (n, m), x (m, b)
 *
 * @param yq             8 bit quantised outputs
 * @param xq             8 bit quantised inputs
 * @param wq             1 bit quantised weight matrix
 * @param out_dim    (n) num rows in weight matrix
 * @param in_dim     (m) num cols in weight matrix/rows in activation x
 * @param batch_size (b) num cols in matrix x
 */
void bitmatmul_fwd(int8_t* yq, const int8_t* xq, const uint8_t* wq,
                   size_t n, size_t m, size_t b) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < b; j++) {
            const int8_t* xq_ptr = xq + j; // pointer to first row at our current column
            const uint8_t* wq_ptr = wq + i * ((m + 7) / 8); // pointer to start of row of weight matrix
            int32_t acc = 0; // accumulate results in larger int to prevent overflow
            for (size_t k = 0; k < m; k += 8) {
                // multiply the sign bits of current wq byte with input activations
                size_t l = 0;
                do {
                    uint8_t mask = 0x80 >> l; // mask out unused bits
                    // weight bit 1 = -1, bit 0 = 1
                    if ((wq_ptr[k / 8] & mask) != 0)
                        acc -= (int32_t) (*xq_ptr);
                    else
                        acc += (int32_t) (*xq_ptr);
                    xq_ptr += b; // move on to the next row in matrix
                    l++;
                } while (k * 8 + l < m); // if number of colums is not multiple of 8, extra bits in last byte of each row will be unused
            }
            yq[i * b + j] = (int8_t) acc;
        }
    }
}


/**
 * @brief Compute gradients of weight and activations of matrix multiplication.
 *        This is used for bit matrices as well to update bit weights during
 *        the training loop.
 * 
 * @param dw          gradient of weights with full precision floats
 * @param dx          gradient of inputs
 * @param dy          gradient of loss
 * @param w           full precision weights of Bit matrix
 * @param x           input activations
 * @param out_dim     num rows in weight matrix
 * @param in_dim      num cols in weight matrix/rows in activation x
 * @param batch_size  num cols in matrix x
 */
void matmul_bkwd(float* dw, float* dx,
                 const float* dy, const float* w, const float* x,
                 size_t out_dim, size_t in_dim, size_t batch_size) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* dy_b = dy + b * out_dim;
        const float* x_b = x + b * in_dim;
        for (size_t j = 0; j < in_dim; j++) {
            for (size_t i = 0; i < out_dim; i++) {
                dw[i * in_dim + j] += dy_b[i] * x_b[j];
                dx[j] += w[i * in_dim + j] * dy_b[i];
            }
        }
    }
}


// Elementwise addition of 2 matrices of the same size.
void matadd_fwd(float* y, const float* m1, const float* m2, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * cols + j;
            y[idx] = m1[idx] + m2[idx];
        }
    }
}


// Compute gradient for weights and inputs for matrix addition.
void matadd_bkwd(float* dm1, float* dm2,
                 const float* dy, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            size_t idx = i * rows + j;
            dm1[idx] = dy[idx];
            dm2[idx] = dy[idx];
        }
    }
}


/**
 * @brief Kaiming initialization with variance of 1. This sets the weights such
 *        that the variance remains the same across every layers' outputs in the
 *        forward pass, addressing the vanishing gradient problem. This is used
 *        instead of Xavier initialization since GELU activation is used.
 *
 * @param x pointer to matrix
 * @param len length of matrix
 */
void mat_init_kaiming(float* x, size_t len) {
    for (size_t i = 0; i < len; i++) x[i] = std_norm();
}


/**
 * @brief Initialize the given matrix to random weights [0, 1).
 *
 * @param x pointer to matrix
 * @param len length of matrix
 */
void mat_init_rand(float* x, size_t len) {
    for (size_t i = 0; i < len; i++) x[i] = rand_float();
}


size_t bitmat_bytes(size_t in_dim, size_t out_dim) {
    return ((in_dim + 7) / 8) * out_dim;
}

#endif
