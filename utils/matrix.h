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
 * @param y output matrix
 * @param x input matrix
 * @param w weight matrix
 * @param n output dims, number of rows in weight matrix, dimensionality of output matrix
 * @param m input dims, number of cols in weight matrix, rows in x
 * @param b batch size, num cols in matrix x
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
 *        We assume that m is multiple of 8 which is fine since dims are
 *        large powers of 2 by convention.
 *
 * @param yq 8 bit quantised outputs
 * @param xq 8 bit quantised inputs
 * @param wq 1 bit quantised weight matrix
 * @param n  output dims
 * @param m  input dims, must be a multiple of 8
 * @param b  batch size, num cols in matrix x
 */
void bitmatmul_fwd(int8_t* yq, const int8_t* xq, const uint8_t* wq,
                   size_t n, size_t m, size_t b) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < b; j++) {
            #ifdef DEBUG
                printf("Elem (%zu, %zu):\n", i, j);
            #endif
            const int8_t* xq_ptr = xq + j; // pointer to first row at our current column
            const uint8_t* wq_ptr = wq + i * ((m + 7) / 8); // pointer to start of row of weight matrix
            int32_t acc = 0; // accumulate results in larger int to prevent overflow
            for (size_t k = 0; k < m; k += 8) {
                #ifdef DEBUG
                    printf("Byte: ");
                    printbin8(wq_ptr[k / 8]);
                    printf("\n");
                #endif
                // multiply the sign bits of current wq byte with input activations
                size_t l = 0;
                do {
                    uint8_t mask = 0x80 >> l; // mask out the bits that are not in use
                    // bit of 1 represents negative number
                    if ((wq_ptr[k / 8] & mask) != 0)
                        acc -= (int32_t) (*xq_ptr);
                    else
                        acc += (int32_t) (*xq_ptr);
                    xq_ptr += b; // move on to the next row in matrix
                    l++;
                } while (k * 8 + l < m); // this version should support any number of columns in the weight matrix
            }
            yq[i * b + j] = (int8_t) acc;
        }
    }
}


/**
 * @brief Compute gradients of weight and activations of matrix multiplication.
 * 
 * @param dw gradient of weights with full precision floats
 * @param dx gradient of inputs to BitLinear layer
 * @param dy gradient of loss function wrt output of bitlinear layer
 * @param w  full precision weights of BitLinear layer
 * @param x  input activations to bitlinear layer
 * @param n  output dim, num rows in weight matrix
 * @param m  input dim, num cols in weight matrix/rows in activation x
 * @param b  batch size, num cols in matrix x
 */
void matmul_bkwd(float* dw, float* dx,
                 const float* dy, const float* w, const float* x,
                 size_t n, size_t m, size_t b) {
#ifdef DEBUG
    printf("Matmul backprop:\n");
    printf("dW:");
    print_mat(dw, n, m);
    printf("dx:");
    print_mat(dx, m, b);
    printf("dy:");
    print_mat(dy, b, m);
#endif
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < m; k++) {
            for (size_t j = 0; j < b; j++) {
                // dw_ik = sum(dL / dy_ij * dx_kj), j in [1, b]
                dw[i * m + k] += dy[i * b + j] * x[k * b + j];
                // dx_kj = sum(dL / dy_ij * dy_ij / dx_ij), j in [1, b]
                dx[k * b + j] += w[i * m + k] * dy[i * b + j];
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
 *        instead of Xavier initialization since we are using GELU activations
 *        in our architecture.
 * @param x pointer to matrix
 * @param len length of matrix
 */
void mat_init_kaiming(float* x, size_t len) {
    for (size_t i = 0; i < len; i++) x[i] = std_norm();
}


/**
 * @brief Initialize the given matrix to random weights [0, 1).
 * @param x pointer to matrix
 * @param len length of matrix
 */
void mat_init_rand(float* x, size_t len) {
    for (size_t i = 0; i < len; i++) x[i] = rand_float();
}

#endif
