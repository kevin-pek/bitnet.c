#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <stdio.h>
#include "../utils/matrix.h"

#define SQRT_2_OVER_PI (sqrtf(M_2_PI))
#define A 0.044715f


// Apply softmax to input logits.
void softmax_fwd(float* probs, const float* logits, int d, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += expf(logits[b * d + i]);
        }

        for (int i = 0; i < d; i++) {
            probs[b * d + i] /= sum;
        }
    }
}


// Backpropagation for softmax activations.
void softmax_bkwd(float* dx, const float* y, const float* dy, int dim, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        float* dx_b = dx + b * dim;
        const float* y_b = y + b * dim;
        const float* dy_b = dy + b * dim;
        for (int i = 0; i < dim; i++) {
            dx_b[i] = 0;
            for (int j = 0; j < dim; j++)
                dx_b[i] += (i == j ? y_b[i] * (1 - y_b[i]) : -y_b[i] * y_b[j]) * dy_b[j];
        }
    }
}


// Activation function for feedforward layer in bitlinear network. This uses the
// tanh approximation of the GELU activation.
void gelu_fwd(float* y, const float *in, int dim, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < dim; i++) {
            int idx = b * dim + i;
            float x = in[idx];
            y[idx] = 0.5 * x * tanhf(SQRT_2_OVER_PI * (x + A * x * x * x));
        }
    }
}


// Backpropagation through GELU activation function.
// Parameters:
// dx array where computed gradients of loss with respect to inputs will be stored
// dy array containing loss with respect to output of GELU activation
// in input array that was passed to the activation function
// dim number of elements in the arrays
// batch_size
void gelu_bkwd(float* dx, const float* dy, const float* in, int dim, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < dim; i++) {
            int idx = b * dim + i;
            float x = in[idx];
            float tanh_out = tanhf(SQRT_2_OVER_PI * (x + A * x * x * x));
            float dtanh = 1 - tanh_out * tanh_out;
            float dgelu = 0.5 * ((1 + tanh_out) + x * dtanh * SQRT_2_OVER_PI * (1 + 3 * A * x * x));
            dx[idx] = dy[idx] * dgelu;
        }
    }
}

#endif
