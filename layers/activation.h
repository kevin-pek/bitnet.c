#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <stdio.h>
#include "../utils/matrix.h"

// Scales the values within x (QK^T) by the sqrt of the model dimension, before the
// softmax is applied to the individual elements.
void softmax_scaled(float *x, int size) {
    float norm = sqrtf(size);
    float max_val = x[0] / norm;
    for (int i = 1; i < size; i++) {
        max_val = fmaxf(x[i] / norm, max_val);
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void softmax_fwd(float* probs, float* logits, int d) {}


/// @brief Backpropagation for softmax activations.
void softmax_bkwd(float* out, float* grad_out, float* grad_in, int size) {
    for (int i = 0; i < size; i++) {
        grad_in[i] = 0;
        for (int j = 0; j < size; j++)
            grad_in[i] += (i == j ? out[i] * (1 - out[i]) : -out[i] * out[j]) * grad_out[j];
    }
}


float sigmoid(float x) {
    return (1 / (1 + expf(-x)));
}


void sigmoid_bkwd(float* out, float* grad_out, float* grad_in, int size) {
    for (int i = 0; i < size; i++)
        grad_in[i] = grad_out[i] * out[i] * (1 - out[i]);
}

#define SQRT_2_OVER_PI (sqrtf(M_2_PI))
#define A 0.044715f

/// @brief Activation function for feedforward layer in bitlinear network. This
///        uses the tanh approximation of the GELU activation.
void gelu_fwd(float* out, const float *in, int size) {
    for (int i = 0; i < size; i++) {
        float x = in[i];
        out[i] = 0.5 * x * tanhf(SQRT_2_OVER_PI * (x + A * x * x * x));
    }
}


/// @brief Backpropagation through GELU activation function.
/// @param grad_in array where computed gradients of loss with respect to inputs will be stored
/// @param grad_out array containing loss with respect to output of GELU activation
/// @param in input array that was passed to the activation function
/// @param size number of elements in the arrays
void gelu_bkwd(float* grad_in, const float* grad_out, const float* in, int size) {
	for (int i = 0; i < size; i++) {
		float x = in[i];
		float tanh_out = tanhf(SQRT_2_OVER_PI * (x + A * x * x * x));
        float grad_tanh = 1 - tanh_out * tanh_out;
		float grad_gelu = 0.5 * ((1 + tanh_out) + x * grad_tanh * SQRT_2_OVER_PI * (1 + 3 * A * x * x));
		grad_in[i] = grad_out[i] * grad_gelu;
	}
}

#endif
