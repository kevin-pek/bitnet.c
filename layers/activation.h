#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <stdio.h>
#include "../utils/logging.h"
#include "../utils/loss.h"
#include "../utils/matrix.h"

#define SQRT_2_OVER_PI (sqrtf(2.0f / PI))
#define A 0.044715f


/**
 * @brief Apply softmax to input logits.
 *
 * @param probs  output probabilities
 * @param logits input logits
 * @param dim    number of dimensions, same as number of labels
 * @param batch_size
 */
void softmax_fwd(float* probs, const float* logits, size_t dim, size_t batch_size) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* logits_b = logits + b * dim;
        float* probs_b = probs + b * dim;

        float lse = log_sum_exp(logits_b, dim);
        for (size_t i = 0; i < dim; i++)
            probs_b[i] = expf(logits_b[i] - lse);
    }
}


/**
 * @brief Backpropagation step to compute the gradient of the loss w.r.t probabilities.
 *        Gradient is given by p_i - y_i, where y_i = 1 if correct class, 0 otherwise.
 *
 * @param dloss      gradient of softmax inputs w.r.t loss
 * @param probs      output probabilities
 * @param targets    ground truth labels
 * @param dim        number of dimensions, same as number of labels
 * @param batch_size
 */
void softmax_bkwd(float* dloss, const float* probs, const uint32_t* targets, size_t dim, size_t batch_size) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* probs_b = probs + b * dim;
        for (size_t i = 0; i < dim; i++)
            dloss[i] += probs_b[i] - (i == targets[b] ? 1.0f : 0.0f);
    }
}


// Activation function for feedforward layer in bitlinear network. This uses the
// tanh approximation of the GELU activation.
void gelu_fwd(float* y, const float *in, size_t dim, size_t batch_size) {
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < dim; i++) {
            int idx = b * dim + i;
            float x = in[idx];
            y[idx] = 0.5 * x * tanhf(SQRT_2_OVER_PI * (x + A * x * x * x));
        }
    }
}


/**
 * @brief Backpropagation through GELU activation function.
 * 
 * @param dx array where computed gradients of loss with respect to inputs will be stored
 * @param dy array containing loss with respect to output of GELU activation
 * @param in input array that was passed to the activation function
 * @param dim number of elements in the arrays
 * @param batch_size
 */
void gelu_bkwd(float* dx, const float* dy, const float* in, size_t dim, size_t batch_size) {
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < dim; i++) {
            size_t idx = b * dim + i;
            float x = in[idx];
            float tanh_out = tanhf(SQRT_2_OVER_PI * (x + A * x * x * x));
            float dtanh = 1 - tanh_out * tanh_out;
            float dgelu = 0.5 * ((1 + tanh_out) + x * dtanh * SQRT_2_OVER_PI * (1 + 3 * A * x * x));
            dx[i] += dy[i] * dgelu;
        }
    }
}

#endif
