#ifndef LOSS_H
#define LOSS_H

#include <math.h>


/**
 * @brief Compute loss of the predictions probabilities vs targets with batch norm.
 *        This is used for logging and is not needed for the backward pass.
 * 
 * @param probs output probabilities
 * @param targets list of class labels for current batch
 * @param n_labels output dimension of model, corresponds to the number of classes
 * @param batch_size
 * @return crossentropy loss value
 */
float crossentropy_fwd(const float* probs, const uint32_t* targets, size_t n_labels, size_t batch_size) {
    float loss = 0.0f;
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < n_labels; i++) {
            loss -= logf(probs[targets[i]]);
        }
    }
    return loss / (n_labels * batch_size);
}


/**
 * @brief Compute gradient of logits using gradient of loss, prediction probability
 *        and target id. Gradient of logit is given by p_i - y_i.
 * 
 * @param dloss gradients of inputs wrt loss
 * @param probs output probabilities
 * @param targets list of class labels for current batch
 * @param n_labels output dimension of model, corresponds to the number of classes
 * @param batch_size
 */
void crossentropy_bkwd(float* dloss, const float* probs, const uint32_t* targets, size_t n_labels, size_t batch_size) {
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < n_labels; i++) {
            float y_i = i == targets[b] ? 1.0f : 0.0f;
            dloss[i] = probs[i] - y_i;
        }
    }
}

#endif
