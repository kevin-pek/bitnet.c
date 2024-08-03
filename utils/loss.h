#ifndef LOSS_H
#define LOSS_H

#include <math.h>
#include <stdint.h>


// Compute log-sum-exp of a float array.
float log_sum_exp(const float* logits, size_t dim) {
    float max_logit = logits[0];
    for (size_t i = 1; i < dim; i++)
        if (logits[i] > max_logit)
            max_logit = logits[i];

    float sum_exp = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }

    return max_logit + logf(sum_exp);
}


/**
 * @brief Compute loss of the predictions probabilities vs targets with batch norm.
 *        This is used for logging and is not needed for the backward pass.
 * 
 * @param logits output logits
 * @param targets list of class labels for current batch
 * @param n_labels output dimension of model, corresponds to the number of classes
 * @param batch_size
 * @return crossentropy loss value normalized across the batch
 */
float cross_entropy_loss(const float* logits, const uint32_t* targets, size_t n_labels, size_t batch_size) {
    float loss = 0.0f;
    for (size_t b = 0; b < batch_size; b++) {
        const float* logits_b = logits + b * n_labels;

        float lse = log_sum_exp(logits_b, n_labels);
        loss -= logits_b[targets[b]] - lse;
    }
    return loss / batch_size;
}

#endif
