#ifndef LOSS_H
#define LOSS_H

#include <math.h>

/// @brief Compute mean loss of the predictions probabilities vs targets. This is
///        used for logging and is not needed for the backward pass.
float crossentropy_fwd(float* probs, uint32_t* targets, int v) {
    float loss = 0.0f;
    for (int i = 0; i < v; i++) {
        loss -= logf(probs[targets[i]]);
    }
    return loss / (float) v;
}


/// @brief Compute gradient of logits using gradient of loss, prediction probability
///        and target id. Gradient of loss is fixed as 1 / (batch_size * seq_len)
///        Gradient of logit is given by p_i - y_i.
void crossentropy_bkwd(float* dlogits,
            const float dloss, const float* probs, const uint32_t target,
            int v) {
    for (int i = 0; i < v; i++) {
        float y_i = i == target ? 1.0f : 0.0f;
        dlogits[i] = (probs[i] - y_i) * dloss;
    }
}

#endif
