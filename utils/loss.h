#ifndef LOSS_H
#define LOSS_H

#include <math.h>


// Compute loss of the predictions probabilities vs targets with batch norm.
// This is used for logging and is not needed for the backward pass.
float crossentropy_fwd(float* probs, uint32_t* targets, int n_labels, int batch_size) {
    float loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n_labels; i++) {
            loss -= logf(probs[targets[i]]);
        }
    }
    return loss / (n_labels * batch_size);
}


// Compute gradient of logits using gradient of loss, prediction probability
// and target id. Gradient of loss is fixed as 1 / (batch_size * seq_len)
// Gradient of logit is given by p_i - y_i.
void crossentropy_bkwd(float* dloss,
            /* const float* dlosses, */ const float* probs, const uint32_t* targets,
            int n_labels, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n_labels; i++) {
            float y_i = i == targets[b] ? 1.0f : 0.0f;
            dloss[i] = probs[i] - y_i;
            // dlogits[i] = (probs[i] - y_i) * dlosses[b];
        }
    }
}

#endif
