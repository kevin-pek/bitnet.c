/**
 * @file Implementation of AdamW optimizer.
 */

#ifndef OPTIM_H
#define OPTIM_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float* m; // first moment estimates
    float* v; // second moment estimates
    size_t n_params; // number of parameters we need to update
} adamw_t;


/**
 * @brief Initialize training parameters and allocate memory for the AdamW optimizer.
 * 
 * @param optim pointer to optimizer
 * @param n_params number of model parameters that we need to update
 * @return 0 on success, non-zero on failure
 */
int adamw_alloc(adamw_t* optim, size_t n_params) {
    optim->n_params = n_params;
    optim->m = (float*) calloc(n_params, sizeof(float));
    if (optim->m == NULL) {
        fprintf(stderr, "Failed to allocate memory for first moment estimate!");
        return 1;
    }

    optim->v = (float*) calloc(n_params, sizeof(float));
    if (optim->v == NULL) {
        fprintf(stderr, "Failed to allocate memory for second moment estimate!");
        free(optim->m);
        return 2;
    }

    return 0;
}


/**
 * @brief Initialize parameters for optimizer.
 * 
 * @param optim pointer to optimizer
 * @param lr learning rate
 * @param beta1
 * @param beta2
 * @param eps
 * @param weight_decay
 */
void adamw_init(adamw_t* optim, float lr, float beta1, float beta2, float eps, float weight_decay) {
    optim->lr = lr;
    optim->beta1 = beta1;
    optim->beta2 = beta2;
    optim->eps = eps;
    optim->weight_decay = weight_decay;
}


/**
 * @brief Update model parameters.
 * 
 * @param optim pointer to the optimizer
 * @param params model parameters to be updated
 * @param grads gradients of the loss with respect to the model parameters
 * @param t current time step (epoch number)
 */
void adamw_update(adamw_t* optim, float* params, float* grads, int t) {
    for (int i = 0; i < optim->n_params; i++) {
        optim->m[i] = optim->beta1 * optim->m[i] + (1.0f - optim->beta1) * grads[i];
        optim->v[i] = optim->beta2 * optim->v[i] + (1.0f - optim->beta2) * grads[i] * grads[i];
        float m_hat = optim->m[i] / (1.0f - powf(optim->beta1, (float) t)); // bias-corrected estimates using epoch, starting from 1
        float v_hat = optim->v[i] / (1.0f - powf(optim->beta2, (float) t));
        params[i] -= optim->lr * ((m_hat / (sqrtf(v_hat) + optim->eps)) + optim->weight_decay * params[i]);
    }
}


void adamw_free(adamw_t* optim) {
    free(optim->m);
    free(optim->v);
}

#endif
