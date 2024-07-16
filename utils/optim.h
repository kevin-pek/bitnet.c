// Implementation of AdamW optimizer

#ifndef OPTIM_H
#define OPTIM_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Keep track of moment estimates for each parameter we have in our model.
typedef struct {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float* m;
    float* v;
    size_t n_params;
} adamw_t;


int adamw_alloc_init(adamw_t* optim, float lr, float beta1, float beta2, float eps, float weight_decay, size_t n_params) {
    optim->m = (float*) malloc(sizeof(float) * n_params);
    if (optim->m == NULL) {
        printf("Failed to allocate memory for first moment estimate!");
        return 1;
    }

    optim->v = (float*) malloc(sizeof(float) * n_params);
    if (optim->m == NULL) {
        printf("Failed to allocate memory for first moment estimate!");
        free(optim->m);
        return 2;
    }

    optim->lr = lr;
    optim->beta1 = beta1;
    optim->beta2 = beta2;
    optim->eps = eps;
    optim->weight_decay = weight_decay;
    optim->n_params = n_params;

    return 0;
}


void adamw_update(adamw_t* optim, float* params, float* grads, size_t len, int t) {
    for (int i = 0; i < optim->n_params; i++) {
        optim->m[i] = optim->beta1 * optim->m[i] + (1 - optim->beta1) * grads[i];
        optim->v[i] = optim->beta2 * optim->v[i] + (1 - optim->beta2) * grads[i] * grads[i];
        float m_hat = optim->m[i] / (1 - powf(optim->beta1, (float) t)); // bias-corrected estimates using epoch, starting from 1
        float v_hat = optim->v[i] / (1 - powf(optim->beta2, (float) t));
        params[i] -= optim->lr * ((m_hat / (sqrtf(v_hat) + optim->eps)) - optim->weight_decay * params[i]);
    }
}


void adamw_free(adamw_t* optim) {
    free(optim->m);
    free(optim->v);
}

#endif
