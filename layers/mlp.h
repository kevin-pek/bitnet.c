#ifndef FFN_H
#define FFN_H

#include "activation.h"
#include "bitlinear.h"
#include <string.h>

// stores all we need to do backpropagation of a single bitlinear layer
typedef struct {
    bitlinear_mem_t lin1;
    float* x_gelu;  // input to gelu activation
    bitlinear_mem_t lin2;
    float* logits;  // output of second bitlinear layer
    float* probs;   // output probabilities
} bitmlp_mem_t;

typedef struct {
    bitlinear_grad_t lin1;
    bitlinear_grad_t lin2;
    float* dy;
} bitmlp_grad_t;

typedef struct {
    bitlinear_t lin1;
    bitlinear_t lin2;
} bitmlp_t;


void mlp_alloc(bitmlp_t* mlp, )


/// @brief Initialise weights for MLP using 1 contiguous block of memory.
/// @param d input dimension
/// @param h hidden dimension
/// @param o output dimension
/// @param b batch size
void mlp_init(bitmlp_t* mlp, bitmlp_mem_t* mem, float* arr, int d, int h, int o, int b) {
    size_t bitlin1_params = b * d * (2 * h + 4);
    size_t gelu_params = b * d;
    size_t bitlin2_params = b * h * (2 * o + 4);
    size_t out_params = b * o;
    bitlinear_init(&mlp->lin1, &mem->lin1, arr, d, h, b);
    mlp->x_gelu = arr + bitlin1_params;
    bitlinear_init(&mlp->lin2, arr + bitlin1_params + gelu_params, h, o, b);
    mlp->logits = arr + bitlin1_params + bitlin2_params + gelu_params;
    memset(mlp->logits, 0, sizeof(float) * out_params);
    mlp->probs = arr + bitlin1_params + bitlin2_params + gelu_params +  out_params;
    memset(mlp->probs, 0, sizeof(float) * out_params);
}


void mlp_train_init(bitmlp_mem_t* mem, bitlinear_grad_t* grads, size_t batch_size) {

}


/**
 * @brief MLP layer consisting of 2 BitLinear layers. Based on architecture from the BitNet paper.
 * 
 * @param y    output array
 * @param gelu intermediate result of first bitlinear layer, input to GELU
 * @param rms1 store intermediate results of rmsnorm in first bitlinear
 * @param rms2 store intermediate results of rmsnorm in second bitlinear
 * @param x2   output of GELU, input to second bitlinear layer
 * @param x1   input matrix
 * @param w1   weight matrix of first bitlinear layer
 * @param w2   weight matrix of second bitlinear layer
 * @param g1 scaling factors for rmsnorm of first bitlinear layer
 * @param g2 scaling factors for rmsnorm of second bitlinear layer
 * @param d    input dimensions
 * @param h    hidden dimension
 * @param o    output dimension
 * @param b    batch size
 */
void mlp_fwd(float* y, float* gelu, float* rms1, float* rms2, float* x2,
             const float* x1, const float* w1, const float* w2, const float* g1, const float* g2,
             int d, int h, int o, int b) {
    bitlinear_fwd(gelu, rms1, x1, w1, g1, d, h, b);
    gelu_fwd(x2, gelu, h, b);
    bitlinear_fwd(y, rms2, x2, w2, g2, h, o, b);
}


void mlp_bkwd(float* dx, float* dw1, float* dw2, float* dg1, float* dg2, const float* dy,
              const float* x2, const float* w2, const float* g2, const float* rms2,
              const float* x_gelu,
              const float* x1, const float* w1, const float* g1, const float* rms1,
              int d, int h, int o, int b) {
    bitlinear_bkwd(dx, dw2, dg2, dy, x2, w2, g2, rms2, h, o, b);
    gelu_bkwd(dx, dy, x_gelu, h, b);
    bitlinear_bkwd(dx, dw1, dg1, dy, x1, w1, g1, rms1, d, h, b);
}

#endif
