#ifndef FFN_H
#define FFN_H

#include "activation.h"
#include "bitlinear.h"
#include <stdint.h>
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


void mlp_alloc(bitmlp_t* mlp, size_t in_dim, size_t out_dim);


/**
 * @brief Initialise weights for MLP using 1 contiguous block of memory.
 *
 * @param d input dimension
 * @param h hidden dimension
 * @param o output dimension
 * @param b batch size
 */
void mlp_init(bitmlp_t* mlp, bitmlp_mem_t* mem, size_t d, size_t h, size_t o, size_t b) {
    bitlinear_init(&mlp->lin1, &mem->lin1, d, h, b);
    bitlinear_init(&mlp->lin2, &mem->lin2, h, o, b);
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
 * @param g1   scaling factors for rmsnorm of first bitlinear layer
 * @param g2   scaling factors for rmsnorm of second bitlinear layer
 * @param wq1
 * @param yq1
 * @param xq1
 * @param wq2
 * @param yq2
 * @param xq2
 * @param d    input dimensions
 * @param h    hidden dimension
 * @param o    output dimension
 * @param b    batch size
 */
void mlp_fwd(float* y, float* gelu, float* rms1, float* rms2, float* x2,
             const float* x1, const float* w1, const float* w2, const float* g1, const float* g2,
             uint8_t* wq1, int8_t* yq1, int8_t* xq1, uint8_t* wq2, int8_t* yq2, int8_t* xq2,
             size_t d, size_t h, size_t o, size_t b) {
    bitlinear_fwd(gelu, rms1, x1, w1, g1, wq1, yq1, xq1, d, h, b);
    gelu_fwd(x2, gelu, h, b);
    bitlinear_fwd(y, rms2, x2, w2, g2, wq2, yq2, xq2, h, o, b);
}


void mlp_bkwd(float* dx, float* dw1, float* dw2, float* dg1, float* dg2, const float* dy,
              const float* x2, const float* w2, const float* g2, const float* rms2,
              const float* x_gelu,
              const float* x1, const float* w1, const float* g1, const float* rms1,
              size_t d, size_t h, size_t o, size_t b) {
    bitlinear_bkwd(dx, dw2, dg2, dy, x2, w2, g2, rms2, h, o, b);
    gelu_bkwd(dx, dy, x_gelu, h, b);
    bitlinear_bkwd(dx, dw1, dg1, dy, x1, w1, g1, rms1, d, h, b);
}

#endif
