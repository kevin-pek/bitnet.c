#ifndef FFN_H
#define FFN_H

#include "activation.h"
#include "batchnorm.h"
#include "bitlinear.h"
#include <stdint.h>
#include <string.h>

// cached values for backpropagation
typedef struct {
    float* dx; // gradient of inputs to mlp layer
    bitlinear_mem_t lin1;
    float* dx_gelu; // gradient of loss wrt output of GELU
    float* dy_gelu; // gradient of loss wrt output of GELU
    bitlinear_mem_t lin2;
    float* dy; // gradient of loss wrt outputs of the mlp
} bitmlp_mem_t;

typedef struct {
    bitlinear_grad_t lin1;
    bitlinear_grad_t lin2;
} bitmlp_grad_t;

typedef struct {
    bitlinear_t lin1;
    bitlinear_t lin2;
} bitmlp_t;


/**
 * @brief Initialise training weights for MLP.
 */
void mlp_init(bitmlp_t* mlp, bitmlp_mem_t* mem, size_t in_dim, size_t hidden_dim, size_t out_dim) {
    bitlinear_init(&mlp->lin1, &mem->lin1, in_dim, hidden_dim);
    bitlinear_init(&mlp->lin2, &mem->lin2, hidden_dim, out_dim);
}


/**
 * @brief MLP layer consisting of 2 BitLinear layers. Based on FFN layer from the BitNet paper.
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


/**
 * @brief Backpropagation for MLP implementation based on the BitNet paper with GELU activation.
 *
 * @param dx        gradient of inputs
 * @param dw1       gradient of weights for first bitlinear layer
 * @param dw2       gradient of weights for second bitlinear layer
 * @param dg1       gradient of rms scaling factors for first bitlinear layer
 * @param dy_rms1   gradient of loss wrt first rms outputs
 * @param dg2       gradient of rms scaling factors for second bitlinear layer
 * @param dy_rms2   gradient of loss wrt second rms outputs
 * @param dy_gelu   gradient of loss wrt GELU outputs
 * @param dx_gelu   gradient of loss wrt GELU inputs, output of first bitlinear layer
 * @param dy2       gradient of loss wrt outputs of entire MLP
 */
void mlp_bkwd(float* dx, float* dw1, float* dw2, float* dg1, float* dy_rms1, float* dg2, float* dy_rms2,
              float* dy_gelu, float* dx_gelu, const float* dy2,
              const float* x2, const float* w2, const float* g2, const float* rms2,
              const float* y1,
              const float* x1, const float* w1, const float* g1, const float* rms1,
              size_t d, size_t h, size_t o, size_t b) {
    bitlinear_bkwd(dy_gelu, dw2, dg2, dy_rms2, dy2, x2, w2, g2, rms2, h, o, b);
    gelu_bkwd(dx_gelu, dy_gelu, y1, h, b);
    batchnorm(dx_gelu, h, b);

    bitlinear_bkwd(dx, dw1, dg1, dy_rms1, dx_gelu, x1, w1, g1, rms1, d, h, b);
}

#endif
