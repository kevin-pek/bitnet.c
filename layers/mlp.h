#ifndef FFN_H
#define FFN_H

#include "activation.h"
#include "bitlinear.h"
#include <stdint.h>
#include <string.h>

// stores all we need to do backpropagation of a single bitlinear layer
typedef struct {
    float* dx; // gradient of inputs to mlp layer
    bitlinear_mem_t lin1;
    float* x_gelu;    // input to gelu activation
    float* dy_hidden; // gradient of loss for hidden layer
    bitlinear_mem_t lin2;
    float* logits;  // output of second bitlinear layer
    float* probs;   // output probabilities
} bitmlp_mem_t;

typedef struct {
    bitlinear_grad_t lin1;
    bitlinear_grad_t lin2;
} bitmlp_grad_t;

typedef struct {
    bitlinear_t lin1;
    bitlinear_t lin2;
} bitmlp_t;


// Allocate memory for BitMLP parameters and assigning pointers for inference.
int mlp_alloc(bitmlp_t* mlp, size_t in_dim, size_t hidden_dim, size_t out_dim, size_t batch_size) {
    float* float_params = (float*) calloc(batch_size * (in_dim + hidden_dim), sizeof(float));
    if (float_params == NULL) {
        return 1;
    }
    float* float_ptr = float_params;
    mlp->lin1.g = float_ptr;
    float_ptr += batch_size * in_dim;
    mlp->lin2.g = float_ptr;

    // verify this is correct size
    uint8_t* uint8_params = (uint8_t*) calloc(batch_size * (((in_dim + 7) / 8) * hidden_dim + ((hidden_dim + 7) / 8) * out_dim), sizeof(uint8_t));
    if (uint8_params == NULL) {
        free(float_params);
        return 2;
    }
    uint8_t* uint8_ptr = uint8_params;
    mlp->lin1.wq = uint8_ptr;
    uint8_ptr += batch_size * ((in_dim + 7) / 8) * hidden_dim;
    mlp->lin2.wq = uint8_ptr;

    int8_t* int8_params = (int8_t*) calloc(batch_size * 2 * in_dim * out_dim, sizeof(int8_t));
    if (int8_params == NULL) {
        free(float_params);
        free(uint8_params);
        return 3;
    }
    int8_t* int8_ptr = int8_params;
    mlp->lin1.yq = int8_ptr;
    int8_ptr += batch_size * in_dim;
    mlp->lin1.xq = int8_ptr;
    int8_ptr += batch_size * in_dim;
    mlp->lin2.yq = int8_ptr;
    int8_ptr += batch_size * out_dim;
    mlp->lin2.xq = int8_ptr;

    return 0;
}


/**
 * @brief Initialise weights for MLP using 1 contiguous block of memory.
 *
 * @param d input dimension
 * @param h hidden dimension
 * @param o output dimension
 */
void mlp_init(bitmlp_t* mlp, bitmlp_mem_t* mem, size_t d, size_t h, size_t o) {
    bitlinear_init(&mlp->lin1, &mem->lin1, d, h);
    bitlinear_init(&mlp->lin2, &mem->lin2, h, o);
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
#ifdef DEBUG
    printf("BitMLP inputs:\n");
    print_mat(x1, b, d);
#endif
    bitlinear_fwd(gelu, rms1, x1, w1, g1, wq1, yq1, xq1, d, h, b);
    gelu_fwd(x2, gelu, h, b);
    bitlinear_fwd(y, rms2, x2, w2, g2, wq2, yq2, xq2, h, o, b);
#ifdef DEBUG
    printf("BitMLP outputs:\n");
    print_mat(y, b, o);
#endif
}


void mlp_bkwd(float* dx, float* dw1, float* dw2, float* dg1, float* dg2, float* dy1,
              const float* dy2,
              const float* x2, const float* w2, const float* g2, const float* rms2,
              const float* x_gelu,
              const float* x1, const float* w1, const float* g1, const float* rms1,
              size_t d, size_t h, size_t o, size_t b) {
    bitlinear_bkwd(dy1, dw2, dg2, dy2, x2, w2, g2, rms2, h, o, b);
    gelu_bkwd(dy1, dy1, x_gelu, h, b);
    bitlinear_bkwd(dx, dw1, dg1, dy1, x1, w1, g1, rms1, d, h, b);
}

#endif
