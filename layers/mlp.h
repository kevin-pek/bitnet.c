#ifndef FFN_H
#define FFN_H

#include "activation.h"
#include "bitlinear.h"
#include <string.h>

// stores all we need to do backpropagation of a single bitlinear layer
typedef struct {
    float* x;     // input to bitlinear layer
    float* g;     // rmsnorm scaling weights
    float* w;     // bitlinear weight matrix
    float* rms;   // output of rmsnorm layer
    float* gelu;  // output of gelu activation
    float* dg;
    float* dw;
    int d;        // input dimension
    int h;        // hidden dimension
    int o;        // output dimension
    int b;        // batch size
} bitmlp_t;

/// @brief Initialise BitMLP weights with given dimensions.
/// @param d input dimension
/// @param h hidden dimension
/// @param o output dimension
/// @param b batch size
void mlp_init(bitmlp_t* mlp, int d, int h, int o, int b) {
    mlp->d = d;
    mlp->h = h;
    mlp->o = o;
    mlp->b = b;
    size_t elems = b * (2 * d * h + d * 4);
    float* arr = (float*) malloc(sizeof(float) * elems);
    if (arr == NULL) {
        fprintf(stderr, "Error allocating memory for MLP!");
        return;
    }

    float* arr_ptr = arr;
    mlp->x = arr_ptr;
    arr_ptr += d * b;
    mlp->g = arr_ptr;
    arr_ptr += d * b;
    mlp->w = arr_ptr;
    arr_ptr += d * h * b;
    mlp->y_rms = arr_ptr;
    arr_ptr += d * b;
    mlp->dg = arr_ptr;
    arr_ptr += d * b;
    mlp->dw = arr_ptr;
}


/// @brief MLP layer consisting of 2 BitLinear layers. Based on BitNet paper.
/// @param y    output array
/// @param gelu intermediate result of first bitlinear layer, input to GELU
/// @param rms1 store intermediate results of rmsnorm in first bitlinear
/// @param rms2 store intermediate results of rmsnorm in second bitlinear
/// @param x2   output of GELU, input to second bitlinear layer
/// @param x1   input matrix
/// @param w1   weight matrix of first bitlinear layer
/// @param w2   weight matrix of second bitlinear layer
/// @param g1 scaling factors for rmsnorm of first bitlinear layer
/// @param g2 scaling factors for rmsnorm of second bitlinear layer
/// @param d    input dimensions
/// @param h    hidden dimension
/// @param o    output dimension
/// @param b    batch size
void mlp_fwd(float* y, float* gelu, float* rms1, float* rms2, float* x2,
             const float* x1, const float* w1, const float* w2, const float* g1, const float* g2,
             int d, int h, int o, int b) {
    bitlinear_fwd(gelu, rms1, x1, w1, g1, d, h);
    gelu_fwd(x2, gelu, h);
    bitlinear_fwd(y, rms2, x2, w2, g2, h, o);
}


void mlp_bkwd(float* dy, float* dw1, float* dw2, float* dg1, float* dg2,
              const float* x2, const float* w2, const float* g2, const float* rms2,
              const float* gelu,
              const float* x1, const float* w1, const float* g1, const float* rms1,
              int d, int h) {
    bitlinear_bkwd(dy, dw2, dg2, dy, x2, w2, g2, rms2, h, d);
    gelu_bkwd(dy, dy, gelu, h);
    bitlinear_bkwd(dy, dw1, dg1, dy, x1, w1, g1, rms1, d, h);
}

#endif
