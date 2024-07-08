#ifndef FFN_H
#define FFN_H

#include "activation.h"
#include "bitlinear.h"
#include <string.h>

// stores all we need to do backpropagation of a single bitlinear layer
typedef struct {
    float* x;     // input to bitlinear layer
    float* g;     // rmsnorm scaling weights
    float* w;     // weight matrix
    float* y_rms; // output of rmsnorm layer
    float* dg;
    float* dw;
    int d;
    int h;
    int b;
} bitmlp_t;

bitmlp_t* mlp_init(int d, int h, int b) {
    bitmlp_t* dbit = (bitmlp_t*) malloc(sizeof(bitmlp_t));
    dbit->d = d;
    dbit->h = h;
    dbit->b = b;
    size_t elems = b * (2 * d * h + d * 4);
    float* arr = (float*) malloc(sizeof(float) * elems);
    float* arr_ptr = arr;
    dbit->x = arr_ptr;
    arr_ptr += d * b;
    dbit->g = arr_ptr;
    arr_ptr += d * b;
    dbit->w = arr_ptr;
    arr_ptr += d * h * b;
    dbit->y_rms = arr_ptr;
    arr_ptr += d * b;
    dbit->dg = arr_ptr;
    arr_ptr += d * b;
    dbit->dw = arr_ptr;
    return dbit;
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
void mlp_fwd(float* y, float* gelu, float* rms1, float* rms2, float* x2,
             const float* x1, const float* w1, const float* w2, const float* g1, const float* g2,
             int d, int h) {
    bitlinear_fwd(gelu, rms1, x1, w1, g1, d, h);
    gelu_fwd(x2, gelu, h);
    bitlinear_fwd(y, rms2, x2, w2, g2, h, d);
}


void mlp_bkwd(float* dx1, float* dx2, float* dw1, float* dw2, float* dg1, float* dg2,
              float* dy,
              const float* x2, const float* w2, const float* g2, const float* rms2,
              const float* gelu,
              const float* x1, const float* w1, const float* g1, const float* rms1,
              int d, int h) {
    bitlinear_bkwd(dx2, dw2, dg2, dy, x2, w2, g2, rms2, h, d);
    gelu_bkwd(dy, dx2, gelu, h);
    bitlinear_bkwd(dx1, dw1, dg1, dy, x1, w1, g1, rms1, d, h);
}

#endif
