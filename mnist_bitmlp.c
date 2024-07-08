// Example training program to classify handwritten digits using BitMLP.

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "layers/mlp.h"
#include "utils/mnist.h"
#include "utils/optim.h"

#define BATCH_SIZE 64
#define EPOCHS 10

void zero_grad(bitmlp_t* model) {

}

void training_step(bitmlp_t* model, mnist_batch_t* batch) {
    // mlp_fwd(float *y, float *gelu, float *rms1, float *rms2, float *x2, const float *x1, const float *w1, const float *w2, const float *g1, const float *g2, int d, int h)
    // zero_grad(model);
    // mlp_bkwd(float *dx1, float *dx2, float *dw1, float *dw2, float *dg1, float *dg2, float *dy, const float *x2, const float *w2, const float *g2, const float *rms2, const float *gelu, const float *x1, const float *w1, const float *g1, const float *rms1, int d, int h);
    // update_weights();
}


int main() {
    // load training images from MNIST dataset
    mnist_dataset_t* trainset = mnist_init_dataset("MNIST_ORG/train-images.idx3-ubyte", "MNIST_ORG/train-labels.idx1-ubyte");
    if (trainset == NULL) {
        fprintf(stderr, "Cannot open train file!\n");
        return 1;
    }
    mnist_dataset_t* testset = mnist_init_dataset("MNIST_ORG/t10k-images.idx3-ubyte", "MNIST_ORG/t10k-labels.idx1-ubyte");
    if (testset == NULL) {
        mnist_free_dataset(trainset);
        fprintf(stderr, "Cannot open test file!\n");
        return 2;
    }

    int d = 512, h = 768;
    bitmlp_t* model = mlp_init(d, h, 1);

    mnist_batch_t* batch = (mnist_batch_t*) malloc(sizeof(mnist_batch_t));
    batch->size = BATCH_SIZE;
    batch->images = (mnist_image_t*) malloc(sizeof(mnist_image_t) * BATCH_SIZE);
    batch->labels = (uint8_t*) malloc(sizeof(uint8_t) * BATCH_SIZE);
    for (int i = 0; i < EPOCHS; i++) {
        while (mnist_get_next_batch(batch, trainset) == 0) {
            training_step(model, batch);
        }
        mnist_reset_dataset(trainset);
        while (mnist_get_next_batch(batch, testset) == 0) {
            validation_step(model, batch);
        }
        mnist_reset_dataset(testset);
    }

    mnist_free_dataset(trainset);
    mnist_free_dataset(testset);

    return 0;
}
