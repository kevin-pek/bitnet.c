// Example training program to classify handwritten digits using BitMLP.

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "layers/mlp.h"
#include "utils/mnist.h"
#include "utils/optim.h"

#define BATCH_SIZE 64
#define EPOCHS 10
#define LR 1e-4

void zero_grad(bitmlp_t* model) {
    memset(model->dg, 0, model->d);
    memset(model->dw, 0, model->d * model->h);
}

void training_step(float* x, bitmlp_t* model, mnist_batch_t* batch) {
    int d = batch->size * model->d;
    for (int i = 0; i < batch->size; i++) {
        for (int j = 0; j < d; j++) {
            int idx = i * d + j;
            x[idx] = (float) batch->images[i].pixels[j];
        }
    }
    float* y = (float*) malloc(sizeof(float) * batch->size * d);

    mlp_fwd(
        y,
        model->gelu,
        model->rms,
        model->rms + d,
        x + d,
        x,
        model->w,
        model->w + d,
        model->g,
        model->g + d,
        model->d,
        model->h,
        model->o,
        batch->size
    );
    zero_grad(model);
    mlp_bkwd(
        dx,
        model->dw,
        model->dw + d,
        model->dg,
        model->dg + d,
        model->gelu,
        model->w + d,
        model->g + d,
        model->rms + d,
        model->x,
        model->w,
        model->g,
        model->rms,
        model->d,
        model->h
    );
}


int main() {
    // load training images from MNIST dataset
    mnist_dataset_t* trainset = mnist_init_dataset(
        "MNIST_ORG/train-images.idx3-ubyte",
        "MNIST_ORG/train-labels.idx1-ubyte"
    );
    if (trainset == NULL) {
        fprintf(stderr, "Cannot open train file!\n");
        return 1;
    }

    mnist_dataset_t* testset = mnist_init_dataset(
        "MNIST_ORG/t10k-images.idx3-ubyte",
        "MNIST_ORG/t10k-labels.idx1-ubyte"
    );
    if (testset == NULL) {
        mnist_free_dataset(trainset);
        fprintf(stderr, "Cannot open test file!\n");
        return 2;
    }

    int h = 512;
    bitmlp_t model;
    mlp_init(&model, MNIST_IMAGE_SIZE, h, MNIST_LABELS, BATCH_SIZE);
    mnist_batch_t batch = {
        .size = BATCH_SIZE,
        .images = (mnist_image_t*) malloc(sizeof(mnist_image_t) * BATCH_SIZE),
        .labels = (uint8_t*) malloc(sizeof(uint8_t) * BATCH_SIZE)
    };

    float* x = (float*) malloc(sizeof(float) * batch.size * MNIST_IMAGE_SIZE);
    for (int i = 0; i < EPOCHS; i++) {
        while (mnist_get_next_batch(&batch, trainset) == 0) {
            training_step(x, &model, &batch);
            update(model, LR, beta1, beta2, EPS, weight_decay, MNIST_IMAGE_SIZE);
        }
        mnist_reset_dataset(trainset);
    }

    while (mnist_get_next_batch(&batch, testset) == 0) {
        validation_step(&model, &batch);
    }

    mnist_free_dataset(trainset);
    mnist_free_dataset(testset);

    return 0;
}
