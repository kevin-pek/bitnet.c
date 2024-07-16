// Example training program to classify handwritten digits using BitMLP.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "layers/activation.h"
#include "layers/bitlinear.h"
#include "layers/mlp.h"
#include "utils/loss.h"
#include "utils/mnist.h"
#include "utils/optim.h"

#define BATCH_SIZE 64
#define HIDDEN_SIZE 512
#define EPOCHS 10
#define LR 1e-4
#define EPS 1e-8
#define BETA1 0.9f
#define BETA2 0.999f
#define WEIGHT_DECAY 1e-2f


typedef struct {
    bitmlp_mem_t* params;
    int d; // input dims
    int h; // hidden dims
    int o; // output dims
} bitmlp_config_t;


typedef struct {
    int correct;
    int wrong;
} classifier_metrics_t;


int save_weights(bitmlp_config_t* model, const char* filepath) {
    FILE* fp = fopen(filepath, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error occurred when opening the file pointer!");
        return 1;
    }

    // write magic numbers to store model information
    fwrite(&model->d, sizeof(int), 1, fp);
    fwrite(&model->h, sizeof(int), 1, fp);
    fwrite(&model->o, sizeof(int), 1, fp);
    bitlinear_save_weights(&model->params->lin1, model->d, model->h, fp);
    bitlinear_save_weights(&model->params->lin2, model->h, model->o, fp);

    return 0;
}


void zero_grad(bitmlp_config_t* model, size_t b) {
    memset(model->params->lin1.dg, 0, sizeof(float) * b * model->d);
    memset(model->params->lin1.dw, 0, sizeof(float) * b * model->d * model->h);
    memset(model->params->lin2.dg, 0, sizeof(float) * b * model->d);
    memset(model->params->lin2.dw, 0, sizeof(float) * b * model->d * model->h);
}


void update_weights(bitmlp_config_t* model, adamw_t* optim) {}


/// @brief Single training step for a batch of inputs for the model.
void training_step(float* y, bitmlp_config_t* model, mnist_batch_t* batch) {
    bitmlp_mem_t* params = model->params;
    uint32_t* labels = (uint32_t*) malloc(sizeof(uint32_t) * batch->size);

    // cast uint8 inputs to float to work with our MLP implementation
    int d = batch->size * model->d;
    for (int i = 0; i < batch->size; i++) {
        for (int j = 0; j < d; j++) {
            int idx = i * d + j;
            params->lin1.x[idx] = (float) batch->images[i].pixels[j];
        }
        labels[i] = (uint32_t) batch->labels[i];
    }

    float* logits = y;
    mlp_fwd(
        logits,
        params->x_gelu,
        params->lin1.rms,
        params->lin2.rms,
        params->lin2.x,
        params->lin1.x,
        params->lin1.w,
        params->lin2.w,
        params->lin1.g,
        params->lin2.g,
        model->d,
        model->h,
        model->o,
        batch->size
    );
    float* probs = y;
    softmax_fwd(probs, logits, model->d, batch->size);
    float loss = crossentropy_fwd(probs, labels, model->o, batch->size);
    printf("Training loss: %.4f", loss); // loss is only used for logging

    zero_grad(model, batch->size);

    float* dy = y;
    // float* dloss = {1 / (batch->size)}; // gradient of loss is fixed as 1 / batch_size
    crossentropy_bkwd(dy, probs, labels, model->o, batch->size);
    softmax_bkwd(dy, y, dy, model->d, batch->size);
    mlp_bkwd(
        dy,
        params->lin1.dw,
        params->lin2.dw,
        params->lin1.dg,
        params->lin2.dg,
        params->lin2.x,
        params->lin2.w,
        params->lin2.g,
        params->lin2.rms,
        params->x_gelu,
        params->lin1.x,
        params->lin1.w,
        params->lin1.g,
        params->lin1.rms,
        model->d,
        model->h,
        model->o,
        batch->size
    );
}


void validation_step(float* y, bitmlp_config_t* model, mnist_batch_t* batch, classifier_metrics_t* metrics) {
    bitmlp_mem_t* params = model->params;
    uint32_t* labels = (uint32_t*) malloc(sizeof(uint32_t) * batch->size);

    // cast uint8 inputs to float to work with our MLP implementation
    int d = batch->size * model->d;
    for (int i = 0; i < batch->size; i++) {
        for (int j = 0; j < d; j++) {
            int idx = i * d + j;
            params->lin1.x[idx] = (float) batch->images[i].pixels[j];
        }
        labels[i] = (uint32_t) batch->labels[i];
    }

    float* logits = y;
    mlp_fwd(
        logits,
        params->x_gelu,
        params->lin1.rms,
        params->lin2.rms,
        params->lin2.x,
        params->lin1.x,
        params->lin1.w,
        params->lin2.w,
        params->lin1.g,
        params->lin2.g,
        model->d,
        model->h,
        model->o,
        batch->size
    );
    float* probs = y;
    softmax_fwd(probs, logits, model->d, batch->size);

    for (int b = 0; b < batch->size; b++) {
        int pred = b * model->o;
        for (int i = 0; i < model->o; i++) {
            if (probs[b * model->o + i] > probs[pred]) pred = b * model->o + i;
        }
        if (pred == batch->labels[b]) metrics->correct++;
        else metrics->wrong++;
    }
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

    bitmlp_mem_t mlp;
    bitmlp_config_t model = {
        .params = &mlp,
        .d = MNIST_IMAGE_SIZE,
        .h = HIDDEN_SIZE,
        .o = MNIST_LABELS
    };

    mnist_batch_t batch = { .size = BATCH_SIZE };
    batch.images = (mnist_image_t*) malloc(sizeof(mnist_image_t) * batch.size);
    batch.labels = (uint8_t*) malloc(sizeof(uint8_t) * batch.size);

    // Allocate memory for MLP
    size_t bitlin_params = model.d * (2 * model.h + 4);
    size_t gelu_params = model.d;
    size_t n_params = 2 * bitlin_params + gelu_params;
    float* arr = (float*) malloc(sizeof(float) * batch.size * n_params);
    if (arr == NULL) {
        fprintf(stderr, "Error allocating memory for BitMLP!");
        return 1;
    }

    mlp_train_init(&mlp, arr, model.d, model.h, model.o, batch.size);
    adamw_t optim;
    adamw_alloc_init(&optim, LR, BETA1, BETA2, EPS, WEIGHT_DECAY, n_params);

    float* y = (float*) malloc(sizeof(float) * batch.size * model.d);
    for (int i = 0; i < EPOCHS; i++) {
        while (mnist_get_next_batch(&batch, trainset) == 0) {
            training_step(y, &model, &batch);
            update_weights(&model, &optim);
        }
        mnist_reset_dataset(trainset);
    }

    classifier_metrics_t metrics = {0};
    while (mnist_get_next_batch(&batch, testset) == 0) {
        validation_step(y, &model, &batch, &metrics);
    }
    printf("Accuracy: %.4f (%d / %d)", (float) metrics.correct / (float) (metrics.wrong + metrics.correct), metrics.correct, metrics.correct + metrics.wrong);

    mnist_free_dataset(trainset);
    mnist_free_dataset(testset);

    save_weights(&model, "output/mnist_bitmlp.bin");

    adamw_free(&optim);
    free(arr);
    mnist_batch_free(&batch);

    return 0;
}
