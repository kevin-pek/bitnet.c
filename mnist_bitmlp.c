// Example training program to classify handwritten digits using BitMLP.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "layers/activation.h"
#include "layers/bitlinear.h"
#include "layers/mlp.h"
#include "utils/loss.h"
#include "utils/matrix.h"
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
    float* logits; // prediction logits
    float* probs;  // prediction probabilities
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


// Single training step for a batch of inputs for the model.
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

    mlp_fwd(
        model->logits,
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
    printf("MLP Logits:\n");
    print_mat(model->logits, model->d, batch->size);
    softmax_fwd(model->probs, model->logits, model->d, batch->size);
    float loss = crossentropy_fwd(model->probs, labels, model->o, batch->size);
    printf("Training loss: %.4f\n", loss); // loss is only used for logging

    zero_grad(model, batch->size);

    float* dy = y;
    // float* dloss = {1 / (batch->size)}; // gradient of loss is fixed as 1 / batch_size
    crossentropy_bkwd(dy, model->probs, labels, model->o, batch->size);
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
    int exit_code = 0;
    // load training images from MNIST dataset
    mnist_dataset_t* trainset = mnist_init_dataset(
        "MNIST_ORG/train-images.idx3-ubyte",
        "MNIST_ORG/train-labels.idx1-ubyte"
    );
    if (trainset == NULL) { exit_code = 1; goto cleanup; }

    mnist_dataset_t* testset = mnist_init_dataset(
        "MNIST_ORG/t10k-images.idx3-ubyte",
        "MNIST_ORG/t10k-labels.idx1-ubyte"
    );
    if (testset == NULL) { exit_code = 2; goto cleanup; }

    bitmlp_mem_t mlp;
    bitmlp_config_t model = {
        .params = &mlp,
        .d = MNIST_IMAGE_SIZE,
        .h = HIDDEN_SIZE,
        .o = MNIST_LABELS
    };

    mnist_batch_t batch = { .size = BATCH_SIZE };
    batch.images = (mnist_image_t*) malloc(sizeof(mnist_image_t) * batch.size);
    if (batch.images == NULL) { exit_code = 3; goto cleanup; }

    batch.labels = (uint8_t*) malloc(sizeof(uint8_t) * batch.size);
    if (batch.labels == NULL) { exit_code = 4; goto cleanup; }

    // Allocate memory for MLP and model
    size_t bitlin_params = model.d * (2 * model.h + 4);
    size_t gelu_params = model.d;
    size_t n_params = 2 * bitlin_params + gelu_params;
    float* arr = (float*) malloc(sizeof(float) * batch.size * n_params);
    if (arr == NULL) { exit_code = 5; goto cleanup; }

    model.probs = (float*) malloc(sizeof(float) * batch.size * model.o);
    if (model.probs == NULL) { exit_code = 6; goto cleanup; }

    model.logits = (float*) malloc(sizeof(float) * batch.size * model.o);
    if (model.logits == NULL) {
        fprintf(stderr, "Error allocating memory for logits!");
        free(model.probs);
        free(model.probs);
        return 3;
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

    save_weights(&model, "output/mnist_bitmlp.bin");

    adamw_free(&optim);
    free(arr);
    mnist_batch_free(&batch);

cleanup:
    if (exit_code != 0)
        fprintf(stderr, "Error occurred in train script. Exit code: %d", exit_code);
    if (trainset) mnist_free_dataset(trainset);
    if (testset) mnist_free_dataset(testset);
    if (batch.images) free(batch.images);
    if (batch.labels) free(batch.labels);
    if (model.probs) free(model.probs);
    if (model.logits) free(model.logits);

    return 0;
}
