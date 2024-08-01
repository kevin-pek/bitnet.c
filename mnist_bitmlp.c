// Example training program to classify handwritten digits using BitMLP.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "layers/activation.h"
#include "layers/batchnorm.h"
#include "layers/bitlinear.h"
#include "layers/mlp.h"
#include "utils/logging.h"
#include "utils/loss.h"
#include "utils/mnist.h"
#include "utils/optim.h"

#define BATCH_SIZE 32
#define EPOCHS 1
#define LR 1e-3f
#define EPS 1e-8f
#define BETA1 0.9f
#define BETA2 0.99f
#define WEIGHT_DECAY 1e-2f

typedef struct {
    bitmlp_mem_t* mem;
    bitmlp_grad_t* grads;
    bitmlp_t* params;
    float* mem_lin3_dy_gelu; // to store propagated gradients from bitlinear layer 3
    bitlinear_mem_t* mem_lin3;   // extra hidden layer because 1 isn't enough
    bitlinear_grad_t* grads_lin3;
    bitlinear_t* params_lin3;
    float* probs;   // output probabilities
    size_t d; // input dims
    size_t h1; // hidden dims
    size_t h2; // hidden dims
    size_t o; // output dims
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

    // write 32 bit integers to store model information
    uint32_t d = (uint32_t) model->d;
    uint32_t h1 = (uint32_t) model->h1;
    uint32_t h2 = (uint32_t) model->h2;
    uint32_t o = (uint32_t) model->o;
    fwrite(&d, sizeof(uint32_t), 1, fp);
    fwrite(&h1, sizeof(uint32_t), 1, fp);
    fwrite(&h2, sizeof(uint32_t), 1, fp);
    fwrite(&o, sizeof(uint32_t), 1, fp);
    bitlinear_save_weights(&model->params->lin1, &model->mem->lin1, model->d, model->h1, fp);
    bitlinear_save_weights(&model->params->lin2, &model->mem->lin2, model->h1, model->h2, fp);
    bitlinear_save_weights(model->params_lin3, model->mem_lin3, model->h2, model->o, fp);
    fclose(fp);

    return 0;
}


void zero_grad(bitmlp_config_t* model, size_t b) {
    memset(model->grads->lin1.dg, 0, sizeof(float) * model->d);
    memset(model->grads->lin1.dw, 0, sizeof(float) * model->d * model->h1);
    memset(model->grads->lin2.dg, 0, sizeof(float) * model->h1);
    memset(model->grads->lin2.dw, 0, sizeof(float) * model->h1 * model->h2);
    memset(model->grads_lin3->dg, 0, sizeof(float) * model->h2);
    memset(model->grads_lin3->dw, 0, sizeof(float) * model->h2 * model->o);
}


// Single training step for a batch of inputs for the model.
void training_step(bitmlp_config_t* model, mnist_batch_t* batch) {
    bitmlp_mem_t* mem = model->mem;
    bitmlp_grad_t* grads = model->grads;
    bitmlp_t* params = model->params;

    // cast uint8 inputs to float to work with our MLP implementation
    for (size_t i = 0; i < batch->size; i++) {
        for (size_t j = 0; j < model->d; j++) {
            size_t idx = i * model->d + j;
            mem->lin1.x[idx] = (float) batch->images[i].pixels[j];
        }
    }

    mlp_fwd(
        mem->lin2.y, // write to input of next GELU layer
        mem->lin1.y,
        mem->lin1.y_rms,
        mem->lin2.y_rms,
        mem->lin2.x,
        mem->lin1.x,
        mem->lin1.w,
        mem->lin2.w,
        params->lin1.g,
        params->lin2.g,
        params->lin1.wq,
        params->lin1.yq,
        params->lin1.xq,
        params->lin2.wq,
        params->lin2.yq,
        params->lin2.xq,
        model->d,
        model->h1,
        model->h2,
        batch->size
    );
    gelu_fwd(model->mem_lin3->x, mem->lin2.y, model->h1, batch->size);
    bitlinear_fwd(
        model->mem_lin3->y,
        model->mem_lin3->y_rms,
        model->mem_lin3->x,
        model->mem_lin3->w,
        model->params_lin3->g,
        model->params_lin3->wq,
        model->params_lin3->yq,
        model->params_lin3->xq,
        model->h2,
        model->o,
        batch->size
    );
    softmax_fwd(model->probs, model->mem_lin3->y, model->o, batch->size);

    // loss is only used for logging, we only need the logits for backpropagation
    float loss = cross_entropy_loss(model->mem_lin3->y, batch->labels, model->o, batch->size);
    printf("Training loss: %.4f\n", loss);

    zero_grad(model, batch->size);

    float* dloss = model->mem_lin3->y; // reuse memory for logits to propagate gradients for loss
    softmax_bkwd(dloss, model->probs, batch->labels, model->o, batch->size);
    batchnorm(dloss, model->o, batch->size);

    bitlinear_bkwd(
        model->mem_lin3_dy_gelu,
        model->grads_lin3->dw,
        model->grads_lin3->dg,
        model->mem_lin3->dy_rms,
        dloss,
        model->mem_lin3->x,
        model->mem_lin3->w,
        model->params_lin3->g,
        model->mem_lin3->y_rms,
        model->h2,
        model->o,
        batch->size
    );
    gelu_bkwd(model->mem->dy, model->mem_lin3_dy_gelu, mem->lin2.y, model->h2, batch->size);
    batchnorm(model->mem->dy, model->h2, batch->size);

    printf("Address of the array pointer: %p\n", (void*) mem->dx);
    printf("Address of the array pointer: %p\n", (void*) grads->lin1.dw);
    printf("===================\n");
    mlp_bkwd(
        mem->dx,
        grads->lin1.dw,
        grads->lin2.dw,
        grads->lin1.dg,
        mem->lin1.dy_rms,
        grads->lin2.dg,
        mem->lin2.dy_rms,
        mem->dy_gelu,
        mem->dx_gelu,
        mem->dy,
        mem->lin2.x,
        mem->lin2.w,
        params->lin2.g,
        mem->lin2.y_rms,
        mem->lin1.y,
        mem->lin1.x,
        mem->lin1.w,
        params->lin1.g,
        mem->lin1.y_rms,
        model->d,
        model->h1,
        model->h2,
        batch->size
    );
}


void validation_step(bitmlp_config_t* model, mnist_batch_t* batch, classifier_metrics_t* metrics) {
    bitmlp_mem_t* mem = model->mem;
    bitmlp_t* params = model->params;

    // cast uint8 inputs to float to work with our MLP implementation
    for (size_t i = 0; i < batch->size; i++) {
        for (size_t j = 0; j < model->d; j++) {
            size_t idx = i * model->d + j;
            mem->lin1.x[idx] = (float) batch->images[i].pixels[j];
        }
    }

    mlp_fwd(
        mem->lin2.y, // write to input of next GELU layer
        mem->lin1.y,
        mem->lin1.y_rms,
        mem->lin2.y_rms,
        mem->lin2.x,
        mem->lin1.x,
        mem->lin1.w,
        mem->lin2.w,
        params->lin1.g,
        params->lin2.g,
        params->lin1.wq,
        params->lin1.yq,
        params->lin1.xq,
        params->lin2.wq,
        params->lin2.yq,
        params->lin2.xq,
        model->d,
        model->h1,
        model->h2,
        batch->size
    );
    gelu_fwd(model->mem_lin3->x, mem->lin2.y, model->h1, batch->size);
    bitlinear_fwd(
        model->mem_lin3->y,
        model->mem_lin3->y_rms,
        model->mem_lin3->x,
        model->mem_lin3->w,
        model->params_lin3->g,
        model->params_lin3->wq,
        model->params_lin3->yq,
        model->params_lin3->xq,
        model->h2,
        model->o,
        batch->size
    );
    softmax_fwd(model->probs, model->mem_lin3->y, model->o, batch->size);

    for (int b = 0; b < batch->size; b++) {
        uint32_t argmax = b * model->o;
        for (int i = 0; i < model->o; i++) {
            if (model->probs[b * model->o + i] > model->probs[argmax])
                argmax = b * model->o + i;
        }
        if ((argmax % model->o) == batch->labels[b]) metrics->correct++;
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

    bitmlp_t mlp;
    bitmlp_mem_t mem;
    bitmlp_grad_t grads;
    bitlinear_mem_t lin3_mem;   // extra hidden layer because 1 isn't enough
    bitlinear_grad_t lin3_grads;
    bitlinear_t lin3_params;
    bitmlp_config_t model = {
        .mem = &mem,
        .grads = &grads,
        .params = &mlp,
        .mem_lin3 = &lin3_mem,
        .grads_lin3 = &lin3_grads,
        .params_lin3 = &lin3_params,
        .d = MNIST_IMAGE_SIZE,
        .h1 = 256,
        .h2 = 128,
        .o = MNIST_LABELS
    };

    mnist_batch_t batch = { .size = BATCH_SIZE };
    batch.images = (mnist_image_t*) calloc(batch.size, sizeof(mnist_image_t));
    if (batch.images == NULL) { exit_code = 3; goto cleanup; }

    batch.labels = (uint32_t*) calloc(batch.size, sizeof(uint32_t));
    if (batch.labels == NULL) { exit_code = 4; goto cleanup; }

    // Allocate memory for MLP training params
    size_t n_params = model.d  + model.d  * model.h1
                    + model.h1 + model.h1 * model.h2
                    + model.h2 + model.h2 * model.o;
    float* params = (float*) calloc(n_params, sizeof(float));
    if (params == NULL) { exit_code = 5; goto cleanup; }

    // Assign pointers for model parameters and intermediate values to contiguous memory block
    float* arr_ptr = params;
    mlp.lin1.g = arr_ptr;
    arr_ptr += model.d;
    mem.lin1.w = arr_ptr;
    arr_ptr += model.d * model.h1;
    mlp.lin2.g = arr_ptr;
    arr_ptr += model.h1;
    mem.lin2.w = arr_ptr;
    arr_ptr += model.h1 * model.h2;
    lin3_params.g = arr_ptr;
    arr_ptr += model.h2;
    lin3_mem.w = arr_ptr;

    float* grad_params = (float*) calloc(n_params, sizeof(float));
    if (grad_params == NULL) { exit_code = 6; goto cleanup; }

    // Assign pointers for gradients to contiguous memory block
    float* grad_ptr = grad_params;
    grads.lin1.dg = grad_ptr;
    grad_ptr += model.d;
    grads.lin1.dw = grad_ptr;
    grad_ptr += model.d * model.h1;
    grads.lin2.dg = grad_ptr;
    grad_ptr += model.h1;
    grads.lin2.dw = grad_ptr;
    grad_ptr += model.h1 * model.h2;
    lin3_grads.dg = grad_ptr;
    grad_ptr += model.h2;
    lin3_grads.dw = grad_ptr;

    size_t n_bit_params = bitmat_bytes(model.d, model.h1) + bitmat_bytes(model.h1, model.h2) + bitmat_bytes(model.h2, model.o);
    uint8_t* uint8_params = (uint8_t*) calloc(batch.size * n_bit_params, sizeof(uint8_t));
    if (uint8_params == NULL) { exit_code = 7; goto cleanup; }

    uint8_t* uint8_ptr = uint8_params;
    mlp.lin1.wq = uint8_ptr;
    uint8_ptr += batch.size * bitmat_bytes(model.d, model.h1);
    mlp.lin2.wq = uint8_ptr;
    uint8_ptr += batch.size * bitmat_bytes(model.h1, model.h2);
    lin3_params.wq = uint8_ptr;

    int8_t* int8_params = (int8_t*) calloc(batch.size * (model.d + 2 * model.h1 + 2 * model.h2 + model.o), sizeof(int8_t));
    if (int8_params == NULL) { exit_code = 8; goto cleanup; }

    int8_t* int8_ptr = int8_params;
    mlp.lin1.xq = int8_ptr;
    int8_ptr += batch.size * model.d;
    mlp.lin1.yq = int8_ptr;
    int8_ptr += batch.size * model.h1;
    mlp.lin2.xq = int8_ptr;
    int8_ptr += batch.size * model.h1;
    mlp.lin2.yq = int8_ptr;
    int8_ptr += batch.size * model.h2;
    lin3_params.xq = int8_ptr;
    int8_ptr += batch.size * model.h2;
    lin3_params.yq = int8_ptr;

    // Allocate memory for storing intermediate results
    size_t n_mems = bitlinear_mem_size(model.d, model.h1)
                  + bitlinear_mem_size(model.h1, model.h2)
                  + bitlinear_mem_size(model.h2, model.o)
                  + model.o;
    size_t n_mem_grads = 2 * model.d + 3 * (model.h1 + model.h2);
    float* mem_params = (float*) calloc(batch.size * n_mems + n_mem_grads, sizeof(float));
    if (params == NULL) { exit_code = 9; goto cleanup; }

    float* mem_ptr = mem_params;
    mem.dx = mem_ptr;
    mem_ptr += model.d;
    mem.lin1.dy_rms = mem_ptr;
    mem_ptr += model.d;
    mem.lin1.x = mem_ptr;
    mem_ptr += batch.size * model.d;
    mem.lin1.y_rms = mem_ptr;
    mem_ptr += batch.size * model.d;
    mem.lin1.y = mem_ptr;
    mem_ptr += batch.size * model.h1;

    mem.dx_gelu = mem_ptr;
    mem_ptr += model.h1;
    mem.dy_gelu = mem_ptr;
    mem_ptr += model.h1;

    mem.lin1.dy_rms = mem_ptr;
    mem_ptr += model.h1;
    mem.lin2.x = mem_ptr;
    mem_ptr += batch.size * model.h1;
    mem.lin2.y_rms = mem_ptr;
    mem_ptr += batch.size * model.h1;
    mem.lin2.y = mem_ptr;
    mem_ptr += batch.size * model.h2;

    mem.dy = mem_ptr;
    mem_ptr += model.h2;
    model.mem_lin3_dy_gelu = mem_ptr; // allocate pointer directly to bitmlp_config_t
    mem_ptr += model.h2;

    lin3_mem.dy_rms = mem_ptr;
    mem_ptr += model.h2;
    lin3_mem.x = mem_ptr;
    mem_ptr += batch.size * model.h2;
    lin3_mem.y_rms = mem_ptr;
    mem_ptr += batch.size * model.h2;
    lin3_mem.y = mem_ptr;
    mem_ptr += batch.size * model.o;
    model.probs = mem_ptr; // batch_size * model.o elems

    // Initialize training parameters
    mlp_init(&mlp, &mem, model.d, model.h1, model.h2);
    bitlinear_init(&lin3_params, &lin3_mem, model.h2, model.o);

    adamw_t optim;
    if (adamw_alloc(&optim, n_params) != 0) { exit_code = 10; goto cleanup; }
    adamw_init(&optim, LR, BETA1, BETA2, EPS, WEIGHT_DECAY);

    classifier_metrics_t metrics = {0};
    // Get zero shot accuracy for initial weight parameters
    // while (mnist_get_next_batch(&batch, testset) == 0)
    //     validation_step(&model, &batch, &metrics);
    // printf("Zero-shot Accuracy: %.4f (%d / %d)\n", (float) metrics.correct / (float) (metrics.wrong + metrics.correct), metrics.correct, metrics.correct + metrics.wrong);
    // mnist_reset_dataset(testset);
    // batch.size = BATCH_SIZE;

    for (int i = 0; i < EPOCHS; i++) {
        int j = 0;
        while (mnist_get_next_batch(&batch, trainset) == 0) {
            printf("Epoch: %d, Batch %d, ", i, j++);
            training_step(&model, &batch);
            adamw_update(&optim, params, grad_params, i + 1);
        }
        mnist_reset_dataset(trainset);
        batch.size = BATCH_SIZE; // reset batch size in case last batch set it lower
    }

    print_mat(model.params->lin1.g, model.d, model.h1);
    print_mat(model.params->lin2.g, model.h1, model.h2);
    print_mat(model.mem_lin3->w, model.h2, model.o);

    metrics.wrong = 0;
    metrics.correct = 0;
    while (mnist_get_next_batch(&batch, testset) == 0)
        validation_step(&model, &batch, &metrics);
    printf("Validation Accuracy: %.4f (%d / %d)\n", (float) metrics.correct / (float) (metrics.wrong + metrics.correct), metrics.correct, metrics.correct + metrics.wrong);

    save_weights(&model, "mnist_bitmlp.bin");

cleanup:
    if (exit_code != 0) {
        fprintf(stderr, "Error occurred in train script. Exit code: %d\n", exit_code);
    }
    adamw_free(&optim);
    free(params);
    free(grad_params);
    free(mem_params);
    free(uint8_params);
    free(int8_params);
    if (trainset) mnist_free_dataset(trainset);
    if (testset) mnist_free_dataset(testset);
    mnist_batch_free(&batch);

    return 0;
}
