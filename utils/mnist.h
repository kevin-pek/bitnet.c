/*
This implementation is mostly based from this repo: https://github.com/AndrewCarterUK/mnist-neural-network-plain-c
*/

#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT
#define MNIST_LABELS 10

typedef struct {
    uint8_t pixels[MNIST_IMAGE_SIZE];
} mnist_image_t;

// represents the dataaet through the file pointers
typedef struct {
    FILE* images;
    FILE* labels;
    size_t idx;
    size_t size;
} mnist_dataset_t;

// represents a batch of samples
typedef struct {
    mnist_image_t* images;
    unsigned char* labels;
    size_t size;
} mnist_batch_t;


int read_int(FILE* fp) {
    unsigned char bytes[4];
    if (fread(bytes, 4, 1, fp) != 1) {
        return 0;
    }
    // Convert big endian to little endian
    return (int)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}


mnist_dataset_t* mnist_init_dataset(const char* imagespath, const char* labelspath) {
    FILE* img_fp = fopen(imagespath, "rb");
    if (img_fp == NULL) {
        return NULL;
    }
    FILE* labels_fp = fopen(labelspath, "rb");
    if (labels_fp == NULL) {
        fclose(img_fp);
        return NULL;
    }

    int magic_number = read_int(img_fp);
    int num_items = read_int(img_fp);
    int num_rows = read_int(img_fp);
    int num_cols = read_int(img_fp);
    printf("Magic Number: %d, Number of Images: %d, Rows: %d, Columns: %d\n",
            magic_number, num_items, num_rows, num_cols);

    magic_number = read_int(labels_fp);
    num_items = read_int(labels_fp);
    printf("Magic Number: %d, Number of Labels: %d\n",
            magic_number, num_items);

    mnist_dataset_t *dataset = (mnist_dataset_t*) malloc(sizeof(mnist_dataset_t));
    dataset->idx = 0;
    dataset->size = num_items;
    dataset->images = img_fp;
    dataset->labels = labels_fp;
    return dataset;
}


int mnist_get_next_batch(mnist_batch_t* batch, mnist_dataset_t* dataset) {
    size_t n_samples = batch->size;
    // Decrement samples to read if remaining samples are less than the dataset size.
    if (dataset->idx + batch->size > dataset->size) {
        n_samples = dataset->size - dataset->idx;
    }

    size_t bytes_read = fread(batch->images, sizeof(mnist_image_t), n_samples, dataset->images);
    if (bytes_read != n_samples) {
        fprintf(stderr, "Failed to read image data!\n");
        return 1;
    }

    if (fread(batch->labels, 1, n_samples, dataset->images) != n_samples) {
        fprintf(stderr, "Failed to read label data!\n");
        return 2;
    }
    printf("label: %hhu\n", batch->labels[0]);

    return 0;
}


// Rewind file pointer to the start of the file.
void mnist_reset_dataset(mnist_dataset_t* dataset) {
    dataset->idx = 0;
    rewind(dataset->images);
    rewind(dataset->labels);
}


void mnist_free_dataset(mnist_dataset_t* dataset) {
    fclose(dataset->images);
    fclose(dataset->labels);
    free(dataset);
}


// Free memory for batch. We do not free the pointer to the batch as it is
// allocated on the stack.
void mnist_batch_free(mnist_batch_t* batch) {
    free(batch->images);
    free(batch->labels);
}

#endif
