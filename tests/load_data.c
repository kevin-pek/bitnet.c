#include "../utils/mnist.h"
#include <stdio.h>

int main() {
    mnist_dataset_t* dataset = mnist_init_dataset(
        "MNIST_ORG/t10k-images.idx3-ubyte",
        "MNIST_ORG/t10k-labels.idx1-ubyte"
    );
    if (dataset == NULL) { return 1; }

    mnist_batch_t batch = { .size = 8 };
    batch.images = (mnist_image_t*) calloc(batch.size, sizeof(mnist_image_t));
    if (batch.images == NULL) { return 2; }

    batch.labels = (uint32_t*) calloc(batch.size, sizeof(uint32_t));
    if (batch.labels == NULL) { return 3; }

    int i = 0;
    while (mnist_get_next_batch(&batch, dataset) == 0) {
        i += batch.size;
    }
    for (size_t i = 0; i < batch.size; i++) printf("%d, ", batch.labels[i]);
    printf("Number of samples read: %d\n", i);
}
