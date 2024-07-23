/**
 * @file Test program to verify Matrix initialization has no nan values.
 */

#include <stdio.h>
#include "../utils/matrix.h"

int main() {
    size_t rows = 8, cols = 8;
    float* mat = (float*) calloc(rows * cols, sizeof(float));
    printf("Random initialization:\n");
    mat_init_rand(mat, rows * cols);
    print_mat(mat, rows, cols);

    printf("Kaiming initialization:\n");
    mat_init_kaiming(mat, rows * cols);
    print_mat(mat, rows, cols);
}
