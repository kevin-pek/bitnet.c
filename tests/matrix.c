#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "../utils/matrix.h"
#ifdef DEBUG
#include "../utils/logging.h"
#endif

#define TOLERANCE 1e-5


void assert_float_mat_eq(const float* res, const float* exp, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            assert(fabs(res[i * cols + j] - exp[i * cols + j]) <= TOLERANCE);
    printf("Test case passed!\n");
}


void assert_int8_mat_eq(const int8_t* res, const int8_t* exp, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            assert(res[i * cols + j] == exp[i * cols + j]);
    printf("Test case passed!\n");
}


void test_matmul() {
    printf("Test matmul:\n");
    float w[5 * 8] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40
    };
    float x[8 * 2] = {
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        13, 14,
        15, 16
    };
    float y[5 * 2] = {0};
    float expected_y[5 * 2] = {
        372, 408,
        884, 984,
        1396, 1560,
        1908, 2136,
        2420, 2712
    };
    size_t n = 5, m = 8, b = 2;
    printf("Test Case 1:\n");
    matmul_fwd(y, x, w, n, m, b);
    print_mat(y, n, b);
    assert_float_mat_eq(y, expected_y, n, b);
    printf("\n");
}


void test_bitmatmul() {
    printf("Test bitmatmul:\n");
    // sum all entries in the matrix
    int8_t xq1[] = {1, 0, -1, 0, 1, 1, -1, -1};
    uint8_t wq1[] = {0b00000000, 0b00000000};
    size_t rows_w1 = 2;
    size_t cols_w1 = 8;
    size_t cols_x1 = 1;
    int8_t yq1[2 * 1] = {0};
    int8_t expected_yq1[2 * 1] = {0};
    #ifdef DEBUG
        for (size_t i = 0; i < cols_w1; i++) {
            printbin8(xq1[i]);
            printf("\n");
        }
        printf("Weights\n");
        for (size_t i = 0; i < rows_w1; i++) {
            printbin8(wq1[i]);
            printf("\n");
        }
    #endif
    printf("Test Case 1:\n");
    bitmatmul_fwd(yq1, xq1, wq1, rows_w1, cols_w1, cols_x1);
    print_qmat(yq1, rows_w1, cols_x1);
    assert_int8_mat_eq(yq1, expected_yq1, rows_w1, cols_x1);
    printf("\n");

    // normal matrix multiplication
    int8_t xq2[] = {
        10, -10,
        98, -98,
        -13, 13,
        27, -27,
        -120, 120,
        -47, 47,
        32, -32,
        0, 0
    };
    uint8_t wq2[] = {0b00000000, 0b11111111, 0b00100000};
    size_t rows_w2 = 3;
    size_t cols_w2 = 8;
    size_t cols_x2 = 2;
    int8_t yq2[3 * 2] = {0};
    int8_t expected_yq2[3 * 2] = {
        -13, 13,
        13, -13,
        13, -13
    };
    printf("Test Case 2:\n");
    bitmatmul_fwd(yq2, xq2, wq2, rows_w2, cols_w2, cols_x2);
    print_qmat(yq2, rows_w2, cols_x2);
    assert_int8_mat_eq(yq2, expected_yq2, rows_w2, cols_x2);
    printf("\n");
}


int main() {
    test_matmul();
    test_bitmatmul();
}
