#ifndef LOG_H
#define LOG_H

#include <stdint.h>
#include <stdio.h>


inline void printbin8(uint8_t num) {
    for (int i = 7; i >= 0; i--)
        printf("%d", (num >> i) & 1);
}


inline void printbin32(uint32_t num) {
    int n_bits = sizeof(num) * 8;
    for (int i = n_bits - 1; i >= 0; i--) {
        unsigned int bit = (num >> i) & 1;
        printf("%u", bit);
    }
}

void print_mat(const float* mat, size_t rows, size_t cols) {
    printf("[\n");
    for (size_t i = 0; i < rows; i++) {
        printf("\t[");
        for (size_t j = 0; j < cols; j++) {
            printf("%.2f,", mat[i * cols + j]);
        }
        printf("]");
        printf("\n");
    }
    printf("]\n");
}


// Print 8 bit quantised matrix in binary.
void print_qmat(int8_t* mat, size_t rows, size_t cols) {
    printf("[\n");
    for (size_t i = 0; i < rows; i++) {
        printf("\t[");
        for (size_t j = 0; j < cols; j++) {
            printf("%4d,", mat[i * cols + j]);
        }
        printf("]");
        printf("\n");
    }
    printf("]\n");
}


#endif
