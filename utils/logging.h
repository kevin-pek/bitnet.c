#ifndef LOG_H
#define LOG_H

#include <stdint.h>
#include <stdio.h>


void printbin8(uint8_t num) {
    for (int i = 7; i >= 0; i--)
        fprintf(stderr, "%d", (num >> i) & 1);
}


void printbin32(uint32_t num) {
    int n_bits = sizeof(num) * 8;
    for (int i = n_bits - 1; i >= 0; i--) {
        unsigned int bit = (num >> i) & 1;
        fprintf(stderr, "%u", bit);
    }
}

void print_mat(const float* mat, size_t rows, size_t cols) {
    fprintf(stderr, "[\n");
    for (size_t i = 0; i < rows; i++) {
        fprintf(stderr, "\t[");
        for (size_t j = 0; j < cols; j++) {
            fprintf(stderr, "%.2f, ", mat[i * cols + j]);
        }
        fprintf(stderr, "]");
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "]\n");
}


// Print 8 bit quantised matrix in binary.
void print_qmat(int8_t* mat, size_t rows, size_t cols) {
    fprintf(stderr, "[\n");
    for (size_t i = 0; i < rows; i++) {
        fprintf(stderr, "\t[");
        for (size_t j = 0; j < cols; j++) {
            fprintf(stderr, "%4d, ", mat[i * cols + j]);
        }
        fprintf(stderr, "]");
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "]\n");
}


#endif
