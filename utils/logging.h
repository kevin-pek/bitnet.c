#ifndef LOG_H
#define LOG_H

#include <stdint.h>
#include <stdio.h>


inline void printbin8(char num) {
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

#endif
