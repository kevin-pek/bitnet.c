/**
 * base64 decode function adapted into C from the C++ implementation provided in
 * https://stackoverflow.com/questions/180947/base64-decode-snippet-in-c/13935718
 */

#ifndef BASE64_H
#define BASE64_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* b64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static const int b64index[256] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  62, 63, 62, 62, 63,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0,  0,  0,  0,  0,  0,
    0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0,  0,  0,  0,  63,
    0,  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
};

typedef struct {
    size_t len;
    char* str;
} DecodedString;

/**
 * @brief Decode given string of base64 characters into a new character array.
 *        Assumes that the input is a valid base64 string.
 *
 * @param b64str
 * @return Decoded char byte array. `str` attribute needs to be deallocated.
 */
static inline DecodedString b64_decode(const char* b64str) {
    size_t len = strlen(b64str);
    if (len == 0) return (DecodedString) {};
    unsigned char* p = (unsigned char*) b64str;
    size_t j = 0,
        pad1 = len % 4 || p[len - 1] == '=',
        pad2 = pad1 && (len % 4 > 2 || p[len - 2] != '=');
    const size_t last = (len - pad1) / 4 << 2;
    char* result = (char*) malloc(last / 4 * 3 + pad1 + pad2 + 1);
    if (result == NULL) {
        fprintf(stderr, "Failed to allocate memory for subtoken.\n");
        return (DecodedString) {};
    }
    unsigned char* str = (unsigned char*) result; // use this pointer for bitwise operations

    for (size_t i = 0; i < last; i += 4) {
        int n = b64index[p[i]] << 18 | b64index[p[i + 1]] << 12 | b64index[p[i + 2]] << 6 | b64index[p[i + 3]];
        str[j++] = n >> 16;
        str[j++] = n >> 8 & 0xFF;
        str[j++] = n & 0xFF;
    }
    if (pad1) {
        int n = b64index[p[last]] << 18 | b64index[p[last + 1]] << 12;
        str[j++] = n >> 16;
        if (pad2) {
            n |= b64index[p[last + 2]] << 6;
            str[j++] = n >> 8 & 0xFF;
        }
    }
    result[j] = '\0';
    return (DecodedString) { .len = j, .str = result };
}

#endif
