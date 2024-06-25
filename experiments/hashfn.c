/*
Test hash function on †he saved tiktoken tokenizer.
*/

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../utils/b64.h"
#include "../utils/hashtable.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void test_hash(int len) {
    FILE *file = fopen("tokenizer.model", "rb");
    if (!file) {
        perror("Failed to open file.\n");
        return;
    }

    char** token_map = (char**) malloc(sizeof(char**) * len);
    int* count_map = (int*) malloc(sizeof(int) * len);
    for (int i = 0; i < len; i++) {
        token_map[i] = NULL;
        count_map[i] = 0;
    }

    // go line by line to decode base64 subword and parse token ids
    char buffer[1024];
    char *b64_str;
    size_t n_chars_vocab = 0, maxlen = 0;
    float n_collisions = 0.0f;
    int lens[128] = {0};
    
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        b64_str = strtok(buffer, " ");
        printf("Reading string %s\n", b64_str);
        if (b64_str != NULL) {
            DecodedString decoded_str = b64_decode(b64_str);
            maxlen = MAX(maxlen, decoded_str.len);
            lens[decoded_str.len]++;
            printf("Decoded length %zu: %s\n", decoded_str.len, decoded_str.str);
            uint32_t key = hash_fnv1a(decoded_str.str, decoded_str.len) % len;
            if (token_map[key] != NULL) {
                n_collisions++;
                printf("Collision: %s collides with %s\n", decoded_str.str, token_map[key]);
                printf("Hash value: %u\n", key);
            }
            token_map[key] = decoded_str.str;
            count_map[key]++;
        } else {
            printf("NULL string encountered\n");
        }
    }

    printf("Collision rate: %f\n", n_collisions / 128000.0f);
    printf("Max string length: %zu\n", maxlen);

    int gaps[64] = {0};
    for (int i = 0; i < len; i++) {
        printf("index: %d count: %d\n", i, count_map[i]);
        if (count_map[i] > 1) {
            gaps[count_map[i]]++;
            for (int j = 1; j < count_map[i] && (j + i) < len; j++) {
                if (count_map[j + i] != 0) {
                    printf("No available space %d entries away from index %d, which has %d values\n", j, i, count_map[i]);
                    break;
                }
            }
        }
    }

    for (int i=0;i<64;i++)
        if (gaps[i] > 0) printf("%d gaps of size %d\n", gaps[i], i);

    for (int i=0;i<128;i++)
        if (lens[i] > 1) printf("string of length %d occurred %d times\n", i, lens[i]);

    for (int i = 0; i < len; i++) {
        if (token_map[i] != NULL) {
            free(token_map[i]);
        }
    }

    free(token_map);
    free(count_map);

    fclose(file);
}

int main() {
    test_hash(128000); // vocab size
    // test_hash(131072); // prime number near 2^17
    // test_hash(262127); // prime number near 2^18
    // test_hash(262144); // 2^18
    // test_hash(4096000);
}