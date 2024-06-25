/*
Encode the tokenizer on a given input sequence.
*/

#include <assert.h>
#include "../tokenizer.h"

#define MAX_SEQ_LEN 56000 // just use an arbitrary value since this is for testing

int main() {
    Tokenizer t = {
        .vocab_size = 128000,
        .vocab_strlen = 831311, // value precomputed from uniquesyms.c
        .bos_token = 128000,
        .eos_token = 128001
    };
    uint32_t* tokens = (uint32_t*) malloc(sizeof(uint32_t) * MAX_SEQ_LEN);
    tokenizer_init(&t, "tokenizer.model");
    size_t len = tokenizer_encode(tokens, &t, "Hello world! What is the meaning of life?");
    for (int i = 0; i < len; i++) {
        if (i == 0) printf("Encoded: [");
        if (i < len - 1) printf("%u, ", tokens[i]);
        else if (i == len - 1) printf("%u]\n", tokens[i]);
    }
    printf("Decoded: %s\n", tokenizer_decode(&t, tokens, len));
    uint32_t expected[] = {128000, 9906, 1917, 0, 3639, 374, 279, 7438, 315, 2324, 30};
    size_t num_elements = sizeof(expected) / sizeof(expected[0]);
    for (size_t i = 0; i < num_elements; i++) {
        assert(tokens[i] == expected[i]);
    }
    free(tokens);
}
