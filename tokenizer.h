/*
Implementation of BPE loading and inference from tokenizers trained using tiktoken.
LINK: https://github.com/openai/tiktoken
Tokenizer file represents subword as space separated base64 string, token id pairs.
Each line in the file represents a subword and its corresponding token id.
BPE training functions are currently not implemented.
*/

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/b64.h"
#include "utils/hashtable.h"
#include "utils/priorityqueue.h"

// llama 3 special tokens, starting from id 128000. The hugging face repo contains
// a whole range of special tokens from 128000 to 128256, but the official model
// card from meta only uses the ones defined in the website linked below.
// https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
const char* special_tokens[] = {
    "<|begin_of_text|>",   // BOS token, 128000
    "<|end_of_text|>",     // EOS token, 128001, last token in sequence
    // These reserved special tokens are not used by the llama 3 tokenizer, but
    // I kept them here for clarity on the token id mappings.
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|reserved_special_token_2|>",
    "<|reserved_special_token_3|>",
    // Header id tokens enclose the role for a particular message. This can be
    // one of the following: system, user, assistant
    "<|start_header_id|>", // 128006
    "<|end_header_id|>",   // 128007
    "<|reserved_special_token_4|>",
    "<|eot_id|>"           // 128009, signifies the end of message in a turn.
};
// there are more reserved special tokens ranging from 5 to 250, but I will
// ignore them in this implementation

typedef struct {
    char*    vocab;         // represent vocab as long contiguous array of chars
    size_t*  id_map;        // map token ids to the starting index in vocab
    HT*      token_map;     // hashtable mapping subwords to their token ids
    size_t   vocab_size;
    size_t   vocab_strlen;  // number of chars within the entire vocab char array
    uint32_t bos_token;     // beginning of sentence token
    uint32_t eos_token;     // end of sentence token
} Tokenizer;

typedef struct {
    uint32_t start;   // char index in input string that symbol starts from
    int32_t  len;     // length of symbol in chars
    int32_t  prev;    // index of previous symbol in symbols array, -1 for first symbol
    int32_t  next;    // index of next symbol in symbols array, -1 for last symbol
} Symbol;

typedef struct {
    int32_t  len;     // length of the pair in number of chars
    int32_t  left;    // index of left symbol in symbols array
    int32_t  right;   // index of right symbol in symbols array
    uint32_t rank;    // used to sort pairs in PriorityQueue, same value as token id
} SymbolPair;


/// @brief Loads from the given filepath tokenizer to the given pointer.
/// @param t pointer to tokenizer to initialize
/// @param filepath filepath to tokenizer file
/// @returns 0 if initialized successfully, otherwise error code is returned
int tokenizer_init(Tokenizer* t, const char* filepath) {
    FILE *file = fopen(filepath, "rb");
    int error_code = 0;

    t->id_map = (size_t*) malloc(sizeof(size_t) * t->vocab_size);
    t->token_map = hash_init(t->vocab_size, hash_fnv1a); // set size of table to prime number close to 2^18
    t->vocab = (char*) malloc(sizeof(char) * t->vocab_strlen);

    char buffer[256]; // longest decoded token is 128 bytes long from hashfn.c
    char *b64_str;
    int i = 0;
    size_t stridx = 0; // starting index of subword string

    if (!file) { error_code = 1; goto cleanup; }
    if (!t->id_map) { error_code = 2; goto cleanup; }
    if (!t->token_map) { error_code = 3; goto cleanup; }
    if (!t->vocab) { error_code = 4; goto cleanup; }

    // go line by line to decode base64 subword and parse token ids
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        b64_str = strtok(buffer, " ");
        if (b64_str != NULL) {
            DecodedString decoded_str = b64_decode(b64_str);
            if (!decoded_str.str) { error_code = 5; goto cleanup; }
            #ifdef DEBUG
            fprintf(stderr, "Decoded string with length %zu: %s with rank/id %d\n", decoded_str.len, decoded_str.str, i);
            #endif

            // add decoded string to the subword hash table
            if (hash_insert(t->token_map, decoded_str.str, decoded_str.len, (uint32_t) i)) {
                free(decoded_str.str);
                error_code = 6;
                goto cleanup;
            }

            // store decoded string to the token id map
            t->id_map[i++] = stridx;
            // copy the string over to vocab string, exclude the null byte
            memcpy(&t->vocab[stridx], decoded_str.str, decoded_str.len);
            stridx += decoded_str.len;
            free(decoded_str.str);
        }
    }

cleanup:
    if (error_code != 0) {
        fprintf(stderr, "Error occurred while allocating memory for tokenizer, error code: %d\n", error_code);
        if (t->id_map) free(t->id_map);
        if (t->token_map) hash_free(t->token_map);
        if (t->vocab) free(t->vocab);
    }
    if (file) fclose(file);
    return error_code;
}


/// @brief Decode sequence of token ids to string.
/// @param t pointer to tokenizer
/// @param tokens array of token ids to decode
/// @param n_tokens number of tokens in token id array
/// @return Deocded string output.
char* tokenizer_decode(Tokenizer *t, uint32_t* tokens, size_t n_tokens) {
    // get length of output string sequence
    size_t strlen = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] == 128000 || tokens[i] == 128001) continue; // skip special tokens
        size_t start_idx = t->id_map[tokens[i]];
        // if token id is last token in vocabulary, set end index to end of vocab string
        size_t end_idx = ((tokens[i] + 1) == t->vocab_size) ? t->vocab_strlen : t->id_map[tokens[i] + 1];
        strlen += end_idx - start_idx;
    }

    // allocate space for output string
    char* outstr = (char*) malloc(sizeof(char) * (strlen + 1));
    if (!outstr) return NULL;

    // copy characters from the vocab string over to the output string
    int stridx = 0;
    for (int i = 0; i < n_tokens; i++) {
        size_t start_idx = t->id_map[tokens[i]];
        // end_idx is value in next entry in id_map, and handle edge case for last token id in vocab
        size_t end_idx = ((tokens[i] + 1) == t->vocab_size) ? t->vocab_strlen : t->id_map[tokens[i] + 1];
        for (size_t j = start_idx; j < end_idx; j++) {
            outstr[stridx++] = t->vocab[j];
        }
    }
    outstr[stridx] = '\0';

    return outstr;
}


// Compare rank of 2 given SymbolPairs, used to rank which to merge first in PriorityQueues.
static inline int compare_pairs(const void* a, const void* b) {
    const SymbolPair* pair1 = (const SymbolPair*) a;
    const SymbolPair* pair2 = (const SymbolPair*) b;
    return pair1->rank - pair2->rank;
}


/// @brief Merge the 2 provided symbols together and returns new Symbol.
/// @param left pointer to left symbol
/// @param right pointer to right symbol
/// @param size size of new merged symbol
/// @return new symbol, returns unintialized Symbol if invalid merge is found
static inline Symbol merge_symbols(Symbol* left, Symbol* right, int32_t len) {
    // validation step, size of symbols to merge must be same as size of new pair
    if (left->len == 0 || right->len == 0) {
        #ifdef DEBUG
        fprintf(stderr, "Invalid pair with invalid symbol, skipping merge\n");
        #endif
        return (Symbol) {};
    } else if ((left->len + right->len) != len) {
        #ifdef DEBUG
        fprintf(stderr, "Invalid pair with unequal length, skipping merge\n");
        #endif
        return (Symbol) {};
    }
    #ifdef DEBUG
    fprintf(stderr, "Symbols with length %d, %d are merged to form pair %d\n", left->len, right->len, len);
    #endif
    Symbol symbol = { .start = left->start, .len = len, .prev = left->prev, .next = right->next };
    *right = (Symbol) {}; // mark entry as invalid, alternatively use memset(right, 0, sizeof(Symbol));
    *left = symbol;
    return symbol;
}


/// @brief Create new SymbolPair based on parameters, and add it into the PriorityQueue
///        if it is a valid token.
/// @param left index of left symbol in symbols array
/// @param right index of right symbol in symbols array
/// @param text input string
/// @param t tokenizer
/// @param pq priority queue of SymbolPairs
static inline void maybe_add_pair(int32_t left, int32_t right, Symbol* symbols, char* text, Tokenizer* t, PQ* pq) {
    // validation for index value of symbols array
    if (left == -1 || right == -1) return;
    Symbol ls = symbols[left];
    Symbol rs = symbols[right];
    int32_t pairlen = ls.len + rs.len;
    char bytes[pairlen];
    for (int32_t i = 0; i < pairlen; i++) {
        bytes[i] = text[i + ls.start];
    }

    uint32_t rank = hash_lookup(t->token_map, bytes, (size_t) pairlen);
    if (rank == INVALID_HASH) {
        #ifdef DEBUG
        fprintf(stderr, "Invalid token\n");
        #endif
        return;
    }

    SymbolPair* pair = (SymbolPair*) malloc(sizeof(SymbolPair));
    if (!pair) {
        #ifdef DEBUG
        fprintf(stderr, "Could not allocate memory for new pair\n");
        #endif
        return;
    }

    pair->left = left;
    pair->right = right;
    pair->len = pairlen;
    pair->rank = rank;
    pq_push(pq, pair);
}

#ifdef DEBUG
void print_pair(SymbolPair* pair, char* text) {
    char* bytes = (char*) malloc(pair->len + 1);
    for (int i = 0; i < (pair->len); i++) {
        bytes[i] = text[i + pair->left];
    }
    bytes[pair->len] = '\0';
    fprintf(stderr, "%s\n", bytes);
    free(bytes);
}


void print_symbol(Symbol* symbol, char* text) {
    char* bytes = (char*) malloc(symbol->len + 1);
    for (int i = 0; i < (symbol->len); i++) {
        bytes[i] = text[i + symbol->start];
    }
    bytes[symbol->len] = '\0';
    fprintf(stderr, "%s", bytes);
    free(bytes);
}
#endif

/// @brief Encode input text into sequence of token ids.
/// @param tokens pointer to output array of token ids
/// @param t tokenizer
/// @param text input prompt
/// @return length of input sequence
size_t tokenizer_encode(uint32_t* tokens, Tokenizer* t, char* text) {
    size_t i, n_symbols = strlen(text);
    if (n_symbols == 0) return 0;
    #ifdef DEBUG
    printf("Input string %s has %zu symbols\n", text, n_symbols);
    #endif

    // initialize symbols array and fill it with individual characters
    Symbol* symbols = (Symbol*) malloc(sizeof(Symbol) * n_symbols);
    if (!symbols) {
        fprintf(stderr, "Failed to allocate symbols array");
        return 0;
    }
    PQ* pairs = pq_init(n_symbols, compare_pairs);
    if (!pairs) {
        free(symbols);
        fprintf(stderr, "Failed to initialize priority queue");
        return 0;
    }

    for (i = 0; i < n_symbols; i++) {
        symbols[i] = (Symbol) {
            .start = (uint32_t) i,
            .len = 1,
            .prev = (int32_t) (i - 1),
            .next = (int32_t) (i + 1),
        };
        // pre-populate priority queue with pairs of symbols
        if (i > 0) maybe_add_pair(i - 1, i, symbols, text, t, pairs);
    }
    symbols[n_symbols - 1].next = -1; // not needed but we set this for clarity

    // iteratively merge symbol pairs, starting from the lowest ranking number
    SymbolPair* pair;
    while ((pair = (SymbolPair*) pq_pop(pairs)) != NULL) {
        #ifdef DEBUG
        fprintf(stderr, "Pair: '");
        print_symbol(&symbols[pair->left], text);
        print_symbol(&symbols[pair->right], text);
        fprintf(stderr, "' with rank %u\n", pair->rank);
        fprintf(stderr, "Symbols: [");
        for (i = 0; i < n_symbols; i++) {
            print_symbol(&symbols[i], text);
            if (i < n_symbols - 1) printf(", ");
        }
        fprintf(stderr, "]\n");
        fprintf(stderr, "Indices: [");
        for (i = 0; i < n_symbols; i++) {
            fprintf(stderr, "%" PRId32, symbols[i].next);
            if (i < n_symbols - 1) printf(", ");
        }
        fprintf(stderr, "]\n");
        #endif

        // make sure that we do not merge pairs that exceed the edges of the array
        if (pair->left != -1 && pair->right != -1) {
            Symbol symbol = merge_symbols(&symbols[pair->left], &symbols[pair->right], pair->len);
            // whole struct should be zero initialised if invalid, we only check length
            if (symbol.len != 0) {
                maybe_add_pair(symbol.prev, pair->left, symbols, text, t, pairs);
                maybe_add_pair(pair->left, symbol.next, symbols, text, t, pairs);
            }
        }
        free(pair);
    }

    // loop through the finalised symbols array to encode the input text
    // allocate memory for the input sequence based on final symbols array
    tokens[0] = t->bos_token;
    size_t j = 1;
    for (i = 0; i < n_symbols; i = symbols[i].next) {
        char* substr = (char*) malloc(symbols[i].len + 1);
        memcpy(substr, &text[symbols[i].start], symbols[i].len);
        substr[symbols[i].len] = '\0';
        uint32_t token_id = hash_lookup(t->token_map, substr, symbols[i].len);
        free(substr);
        if (token_id == INVALID_HASH) continue;
        #ifdef DEBUG
        fprintf(stderr, "substring %s has token id %u\n", substr, token_id);
        #endif
        tokens[j++] = token_id;
    }

    free(symbols);
    pq_free(pairs);

    return j;
}

#endif
