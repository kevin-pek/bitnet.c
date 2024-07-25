/**
 * Hash table implementation for insertion and lookup from string keys to integer values.
 * Simple implementation with linear probing and single element buckets.
 */

#ifndef HASH_H
#define HASH_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "logging.h"

#define INVALID_HASH UINT32_MAX

typedef struct {
    uint32_t key;
    uint32_t val; // unsigned int as max value of signed int is not large enough
    uint8_t occupied;
} HTBucket;

typedef struct {
    HTBucket* buckets;
    uint32_t (*hashfn)(const char*, size_t);
    uint32_t len;
} HT;


HT* hash_init(size_t len, uint32_t (*hashfn)(const char*, size_t)) {
    HT* ht = (HT*) malloc(sizeof(HT));
    if (!ht) return NULL;
    ht->len = len;
    ht->hashfn = hashfn;
    ht->buckets = (HTBucket*) malloc(sizeof(HTBucket) * len);
    if (!ht->buckets) {
        free(ht);
        return NULL;
    }
    memset(ht->buckets, 0, sizeof(HTBucket) * len); // use 0 to indicate unassigned entry
    return ht;
}


void hash_free(HT* ht) {
    free(ht->buckets);
}


static inline uint32_t hash_dbj2(const char* str, size_t len) {
    uint32_t hashval = 5381;
    for (int i = 0; i < len; i++) {
        hashval = ((hashval << 5) + hashval) + str[i]; // hash X 33 + char ascii value
    }
    return hashval;
}


static inline uint32_t hash_fnv1a(const char* str, size_t len) {
    uint32_t hashval = 2166136261u;
    for (int i = 0; i < len; i++) {
        hashval ^= (uint32_t) str[i];
        hashval *= 16777619u;
    }
    return hashval;
}

// murmurhash3 function. Taken from https://github.com/jwerle/murmurhash.c/blob/master/murmurhash.c
static inline uint32_t hash_murmur3(const char* str, size_t len) {
    uint32_t c1 = 0xcc9e2d51;
    uint32_t c2 = 0x1b873593;
    uint32_t r1 = 15;
    uint32_t r2 = 13;
    uint32_t m = 5;
    uint32_t n = 0xe6546b64;

    uint32_t hashval = 5381;

    const int nblocks = len / 4;
    const uint32_t *blocks = (const uint32_t *)(str);
    int i;
    for (i = 0; i < nblocks; i++) {
        uint32_t k = blocks[i];

        k *= c1;
        k = (k << r1) | (k >> (32 - r1));
        k *= c2;

        hashval ^= k;
        hashval = (hashval << r2) | (hashval >> (32 - r2));
        hashval = hashval * m + n;
    }

    const uint8_t *tail = (const uint8_t *)(str + nblocks * 4);
    uint32_t k1 = 0;

    switch (len & 3) {
        case 3:
            k1 ^= tail[2] << 16;
        case 2:
            k1 ^= tail[1] << 8;
        case 1:
            k1 ^= tail[0];
            k1 *= c1;
            k1 = (k1 << r1) | (k1 >> (32 - r1));
            k1 *= c2;
            hashval ^= k1;
    }

    hashval ^= len;

    hashval ^= hashval >> 16;
    hashval *= 0x85ebca6b;
    hashval ^= hashval >> 13;
    hashval *= 0xc2b2ae35;
    hashval ^= hashval >> 16;

    return hashval;
}

/// @brief Inserts or updates value for a given string key.
/// @param ht pointer to hash table struct
/// @param str string key to hash
/// @param value value to store in hash table entry/bucket
/// @returns exit code, 0 if successful
int hash_insert(HT* ht, char* str, size_t len, uint32_t value) {
    #ifdef DEBUG
    clock_t start, end;
    start = clock();
    #endif

    uint32_t key = ht->hashfn(str, len);
    int i = key % ht->len;
    int initial = i;
    while (ht->buckets[i].occupied && ht->buckets[i].key != key) {
        i = (i + 1) % ht->len; // linear probing, wrap around to start if exceed size
        if (initial == i) return 1; // return error when we have traversed the entire table
    }
    HTBucket entry = { .key = key, .val = value, .occupied = 1 };
    ht->buckets[i] = entry;

    #ifdef DEBUG
    end = clock();
    fprintf(stderr, "Hash insertion took %f\n", ((double) (end - start)) / CLOCKS_PER_SEC);
    #endif

    return 0;
}


uint32_t hash_lookup(HT* ht, char* str, size_t len) {
    uint32_t key = ht->hashfn(str, len);
    int i = key % ht->len;
    while (ht->buckets[i].occupied && ht->buckets[i].key != key)
        i = (i + 1) % ht->len; // linear probing, wrap around to start if exceed size
    if (ht->buckets[i].occupied == 0) {
        #ifdef DEBUG
        fprintf(stderr, "No value found for %s\n", str);
        #endif
        return INVALID_HASH;
    }
    return ht->buckets[i].val;
}

#endif
