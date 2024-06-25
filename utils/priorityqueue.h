/*
Min Heap implementation to get lowest rank SymbolPair to merge during tokenization.
*/

#ifndef PQ_H
#define PQ_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t size;
    size_t capacity;
    void** heap;
    int (*compare)(const void*, const void*);
} PQ;


PQ* pq_init(size_t capacity, int (*compare)(const void*, const void*)) {
    PQ* pq = (PQ*) malloc(sizeof(PQ));
    if (!pq) {
        return NULL;
    }
    pq->size = 0;
    pq->capacity = capacity;
    pq->heap = (void**) malloc(sizeof(void*) * capacity);
    if (!pq->heap) {
        free(pq);
        return NULL;
    }
    pq->compare = compare;
    return pq;
}


/// @brief Push new item into the priority queue. This is done by inserting the
///        new item at the last entry of the heap array, then calling the bubble
///        up operation.
/// @param pq pointer to PriorityQueue struct
/// @param item pointer to object stored in PriorityQueue
void pq_push(PQ* pq, void* item) {
    if (pq->size >= pq->capacity) {
        fprintf(stderr, "Heap is already full!\n");
        return;
    }
    int p = pq->size;
    pq->heap[p] = item;
    // bubble up operation
    while (p > 0) {
        int parent = (p - 1) / 2;
        // if parent is larger than current element, swap and shift parent down
        if (pq->compare(pq->heap[p], pq->heap[parent]) < 0) {
            void* temp = pq->heap[p];
            pq->heap[p] = pq->heap[parent];
            pq->heap[parent] = temp;
            p = parent;
        } else {
            break;
        }
    }
    pq->size++;
}


/// @brief Bubble down the item at specified index of heap.
/// @param pq 
/// @param i 
static inline void shift_down(PQ* pq, int i) {
    int l = 2 * i + 1; // index of left child
    while (l < pq->size) {
        int r = l + 1;
        // if right child exists and has lower value than left child, we compare
        // our item i against the right child instead
        if (r < pq->size && pq->compare(pq->heap[r], pq->heap[l]) < 0)
            l = r;

        // if current child is smaller than item i, swap their positions
        if (pq->compare(&pq->heap[l], pq->heap[i]) < 0) {
            void* temp = pq->heap[i];
            pq->heap[i] = pq->heap[l];
            pq->heap[l] = temp;
            i = l;
            l = 2 * i + 1;
        } else {
            break;
        }
    }
}


/// @brief Pops smallest element within the heap. We do this by taking the root
///        item (index 0), replacing the root element with the last item, then
///        calling the bubble down operation on it.
/// @param pq pointer to PriorityQueue struct
void* pq_pop(PQ* pq) {
    if (pq->size == 0) {
        return NULL;
    }
    void* root = pq->heap[0];
    pq->heap[0] = pq->heap[--pq->size];
    shift_down(pq, 0);
    return root;
}


void pq_free(PQ* pq) {
    free(pq->heap);
    free(pq);
}

#endif
