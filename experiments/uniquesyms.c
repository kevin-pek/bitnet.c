/*
Determine the number of unique starting characters within the tokenizer vocab.
*/

#include <stdio.h>
#include <string.h>
#include "../utils/b64.h"

#define SYMBOLS 256 // max number of values represented by a single byte

int main() {
    FILE *file = fopen("tokenizer.model", "rb");
    if (!file) {
        perror("Failed to open file.\n");
        return 1;
    }

    int symbols[SYMBOLS] = {0};
    int start_sym[SYMBOLS] = {0};
    int total_len = 0;
    char buffer[1024];
    char *b64_str;
    
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        b64_str = strtok(buffer, " ");
        if (b64_str != NULL) {
            DecodedString decoded_str = b64_decode(b64_str);
            total_len += decoded_str.len;
            start_sym[(unsigned char) decoded_str.str[0]]++;
            unsigned char* p = (unsigned char*) decoded_str.str;
            int c;
            while ((c = *p++))
                symbols[c]++;

            free(decoded_str.str);
        } else {
            printf("NULL string encountered\n");
        }
    }

    for (int i = 0; i < SYMBOLS; i++)
        printf("character %c (ASCII %d) is starting symbol for %d subwords\n", i, i, start_sym[i]);

    int sum = 0;
    for (int i = 0; i < SYMBOLS; i++) {
        printf("character %c (ASCII %d) has frequency %d\n", i, i, symbols[i]);
        sum += symbols[i];
    }

    printf("Total characters in corpus: %d\n", total_len);

    fclose(file);
}