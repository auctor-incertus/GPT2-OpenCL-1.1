#ifndef GPT2_TOKENIZER_H
#define GPT2_TOKENIZER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpt2_tokenizer gpt2_tokenizer;

/* Load vocab.json and merges.txt from the given directory. NULL on failure. */
gpt2_tokenizer *gpt2_tokenizer_load(const char *dir);
void            gpt2_tokenizer_free(gpt2_tokenizer *t);

/* Encode UTF-8 text to token IDs. Returns count written, or -1 on error. */
int gpt2_encode(const gpt2_tokenizer *t, const char *text,
                int *ids, int max_ids);

/* Decode token IDs to UTF-8. Returns bytes written (excluding NUL). */
int gpt2_decode(const gpt2_tokenizer *t, const int *ids, int n,
                char *out, int max_bytes);

int gpt2_tokenizer_vocab_size(const gpt2_tokenizer *t);

#ifdef __cplusplus
}
#endif

#endif
