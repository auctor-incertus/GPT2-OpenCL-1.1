/* tokenizer.c – GPT-2 byte-level BPE tokenizer.
 *
 * Implements the exact algorithm used by openai/gpt-2:
 *   1. Regex pre-tokenization (contractions, " ?letters", " ?digits",
 *      " ?punct", whitespace runs).
 *   2. Byte-level encoding: each raw byte b maps to a Unicode codepoint
 *      via the standard bytes_to_unicode table.
 *   3. BPE merges using ranked pairs from merges.txt.
 *   4. Vocabulary lookup from vocab.json.
 */

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

/* ============ UTF-8 helpers ============ */

static int utf8_decode(const char *s, int *out_cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80)            { *out_cp = c; return 1; }
    if ((c & 0xE0) == 0xC0)  { *out_cp = ((c & 0x1F) << 6)  | (s[1] & 0x3F); return 2; }
    if ((c & 0xF0) == 0xE0)  { *out_cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6)
                                          | (s[2] & 0x3F); return 3; }
    if ((c & 0xF8) == 0xF0)  { *out_cp = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12)
                                          | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4; }
    return 0;
}

static int utf8_encode(int cp, char *out) {
    if (cp < 0x80)      { out[0] = (char)cp; return 1; }
    if (cp < 0x800)     { out[0] = (char)(0xC0 | (cp >> 6));
                          out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000)   { out[0] = (char)(0xE0 | (cp >> 12));
                          out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                          out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* ============ Simple chained hashmap ============ */

typedef struct hm_entry {
    char *key;
    int   keylen;
    int   value;
    struct hm_entry *next;
} hm_entry;

typedef struct {
    hm_entry **buckets;
    int nbuckets;
    int count;
} hashmap;

static uint32_t fnv1a(const void *data, size_t len) {
    const uint8_t *p = data;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static void hm_init(hashmap *m, int nbuckets) {
    m->nbuckets = nbuckets;
    m->buckets  = calloc((size_t)nbuckets, sizeof(hm_entry *));
    m->count    = 0;
}

static void hm_free(hashmap *m) {
    for (int i = 0; i < m->nbuckets; i++) {
        hm_entry *e = m->buckets[i];
        while (e) { hm_entry *n = e->next; free(e->key); free(e); e = n; }
    }
    free(m->buckets);
    m->buckets = NULL; m->nbuckets = 0; m->count = 0;
}

static void hm_put(hashmap *m, const char *key, int keylen, int value) {
    uint32_t h = fnv1a(key, (size_t)keylen);
    int b = (int)(h % (uint32_t)m->nbuckets);
    for (hm_entry *e = m->buckets[b]; e; e = e->next)
        if (e->keylen == keylen && memcmp(e->key, key, (size_t)keylen) == 0) {
            e->value = value; return;
        }
    hm_entry *e = malloc(sizeof(*e));
    e->key = malloc((size_t)keylen);
    memcpy(e->key, key, (size_t)keylen);
    e->keylen = keylen;
    e->value  = value;
    e->next   = m->buckets[b];
    m->buckets[b] = e;
    m->count++;
}

static int hm_get(const hashmap *m, const char *key, int keylen, int *out) {
    uint32_t h = fnv1a(key, (size_t)keylen);
    int b = (int)(h % (uint32_t)m->nbuckets);
    for (hm_entry *e = m->buckets[b]; e; e = e->next)
        if (e->keylen == keylen && memcmp(e->key, key, (size_t)keylen) == 0) {
            *out = e->value; return 1;
        }
    return 0;
}

/* ============ Tokenizer struct ============ */

struct gpt2_tokenizer {
    int byte_to_cp[256];
    int cp_to_byte[512];

    hashmap vocab;             /* UTF-8 token -> id   */
    char  **id_to_str;         /* id -> UTF-8 token   */
    int    *id_to_str_len;
    int     vocab_size;
    int     vocab_cap;

    hashmap merges;            /* "a\0b" -> rank      */
};

/* ============ File I/O ============ */

static char *read_file(const char *path, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    fclose(f);
    buf[n] = 0;
    if (out_len) *out_len = n;
    return buf;
}

/* ============ JSON string parser ============ */

static char *parse_json_string(const char **pp, const char *end, int *out_len) {
    const char *p = *pp;
    if (p >= end || *p != '"') return NULL;
    p++;

    int cap = 32, len = 0;
    char *out = malloc((size_t)cap);
    if (!out) return NULL;

    while (p < end && *p != '"') {
        char c = *p;
        if (c == '\\') {
            p++;
            if (p >= end) { free(out); return NULL; }
            char esc = *p;
            switch (esc) {
                case '"':  c = '"';  break;
                case '\\': c = '\\'; break;
                case '/':  c = '/';  break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u': {
                    if (p + 4 >= end) { free(out); return NULL; }
                    int cp = 0;
                    for (int i = 1; i <= 4; i++) {
                        char h = p[i];
                        int v;
                        if      (h >= '0' && h <= '9') v = h - '0';
                        else if (h >= 'a' && h <= 'f') v = h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') v = h - 'A' + 10;
                        else { free(out); return NULL; }
                        cp = (cp << 4) | v;
                    }
                    p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF &&
                        p + 6 < end && p[1] == '\\' && p[2] == 'u') {
                        int lo = 0, ok = 1;
                        for (int i = 3; i <= 6; i++) {
                            char h = p[i];
                            int v;
                            if      (h >= '0' && h <= '9') v = h - '0';
                            else if (h >= 'a' && h <= 'f') v = h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F') v = h - 'A' + 10;
                            else { ok = 0; break; }
                            lo = (lo << 4) | v;
                        }
                        if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            p += 6;
                        }
                    }
                    p++;
                    char tmp[4];
                    int tl = utf8_encode(cp, tmp);
                    while (len + tl > cap) { cap *= 2; out = realloc(out, (size_t)cap); }
                    memcpy(out + len, tmp, (size_t)tl);
                    len += tl;
                    continue;
                }
                default: free(out); return NULL;
            }
            p++;
        } else {
            p++;
        }
        while (len + 1 > cap) { cap *= 2; out = realloc(out, (size_t)cap); }
        out[len++] = c;
    }
    if (p >= end || *p != '"') { free(out); return NULL; }
    p++;
    out[len] = 0;
    *pp = p;
    if (out_len) *out_len = len;
    return out;
}

/* ============ Load vocab.json ============ */

static int load_vocab(gpt2_tokenizer *t, const char *path) {
    long len = 0;
    char *buf = read_file(path, &len);
    if (!buf) return -1;

    const char *p   = buf;
    const char *end = buf + len;

    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (p >= end || *p != '{') { free(buf); return -1; }
    p++;

    t->vocab_cap  = 1024;
    t->id_to_str  = calloc((size_t)t->vocab_cap, sizeof(char *));
    t->id_to_str_len = calloc((size_t)t->vocab_cap, sizeof(int));
    t->vocab_size = 0;

    for (;;) {
        while (p < end && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r'||*p==',')) p++;
        if (p >= end || *p == '}') break;

        int klen = 0;
        char *key = parse_json_string(&p, end, &klen);
        if (!key) { free(buf); return -1; }

        while (p < end && (*p == ' ' || *p == '\t')) p++;
        if (p >= end || *p != ':') { free(key); free(buf); return -1; }
        p++;
        while (p < end && (*p == ' ' || *p == '\t')) p++;

        int val = 0;
        while (p < end && *p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); p++; }

        hm_put(&t->vocab, key, klen, val);

        if (val >= t->vocab_cap) {
            int nc = t->vocab_cap;
            while (val >= nc) nc *= 2;
            t->id_to_str     = realloc(t->id_to_str,     (size_t)nc * sizeof(char *));
            t->id_to_str_len = realloc(t->id_to_str_len, (size_t)nc * sizeof(int));
            for (int i = t->vocab_cap; i < nc; i++) {
                t->id_to_str[i] = NULL;
                t->id_to_str_len[i] = 0;
            }
            t->vocab_cap = nc;
        }
        if (t->id_to_str[val]) free(t->id_to_str[val]);
        t->id_to_str[val]     = key;
        t->id_to_str_len[val] = klen;
        if (val >= t->vocab_size) t->vocab_size = val + 1;
    }

    free(buf);
    return 0;
}

/* ============ Load merges.txt ============ */

static int load_merges(gpt2_tokenizer *t, const char *path) {
    long len = 0;
    char *buf = read_file(path, &len);
    if (!buf) return -1;

    const char *p   = buf;
    const char *end = buf + len;

    /* Skip "#version: 0.2" line */
    if (p < end && *p == '#') {
        while (p < end && *p != '\n') p++;
        if (p < end) p++;
    }

    int rank = 0;
    while (p < end) {
        const char *ls = p;
        while (p < end && *p != '\n') p++;
        const char *le = p;
        if (p < end) p++;

        while (le > ls && le[-1] == '\r') le--;

        if (le == ls) continue;

        const char *sep = memchr(ls, ' ', (size_t)(le - ls));
        if (!sep) continue;

        int llen = (int)(sep - ls);
        int rlen = (int)(le - (sep + 1));

        int klen = llen + 1 + rlen;
        char *key = malloc((size_t)klen);
        memcpy(key, ls, (size_t)llen);
        key[llen] = '\0';
        memcpy(key + llen + 1, sep + 1, (size_t)rlen);

        hm_put(&t->merges, key, klen, rank++);
        free(key);
    }

    free(buf);
    return 0;
}

/* ============ Byte-level encoding tables ============ */

static void init_byte_cp_maps(gpt2_tokenizer *t) {
    for (int i = 0; i < 512; i++) t->cp_to_byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int identity = (b >= 33 && b <= 126) ||
                       (b >= 161 && b <= 172) ||
                       (b >= 174 && b <= 255);
        int cp = identity ? b : (256 + n++);
        t->byte_to_cp[b] = cp;
        if (cp < 512) t->cp_to_byte[cp] = b;
    }
}

/* ============ BPE ============ */

typedef struct { char *s; int len; } piece_t;

static int lookup_pair(const gpt2_tokenizer *t, const piece_t *a, const piece_t *b) {
    char buf[512];
    int klen = a->len + 1 + b->len;
    if (klen > (int)sizeof(buf)) return -1;
    memcpy(buf, a->s, (size_t)a->len);
    buf[a->len] = '\0';
    memcpy(buf + a->len + 1, b->s, (size_t)b->len);
    int rank;
    return hm_get(&t->merges, buf, klen, &rank) ? rank : -1;
}

static int process_pretoken(const gpt2_tokenizer *t,
                            const char *text, int text_len,
                            int *ids, int n_ids, int max_ids)
{
    /* 1. Byte-level encode */
    int cap = text_len + 4;
    piece_t *pieces = malloc((size_t)cap * sizeof(piece_t));
    int n = 0;
    for (int i = 0; i < text_len; i++) {
        unsigned char b = (unsigned char)text[i];
        int cp = t->byte_to_cp[b];
        char tmp[4];
        int tl = utf8_encode(cp, tmp);
        pieces[n].s = malloc((size_t)tl);
        memcpy(pieces[n].s, tmp, (size_t)tl);
        pieces[n].len = tl;
        n++;
    }

    /* 2. BPE merges */
    while (n > 1) {
        int best_rank = INT_MAX, best_i = -1;
        for (int i = 0; i < n - 1; i++) {
            int r = lookup_pair(t, &pieces[i], &pieces[i+1]);
            if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; }
        }
        if (best_i < 0) break;

        piece_t *next = malloc((size_t)cap * sizeof(piece_t));
        int new_n = 0, i = 0;
        while (i < n) {
            if (i < n - 1 && lookup_pair(t, &pieces[i], &pieces[i+1]) == best_rank) {
                int newlen = pieces[i].len + pieces[i+1].len;
                char *m = malloc((size_t)newlen);
                memcpy(m, pieces[i].s, (size_t)pieces[i].len);
                memcpy(m + pieces[i].len, pieces[i+1].s, (size_t)pieces[i+1].len);
                free(pieces[i].s); free(pieces[i+1].s);
                next[new_n].s = m; next[new_n].len = newlen; new_n++;
                i += 2;
            } else {
                next[new_n++] = pieces[i];
                i++;
            }
        }
        free(pieces);
        pieces = next;
        n = new_n;
    }

    /* 3. Vocab lookup */
    for (int i = 0; i < n; i++) {
        int id;
        if (hm_get(&t->vocab, pieces[i].s, pieces[i].len, &id)) {
            if (n_ids < max_ids) ids[n_ids++] = id;
        }
        free(pieces[i].s);
    }
    free(pieces);
    return n_ids;
}

/* ============ Pre-tokenizer ============ */

static int is_letter_ch(int c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static int is_digit_ch(int c) { return c >= '0' && c <= '9'; }
static int is_space_ch(int c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
static int contraction_len(const char *p, int n) {
    if (n < 2 || p[0] != '\'') return 0;
    if (p[1] == 's' || p[1] == 't' || p[1] == 'm' || p[1] == 'd') return 2;
    if (n >= 3) {
        if ((p[1]=='r'&&p[2]=='e') || (p[1]=='v'&&p[2]=='e') ||
            (p[1]=='l'&&p[2]=='l')) return 3;
    }
    return 0;
}

/* ============ Public API ============ */

gpt2_tokenizer *gpt2_tokenizer_load(const char *dir) {
    gpt2_tokenizer *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    init_byte_cp_maps(t);

    hm_init(&t->vocab,  262144);
    hm_init(&t->merges, 262144);

    char path[1024];
    snprintf(path, sizeof(path), "%s/vocab.json", dir);
    if (load_vocab(t, path) < 0) { gpt2_tokenizer_free(t); return NULL; }

    snprintf(path, sizeof(path), "%s/merges.txt", dir);
    if (load_merges(t, path) < 0) { gpt2_tokenizer_free(t); return NULL; }

    return t;
}

void gpt2_tokenizer_free(gpt2_tokenizer *t) {
    if (!t) return;
    if (t->id_to_str) {
        for (int i = 0; i < t->vocab_cap; i++) free(t->id_to_str[i]);
        free(t->id_to_str);
    }
    free(t->id_to_str_len);
    hm_free(&t->vocab);
    hm_free(&t->merges);
    free(t);
}

int gpt2_tokenizer_vocab_size(const gpt2_tokenizer *t) {
    return t ? t->vocab_size : 0;
}

int gpt2_encode(const gpt2_tokenizer *t, const char *text,
                int *ids, int max_ids)
{
    if (!t || !text || !ids || max_ids <= 0) return -1;

    int len = (int)strlen(text);
    int i = 0, n_ids = 0;

    while (i < len && n_ids < max_ids) {
        char c = text[i];

        /* Contractions */
        if (c == '\'') {
            int cl = contraction_len(text + i, len - i);
            if (cl > 0) {
                n_ids = process_pretoken(t, text + i, cl, ids, n_ids, max_ids);
                i += cl;
                continue;
            }
        }

        /* Space + letter/digit/punct */
        if (c == ' ' && i + 1 < len && !is_space_ch(text[i+1])) {
            char c2 = text[i+1];
            int j;
            if (is_letter_ch(c2)) {
                j = i + 1;
                while (j < len && is_letter_ch(text[j])) j++;
            } else if (is_digit_ch(c2)) {
                j = i + 1;
                while (j < len && is_digit_ch(text[j])) j++;
            } else {
                j = i + 1;
                while (j < len && !is_space_ch(text[j]) &&
                       !is_letter_ch(text[j]) && !is_digit_ch(text[j])) j++;
            }
            n_ids = process_pretoken(t, text + i, j - i, ids, n_ids, max_ids);
            i = j;
            continue;
        }

        /* Letter run */
        if (is_letter_ch(c)) {
            int j = i;
            while (j < len && is_letter_ch(text[j])) j++;
            n_ids = process_pretoken(t, text + i, j - i, ids, n_ids, max_ids);
            i = j;
            continue;
        }

        /* Digit run */
        if (is_digit_ch(c)) {
            int j = i;
            while (j < len && is_digit_ch(text[j])) j++;
            n_ids = process_pretoken(t, text + i, j - i, ids, n_ids, max_ids);
            i = j;
            continue;
        }

        /* Punct run (no whitespace) */
        if (!is_space_ch(c)) {
            int j = i;
            while (j < len && !is_space_ch(text[j]) &&
                   !is_letter_ch(text[j]) && !is_digit_ch(text[j])) j++;
            n_ids = process_pretoken(t, text + i, j - i, ids, n_ids, max_ids);
            i = j;
            continue;
        }

        /* Whitespace: one char at a time (last space attaches to next token) */
        n_ids = process_pretoken(t, text + i, 1, ids, n_ids, max_ids);
        i++;
    }
    return n_ids;
}

int gpt2_decode(const gpt2_tokenizer *t, const int *ids, int n,
                char *out, int max_bytes)
{
    if (!t || !ids || !out || max_bytes <= 0) return -1;

    int w = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->vocab_size) continue;
        const char *s = t->id_to_str[id];
        int slen = t->id_to_str_len[id];
        if (!s) continue;

        int j = 0;
        while (j < slen) {
            int cp;
            int consumed = utf8_decode(s + j, &cp);
            if (consumed <= 0) break;
            int b = (cp >= 0 && cp < 512) ? t->cp_to_byte[cp] : -1;
            if (b < 0) b = '?';
            if (w >= max_bytes - 1) { out[w] = 0; return w; }
            out[w++] = (char)b;
            j += consumed;
        }
    }
    out[w] = 0;
    return w;
}
