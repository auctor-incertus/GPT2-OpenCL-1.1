/* main.c – Hardware-agnostic GPT-2 / DistilGPT2 inference, OpenCL 1.1.
 *
 * Compile: gcc -O3 -o gpt2-opencl11 main.c tokenizer.c -lOpenCL -lm
 * Run:     ./gpt2-opencl11 --model ./gpt2 [options]
 *
 * Model directory must contain model.safetensors, merges.txt, vocab.json
 * and config.json.
 *
 * Options:
 *   --model DIR        model directory (required unless --list-devices)
 *   --device N         force OpenCL device index (see --list-devices)
 *   --list-devices     enumerate OpenCL GPUs and exit
 *   --temperature T    sampling temperature (default 0.8; 0 = greedy)
 *   --top-k K          top-K filter (default 40; 0 = off)
 *   --rep-penalty P    repetition penalty (default 1.1; 1.0 = off)
 *   --max-tokens N     stop after N generated tokens (default 32)
 *   --min-tokens N     forbid EOS for the first N tokens (default 1)
 *   --seed N           RNG seed (default: time(NULL))
 *   --greedy           shorthand for --temperature 0
 *   --prompt TEXT      run one generation and exit
 */

#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

#include <CL/cl.h>
#include "tokenizer.h"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

/* ---------- Hard ceilings ---------- */
#define MAX_EMBED_DIM   1280
#define MAX_LAYERS      36
#define MAX_HEADS       20
#define MAX_VOCAB       50304
#define MAX_SEQ_LEN     1024
#define MAX_FFN         5120

#define MAX_KSLICE      8
#define MAX_TOPK        256
#define MAX_DEVICES     16

#define EOS_TOKEN_ID    50256

/* ---------- Model configuration ---------- */
typedef struct {
    int n_layer, n_embd, n_head, n_ctx, vocab_size, ffn_dim;
} ModelConfig;
static ModelConfig cfg;

static int padded_vocab = 0;

/* ---------- Sampling parameters ---------- */
static float    g_temperature  = 0.8f;
static int      g_top_k        = 40;
static float    g_rep_penalty  = 1.1f;
static unsigned g_seed         = 0;
static int      g_max_tokens   = 32;
static int      g_min_tokens   = 1;
static int      g_device_choice = -1;   /* -1 = auto-select fastest */

/* ---------- Device info ---------- */
typedef enum { FMT_FP32, FMT_FP64 } ComputeFormat;

typedef struct {
    cl_device_id   device;
    cl_platform_id platform;
    char           platform_name[128];
    char           name[256];
    char           vendor[256];
    cl_uint        max_compute_units;
    cl_uint        max_clock_freq;
    cl_ulong       global_mem_size;
    cl_ulong       local_mem_size;
    size_t         max_work_group_size;
    cl_bool        fp16_supported;
    cl_bool        fp64_supported;
    ComputeFormat  best_format;
    double         quick_ms;     /* 256x256 FP32 matmul, for device ranking */
} DeviceInfo;

static DeviceInfo  all_devices[MAX_DEVICES];
static int         ndevices = 0;
static DeviceInfo  dev;             /* the selected device */

/* ---------- Per-shape tuning (matvec decode path) ---------- */
enum ShapeId {
    SHAPE_ATTN_PROJ,   /* N = n_embd,       K = n_embd   */
    SHAPE_QKV,         /* N = 3*n_embd,     K = n_embd   */
    SHAPE_FFN1,        /* N = ffn_dim,      K = n_embd   */
    SHAPE_FFN2,        /* N = n_embd,       K = ffn_dim  */
    SHAPE_LM_HEAD,     /* N = padded_vocab, K = n_embd   */
    SHAPE_COUNT
};

typedef struct {
    int N, K;
    int lws;
    int ks;
} TuneShape;

static TuneShape tune_shapes[SHAPE_COUNT];

/* ---------- Kernel launch parameters for attention / elementwise ---------- */
typedef struct {
    int attn_lws;
    int red_lws;
} DeviceTuning;

static DeviceTuning tune = { 64, 64 };

/* ---------- Matmul tile parameters (chosen by sweep) ---------- */
static int mm_bm  = 64;
static int mm_bn  = 64;
static int mm_wgx = 16;
static int mm_wgy = 16;

static cl_context       ctx;
static cl_command_queue queue;
static cl_program       program;

/* ---------- Safetensors (streaming) ---------- */
static FILE            *st_fp          = NULL;
static uint8_t         *st_header      = NULL;
static int64_t          st_header_size = 0;
static int64_t          st_data_start  = 0;
static safetensors_File st_file;

/* ---------- Kernels ---------- */
static cl_kernel k_matmul, k_matvec, k_matvec4,
                 k_matvec4_sk, k_reduce_partial,
                 k_layernorm, k_gelu, k_embed, k_add,
                 k_split_kv, k_attn_scores, k_attn_softmax, k_attn_out;

/* ---------- Model tensors ---------- */
typedef struct {
    cl_mem wte, wpe;
    cl_mem ln1_w[MAX_LAYERS], ln1_b[MAX_LAYERS];
    cl_mem qkv_w[MAX_LAYERS], qkv_b[MAX_LAYERS];
    cl_mem attn_proj_w[MAX_LAYERS], attn_proj_b[MAX_LAYERS];
    cl_mem ln2_w[MAX_LAYERS], ln2_b[MAX_LAYERS];
    cl_mem ffn1_w[MAX_LAYERS], ffn1_b[MAX_LAYERS];
    cl_mem ffn2_w[MAX_LAYERS], ffn2_b[MAX_LAYERS];
    cl_mem lnf_w, lnf_b;
} Model;
static Model model;
static size_t gpu_bytes_used = 0;
static gpt2_tokenizer *tok = NULL;

static int file_is_conv1d = 0;

static cl_mem lm_head_weight = NULL;
static int    lm_head_is_wte = 0;

/* ---------- Preallocated inference workspaces ---------- */
static cl_mem buf_hidden;
static cl_mem buf_ln_out;
static cl_mem buf_attn;
static cl_mem buf_attn_proj;
static cl_mem buf_ffn1;
static cl_mem buf_ffn2;
static cl_mem buf_qkv_new;
static cl_mem buf_qkv_full;
static cl_mem buf_last_hidden;
static cl_mem buf_logits;
static cl_mem buf_ids;
static cl_mem buf_partial;
static cl_mem buf_scores;
static cl_mem k_cache[MAX_LAYERS];
static cl_mem v_cache[MAX_LAYERS];

/* ---------- Utilities ---------- */
#define CL_CHECK(err, msg) do { \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %d at %s:%d – %s\n", \
                err, __FILE__, __LINE__, msg); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static inline size_t round_up(size_t v, size_t m) {
    return ((v + m - 1) / m) * m;
}

static inline void make_gws2(size_t out[2], size_t nx, size_t ny,
                             const size_t lws[2])
{
    out[0] = round_up(nx, lws[0]);
    out[1] = round_up(ny, lws[1]);
}

static void make_gws_mm(size_t gws[2], int M, int N)
{
    gws[0] = (size_t)((N + 63) / 64) * 16;
    gws[1] = (size_t)((M + 63) / 64) * 16;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static char *slurp(const char *path, int64_t *out_size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return NULL; }
    if (fseeko(fp, 0, SEEK_END) != 0) { perror("fseeko END"); fclose(fp); return NULL; }
    off_t sz_off = ftello(fp);
    if (sz_off < 0) { perror("ftello"); fclose(fp); return NULL; }
    if (fseeko(fp, 0, SEEK_SET) != 0) { perror("fseeko SET"); fclose(fp); return NULL; }

    size_t sz = (size_t)sz_off;
    char *buf = malloc(sz + 1);
    if (!buf) {
        fprintf(stderr, "Cannot allocate %zu bytes for %s\n", sz, path);
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, sz, fp);
    fclose(fp);
    if (got != sz) { fprintf(stderr, "Short read on %s\n", path); free(buf); return NULL; }
    buf[sz] = 0;
    if (out_size) *out_size = (int64_t)sz;
    return buf;
}

static int json_get_int(const char *json, const char *key, int defval) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return defval;
    p = strchr(p, ':');
    if (!p) return defval;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return atoi(p);
}

static void load_config(const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    int64_t sz = 0;
    char *json = slurp(path, &sz);
    if (!json) {
        fprintf(stderr, "Cannot read %s – using GPT-2 small defaults.\n", path);
        cfg.n_layer = 12; cfg.n_embd = 768; cfg.n_head = 12;
        cfg.n_ctx = 1024; cfg.vocab_size = 50257; cfg.ffn_dim = 3072;
        padded_vocab = (cfg.vocab_size + 3) & ~3;
        return;
    }
    cfg.n_layer    = json_get_int(json, "n_layer",    12);
    cfg.n_embd     = json_get_int(json, "n_embd",     768);
    cfg.n_head     = json_get_int(json, "n_head",     12);
    cfg.n_ctx      = json_get_int(json, "n_ctx",      1024);
    cfg.vocab_size = json_get_int(json, "vocab_size", 50257);
    cfg.ffn_dim    = 4 * cfg.n_embd;
    free(json);

    if (cfg.n_embd > MAX_EMBED_DIM || cfg.n_layer > MAX_LAYERS ||
        cfg.n_head > MAX_HEADS || cfg.vocab_size > MAX_VOCAB ||
        cfg.ffn_dim > MAX_FFN   || cfg.n_ctx > MAX_SEQ_LEN) {
        fprintf(stderr, "Config exceeds compiled-in ceilings.\n");
        exit(EXIT_FAILURE);
    }

    padded_vocab = (cfg.vocab_size + 3) & ~3;

    printf("Config: n_layer=%d n_embd=%d n_head=%d ffn=%d vocab=%d ctx=%d\n",
           cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.ffn_dim,
           cfg.vocab_size, cfg.n_ctx);
}

/* ============================================================
 * Device enumeration
 *
 * Every GPU on every platform is enumerated and benchmarked on a
 * 256x256 FP32 matmul.  The fastest is selected, unless --device N
 * forces a specific index.
 * ============================================================ */

static double bench_device(cl_platform_id plat, cl_device_id d)
{
    (void)plat;
    cl_int err;
    cl_context c = clCreateContext(NULL, 1, &d, NULL, NULL, &err);
    if (err != CL_SUCCESS) return 1e30;
    cl_command_queue q = clCreateCommandQueue(c, d, 0, &err);
    if (err != CL_SUCCESS) { clReleaseContext(c); return 1e30; }

    const char *src =
        "__kernel void bench(__global const float *A, __global const float *B,\n"
        "                    __global float *C, const int N) {\n"
        "    int i = get_global_id(0), j = get_global_id(1);\n"
        "    if (i >= N || j >= N) return;\n"
        "    float s = 0;\n"
        "    for (int k = 0; k < N; ++k) s += A[i*N+k] * B[j*N+k];\n"
        "    C[i*N+j] = s;\n"
        "}\n";
    cl_program prog = clCreateProgramWithSource(c, 1, &src, NULL, &err);
    if (err != CL_SUCCESS) { clReleaseCommandQueue(q); clReleaseContext(c); return 1e30; }
    if (clBuildProgram(prog, 1, &d, NULL, NULL, NULL) != CL_SUCCESS) {
        clReleaseProgram(prog);
        clReleaseCommandQueue(q);
        clReleaseContext(c);
        return 1e30;
    }
    cl_kernel k = clCreateKernel(prog, "bench", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        clReleaseCommandQueue(q);
        clReleaseContext(c);
        return 1e30;
    }

    const int N = 256;
    size_t bytes = (size_t)N * N * sizeof(float);
    cl_mem A = clCreateBuffer(c, CL_MEM_READ_ONLY,  bytes, NULL, &err);
    cl_mem B = clCreateBuffer(c, CL_MEM_READ_ONLY,  bytes, NULL, &err);
    cl_mem C = clCreateBuffer(c, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(k); clReleaseProgram(prog);
        clReleaseCommandQueue(q); clReleaseContext(c);
        return 1e30;
    }

    float *host = malloc(bytes);
    for (int i = 0; i < N * N; ++i) host[i] = 1.0f;
    clEnqueueWriteBuffer(q, A, CL_TRUE, 0, bytes, host, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, B, CL_TRUE, 0, bytes, host, 0, NULL, NULL);
    free(host);

    clSetKernelArg(k, 0, sizeof(cl_mem), &A);
    clSetKernelArg(k, 1, sizeof(cl_mem), &B);
    clSetKernelArg(k, 2, sizeof(cl_mem), &C);
    clSetKernelArg(k, 3, sizeof(int), &N);

    size_t lws[2] = { 16, 16 };
    size_t gws[2] = { (size_t)N, (size_t)N };

    clEnqueueNDRangeKernel(q, k, 2, NULL, gws, lws, 0, NULL, NULL);
    clFinish(q);

    const int reps = 20;
    double t0 = now_sec();
    for (int r = 0; r < reps; ++r)
        clEnqueueNDRangeKernel(q, k, 2, NULL, gws, lws, 0, NULL, NULL);
    clFinish(q);
    double dt = (now_sec() - t0) / reps;

    clReleaseMemObject(A); clReleaseMemObject(B); clReleaseMemObject(C);
    clReleaseKernel(k); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(c);
    return dt;
}

static void fill_device_info(DeviceInfo *d) {
    CL_CHECK(clGetDeviceInfo(d->device, CL_DEVICE_NAME, sizeof(d->name), d->name, NULL), "name");
    CL_CHECK(clGetDeviceInfo(d->device, CL_DEVICE_VENDOR, sizeof(d->vendor), d->vendor, NULL), "vendor");
    CL_CHECK(clGetDeviceInfo(d->device, CL_DEVICE_MAX_COMPUTE_UNITS,
              sizeof(d->max_compute_units), &d->max_compute_units, NULL), "cus");
    CL_CHECK(clGetDeviceInfo(d->device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
              sizeof(d->max_clock_freq), &d->max_clock_freq, NULL), "clock");
    CL_CHECK(clGetDeviceInfo(d->device, CL_DEVICE_GLOBAL_MEM_SIZE,
              sizeof(d->global_mem_size), &d->global_mem_size, NULL), "gmem");
    CL_CHECK(clGetDeviceInfo(d->device, CL_DEVICE_LOCAL_MEM_SIZE,
              sizeof(d->local_mem_size), &d->local_mem_size, NULL), "lmem");
    CL_CHECK(clGetDeviceInfo(d->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
              sizeof(d->max_work_group_size), &d->max_work_group_size, NULL), "wg");

    size_t ext_size = 0;
    CL_CHECK(clGetDeviceInfo(d->device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_size), "ext size");
    char *ext = malloc(ext_size);
    CL_CHECK(clGetDeviceInfo(d->device, CL_DEVICE_EXTENSIONS, ext_size, ext, NULL), "ext");
    d->fp16_supported = (strstr(ext, "cl_khr_fp16") != NULL);
    d->fp64_supported = (strstr(ext, "cl_khr_fp64") != NULL);
    free(ext);

    d->best_format = FMT_FP32;
}

static void enumerate_devices(void) {
    cl_uint np = 0;
    if (clGetPlatformIDs(0, NULL, &np) != CL_SUCCESS || np == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        exit(EXIT_FAILURE);
    }
    cl_platform_id *plats = malloc((size_t)np * sizeof(*plats));
    clGetPlatformIDs(np, plats, NULL);

    printf("Enumerating OpenCL GPU devices...\n");

    for (cl_uint pi = 0; pi < np && ndevices < MAX_DEVICES; ++pi) {
        char pname[128] = {0};
        clGetPlatformInfo(plats[pi], CL_PLATFORM_NAME, sizeof pname, pname, NULL);

        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_GPU, 0, NULL, &nd) != CL_SUCCESS || nd == 0)
            continue;
        cl_device_id *devs = malloc((size_t)nd * sizeof(*devs));
        clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_GPU, nd, devs, NULL);

        for (cl_uint di = 0; di < nd && ndevices < MAX_DEVICES; ++di) {
            DeviceInfo *d = &all_devices[ndevices];
            memset(d, 0, sizeof *d);
            d->platform = plats[pi];
            d->device   = devs[di];
            snprintf(d->platform_name, sizeof d->platform_name, "%s", pname);
            fill_device_info(d);
            d->quick_ms = bench_device(d->platform, d->device);
            ndevices++;
        }
        free(devs);
    }
    free(plats);

    if (ndevices == 0) {
        fprintf(stderr, "No OpenCL GPU devices available.\n");
        exit(EXIT_FAILURE);
    }
}

static void print_devices(void) {
    printf("\n");
    printf("  [idx] platform / device                             ms    CUs   VRAM    fp16 fp64\n");
    printf("  -------------------------------------------------------------------------------\n");
    for (int i = 0; i < ndevices; ++i) {
        DeviceInfo *d = &all_devices[i];
        printf("  [%3d] %-22s %-28s %6.3f  %4u  %4.1fG   %s   %s\n",
               i,
               d->platform_name,
               d->name,
               d->quick_ms,
               d->max_compute_units,
               d->global_mem_size / 1073741824.0,
               d->fp16_supported ? "Y" : "n",
               d->fp64_supported ? "Y" : "n");
    }
    printf("\n");
}

static void select_device(void) {
    if (g_device_choice >= 0) {
        if (g_device_choice >= ndevices) {
            fprintf(stderr, "Error: --device %d out of range (0..%d)\n",
                    g_device_choice, ndevices - 1);
            exit(EXIT_FAILURE);
        }
        dev = all_devices[g_device_choice];
        printf("Using device [%d] (forced): %s (%s)\n",
               g_device_choice, dev.name, dev.vendor);
    } else {
        int best = 0;
        for (int i = 1; i < ndevices; ++i)
            if (all_devices[i].quick_ms < all_devices[best].quick_ms) best = i;
        dev = all_devices[best];
        printf("Using device [%d] (fastest, %.3f ms): %s (%s)\n",
               best, dev.quick_ms, dev.name, dev.vendor);
    }

    printf("  Compute units : %u\n", dev.max_compute_units);
    printf("  Clock (MHz)   : %u\n", dev.max_clock_freq);
    printf("  Global mem    : %.2f GiB\n", dev.global_mem_size / 1073741824.0);
    printf("  Local mem     : %llu KiB\n", (unsigned long long)(dev.local_mem_size / 1024));
    printf("  Max WG size   : %zu\n", dev.max_work_group_size);
    printf("  fp16 / fp64   : %s / %s\n",
           dev.fp16_supported ? "yes" : "no",
           dev.fp64_supported ? "yes" : "no");
}

/* ============================================================
 * OpenCL program setup
 * ============================================================ */

static void init_opencl(void) {
    cl_int err;

    ctx = clCreateContext(NULL, 1, &dev.device, NULL, NULL, &err);
    CL_CHECK(err, "context");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    queue = clCreateCommandQueue(ctx, dev.device, CL_QUEUE_PROFILING_ENABLE, &err);
#pragma GCC diagnostic pop
    CL_CHECK(err, "queue");

    int64_t fsize = 0;
    char *src = slurp("math.cl", &fsize);
    if (!src) { perror("math.cl"); exit(1); }

    program = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err);
    CL_CHECK(err, "program");
    err = clBuildProgram(program, 1, &dev.device, "-DUSE_FP32", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, dev.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size + 1);
        clGetProgramBuildInfo(program, dev.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    free(src);

    /* matmul kernel is built separately by sweep_matmul_tile(). */
    k_matmul       = NULL;
    k_matvec       = clCreateKernel(program, "matvec",         &err); CL_CHECK(err, "matvec");
    k_layernorm    = clCreateKernel(program, "layernorm",      &err); CL_CHECK(err, "layernorm");
    k_gelu         = clCreateKernel(program, "gelu",           &err); CL_CHECK(err, "gelu");
    k_embed        = clCreateKernel(program, "embed",          &err); CL_CHECK(err, "embed");
    k_add          = clCreateKernel(program, "add_inplace",    &err); CL_CHECK(err, "add");
    k_split_kv     = clCreateKernel(program, "split_kv",       &err); CL_CHECK(err, "split_kv");
    k_attn_scores  = clCreateKernel(program, "attn_scores_1q", &err); CL_CHECK(err, "attn_scores_1q");
    k_attn_softmax = clCreateKernel(program, "attn_softmax_1q",&err); CL_CHECK(err, "attn_softmax_1q");
    k_attn_out     = clCreateKernel(program, "attn_out_1q",    &err); CL_CHECK(err, "attn_out_1q");

    k_matvec4 = clCreateKernel(program, "matvec4", &err);
    if (err != CL_SUCCESS) { k_matvec4 = NULL; printf("Note: matvec4 not found.\n"); }

    k_matvec4_sk     = clCreateKernel(program, "matvec4_sk",     &err);
    if (err != CL_SUCCESS) k_matvec4_sk = NULL;
    k_reduce_partial = clCreateKernel(program, "reduce_partial", &err);
    if (err != CL_SUCCESS) k_reduce_partial = NULL;
    if (k_matvec4_sk == NULL || k_reduce_partial == NULL) {
        if (k_matvec4_sk)     clReleaseKernel(k_matvec4_sk);
        if (k_reduce_partial) clReleaseKernel(k_reduce_partial);
        k_matvec4_sk = k_reduce_partial = NULL;
        printf("Note: split-K kernels not found.\n");
    }
}

/* ============================================================
 * Matmul tile sweep
 *
 * The matmul kernel's tiling constants are compile-time.  To tune them
 * we build a standalone program containing only the matmul kernel,
 * with tile parameters injected as #defines.  Each candidate config is
 * built, benchmarked on a prefill-shaped workload, and the winner is
 * kept as k_matmul.
 * ============================================================ */

static const char *MATMUL_SRC_FMT =
"#define MM_BM %d\n"
"#define MM_BN %d\n"
"#define MM_BK %d\n"
"#define MM_TM %d\n"
"#define MM_TN %d\n"
"#define MM_WGX (MM_BN / MM_TN)\n"
"#define MM_WGY (MM_BM / MM_TM)\n"
"#define MM_NTH (MM_WGX * MM_WGY)\n"
"\n"
"__kernel void matmul(\n"
"    __global const float *A,\n"
"    __global const float *B,\n"
"    __global const float *bias,\n"
"    __global float       *C,\n"
"    const int M, const int N, const int K,\n"
"    const int TILE)\n"
"{\n"
"    (void)TILE;\n"
"    const int tx = get_local_id(0);\n"
"    const int ty = get_local_id(1);\n"
"    const int gx = get_group_id(0);\n"
"    const int gy = get_group_id(1);\n"
"    const int tid = ty * MM_WGX + tx;\n"
"\n"
"    __local float Asub[MM_BK][MM_BM + 1];\n"
"    __local float Bsub[MM_BK][MM_BN + 1];\n"
"\n"
"    const int aRowBase = gy * MM_BM;\n"
"    const int bColBase = gx * MM_BN;\n"
"\n"
"    float acc[MM_TM][MM_TN];\n"
"    for (int m = 0; m < MM_TM; ++m)\n"
"        for (int n = 0; n < MM_TN; ++n)\n"
"            acc[m][n] = 0.0f;\n"
"\n"
"    const int nKtiles = (K + MM_BK - 1) / MM_BK;\n"
"\n"
"    for (int kt = 0; kt < nKtiles; ++kt) {\n"
"        const int kBase = kt * MM_BK;\n"
"\n"
"        for (int l = 0; l < (MM_BM * MM_BK) / MM_NTH; ++l) {\n"
"            const int idx = tid + l * MM_NTH;\n"
"            const int r = idx / MM_BK;\n"
"            const int c = idx % MM_BK;\n"
"            const int aRow = aRowBase + r;\n"
"            const int aCol = kBase + c;\n"
"            float v = 0.0f;\n"
"            if (aRow < M && aCol < K) v = A[(size_t)aRow * K + aCol];\n"
"            Asub[c][r] = v;\n"
"        }\n"
"\n"
"        for (int l = 0; l < (MM_BK * MM_BN) / MM_NTH; ++l) {\n"
"            const int idx = tid + l * MM_NTH;\n"
"            const int r = idx / MM_BN;\n"
"            const int c = idx % MM_BN;\n"
"            const int bRow = kBase + r;\n"
"            const int bCol = bColBase + c;\n"
"            float v = 0.0f;\n"
"            if (bRow < K && bCol < N) v = B[(size_t)bRow * N + bCol];\n"
"            Bsub[r][c] = v;\n"
"        }\n"
"\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"        for (int lk = 0; lk < MM_BK; ++lk) {\n"
"            float aReg[MM_TM];\n"
"            float bReg[MM_TN];\n"
"            for (int m = 0; m < MM_TM; ++m) aReg[m] = Asub[lk][ty * MM_TM + m];\n"
"            for (int n = 0; n < MM_TN; ++n) bReg[n] = Bsub[lk][tx * MM_TN + n];\n"
"            for (int m = 0; m < MM_TM; ++m)\n"
"                for (int n = 0; n < MM_TN; ++n)\n"
"                    acc[m][n] += aReg[m] * bReg[n];\n"
"        }\n"
"\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"\n"
"    const int cRowBase = aRowBase + ty * MM_TM;\n"
"    const int cColBase = bColBase + tx * MM_TN;\n"
"    for (int m = 0; m < MM_TM; ++m) {\n"
"        const int r = cRowBase + m;\n"
"        if (r >= M) continue;\n"
"        for (int n = 0; n < MM_TN; ++n) {\n"
"            const int c = cColBase + n;\n"
"            if (c >= N) continue;\n"
"            float v = acc[m][n];\n"
"            if (bias != NULL) v += bias[c];\n"
"            C[(size_t)r * N + c] = v;\n"
"        }\n"
"    }\n"
"}\n";

static cl_kernel build_matmul_kernel(int bm, int bn, int bk, int tm, int tn)
{
    char src[8192];
    snprintf(src, sizeof src, MATMUL_SRC_FMT, bm, bn, bk, tm, tn);

    cl_int err;
    const char *sp = src;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &sp, NULL, &err);
    if (err != CL_SUCCESS) return NULL;
    err = clBuildProgram(prog, 1, &dev.device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return NULL;
    }
    cl_kernel k = clCreateKernel(prog, "matmul", &err);
    clReleaseProgram(prog);
    if (err != CL_SUCCESS) return NULL;
    return k;
}

static void sweep_matmul_tile(void)
{
    typedef struct { int bm, bn, bk, tm, tn; const char *name; } Cfg;
    static const Cfg cfgs[] = {
        { 64,  64, 16, 4, 4, "64x64x16/4x4"   },
        { 32,  32, 16, 4, 4, "32x32x16/4x4"   },
        { 64,  64,  8, 4, 4, "64x64x8/4x4"    },
        { 128, 64,  8, 4, 4, "128x64x8/4x4"   },
        { 64, 128,  8, 4, 4, "64x128x8/4x4"   },
        { 32,  64, 16, 4, 8, "32x64x16/4x8"   },
        { 64,  32, 16, 8, 4, "64x32x16/8x4"   },
    };
    const int ncfg = (int)(sizeof(cfgs) / sizeof(cfgs[0]));

    const int M = 64;
    const int N = cfg.n_embd;
    const int K = cfg.ffn_dim;

    cl_int err;
    const size_t a_bytes = (size_t)M * K * sizeof(float);
    const size_t b_bytes = (size_t)K * N * sizeof(float);
    const size_t c_bytes = (size_t)M * N * sizeof(float);

    cl_mem A = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  a_bytes, NULL, &err);
    CL_CHECK(err, "sweep A");
    cl_mem B = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  b_bytes, NULL, &err);
    CL_CHECK(err, "sweep B");
    cl_mem C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, c_bytes, NULL, &err);
    CL_CHECK(err, "sweep C");

    float *host = malloc(a_bytes > b_bytes ? a_bytes : b_bytes);
    if (!host) { fprintf(stderr, "OOM in tile sweep\n"); exit(EXIT_FAILURE); }
    for (size_t i = 0; i < a_bytes / 4; ++i) host[i] = 0.01f;
    clEnqueueWriteBuffer(queue, A, CL_TRUE, 0, a_bytes, host, 0, NULL, NULL);
    for (size_t i = 0; i < b_bytes / 4; ++i) host[i] = 0.01f;
    clEnqueueWriteBuffer(queue, B, CL_TRUE, 0, b_bytes, host, 0, NULL, NULL);
    free(host);

    printf("  Matmul tile sweep (M=%d N=%d K=%d):\n", M, N, K);

    double best_t = 1e30;
    int    best_i = -1;
    cl_kernel kers[16];
    const Cfg *ker_cfg[16];
    int nk = 0;

    for (int i = 0; i < ncfg; ++i) {
        cl_kernel k = build_matmul_kernel(cfgs[i].bm, cfgs[i].bn,
                                          cfgs[i].bk, cfgs[i].tm, cfgs[i].tn);
        if (!k) {
            printf("    %-18s : build failed\n", cfgs[i].name);
            continue;
        }

        const int wgx = cfgs[i].bn / cfgs[i].tn;
        const int wgy = cfgs[i].bm / cfgs[i].tm;

        if (wgx * wgy > (int)dev.max_work_group_size) {
            printf("    %-18s : skipped (wg %d > max %zu)\n",
                   cfgs[i].name, wgx * wgy, dev.max_work_group_size);
            clReleaseKernel(k);
            continue;
        }

        size_t lws[2] = { (size_t)wgx, (size_t)wgy };
        size_t gws[2] = {
            (size_t)((N + cfgs[i].bn - 1) / cfgs[i].bn) * (size_t)wgx,
            (size_t)((M + cfgs[i].bm - 1) / cfgs[i].bm) * (size_t)wgy,
        };

        cl_int Mi = M, Ni = N, Ki = K, tile = 0;
        clSetKernelArg(k, 0, sizeof(cl_mem), &A);
        clSetKernelArg(k, 1, sizeof(cl_mem), &B);
        clSetKernelArg(k, 2, sizeof(cl_mem), NULL);
        clSetKernelArg(k, 3, sizeof(cl_mem), &C);
        clSetKernelArg(k, 4, sizeof(cl_int), &Mi);
        clSetKernelArg(k, 5, sizeof(cl_int), &Ni);
        clSetKernelArg(k, 6, sizeof(cl_int), &Ki);
        clSetKernelArg(k, 7, sizeof(cl_int), &tile);

        clEnqueueNDRangeKernel(queue, k, 2, NULL, gws, lws, 0, NULL, NULL);
        clFinish(queue);

        const int reps = 20;
        double t0 = now_sec();
        for (int r = 0; r < reps; ++r)
            clEnqueueNDRangeKernel(queue, k, 2, NULL, gws, lws, 0, NULL, NULL);
        clFinish(queue);
        double dt = (now_sec() - t0) / reps;

        printf("    %-18s : %8.3f ms\n", cfgs[i].name, dt * 1000.0);

        kers[nk]    = k;
        ker_cfg[nk] = &cfgs[i];
        if (dt < best_t) { best_t = dt; best_i = nk; }
        nk++;
    }

    if (nk == 0) {
        fprintf(stderr, "Matmul sweep failed; no kernel built.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nk; ++i) {
        if (i == best_i) {
            k_matmul = kers[i];
            mm_bm    = ker_cfg[i]->bm;
            mm_bn    = ker_cfg[i]->bn;
            mm_wgx   = ker_cfg[i]->bn / ker_cfg[i]->tn;
            mm_wgy   = ker_cfg[i]->bm / ker_cfg[i]->tm;
        } else {
            clReleaseKernel(kers[i]);
        }
    }
    printf("  Selected matmul tile: %s (%.3f ms)\n",
           ker_cfg[best_i]->name, best_t * 1000.0);

    clReleaseMemObject(A);
    clReleaseMemObject(B);
    clReleaseMemObject(C);
}

/* ============================================================
 * Diagnostic format benchmark (FP32 vs FP64)
 * ============================================================ */

static void benchmark_formats(void) {
    const char *src =
        "__kernel void mm32(__global const float *A, __global const float *B,\n"
        "                   __global float *C, const int N) {\n"
        "    int i = get_global_id(0), j = get_global_id(1);\n"
        "    if (i >= N || j >= N) return;\n"
        "    float s = 0;\n"
        "    for (int k = 0; k < N; ++k) s += A[i*N+k] * B[j*N+k];\n"
        "    C[i*N+j] = s;\n"
        "}\n"
        "#ifdef cl_khr_fp64\n"
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
        "#endif\n"
        "__kernel void mm64(__global const double *A, __global const double *B,\n"
        "                   __global double *C, const int N) {\n"
        "    int i = get_global_id(0), j = get_global_id(1);\n"
        "    if (i >= N || j >= N) return;\n"
        "    double s = 0;\n"
        "    for (int k = 0; k < N; ++k) s += A[i*N+k] * B[j*N+k];\n"
        "    C[i*N+j] = s;\n"
        "}\n";

    cl_int err;
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    CL_CHECK(err, "probe program");
    err = clBuildProgram(prog, 1, &dev.device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); return; }

    cl_kernel k32 = clCreateKernel(prog, "mm32", &err);
    cl_kernel k64 = NULL;
    if (dev.fp64_supported) k64 = clCreateKernel(prog, "mm64", &err);

    const int N = 256;
    size_t bytes_f = (size_t)N * N * sizeof(float);
    size_t bytes_d = (size_t)N * N * sizeof(double);

    cl_mem A32 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes_f, NULL, &err);
    cl_mem B32 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes_f, NULL, &err);
    cl_mem C32 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes_f, NULL, &err);
    cl_mem A64 = NULL, B64 = NULL, C64 = NULL;
    if (k64) {
        A64 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes_d, NULL, &err);
        B64 = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes_d, NULL, &err);
        C64 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes_d, NULL, &err);
    }

    float *hf = malloc(bytes_f);
    for (int i = 0; i < N * N; ++i) hf[i] = 1.0f;
    clEnqueueWriteBuffer(queue, A32, CL_TRUE, 0, bytes_f, hf, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, B32, CL_TRUE, 0, bytes_f, hf, 0, NULL, NULL);
    free(hf);
    if (k64) {
        double *hd = malloc(bytes_d);
        for (int i = 0; i < N * N; ++i) hd[i] = 1.0;
        clEnqueueWriteBuffer(queue, A64, CL_TRUE, 0, bytes_d, hd, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, B64, CL_TRUE, 0, bytes_d, hd, 0, NULL, NULL);
        free(hd);
    }

    size_t lws[2] = { 16, 16 };
    size_t gws[2];
    make_gws2(gws, (size_t)N, (size_t)N, lws);

    clSetKernelArg(k32, 0, sizeof(cl_mem), &A32);
    clSetKernelArg(k32, 1, sizeof(cl_mem), &B32);
    clSetKernelArg(k32, 2, sizeof(cl_mem), &C32);
    clSetKernelArg(k32, 3, sizeof(int), &N);
    if (k64) {
        clSetKernelArg(k64, 0, sizeof(cl_mem), &A64);
        clSetKernelArg(k64, 1, sizeof(cl_mem), &B64);
        clSetKernelArg(k64, 2, sizeof(cl_mem), &C64);
        clSetKernelArg(k64, 3, sizeof(int), &N);
    }

    clEnqueueNDRangeKernel(queue, k32, 2, NULL, gws, lws, 0, NULL, NULL);
    if (k64) clEnqueueNDRangeKernel(queue, k64, 2, NULL, gws, lws, 0, NULL, NULL);
    clFinish(queue);

    const int reps = 20;
    double t0 = now_sec();
    for (int r = 0; r < reps; ++r)
        clEnqueueNDRangeKernel(queue, k32, 2, NULL, gws, lws, 0, NULL, NULL);
    clFinish(queue);
    double t32 = now_sec() - t0;

    double t64 = 1e9;
    if (k64) {
        t0 = now_sec();
        for (int r = 0; r < reps; ++r)
            clEnqueueNDRangeKernel(queue, k64, 2, NULL, gws, lws, 0, NULL, NULL);
        clFinish(queue);
        t64 = now_sec() - t0;
    }

    printf("  Format benchmark (%dx%d matmul, %d reps, diagnostic):\n", N, N, reps);
    printf("    FP32 : %8.3f ms\n", t32 * 1000.0 / reps);
    if (k64) printf("    FP64 : %8.3f ms\n", t64 * 1000.0 / reps);

    clReleaseMemObject(A32); clReleaseMemObject(B32); clReleaseMemObject(C32);
    if (A64) clReleaseMemObject(A64);
    if (B64) clReleaseMemObject(B64);
    if (C64) clReleaseMemObject(C64);
    clReleaseKernel(k32);
    if (k64) clReleaseKernel(k64);
    clReleaseProgram(prog);
}

/* ============================================================
 * Per-shape matvec tuning
 *
 * Sweeps (lws, ks) on each distinct decode-time matvec shape and stores
 * the winner in tune_shapes[].  The LM head inherits FFN2's lws and
 * always uses ks=1 (N ≈ 50000 already provides plenty of parallelism).
 * ============================================================ */

static int sweep_one_shape(int N, int K, int *out_lws, int *out_ks)
{
    *out_lws = 128;
    *out_ks  = 1;

    if (N <= 0 || K <= 0 || (N & 3) != 0 || k_matvec4 == NULL) return -1;

    const int N4 = N >> 2;
    cl_int err;
    const size_t a_bytes = (size_t)K * sizeof(float);
    const size_t b_bytes = (size_t)K * (size_t)N * sizeof(float);
    const size_t c_bytes = (size_t)N * sizeof(float);
    const size_t p_bytes = (size_t)MAX_KSLICE * (size_t)N4 * sizeof(cl_float4);

    cl_mem A = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  a_bytes, NULL, &err);
    cl_mem B = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  b_bytes, NULL, &err);
    cl_mem C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, c_bytes, NULL, &err);
    cl_mem P = clCreateBuffer(ctx, CL_MEM_READ_WRITE, p_bytes, NULL, &err);
    if (err != CL_SUCCESS) return -1;

    float *hostA = malloc(a_bytes);
    float *hostB = malloc(b_bytes);
    if (!hostA || !hostB) { free(hostA); free(hostB); return -1; }
    for (int i = 0; i < K; ++i) hostA[i] = 0.01f;
    for (int i = 0; i < K * N; ++i) hostB[i] = 0.01f;
    clEnqueueWriteBuffer(queue, A, CL_TRUE, 0, a_bytes, hostA, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, B, CL_TRUE, 0, b_bytes, hostB, 0, NULL, NULL);
    free(hostA);
    free(hostB);

    static const int cand_lws[] = { 32, 64, 128, 256 };
    static const int cand_ks[]  = { 1, 2, 4, 8 };
    const int n_lws = (int)(sizeof(cand_lws) / sizeof(cand_lws[0]));
    const int n_ks  = (int)(sizeof(cand_ks)  / sizeof(cand_ks[0]));

    const int have_splitk = (k_matvec4_sk != NULL && k_reduce_partial != NULL);

    int    best_lws = 128;
    int    best_ks  = have_splitk ? 4 : 1;
    double best_t   = 1e30;

    for (int i = 0; i < n_lws; ++i) {
        const int lws = cand_lws[i];
        if ((size_t)lws > dev.max_work_group_size) continue;

        for (int j = 0; j < n_ks; ++j) {
            const int ks = cand_ks[j];
            if (ks > 1 && !have_splitk) continue;
            if (ks > MAX_KSLICE) continue;

            cl_int Ni = N, Ki = K, Ks = ks, G = 0;

            clSetKernelArg(k_matvec4, 0, sizeof(cl_mem), &A);
            clSetKernelArg(k_matvec4, 1, sizeof(cl_mem), &B);
            clSetKernelArg(k_matvec4, 2, sizeof(cl_mem), NULL);
            clSetKernelArg(k_matvec4, 3, sizeof(cl_mem), NULL);
            clSetKernelArg(k_matvec4, 4, sizeof(cl_mem), &C);
            clSetKernelArg(k_matvec4, 5, sizeof(cl_int), &Ni);
            clSetKernelArg(k_matvec4, 6, sizeof(cl_int), &Ki);
            clSetKernelArg(k_matvec4, 7, sizeof(cl_int), &G);

            size_t lws1 = (size_t)lws;
            size_t gws1 = round_up((size_t)N4, lws1);

            size_t lws2[2] = { (size_t)lws, 1 };
            size_t gws2[2] = { round_up((size_t)N4, lws2[0]), (size_t)ks };
            size_t r_lws = (size_t)lws;
            size_t r_gws = round_up((size_t)N4, r_lws);

            if (ks > 1) {
                clSetKernelArg(k_matvec4_sk, 0, sizeof(cl_mem), &A);
                clSetKernelArg(k_matvec4_sk, 1, sizeof(cl_mem), &B);
                clSetKernelArg(k_matvec4_sk, 2, sizeof(cl_mem), &P);
                clSetKernelArg(k_matvec4_sk, 3, sizeof(cl_int), &Ni);
                clSetKernelArg(k_matvec4_sk, 4, sizeof(cl_int), &Ki);
                clSetKernelArg(k_matvec4_sk, 5, sizeof(cl_int), &Ks);

                clSetKernelArg(k_reduce_partial, 0, sizeof(cl_mem), &P);
                clSetKernelArg(k_reduce_partial, 1, sizeof(cl_mem), NULL);
                clSetKernelArg(k_reduce_partial, 2, sizeof(cl_mem), NULL);
                clSetKernelArg(k_reduce_partial, 3, sizeof(cl_mem), &C);
                clSetKernelArg(k_reduce_partial, 4, sizeof(cl_int), &Ni);
                clSetKernelArg(k_reduce_partial, 5, sizeof(cl_int), &Ks);
                clSetKernelArg(k_reduce_partial, 6, sizeof(cl_int), &G);

                clEnqueueNDRangeKernel(queue, k_matvec4_sk, 2, NULL, gws2, lws2, 0, NULL, NULL);
                clEnqueueNDRangeKernel(queue, k_reduce_partial, 1, NULL, &r_gws, &r_lws, 0, NULL, NULL);
            } else {
                clEnqueueNDRangeKernel(queue, k_matvec4, 1, NULL, &gws1, &lws1, 0, NULL, NULL);
            }
            clFinish(queue);

            const int reps = 30;
            double t0 = now_sec();
            for (int r = 0; r < reps; ++r) {
                if (ks > 1) {
                    clEnqueueNDRangeKernel(queue, k_matvec4_sk, 2, NULL, gws2, lws2, 0, NULL, NULL);
                    clEnqueueNDRangeKernel(queue, k_reduce_partial, 1, NULL, &r_gws, &r_lws, 0, NULL, NULL);
                } else {
                    clEnqueueNDRangeKernel(queue, k_matvec4, 1, NULL, &gws1, &lws1, 0, NULL, NULL);
                }
            }
            clFinish(queue);
            const double dt = (now_sec() - t0) / reps;

            if (dt < best_t) { best_t = dt; best_lws = lws; best_ks = ks; }
        }
    }

    if (best_t >= 1e29) {
        best_lws = 128;
        best_ks  = have_splitk ? 4 : 1;
    }

    *out_lws = best_lws;
    *out_ks  = best_ks;

    clReleaseMemObject(A);
    clReleaseMemObject(B);
    clReleaseMemObject(C);
    clReleaseMemObject(P);
    return 0;
}

static void benchmark_tuning_shapes(void)
{
    const int E = cfg.n_embd;
    const int F = cfg.ffn_dim;
    const int T = 3 * E;

    struct { int shape; int N; int K; const char *name; } table[] = {
        { SHAPE_ATTN_PROJ, E, E, "attn_proj" },
        { SHAPE_QKV,       T, E, "qkv"       },
        { SHAPE_FFN1,      F, E, "ffn1"      },
        { SHAPE_FFN2,      E, F, "ffn2"      },
    };
    const int ntab = (int)(sizeof(table) / sizeof(table[0]));

    printf("  Per-shape tuning:\n");

    for (int i = 0; i < ntab; ++i) {
        int lws = 128, ks = 4;
        int rc = sweep_one_shape(table[i].N, table[i].K, &lws, &ks);
        if (rc != 0) {
            printf("    %-10s N=%d K=%d : skipped\n",
                   table[i].name, table[i].N, table[i].K);
            lws = 128; ks = 1;
        }
        tune_shapes[table[i].shape].N   = table[i].N;
        tune_shapes[table[i].shape].K   = table[i].K;
        tune_shapes[table[i].shape].lws = lws;
        tune_shapes[table[i].shape].ks  = ks;
        printf("    %-10s N=%-5d K=%-5d -> lws=%3d ks=%d\n",
               table[i].name, table[i].N, table[i].K, lws, ks);
    }

    tune_shapes[SHAPE_LM_HEAD].N   = padded_vocab;
    tune_shapes[SHAPE_LM_HEAD].K   = E;
    tune_shapes[SHAPE_LM_HEAD].lws = tune_shapes[SHAPE_FFN2].lws;
    tune_shapes[SHAPE_LM_HEAD].ks  = 1;
    printf("    %-10s N=%-5d K=%-5d -> lws=%3d ks=%d (inherited)\n",
           "lm_head", tune_shapes[SHAPE_LM_HEAD].N, E,
           tune_shapes[SHAPE_LM_HEAD].lws, 1);
}

/* ============================================================
 * Tensor loading (streaming)
 * ============================================================ */

static void load_safetensors_file(const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);

    st_fp = fopen(path, "rb");
    if (!st_fp) { perror(path); exit(EXIT_FAILURE); }

    uint8_t len_bytes[8];
    if (fread(len_bytes, 1, 8, st_fp) != 8) {
        fprintf(stderr, "Cannot read header length from %s\n", path);
        exit(EXIT_FAILURE);
    }
    uint64_t header_len_u64 = safetensors_read_le_u64(len_bytes);
    if (header_len_u64 > (uint64_t)INT64_MAX - 16) {
        fprintf(stderr, "Header length implausibly large\n");
        exit(EXIT_FAILURE);
    }
    int64_t header_len = (int64_t)header_len_u64;
    st_header_size = 8 + header_len;
    st_data_start  = st_header_size;

    st_header = malloc((size_t)st_header_size + 16);
    if (!st_header) { fprintf(stderr, "Cannot allocate header\n"); exit(EXIT_FAILURE); }
    memcpy(st_header, len_bytes, 8);
    if (fread(st_header + 8, 1, (size_t)header_len, st_fp) != (size_t)header_len) {
        fprintf(stderr, "Short read on header\n");
        exit(EXIT_FAILURE);
    }

    char *err = safetensors_file_init(st_header, st_header_size, &st_file);
    if (err) { fprintf(stderr, "safetensors: %s\n", err); exit(EXIT_FAILURE); }

    printf("Loaded %d tensors (header %lld bytes, data at %lld).\n",
           st_file.num_tensors,
           (long long)st_header_size, (long long)st_data_start);
}

static size_t tensor_bytes(const safetensors_TensorDescriptor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->n_dimensions; ++i) n *= t->shape[i];
    return (size_t)(n * safetensors_dtype_size(t->dtype));
}

static void read_tensor_data(const safetensors_TensorDescriptor *t, void *dst) {
    int64_t off  = st_data_start + t->begin_offset_bytes;
    int64_t size = t->end_offset_bytes - t->begin_offset_bytes;
    if (size <= 0) { fprintf(stderr, "Tensor non-positive size\n"); exit(EXIT_FAILURE); }
    if (fseeko(st_fp, (off_t)off, SEEK_SET) != 0) { perror("fseeko tensor"); exit(EXIT_FAILURE); }
    if (fread(dst, 1, (size_t)size, st_fp) != (size_t)size) {
        fprintf(stderr, "Short read on tensor\n"); exit(EXIT_FAILURE);
    }
}

static int lookup_flexible(const char *name) {
    int idx = safetensors_lookup(&st_file, name);
    if (idx >= 0) return idx;

    char buf[512];
    snprintf(buf, sizeof(buf), "transformer.%s", name);
    idx = safetensors_lookup(&st_file, buf);
    if (idx >= 0) return idx;

    snprintf(buf, sizeof(buf), "model.%s", name);
    idx = safetensors_lookup(&st_file, buf);
    if (idx >= 0) return idx;
    return -1;
}

static cl_mem load_tensor(const char *name, cl_mem_flags flags) {
    int idx = lookup_flexible(name);
    if (idx < 0) { fprintf(stderr, "Tensor '%s' not found.\n", name); exit(EXIT_FAILURE); }
    safetensors_TensorDescriptor *t = &st_file.tensors[idx];
    if (t->dtype != SAFETENSORS_F32) {
        fprintf(stderr, "Tensor '%s' dtype=%s; only F32 supported.\n",
                name, safetensors_dtype_name(t->dtype));
        exit(EXIT_FAILURE);
    }
    size_t bytes = tensor_bytes(t);

    void *stage = malloc(bytes);
    if (!stage) { fprintf(stderr, "OOM staging %s\n", name); exit(EXIT_FAILURE); }
    read_tensor_data(t, stage);

    cl_int err;
    cl_mem buf = clCreateBuffer(ctx, flags, bytes, NULL, &err);
    CL_CHECK(err, "create buffer");
    CL_CHECK(clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, stage,
                                  0, NULL, NULL), "upload tensor");
    free(stage);
    gpu_bytes_used += bytes;
    return buf;
}

static void detect_weight_layout(void) {
    const char *probe = "h.0.attn.c_attn.weight";
    int idx = lookup_flexible(probe);
    if (idx < 0) { fprintf(stderr, "Cannot find %s\n", probe); exit(EXIT_FAILURE); }
    safetensors_TensorDescriptor *t = &st_file.tensors[idx];
    int r = (int)t->shape[0], c = (int)t->shape[1];
    if (r == cfg.n_embd && c == 3 * cfg.n_embd) {
        file_is_conv1d = 1;
        printf("Weight layout: Conv1D [in=%d, out=%d]\n", r, c);
    } else if (r == 3 * cfg.n_embd && c == cfg.n_embd) {
        file_is_conv1d = 0;
        printf("Weight layout: Linear [out=%d, in=%d]\n", r, c);
    } else {
        fprintf(stderr, "Unexpected shape [%d, %d]\n", r, c);
        exit(EXIT_FAILURE);
    }
}

static cl_mem load_linear_weight(const char *name, int out_dim, int in_dim,
                                 cl_mem_flags flags)
{
    int idx = lookup_flexible(name);
    if (idx < 0) { fprintf(stderr, "Tensor '%s' not found.\n", name); exit(EXIT_FAILURE); }
    safetensors_TensorDescriptor *t = &st_file.tensors[idx];
    int fr = (int)t->shape[0], fc = (int)t->shape[1];
    int need_transpose;
    if (file_is_conv1d) {
        if (fr != in_dim || fc != out_dim) {
            fprintf(stderr, "Tensor '%s': expected [%d,%d], got [%d,%d]\n",
                    name, in_dim, out_dim, fr, fc);
            exit(EXIT_FAILURE);
        }
        need_transpose = 0;
    } else {
        if (fr != out_dim || fc != in_dim) {
            fprintf(stderr, "Tensor '%s': expected [%d,%d], got [%d,%d]\n",
                    name, out_dim, in_dim, fr, fc);
            exit(EXIT_FAILURE);
        }
        need_transpose = 1;
    }

    size_t bytes = (size_t)out_dim * (size_t)in_dim * sizeof(float);
    float *raw = malloc(bytes);
    if (!raw) { fprintf(stderr, "OOM staging %s\n", name); exit(EXIT_FAILURE); }
    read_tensor_data(t, raw);

    float *final = raw;
    if (need_transpose) {
        final = malloc(bytes);
        if (!final) { fprintf(stderr, "OOM transpose\n"); exit(EXIT_FAILURE); }
        const float *src = raw;
        for (int n = 0; n < out_dim; ++n)
            for (int k = 0; k < in_dim; ++k)
                final[(size_t)k * out_dim + n] = src[(size_t)n * in_dim + k];
        free(raw);
    }

    cl_int err;
    cl_mem buf = clCreateBuffer(ctx, flags, bytes, NULL, &err);
    CL_CHECK(err, "create buffer");
    CL_CHECK(clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, bytes, final,
                                  0, NULL, NULL), "upload weight");
    free(final);
    gpu_bytes_used += bytes;
    return buf;
}

static cl_mem make_padded_lm_head(const float *src, int K, int N_real, int N_pad) {
    size_t bytes = (size_t)K * (size_t)N_pad * sizeof(float);
    cl_int err;
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, NULL, &err);
    CL_CHECK(err, "lm head buffer");

    const int BN = 128;
    float *stage = malloc((size_t)K * BN * sizeof(float));
    if (!stage) { fprintf(stderr, "OOM staging lm head\n"); exit(EXIT_FAILURE); }

    for (int n0 = 0; n0 < N_pad; n0 += BN) {
        const int bn = (n0 + BN <= N_pad) ? BN : (N_pad - n0);
        for (int k = 0; k < K; ++k) {
            const float *src_row_base = src + k;
            float *stage_row = stage + (size_t)k * bn;
            for (int i = 0; i < bn; ++i) {
                const int n = n0 + i;
                stage_row[i] = (n < N_real) ? src_row_base[(size_t)n * K] : 0.0f;
            }
        }
        size_t buf_origin[3]  = { (size_t)n0 * sizeof(float), 0, 0 };
        size_t host_origin[3] = { 0, 0, 0 };
        size_t region[3]      = { (size_t)bn * sizeof(float), (size_t)K, 1 };
        size_t buf_row_pitch  = (size_t)N_pad * sizeof(float);
        size_t host_row_pitch = (size_t)bn * sizeof(float);
        CL_CHECK(clEnqueueWriteBufferRect(queue, buf, CL_TRUE,
                                          buf_origin, host_origin, region,
                                          buf_row_pitch, 0,
                                          host_row_pitch, 0,
                                          stage, 0, NULL, NULL),
                 "write lm head rect");
    }
    free(stage);
    gpu_bytes_used += bytes;
    return buf;
}

static void load_model(const char *dir) {
    load_safetensors_file(dir);
    detect_weight_layout();

    model.wte = load_tensor("wte.weight", CL_MEM_READ_ONLY);
    model.wpe = load_tensor("wpe.weight", CL_MEM_READ_ONLY);

    const int E = cfg.n_embd, F = cfg.ffn_dim, T = 3 * cfg.n_embd;

    for (int l = 0; l < cfg.n_layer; ++l) {
        char n[256];
        snprintf(n, sizeof(n), "h.%d.ln_1.weight", l); model.ln1_w[l] = load_tensor(n, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.ln_1.bias",   l); model.ln1_b[l] = load_tensor(n, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.attn.c_attn.weight", l);
        model.qkv_w[l] = load_linear_weight(n, T, E, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.attn.c_attn.bias", l);
        model.qkv_b[l] = load_tensor(n, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.attn.c_proj.weight", l);
        model.attn_proj_w[l] = load_linear_weight(n, E, E, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.attn.c_proj.bias", l);
        model.attn_proj_b[l] = load_tensor(n, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.ln_2.weight", l); model.ln2_w[l] = load_tensor(n, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.ln_2.bias",   l); model.ln2_b[l] = load_tensor(n, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.mlp.c_fc.weight", l);
        model.ffn1_w[l] = load_linear_weight(n, F, E, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.mlp.c_fc.bias", l);
        model.ffn1_b[l] = load_tensor(n, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.mlp.c_proj.weight", l);
        model.ffn2_w[l] = load_linear_weight(n, E, F, CL_MEM_READ_ONLY);
        snprintf(n, sizeof(n), "h.%d.mlp.c_proj.bias", l);
        model.ffn2_b[l] = load_tensor(n, CL_MEM_READ_ONLY);
    }
    model.lnf_w = load_tensor("ln_f.weight", CL_MEM_READ_ONLY);
    model.lnf_b = load_tensor("ln_f.bias",   CL_MEM_READ_ONLY);

    int lm_idx = lookup_flexible("lm_head.weight");
    if (lm_idx < 0) {
        int wte_idx = lookup_flexible("wte.weight");
        safetensors_TensorDescriptor *t = &st_file.tensors[wte_idx];
        size_t src_bytes = tensor_bytes(t);
        float *src = malloc(src_bytes);
        if (!src) { fprintf(stderr, "OOM staging wte\n"); exit(EXIT_FAILURE); }
        read_tensor_data(t, src);
        lm_head_weight = make_padded_lm_head(src, cfg.n_embd,
                                             cfg.vocab_size, padded_vocab);
        free(src);
        lm_head_is_wte = 0;
        printf("LM head: tied to wte.weight (padded [%d,%d])\n",
               cfg.n_embd, padded_vocab);
    } else {
        safetensors_TensorDescriptor *t = &st_file.tensors[lm_idx];
        int r = (int)t->shape[0], c = (int)t->shape[1];
        size_t src_bytes = tensor_bytes(t);
        float *src = malloc(src_bytes);
        if (!src) { fprintf(stderr, "OOM staging lm_head\n"); exit(EXIT_FAILURE); }
        read_tensor_data(t, src);
        if (r == cfg.vocab_size && c == cfg.n_embd) {
            lm_head_weight = make_padded_lm_head(src, cfg.n_embd,
                                                 cfg.vocab_size, padded_vocab);
        } else if (r == cfg.n_embd && c == cfg.vocab_size) {
            float *tmp = malloc((size_t)cfg.vocab_size * cfg.n_embd * sizeof(float));
            if (!tmp) { fprintf(stderr, "OOM\n"); exit(EXIT_FAILURE); }
            for (int v = 0; v < cfg.vocab_size; ++v)
                for (int k = 0; k < cfg.n_embd; ++k)
                    tmp[(size_t)v * cfg.n_embd + k] =
                        src[(size_t)k * cfg.vocab_size + v];
            lm_head_weight = make_padded_lm_head(tmp, cfg.n_embd,
                                                 cfg.vocab_size, padded_vocab);
            free(tmp);
        } else {
            fprintf(stderr, "lm_head.weight: unexpected shape\n");
            exit(EXIT_FAILURE);
        }
        free(src);
        lm_head_is_wte = 0;
        printf("LM head: separate (padded [%d,%d])\n",
               cfg.n_embd, padded_vocab);
    }

    printf("Model VRAM: %.1f MiB\n", gpu_bytes_used / 1048576.0);
}

/* ============================================================
 * Preallocated inference workspaces
 * ============================================================ */

static void preallocate_buffers(void) {
    cl_int err;
    const size_t E = (size_t)cfg.n_embd;
    const size_t F = (size_t)cfg.ffn_dim;
    const size_t T = (size_t)3 * cfg.n_embd;
    const size_t S = (size_t)MAX_SEQ_LEN;
    const size_t V = (size_t)padded_vocab;
    const size_t NH = (size_t)cfg.n_head;

    buf_hidden    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S * E * sizeof(float), NULL, &err); CL_CHECK(err, "buf_hidden");
    buf_ln_out    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S * E * sizeof(float), NULL, &err); CL_CHECK(err, "buf_ln_out");
    buf_attn      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S * E * sizeof(float), NULL, &err); CL_CHECK(err, "buf_attn");
    buf_attn_proj = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S * E * sizeof(float), NULL, &err); CL_CHECK(err, "buf_attn_proj");
    buf_ffn1      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S * F * sizeof(float), NULL, &err); CL_CHECK(err, "buf_ffn1");
    buf_ffn2      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S * E * sizeof(float), NULL, &err); CL_CHECK(err, "buf_ffn2");
    buf_qkv_new   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, T * sizeof(float), NULL, &err); CL_CHECK(err, "buf_qkv_new");
    buf_qkv_full  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S * T * sizeof(float), NULL, &err); CL_CHECK(err, "buf_qkv_full");
    buf_last_hidden = clCreateBuffer(ctx, CL_MEM_READ_WRITE, E * sizeof(float), NULL, &err); CL_CHECK(err, "buf_last_hidden");
    buf_logits      = clCreateBuffer(ctx, CL_MEM_READ_WRITE, V * sizeof(float), NULL, &err); CL_CHECK(err, "buf_logits");
    buf_ids         = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  S * sizeof(cl_int), NULL, &err); CL_CHECK(err, "buf_ids");

    buf_partial = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 (size_t)MAX_KSLICE * (V / 4) * sizeof(cl_float4),
                                 NULL, &err);
    CL_CHECK(err, "buf_partial");

    buf_scores = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                NH * S * sizeof(float), NULL, &err);
    CL_CHECK(err, "buf_scores");

    for (int l = 0; l < cfg.n_layer; ++l) {
        k_cache[l] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, E * S * sizeof(float), NULL, &err);
        CL_CHECK(err, "k_cache");
        v_cache[l] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S * E * sizeof(float), NULL, &err);
        CL_CHECK(err, "v_cache");
    }
}

static void release_buffers(void) {
    clReleaseMemObject(buf_hidden);
    clReleaseMemObject(buf_ln_out);
    clReleaseMemObject(buf_attn);
    clReleaseMemObject(buf_attn_proj);
    clReleaseMemObject(buf_ffn1);
    clReleaseMemObject(buf_ffn2);
    clReleaseMemObject(buf_qkv_new);
    clReleaseMemObject(buf_qkv_full);
    clReleaseMemObject(buf_last_hidden);
    clReleaseMemObject(buf_logits);
    clReleaseMemObject(buf_ids);
    clReleaseMemObject(buf_partial);
    clReleaseMemObject(buf_scores);
    for (int l = 0; l < cfg.n_layer; ++l) {
        clReleaseMemObject(k_cache[l]);
        clReleaseMemObject(v_cache[l]);
    }
}

/* ============================================================
 * Kernel launch helpers
 * ============================================================ */

static void launch_matmul(cl_mem A, cl_mem B, cl_mem bias, cl_mem C,
                          int M, int N, int K)
{
    cl_int Mi = M, Ni = N, Ki = K, tile = 0;
    size_t lws[2] = { (size_t)mm_wgx, (size_t)mm_wgy };
    size_t gws[2];
    gws[0] = (size_t)((N + mm_bn - 1) / mm_bn) * (size_t)mm_wgx;
    gws[1] = (size_t)((M + mm_bm - 1) / mm_bm) * (size_t)mm_wgy;
    clSetKernelArg(k_matmul, 0, sizeof(cl_mem), &A);
    clSetKernelArg(k_matmul, 1, sizeof(cl_mem), &B);
    clSetKernelArg(k_matmul, 2, sizeof(cl_mem), &bias);
    clSetKernelArg(k_matmul, 3, sizeof(cl_mem), &C);
    clSetKernelArg(k_matmul, 4, sizeof(cl_int), &Mi);
    clSetKernelArg(k_matmul, 5, sizeof(cl_int), &Ni);
    clSetKernelArg(k_matmul, 6, sizeof(cl_int), &Ki);
    clSetKernelArg(k_matmul, 7, sizeof(cl_int), &tile);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_matmul, 2, NULL, gws, lws, 0, NULL, NULL), "matmul");
}

static void launch_matvec4_lws(cl_mem A, cl_mem B, cl_mem bias, cl_mem residual,
                               cl_mem C, int N, int K, int lws, int apply_gelu)
{
    cl_int Ni = N, Ki = K, G = apply_gelu;
    size_t l = (size_t)lws;
    size_t gws = round_up((size_t)((N + 3) >> 2), l);
    clSetKernelArg(k_matvec4, 0, sizeof(cl_mem), &A);
    clSetKernelArg(k_matvec4, 1, sizeof(cl_mem), &B);
    clSetKernelArg(k_matvec4, 2, sizeof(cl_mem), &bias);
    clSetKernelArg(k_matvec4, 3, sizeof(cl_mem), &residual);
    clSetKernelArg(k_matvec4, 4, sizeof(cl_mem), &C);
    clSetKernelArg(k_matvec4, 5, sizeof(cl_int), &Ni);
    clSetKernelArg(k_matvec4, 6, sizeof(cl_int), &Ki);
    clSetKernelArg(k_matvec4, 7, sizeof(cl_int), &G);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_matvec4, 1, NULL, &gws, &l, 0, NULL, NULL), "matvec4");
}

static void launch_matvec(cl_mem A, cl_mem B, cl_mem bias, cl_mem residual,
                          cl_mem C, int N, int K, int shape_id, int apply_gelu)
{
    int lws = 128;
    if (shape_id >= 0 && shape_id < SHAPE_COUNT) lws = tune_shapes[shape_id].lws;

    if (k_matvec4 != NULL && (N & 3) == 0) {
        launch_matvec4_lws(A, B, bias, residual, C, N, K, lws, apply_gelu);
        return;
    }
    cl_int Ni = N, Ki = K, G = apply_gelu;
    size_t l = (size_t)lws;
    size_t gws = round_up((size_t)N, l);
    clSetKernelArg(k_matvec, 0, sizeof(cl_mem), &A);
    clSetKernelArg(k_matvec, 1, sizeof(cl_mem), &B);
    clSetKernelArg(k_matvec, 2, sizeof(cl_mem), &bias);
    clSetKernelArg(k_matvec, 3, sizeof(cl_mem), &residual);
    clSetKernelArg(k_matvec, 4, sizeof(cl_mem), &C);
    clSetKernelArg(k_matvec, 5, sizeof(cl_int), &Ni);
    clSetKernelArg(k_matvec, 6, sizeof(cl_int), &Ki);
    clSetKernelArg(k_matvec, 7, sizeof(cl_int), &G);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_matvec, 1, NULL, &gws, &l, 0, NULL, NULL), "matvec");
}

static void launch_matvec_splitk(cl_mem A, cl_mem B, cl_mem bias, cl_mem residual,
                                 cl_mem C, int N, int K, int shape_id,
                                 int kslices, int apply_gelu)
{
    int lws = 128, ks = 4;
    if (shape_id >= 0 && shape_id < SHAPE_COUNT) {
        lws = tune_shapes[shape_id].lws;
        ks  = tune_shapes[shape_id].ks;
    }
    if (kslices > 0) ks = kslices;

    if (k_matvec4_sk == NULL || k_reduce_partial == NULL ||
        (N & 3) != 0 || ks <= 1 || ks > MAX_KSLICE) {
        launch_matvec(A, B, bias, residual, C, N, K, shape_id, apply_gelu);
        return;
    }
    cl_int Ni = N, Ki = K, Ks = ks, G = apply_gelu;
    size_t lws2[2] = { (size_t)lws, 1 };
    size_t gws2[2] = { round_up((size_t)(N >> 2), lws2[0]), (size_t)ks };

    clSetKernelArg(k_matvec4_sk, 0, sizeof(cl_mem), &A);
    clSetKernelArg(k_matvec4_sk, 1, sizeof(cl_mem), &B);
    clSetKernelArg(k_matvec4_sk, 2, sizeof(cl_mem), &buf_partial);
    clSetKernelArg(k_matvec4_sk, 3, sizeof(cl_int), &Ni);
    clSetKernelArg(k_matvec4_sk, 4, sizeof(cl_int), &Ki);
    clSetKernelArg(k_matvec4_sk, 5, sizeof(cl_int), &Ks);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_matvec4_sk, 2, NULL,
                                    gws2, lws2, 0, NULL, NULL), "matvec4_sk");

    const size_t r_lws = (size_t)tune.red_lws;
    const size_t rgws = round_up((size_t)(N >> 2), r_lws);
    clSetKernelArg(k_reduce_partial, 0, sizeof(cl_mem), &buf_partial);
    clSetKernelArg(k_reduce_partial, 1, sizeof(cl_mem), &bias);
    clSetKernelArg(k_reduce_partial, 2, sizeof(cl_mem), &residual);
    clSetKernelArg(k_reduce_partial, 3, sizeof(cl_mem), &C);
    clSetKernelArg(k_reduce_partial, 4, sizeof(cl_int), &Ni);
    clSetKernelArg(k_reduce_partial, 5, sizeof(cl_int), &Ks);
    clSetKernelArg(k_reduce_partial, 6, sizeof(cl_int), &G);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_reduce_partial, 1, NULL,
                                    &rgws, &r_lws, 0, NULL, NULL), "reduce_partial");
}

static void launch_layernorm(cl_mem x, cl_mem out, cl_mem gamma, cl_mem beta,
                             int n_embd, int seq_len)
{
    cl_int E = n_embd, S = seq_len;
    size_t lws = (size_t)tune.red_lws;
    size_t gws = round_up((size_t)seq_len, lws);
    clSetKernelArg(k_layernorm, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k_layernorm, 1, sizeof(cl_mem), &out);
    clSetKernelArg(k_layernorm, 2, sizeof(cl_mem), &gamma);
    clSetKernelArg(k_layernorm, 3, sizeof(cl_mem), &beta);
    clSetKernelArg(k_layernorm, 4, sizeof(cl_int), &E);
    clSetKernelArg(k_layernorm, 5, sizeof(cl_int), &S);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_layernorm, 1, NULL, &gws, &lws, 0, NULL, NULL), "layernorm");
}

static void launch_add(cl_mem a, cl_mem b, int n)
{
    cl_int N = n;
    size_t lws = (size_t)tune.red_lws;
    size_t gws = round_up((size_t)n, lws);
    clSetKernelArg(k_add, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k_add, 1, sizeof(cl_mem), &b);
    clSetKernelArg(k_add, 2, sizeof(cl_int), &N);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_add, 1, NULL, &gws, &lws, 0, NULL, NULL), "add");
}

static void launch_gelu(cl_mem x, int n)
{
    cl_int N = n;
    size_t lws = (size_t)tune.red_lws;
    size_t gws = round_up((size_t)n, lws);
    clSetKernelArg(k_gelu, 0, sizeof(cl_mem), &x);
    clSetKernelArg(k_gelu, 1, sizeof(cl_int), &N);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_gelu, 1, NULL, &gws, &lws, 0, NULL, NULL), "gelu");
}

static void launch_embed(cl_mem out, int seq_len, int start_pos, cl_mem ids)
{
    cl_int E = cfg.n_embd, S = seq_len, SP = start_pos;
    size_t lws[2] = { 16, 1 };
    size_t gws[2] = { round_up((size_t)cfg.n_embd, 16), (size_t)seq_len };
    clSetKernelArg(k_embed, 0, sizeof(cl_mem), &model.wte);
    clSetKernelArg(k_embed, 1, sizeof(cl_mem), &model.wpe);
    clSetKernelArg(k_embed, 2, sizeof(cl_mem), &out);
    clSetKernelArg(k_embed, 3, sizeof(cl_int), &E);
    clSetKernelArg(k_embed, 4, sizeof(cl_int), &S);
    clSetKernelArg(k_embed, 5, sizeof(cl_int), &SP);
    clSetKernelArg(k_embed, 6, sizeof(cl_mem), &ids);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_embed, 2, NULL, gws, lws, 0, NULL, NULL), "embed");
}

static void launch_split_kv(cl_mem qkv, cl_mem kc, cl_mem vc,
                            int start_pos, int T)
{
    cl_int SP = start_pos, TT = T, D = cfg.n_embd, S = MAX_SEQ_LEN;
    size_t lws = (size_t)tune.red_lws;
    size_t gws = round_up((size_t)T * cfg.n_embd, lws);
    clSetKernelArg(k_split_kv, 0, sizeof(cl_mem), &qkv);
    clSetKernelArg(k_split_kv, 1, sizeof(cl_mem), &kc);
    clSetKernelArg(k_split_kv, 2, sizeof(cl_mem), &vc);
    clSetKernelArg(k_split_kv, 3, sizeof(cl_int), &SP);
    clSetKernelArg(k_split_kv, 4, sizeof(cl_int), &TT);
    clSetKernelArg(k_split_kv, 5, sizeof(cl_int), &D);
    clSetKernelArg(k_split_kv, 6, sizeof(cl_int), &S);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_split_kv, 1, NULL, &gws, &lws, 0, NULL, NULL), "split_kv");
}

static void launch_attn_scores(cl_mem qkv, cl_mem kc, int q_base, int T_kv)
{
    cl_int QB = q_base, TK = T_kv, D = cfg.n_embd;
    cl_int DH = cfg.n_embd / cfg.n_head, NH = cfg.n_head;
    cl_int S = MAX_SEQ_LEN;
    float scale = 1.0f / sqrtf((float)DH);
    size_t lws = (size_t)tune.attn_lws;
    size_t gws = round_up((size_t)NH * T_kv, lws);
    clSetKernelArg(k_attn_scores, 0, sizeof(cl_mem), &qkv);
    clSetKernelArg(k_attn_scores, 1, sizeof(cl_mem), &kc);
    clSetKernelArg(k_attn_scores, 2, sizeof(cl_mem), &buf_scores);
    clSetKernelArg(k_attn_scores, 3, sizeof(cl_int), &QB);
    clSetKernelArg(k_attn_scores, 4, sizeof(cl_int), &TK);
    clSetKernelArg(k_attn_scores, 5, sizeof(cl_int), &D);
    clSetKernelArg(k_attn_scores, 6, sizeof(cl_int), &DH);
    clSetKernelArg(k_attn_scores, 7, sizeof(cl_int), &NH);
    clSetKernelArg(k_attn_scores, 8, sizeof(cl_int), &S);
    clSetKernelArg(k_attn_scores, 9, sizeof(float), &scale);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_attn_scores, 1, NULL, &gws, &lws, 0, NULL, NULL), "attn_scores");
}

static void launch_attn_softmax(int T_kv)
{
    cl_int TK = T_kv;
    size_t lws = (size_t)tune.attn_lws;
    size_t gws = (size_t)cfg.n_head * lws;
    clSetKernelArg(k_attn_softmax, 0, sizeof(cl_mem), &buf_scores);
    clSetKernelArg(k_attn_softmax, 1, sizeof(cl_int), &TK);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_attn_softmax, 1, NULL, &gws, &lws, 0, NULL, NULL), "attn_softmax");
}

static void launch_attn_out(cl_mem vc, cl_mem out, int out_base, int T_kv)
{
    cl_int TK = T_kv, D = cfg.n_embd, OB = out_base;
    cl_int DH = cfg.n_embd / cfg.n_head;
    size_t lws = (size_t)tune.attn_lws;
    size_t gws = round_up((size_t)cfg.n_embd, lws);
    clSetKernelArg(k_attn_out, 0, sizeof(cl_mem), &buf_scores);
    clSetKernelArg(k_attn_out, 1, sizeof(cl_mem), &vc);
    clSetKernelArg(k_attn_out, 2, sizeof(cl_mem), &out);
    clSetKernelArg(k_attn_out, 3, sizeof(cl_int), &TK);
    clSetKernelArg(k_attn_out, 4, sizeof(cl_int), &D);
    clSetKernelArg(k_attn_out, 5, sizeof(cl_int), &DH);
    clSetKernelArg(k_attn_out, 6, sizeof(cl_int), &OB);
    CL_CHECK(clEnqueueNDRangeKernel(queue, k_attn_out, 1, NULL, &gws, &lws,
                                    0, NULL, NULL), "attn_out");
}

/* ============================================================
 * Inference: prefill + decode
 * ============================================================ */

static void prefill(const int *ids, int prompt_len)
{
    const int E = cfg.n_embd;
    const int T = 3 * E;
    const int F = cfg.ffn_dim;

    clEnqueueWriteBuffer(queue, buf_ids, CL_TRUE, 0,
                         (size_t)prompt_len * sizeof(cl_int), ids,
                         0, NULL, NULL);

    launch_embed(buf_hidden, prompt_len, 0, buf_ids);

    for (int l = 0; l < cfg.n_layer; ++l) {
        launch_layernorm(buf_hidden, buf_ln_out,
                         model.ln1_w[l], model.ln1_b[l], E, prompt_len);

        launch_matmul(buf_ln_out, model.qkv_w[l], model.qkv_b[l],
                      buf_qkv_full, prompt_len, T, E);
        launch_split_kv(buf_qkv_full, k_cache[l], v_cache[l], 0, prompt_len);

        for (int i = 0; i < prompt_len; ++i) {
            launch_attn_scores(buf_qkv_full, k_cache[l], i * 3 * E, i + 1);
            launch_attn_softmax(i + 1);
            launch_attn_out(v_cache[l], buf_attn, i * E, i + 1);
        }

        launch_matmul(buf_attn, model.attn_proj_w[l], model.attn_proj_b[l],
                      buf_attn_proj, prompt_len, E, E);
        launch_add(buf_hidden, buf_attn_proj, prompt_len * E);

        launch_layernorm(buf_hidden, buf_ln_out,
                         model.ln2_w[l], model.ln2_b[l], E, prompt_len);

        launch_matmul(buf_ln_out, model.ffn1_w[l], model.ffn1_b[l],
                      buf_ffn1, prompt_len, F, E);
        launch_gelu(buf_ffn1, prompt_len * F);
        launch_matmul(buf_ffn1, model.ffn2_w[l], model.ffn2_b[l],
                      buf_ffn2, prompt_len, E, F);
        launch_add(buf_hidden, buf_ffn2, prompt_len * E);
    }

    launch_layernorm(buf_hidden, buf_ln_out,
                     model.lnf_w, model.lnf_b, E, prompt_len);

    clEnqueueCopyBuffer(queue, buf_ln_out, buf_last_hidden,
                        (size_t)(prompt_len - 1) * E * sizeof(float),
                        0, (size_t)E * sizeof(float),
                        0, NULL, NULL);

    launch_matvec(buf_last_hidden, lm_head_weight, NULL, NULL,
                  buf_logits, padded_vocab, E, SHAPE_LM_HEAD, 0);
}

static void decode(int new_token_id, int start_pos)
{
    const int E = cfg.n_embd;
    const int T = 3 * E;
    const int F = cfg.ffn_dim;

    clEnqueueWriteBuffer(queue, buf_ids, CL_TRUE, 0,
                         sizeof(cl_int), &new_token_id, 0, NULL, NULL);

    launch_embed(buf_hidden, 1, start_pos, buf_ids);

    for (int l = 0; l < cfg.n_layer; ++l) {
        launch_layernorm(buf_hidden, buf_ln_out,
                         model.ln1_w[l], model.ln1_b[l], E, 1);

        launch_matvec(buf_ln_out, model.qkv_w[l], model.qkv_b[l],
                      NULL, buf_qkv_new, T, E, SHAPE_QKV, 0);
        launch_split_kv(buf_qkv_new, k_cache[l], v_cache[l], start_pos, 1);

        launch_attn_scores(buf_qkv_new, k_cache[l], 0, start_pos + 1);
        launch_attn_softmax(start_pos + 1);
        launch_attn_out(v_cache[l], buf_attn, 0, start_pos + 1);

        launch_matvec_splitk(buf_attn, model.attn_proj_w[l], model.attn_proj_b[l],
                             buf_hidden, buf_hidden, E, E, SHAPE_ATTN_PROJ, 0, 0);

        launch_layernorm(buf_hidden, buf_ln_out,
                         model.ln2_w[l], model.ln2_b[l], E, 1);

        launch_matvec(buf_ln_out, model.ffn1_w[l], model.ffn1_b[l],
                      NULL, buf_ffn1, F, E, SHAPE_FFN1, 1);

        launch_matvec_splitk(buf_ffn1, model.ffn2_w[l], model.ffn2_b[l],
                             buf_hidden, buf_hidden, E, F, SHAPE_FFN2, 0, 0);
    }

    launch_layernorm(buf_hidden, buf_ln_out,
                     model.lnf_w, model.lnf_b, E, 1);

    launch_matvec(buf_ln_out, lm_head_weight, NULL, NULL,
                  buf_logits, padded_vocab, E, SHAPE_LM_HEAD, 0);
}

/* ============================================================
 * Sampling
 * ============================================================ */

static float kth_largest(const float *a, int n, int k) {
    float heap[MAX_TOPK];
    int hn = 0;
    for (int i = 0; i < n; ++i) {
        const float v = a[i];
        if (hn < k) {
            int j = hn++;
            heap[j] = v;
            while (j > 0) {
                const int p = (j - 1) >> 1;
                if (heap[p] <= heap[j]) break;
                const float t = heap[p]; heap[p] = heap[j]; heap[j] = t;
                j = p;
            }
        } else if (v > heap[0]) {
            heap[0] = v;
            int j = 0;
            for (;;) {
                const int l = 2*j + 1, r = 2*j + 2;
                int m = j;
                if (l < k && heap[l] < heap[m]) m = l;
                if (r < k && heap[r] < heap[m]) m = r;
                if (m == j) break;
                const float t = heap[j]; heap[j] = heap[m]; heap[m] = t;
                j = m;
            }
        }
    }
    return heap[0];
}

/* Sample the next token.  If forbid_eos is nonzero, the EOS logit is
 * masked to -infinity before sampling so the model cannot stop yet.
 * Used to enforce a minimum generation length. */
static int sample_next_token(const int *context_ids, int n_context,
                             int forbid_eos)
{
    float *logits = malloc((size_t)cfg.vocab_size * sizeof(float));
    if (!logits) { fprintf(stderr, "OOM\n"); exit(EXIT_FAILURE); }
    CL_CHECK(clEnqueueReadBuffer(queue, buf_logits, CL_TRUE, 0,
                                 (size_t)cfg.vocab_size * sizeof(float),
                                 logits, 0, NULL, NULL), "read logits");

    /* Mask EOS before rep-penalty / temperature / top-k so the sampler
     * cannot pick it, regardless of the other settings. */
    if (forbid_eos && EOS_TOKEN_ID < cfg.vocab_size)
        logits[EOS_TOKEN_ID] = -1e30f;

    if (g_rep_penalty > 1.0f) {
        for (int i = 0; i < n_context; ++i) {
            const int tk = context_ids[i];
            if (tk < 0 || tk >= cfg.vocab_size) continue;
            int dup = 0;
            for (int j = 0; j < i; ++j)
                if (context_ids[j] == tk) { dup = 1; break; }
            if (dup) continue;
            if (logits[tk] > 0.0f) logits[tk] /= g_rep_penalty;
            else                    logits[tk] *= g_rep_penalty;
        }
    }

    if (g_temperature <= 0.0f) {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < cfg.vocab_size; ++i)
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        free(logits);
        return best;
    }

    const float inv_T = 1.0f / g_temperature;
    for (int i = 0; i < cfg.vocab_size; ++i) logits[i] *= inv_T;

    int k = g_top_k;
    if (k <= 0 || k > cfg.vocab_size) k = cfg.vocab_size;
    if (k > MAX_TOPK) k = MAX_TOPK;

    if (k < cfg.vocab_size) {
        const float thresh = kth_largest(logits, cfg.vocab_size, k);
        for (int i = 0; i < cfg.vocab_size; ++i)
            if (logits[i] < thresh) logits[i] = -1e30f;
    }

    float max_v = -1e30f;
    for (int i = 0; i < cfg.vocab_size; ++i)
        if (logits[i] > max_v) max_v = logits[i];

    double sum = 0.0;
    for (int i = 0; i < cfg.vocab_size; ++i) {
        if (logits[i] <= -1e29f) { logits[i] = 0.0f; continue; }
        const float e = expf(logits[i] - max_v);
        logits[i] = e;
        sum += e;
    }

    const double r = ((double)rand() / ((double)RAND_MAX + 1.0)) * sum;
    double acc = 0.0;
    int chosen = cfg.vocab_size - 1;
    for (int i = 0; i < cfg.vocab_size; ++i) {
        acc += logits[i];
        if (acc >= r) { chosen = i; break; }
    }
    free(logits);
    return chosen;
}

/* ============================================================
 * main
 * ============================================================ */

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *prompt_arg = NULL;
    int list_devices = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            g_device_choice = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--list-devices") == 0) {
            list_devices = 1;
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            g_temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            g_top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
            g_rep_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            g_max_tokens = atoi(argv[++i]);
            if (g_max_tokens < 1) g_max_tokens = 1;
        } else if (strcmp(argv[i], "--min-tokens") == 0 && i + 1 < argc) {
            g_min_tokens = atoi(argv[++i]);
            if (g_min_tokens < 1) g_min_tokens = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            g_seed = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--greedy") == 0) {
            g_temperature = 0.0f;
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt_arg = argv[++i];
        } else if (!model_dir && !list_devices) {
            model_dir = argv[i];
        }
    }

    if (!model_dir && !list_devices) {
        fprintf(stderr,
            "Usage: %s --model <dir> [options]\n"
            "       %s --list-devices\n"
            "Options:\n"
            "  --device N         force OpenCL device index\n"
            "  --temperature T    sampling temperature (default 0.8; 0 = greedy)\n"
            "  --top-k K          top-K filter (default 40; 0 = off)\n"
            "  --rep-penalty P    repetition penalty (default 1.1; 1.0 = off)\n"
            "  --max-tokens N     stop after N generated tokens (default 32)\n"
            "  --min-tokens N     forbid EOS for the first N tokens (default 1)\n"
            "  --seed N           RNG seed (default: time(NULL))\n"
            "  --greedy           shorthand for --temperature 0\n"
            "  --prompt TEXT      run one generation and exit\n",
            argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    enumerate_devices();
    print_devices();
    if (list_devices) return EXIT_SUCCESS;
    select_device();

    if (g_seed == 0) g_seed = (unsigned)time(NULL);
    srand(g_seed);

    if (g_temperature <= 0.0f) {
        printf("Sampling: greedy min_tokens=%d%s\n",
               g_min_tokens,
               g_rep_penalty > 1.0f ? " (with repetition penalty)" : "");
    } else {
        printf("Sampling: temperature=%.3f top_k=%d rep_penalty=%.3f "
               "min_tokens=%d seed=%u\n",
               g_temperature, g_top_k, g_rep_penalty, g_min_tokens, g_seed);
    }

    /* Config first so the shape table knows n_embd / ffn_dim. */
    load_config(model_dir);

    init_opencl();
    sweep_matmul_tile();
    benchmark_tuning_shapes();
    benchmark_formats();

    load_model(model_dir);
    preallocate_buffers();

    tok = gpt2_tokenizer_load(model_dir);
    if (!tok) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", model_dir);
        return EXIT_FAILURE;
    }
    printf("Tokenizer: %d tokens.\n", gpt2_tokenizer_vocab_size(tok));

    int verbose = getenv("VERBOSE") != NULL;
    int interactive = isatty(STDIN_FILENO) && prompt_arg == NULL;

    const int EOS      = EOS_TOKEN_ID;
    const int MAX_NEW  = g_max_tokens;
    const int MIN_NEW  = g_min_tokens;

    char prompt[1024];
    do {
        if (interactive) {
            printf("> ");
            fflush(stdout);
        }

        if (prompt_arg) {
            snprintf(prompt, sizeof(prompt), "%s", prompt_arg);
        } else {
            if (!fgets(prompt, sizeof(prompt), stdin)) break;
            prompt[strcspn(prompt, "\n")] = '\0';
        }

        if (prompt[0] == '\0') {
            if (!interactive) break;
            continue;
        }

        int input_ids[MAX_SEQ_LEN];
        int prompt_len = gpt2_encode(tok, prompt, input_ids, MAX_SEQ_LEN);
        if (prompt_len <= 0) {
            if (!interactive) return EXIT_FAILURE;
            continue;
        }
        int seq_len = prompt_len;

        double t_start = now_sec();

        prefill(input_ids, seq_len);

        /* First sampled token: mask EOS if the caller requested a
         * minimum response length. */
        int next = sample_next_token(input_ids, seq_len,
                                     /*forbid_eos=*/0 < MIN_NEW);
        input_ids[seq_len++] = next;

        char dbuf[256];
        int dn = 0;
        if (next != EOS) {
            dn = gpt2_decode(tok, &next, 1, dbuf, sizeof(dbuf));
            if (dn > 0) fwrite(dbuf, 1, (size_t)dn, stdout);
            fflush(stdout);
        }

        double t_first = now_sec();

        if (verbose) {
            fprintf(stderr, "[prefill %d tok: %.3fs]\n",
                    prompt_len, t_first - t_start);
        }

        int generated = 1;
        while (generated < MAX_NEW && next != EOS && seq_len < MAX_SEQ_LEN) {
            double t_step0 = now_sec();
            decode(input_ids[seq_len - 1], seq_len - 1);

            /* Mask EOS while we are still below the minimum length. */
            int forbid_eos = (generated < MIN_NEW);
            next = sample_next_token(input_ids, seq_len, forbid_eos);
            input_ids[seq_len++] = next;
            generated++;

            if (next == EOS) {
                if (verbose) {
                    fprintf(stderr, "[%d: %.3fs]\n", generated - 1,
                            now_sec() - t_step0);
                }
                break;
            }

            dn = gpt2_decode(tok, &next, 1, dbuf, sizeof(dbuf));
            if (dn > 0) fwrite(dbuf, 1, (size_t)dn, stdout);
            fflush(stdout);

            if (verbose) {
                fprintf(stderr, "[%d: %.3fs]\n", generated - 1,
                        now_sec() - t_step0);
            }
        }

        double t_end = now_sec();
        printf("\n");

        if (!verbose) {
            double t_total   = t_end - t_start;
            double t_prefill = t_first - t_start;
            double t_decode  = t_end - t_first;
            int decode_steps = generated - 1;
            double decode_rate = decode_steps > 0
                                 ? decode_steps / t_decode : 0.0;
            fprintf(stderr,
                "Timing:\n"
                "  prefill+first    : %7.3f s\n"
                "  decode (%3d steps): %7.3f s   (%.1f ms/token)\n"
                "  total            : %7.3f s   (%d tokens, %.2f tok/s)\n",
                t_prefill, decode_steps, t_decode,
                decode_steps > 0 ? 1000.0 * t_decode / decode_steps : 0.0,
                t_total, generated, t_total > 0 ? generated / t_total : 0.0);
        }

        if (!interactive) break;
    } while (1);

    if (k_matmul)         clReleaseKernel(k_matmul);
    clReleaseKernel(k_matvec);
    if (k_matvec4)        clReleaseKernel(k_matvec4);
    if (k_matvec4_sk)     clReleaseKernel(k_matvec4_sk);
    if (k_reduce_partial) clReleaseKernel(k_reduce_partial);
    clReleaseKernel(k_layernorm);
    clReleaseKernel(k_gelu);
    clReleaseKernel(k_embed);
    clReleaseKernel(k_add);
    clReleaseKernel(k_split_kv);
    clReleaseKernel(k_attn_scores);
    clReleaseKernel(k_attn_softmax);
    clReleaseKernel(k_attn_out);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    gpt2_tokenizer_free(tok);
    tok = NULL;
    if (!lm_head_is_wte && lm_head_weight) clReleaseMemObject(lm_head_weight);
    release_buffers();
    if (st_fp) fclose(st_fp);
    free(st_header);
    free(st_file.tensors);
    free(st_file.metadata);
    return EXIT_SUCCESS;
}

