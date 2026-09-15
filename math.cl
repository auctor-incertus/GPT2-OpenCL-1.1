/* math.cl – OpenCL 1.1 kernels for GPT-2 inference. */

#if defined(USE_FP16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half scalar_t;
#elif defined(USE_FP64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double scalar_t;
#else
typedef float scalar_t;
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

/* Attention cache dimensions.  Must match MAX_SEQ_LEN in main.c. */
#define ATTN_MAX_SEQ_LEN 1024

/* Small helper for the fused epilogues. */
static float gelu_f(float v) {
    return 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
}

/* ============================================================
 * Register-blocked GEMM:  C[M,N] = A[M,K] @ B[K,N]  (+ bias)
 *
 * Each work-item computes a 4x4 tile of C; a 16x16 work-group
 * produces a 64x64 block of C over a 64x16 K-slab per iteration.
 * Global size must be ( ceil(N/64)*16 , ceil(M/64)*16 ).
 *
 * Used for prefill only.  Decode uses matvec/matvec4/split-K.
 * ============================================================ */
#define MM_BM  64
#define MM_BN  64
#define MM_BK  16
#define MM_TM   4
#define MM_TN   4
#define MM_WGX 16
#define MM_WGY 16
#define MM_NTH (MM_WGX * MM_WGY)

__kernel void matmul(
    __global const scalar_t *A,
    __global const scalar_t *B,
    __global const scalar_t *bias,
    __global scalar_t       *C,
    const int M, const int N, const int K,
    const int TILE)
{
    (void)TILE;
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);
    const int tid = ty * MM_WGX + tx;

    __local scalar_t Asub[MM_BK][MM_BM + 1];
    __local scalar_t Bsub[MM_BK][MM_BN + 1];

    const int aRowBase = gy * MM_BM;
    const int bColBase = gx * MM_BN;

    scalar_t acc[MM_TM][MM_TN];
    #pragma unroll
    for (int m = 0; m < MM_TM; ++m)
        #pragma unroll
        for (int n = 0; n < MM_TN; ++n)
            acc[m][n] = (scalar_t)0;

    const int nKtiles = (K + MM_BK - 1) / MM_BK;

    for (int kt = 0; kt < nKtiles; ++kt) {
        const int kBase = kt * MM_BK;

        #pragma unroll
        for (int l = 0; l < (MM_BM * MM_BK) / MM_NTH; ++l) {
            const int idx = tid + l * MM_NTH;
            const int r = idx / MM_BK;
            const int c = idx % MM_BK;
            const int aRow = aRowBase + r;
            const int aCol = kBase + c;
            scalar_t v = (scalar_t)0;
            if (aRow < M && aCol < K)
                v = A[(size_t)aRow * K + aCol];
            Asub[c][r] = v;
        }

        #pragma unroll
        for (int l = 0; l < (MM_BK * MM_BN) / MM_NTH; ++l) {
            const int idx = tid + l * MM_NTH;
            const int r = idx / MM_BN;
            const int c = idx % MM_BN;
            const int bRow = kBase + r;
            const int bCol = bColBase + c;
            scalar_t v = (scalar_t)0;
            if (bRow < K && bCol < N)
                v = B[(size_t)bRow * N + bCol];
            Bsub[r][c] = v;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int lk = 0; lk < MM_BK; ++lk) {
            scalar_t aReg[MM_TM];
            scalar_t bReg[MM_TN];
            #pragma unroll
            for (int m = 0; m < MM_TM; ++m)
                aReg[m] = Asub[lk][ty * MM_TM + m];
            #pragma unroll
            for (int n = 0; n < MM_TN; ++n)
                bReg[n] = Bsub[lk][tx * MM_TN + n];
            #pragma unroll
            for (int m = 0; m < MM_TM; ++m)
                #pragma unroll
                for (int n = 0; n < MM_TN; ++n)
                    acc[m][n] += aReg[m] * bReg[n];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int cRowBase = aRowBase + ty * MM_TM;
    const int cColBase = bColBase + tx * MM_TN;
    for (int m = 0; m < MM_TM; ++m) {
        const int r = cRowBase + m;
        if (r >= M) continue;
        for (int n = 0; n < MM_TN; ++n) {
            const int c = cColBase + n;
            if (c >= N) continue;
            scalar_t v = acc[m][n];
            if (bias != NULL) v += bias[c];
            C[(size_t)r * N + c] = v;
        }
    }
}

/* ============================================================
 * Matrix-vector products (M = 1, decode mode).
 *
 * Optional epilogues:
 *   residual  : if non-NULL, add residual[i] to the output
 *   apply_gelu: if nonzero, apply GELU to the output
 *
 * Both are applied after the bias.  Passing residual == C lets a
 * caller do `C = matvec + C_old` in a single kernel (the ranges
 * are disjoint per work-item, so the read-then-write is safe).
 * ============================================================ */

/* Scalar matvec (fallback for N not divisible by 4). */
__kernel void matvec(
    __global const scalar_t *A,
    __global const scalar_t *B,
    __global const scalar_t *bias,
    __global const scalar_t *residual,
    __global scalar_t       *C,
    const int N, const int K,
    const int apply_gelu)
{
    const int col = get_global_id(0);
    if (col >= N) return;

    const __global scalar_t *Bcol = B + col;
    scalar_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

    int k = 0;
    for (; k + 4 <= K; k += 4) {
        s0 += A[k+0] * Bcol[(k+0) * N];
        s1 += A[k+1] * Bcol[(k+1) * N];
        s2 += A[k+2] * Bcol[(k+2) * N];
        s3 += A[k+3] * Bcol[(k+3) * N];
    }
    for (; k < K; ++k)
        s0 += A[k] * Bcol[k * N];

    scalar_t sum = s0 + s1 + s2 + s3;
    if (bias != NULL)     sum += bias[col];
    if (residual != NULL) sum += residual[col];
    if (apply_gelu)       sum = gelu_f(sum);
    C[col] = sum;
}

/* Vectorized matvec: each thread computes 4 output columns. */
__kernel void matvec4(
    __global const float *A,
    __global const float4 *B4,
    __global const float *bias,
    __global const float *residual,
    __global float *C,
    const int N, const int K,
    const int apply_gelu)
{
    const int col4 = get_global_id(0);
    const int c0 = col4 << 2;
    if (c0 >= N) return;

    const int N4 = N >> 2;
    const __global float4 *Bcol = B4 + col4;
    float4 s = (float4)0;

    int k = 0;
    for (; k + 4 <= K; k += 4) {
        s += A[k+0] * Bcol[(k+0) * N4];
        s += A[k+1] * Bcol[(k+1) * N4];
        s += A[k+2] * Bcol[(k+2) * N4];
        s += A[k+3] * Bcol[(k+3) * N4];
    }
    for (; k < K; ++k)
        s += A[k] * Bcol[k * N4];

    if (bias != NULL) {
        s.s0 += bias[c0+0]; s.s1 += bias[c0+1];
        s.s2 += bias[c0+2]; s.s3 += bias[c0+3];
    }
    if (residual != NULL) {
        s.s0 += residual[c0+0]; s.s1 += residual[c0+1];
        s.s2 += residual[c0+2]; s.s3 += residual[c0+3];
    }
    if (apply_gelu) {
        s.s0 = gelu_f(s.s0); s.s1 = gelu_f(s.s1);
        s.s2 = gelu_f(s.s2); s.s3 = gelu_f(s.s3);
    }
    C[c0+0] = s.s0; C[c0+1] = s.s1;
    C[c0+2] = s.s2; C[c0+3] = s.s3;
}

/* Split-K matvec: writes only partials.  Epilogues go in the reduce. */
__kernel void matvec4_sk(
    __global const float  *A,
    __global const float4 *B4,
    __global float4       *partial,
    const int N, const int K, const int kslice)
{
    const int col4 = get_global_id(0);
    const int ks   = get_global_id(1);
    const int N4   = N >> 2;
    if (col4 >= N4 || ks >= kslice) return;

    const __global float4 *Bcol = B4 + col4;
    float4 s = (float4)0;
    for (int k = ks; k < K; k += kslice)
        s += A[k] * Bcol[k * N4];

    partial[(size_t)ks * N4 + col4] = s;
}

/* Reduce the split-K partials, add bias + optional residual, optional GELU. */
__kernel void reduce_partial(
    __global const float4 *partial,
    __global const float  *bias,
    __global const float  *residual,
    __global float        *C,
    const int N, const int kslice,
    const int apply_gelu)
{
    const int col4 = get_global_id(0);
    const int N4   = N >> 2;
    if (col4 >= N4) return;

    float4 s = (float4)0;
    for (int ks = 0; ks < kslice; ++ks)
        s += partial[(size_t)ks * N4 + col4];

    const int c0 = col4 << 2;
    if (bias != NULL) {
        s.s0 += bias[c0+0]; s.s1 += bias[c0+1];
        s.s2 += bias[c0+2]; s.s3 += bias[c0+3];
    }
    if (residual != NULL) {
        s.s0 += residual[c0+0]; s.s1 += residual[c0+1];
        s.s2 += residual[c0+2]; s.s3 += residual[c0+3];
    }
    if (apply_gelu) {
        s.s0 = gelu_f(s.s0); s.s1 = gelu_f(s.s1);
        s.s2 = gelu_f(s.s2); s.s3 = gelu_f(s.s3);
    }
    C[c0+0] = s.s0; C[c0+1] = s.s1;
    C[c0+2] = s.s2; C[c0+3] = s.s3;
}

/* ---------- LayerNorm over the feature axis ---------- */
__kernel void layernorm(
    __global const scalar_t *x,
    __global scalar_t       *out,
    __global const scalar_t *gamma,
    __global const scalar_t *beta,
    const int n_embd,
    const int seq_len)
{
    const int s = get_global_id(0);
    if (s >= seq_len) return;

    __global const scalar_t *xi = x   + (size_t)s * n_embd;
    __global scalar_t       *xo = out + (size_t)s * n_embd;

    scalar_t mean = (scalar_t)0;
    for (int i = 0; i < n_embd; ++i) mean += xi[i];
    mean /= (scalar_t)n_embd;

    scalar_t var = (scalar_t)0;
    for (int i = 0; i < n_embd; ++i) {
        scalar_t d = xi[i] - mean;
        var += d * d;
    }
    var /= (scalar_t)n_embd;

    const scalar_t inv_std = (scalar_t)1 / sqrt(var + (scalar_t)1e-5);
    for (int i = 0; i < n_embd; ++i)
        xo[i] = (xi[i] - mean) * inv_std * gamma[i] + beta[i];
}

/* ---------- Residual add ---------- */
__kernel void add_inplace(
    __global scalar_t       *a,
    __global const scalar_t *b,
    const int n)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    a[i] += b[i];
}

/* ---------- GELU (still used by prefill) ---------- */
__kernel void gelu(__global scalar_t *x, const int n)
{
    const int i = get_global_id(0);
    if (i >= n) return;
    x[i] = gelu_f(x[i]);
}

/* ---------- Token + position embedding ---------- */
__kernel void embed(
    __global const scalar_t *wte,
    __global const scalar_t *wpe,
    __global scalar_t       *out,
    const int n_embd,
    const int seq_len,
    const int start_pos,
    __global const int      *ids)
{
    const int dim = get_global_id(0);
    const int i   = get_global_id(1);
    if (dim >= n_embd || i >= seq_len) return;

    const int pos = start_pos + i;
    const int tok = ids[i];
    out[i * n_embd + dim] = wte[tok * n_embd + dim] +
                            wpe[pos * n_embd + dim];
}

/* ============================================================
 * Split attention (decode + prefill).
 *
 * KV cache layout:
 *   k_cache  : [D, MAX_SEQ]   transposed — k_cache[d * MAX_SEQ + pos]
 *   v_cache  : [MAX_SEQ, D]   unchanged  — v_cache[pos * D + d]
 *
 * The transposed K layout is what makes the score kernel coalesced:
 * at fixed (h, k), threads with consecutive j read consecutive
 * addresses.  In the original [pos, D] layout, the same read has
 * stride D between consecutive j.
 * ============================================================ */

/* Split the QKV output of T rows into K and V caches.
 * Q stays in the qkv buffer (read by attn_scores_1q). */
__kernel void split_kv(
    __global const float *qkv,       /* [T, 3*D] */
    __global float       *k_cache,   /* [D, MAX_SEQ] */
    __global float       *v_cache,   /* [MAX_SEQ, D] */
    const int start_pos,
    const int T,
    const int D,
    const int MAX_SEQ)
{
    int n = get_global_id(0);
    int i = n / D;
    int d = n % D;
    if (i >= T) return;
    int pos = start_pos + i;
    k_cache[d * MAX_SEQ + pos] = qkv[i * 3 * D + D + d];
    v_cache[pos * D + d]       = qkv[i * 3 * D + 2 * D + d];
}

/* Attention scores for one query row.  Grid = { NH * T_kv }.
 * Query lives at qkv[q_base .. q_base + 3*D).  Its Q component is
 * the first D floats; head h's Q slice is at offset h*DH. */
__kernel void attn_scores_1q(
    __global const float *qkv,
    __global const float *k_cache,   /* [D, MAX_SEQ] */
    __global float       *scores,    /* [NH, MAX_SEQ] */
    const int q_base,
    const int T_kv,
    const int D,
    const int DH,
    const int NH,
    const int MAX_SEQ,
    const float scale)
{
    int n = get_global_id(0);
    int h = n / T_kv;
    int j = n % T_kv;
    if (h >= NH) return;

    int off = h * DH;
    __global const float *q_row = qkv + q_base;

    float s = 0;
    for (int k = 0; k < DH; k++)
        s += q_row[off + k] * k_cache[(off + k) * MAX_SEQ + j];
    scores[h * MAX_SEQ + j] = s * scale;
}

/* Softmax over scores[h][0..T_kv-1] in place.  One work-group per head. */
__kernel void attn_softmax_1q(
    __global float *scores,          /* [NH, MAX_SEQ] */
    const int T_kv)
{
    __local float scratch[256];
    int h   = get_group_id(0);
    int lid = get_local_id(0);
    int lws = get_local_size(0);
    __global float *row = scores + (size_t)h * ATTN_MAX_SEQ_LEN;

    float mx = -1e30f;
    for (int j = lid; j < T_kv; j += lws)
        if (row[j] > mx) mx = row[j];
    scratch[lid] = mx;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lws >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            float a = scratch[lid], b = scratch[lid + s];
            scratch[lid] = a > b ? a : b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float m = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;
    for (int j = lid; j < T_kv; j += lws) {
        float e = exp(row[j] - m);
        row[j] = e;
        sum += e;
    }
    scratch[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lws >> 1; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.0f / scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = lid; j < T_kv; j += lws)
        row[j] *= inv;
}

/* Weighted sum of V.  Grid = { D }.  One output element per work-item. */
__kernel void attn_out_1q(
    __global const float *scores,    /* [NH, MAX_SEQ] */
    __global const float *v_cache,   /* [MAX_SEQ, D] */
    __global float       *out,       /* [seq_len, D] */
    const int T_kv,
    const int D,
    const int DH,
    const int out_base)
{
    int d = get_global_id(0);
    if (d >= D) return;
    int h = d / DH;
    __global const float *srow = scores + (size_t)h * ATTN_MAX_SEQ_LEN;
    float s = 0;
    for (int j = 0; j < T_kv; j++)
        s += srow[j] * v_cache[(size_t)j * D + d];
    out[out_base + d] = s;
}
