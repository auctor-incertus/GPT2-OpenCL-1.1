> [!NOTE]
> This repository has been vibe-coded with DeepSeek.

# GPT2-OpenCL-1.1

**Hardware-agnostic adaptive GPT-2 inference for ancient outdated OpenCL 1.1 devices**

A high-performance GPT-2 inference engine optimized for outdated graphics cards stuck on OpenCL 1.1—the devices that `llama.cpp` and `ollama` don't support. This project brings modern language model inference to decade-old GPUs through adaptive kernel selection, device-specific tuning, and low-level OpenCL optimization.

[safetensors.h](https://github.com/hsnyder/safetensors.h) is used to read the safetensor files.

## Demonstration

```
$ ~/gpt2-opencl11$ ./gpt2-opencl11 --model ./gpt2-small --max-tokens 512 --min-tokens 64
Enumerating OpenCL GPU devices...

  [idx] platform / device                             ms    CUs   VRAM    fp16 fp64
  -------------------------------------------------------------------------------
  [  0] NVIDIA CUDA            GeForce GT 730                0.008     2   3.9G   n   Y

Using device [0] (fastest, 0.008 ms): GeForce GT 730 (NVIDIA Corporation)
  Compute units : 2
  Clock (MHz)   : 1400
  Global mem    : 3.93 GiB
  Local mem     : 48 KiB
  Max WG size   : 1024
  fp16 / fp64   : no / yes
Sampling: temperature=0.800 top_k=40 rep_penalty=1.100 min_tokens=64 seed=1789469180
Config: n_layer=12 n_embd=768 n_head=12 ffn=3072 vocab=50257 ctx=1024
  Matmul tile sweep (M=64 N=768 K=3072):
    64x64x16/4x4       :    3.886 ms
    32x32x16/4x4       :    4.490 ms
    64x64x8/4x4        :    4.498 ms
    128x64x8/4x4       :    9.079 ms
    64x128x8/4x4       :    5.179 ms
    32x64x16/4x8       :    8.266 ms
    64x32x16/8x4       :   10.525 ms
  Selected matmul tile: 64x64x16/4x4 (3.886 ms)
  Per-shape tuning:
    attn_proj  N=768   K=768   -> lws=128 ks=8
    qkv        N=2304  K=768   -> lws=256 ks=2
    ffn1       N=3072  K=768   -> lws= 64 ks=4
    ffn2       N=768   K=3072  -> lws=128 ks=8
    lm_head    N=50260 K=768   -> lws=128 ks=1 (inherited)
  Format benchmark (256x256 matmul, 20 reps, diagnostic):
    FP32 :    7.846 ms
    FP64 :   19.586 ms
Loaded 160 tensors (header 14291 bytes, data at 14291).
Weight layout: Conv1D [in=768, out=2304]
LM head: tied to wte.weight (padded [768,50260])
Model VRAM: 621.9 MiB
Tokenizer: 50257 tokens.
> Hello.
 The last bit of information you can tell is that the first time it was posted, we received no response; this led to us trying various websites and email accounts with our domain name registration in [...]
So how do one get started? Well let me start by saying two things before i say anything more because my friends arent like most people who write articles but if anyone has seen much discussion around [...]
Timing:
  prefill+first    :   0.149 s
  decode (234 steps):   9.403 s   (40.2 ms/token)
  total            :   9.552 s   (235 tokens, 24.60 tok/s)
```

## Features

- **OpenCL 1.1 compatible**: Runs on decade-old GPUs, integrated graphics, and legacy compute devices
- **Adaptive kernel selection**: Automatically selects optimal compute paths based on device capabilities
- **Hardware tuning**: Profiles device performance and adapts matrix multiplication tile sizes at runtime
- **Format flexibility**: Supports FP32, FP64, and FP16 inference (with appropriate extensions)
- **Safetensors support**: Loads pretrained weights from safetensors format files
- **Efficient kernels**: Specialized compute kernels for prefill (matrix-matrix) and decode (matrix-vector) phases
- **Split-K acceleration**: Advanced GPU optimization for large matrix-vector products
- **Zero external dependencies**: Pure C with only POSIX and OpenCL 1.1 requirements

## Architecture

### Core Components

**`main.c`** — OpenCL runtime and inference orchestration
- Device probing and capability detection
- Model loading from safetensors files
- Benchmark-driven format selection (FP32 vs FP64 vs FP16)
- Device-specific tuning (matrix tile sizes, work-group dimensions)
- Prefill and decode loop implementation
- Attention mechanism orchestration

**`math.cl`** — OpenCL compute kernels
- **Register-blocked GEMM** (`matmul`): 64×64 tile-based matrix multiplication for prompt encoding
- **Scalar matvec** (`matvec`): Fallback matrix-vector product with loop unrolling
- **Vectorized matvec** (`matvec4`): SIMD-friendly 4-column-per-thread implementation
- **Split-K matvec** (`matvec4_sk` + `reduce_partial`): High-bandwidth alternative for decode bottlenecks
- **LayerNorm** (`layernorm`): Per-sequence normalization with optional bias fusing
- **Attention kernels**:
  - `split_kv`: Transposed K-cache layout for coalesced memory access
  - `attn_scores_1q`: Per-head attention score computation
  - `attn_softmax_1q`: Numerically stable softmax with local reduction
  - `attn_out_1q`: Weighted sum over attention cache
- **Utilities**: Element-wise add, GELU activation, token+position embedding

**`tokenizer.c` / `tokenizer.h`** — GPT-2 byte-level BPE tokenizer
- UTF-8 aware string handling
- Byte-level encoding with standard Unicode mapping table
- Ranked pair-based BPE merges
- Vocabulary lookup and string reconstruction
- Handles contractions, multi-word tokens, and punctuation

### Inference Pipeline

```
1. Load model weights (safetensors)
2. Tokenize input text → token IDs
3. Prefill phase:
   - Embed tokens (token + position embeddings)
   - For each transformer block:
     • QKV projection (matmul)
     • Split KV cache, compute attention scores, softmax
     • Attention output
     • Residual add, layer norm
     • FFN (two matvecs with GELU)
4. Decode phase (per token):
   - Use cached KV for new position
   - Matrix-vector products (optimized kernel selection)
   - Softmax + top-k/nucleus sampling
   - Append to cache
5. Detokenize → output text
```

## Building

```bash
gcc -O3 main.c tokenizer.c -o gpt2-opencl11 -lOpenCL -lm 
```

### Requirements
- **C99 compiler** (gcc, clang, MSVC)
- **OpenCL 1.1 headers** (or compatible)
- **OpenCL 1.1 runtime** from your GPU/device vendor
- **POSIX system** (Linux, macOS, *BSD; Windows with POSIX layer)

## Usage

### Basic Inference

```bash
./gpt2-opencl11 <model_dir> "<prompt>"
```

Where `<model_dir>` contains:
- `config.json` — model hyperparameters
- `model.safetensors` — pretrained weights
- `vocab.json` — GPT-2 vocabulary
- `merges.txt` — BPE merge pairs

### Example

```bash
./gpt2-opencl11 ./gpt2-small "The future of AI is"
```

### Runtime Behavior

On startup, the program:

1. **Enumerates OpenCL devices** — lists all available GPUs/accelerators with capabilities and performance estimates
2. **Auto-selects device** — chooses fastest device by benchmarking a small kernel
3. **Benchmarks compute formats** — compares FP32, FP64, and (if supported) FP16 throughput on a 256×256 matrix multiply
4. **Profiles matrix multiplication** — sweeps 6+ tile configurations to find optimal performance for your device
5. **Per-kernel tuning** — benchmarks specific layer dimensions (attention, FFN outputs) to select optimal work-group sizes and split-K parameters
6. **Loads model** — reads weights, detects layout conventions, caches results for future runs
7. **Begins inference** — alternates prefill (all input tokens) and decode (one token at a time)

This adaptive tuning means **first run is ~10–30× slower than subsequent runs** on the same device (tuning results are cached).

## Configuration

### Model Limits

Hard-coded maximums (defined in `main.c`):
- **MAX_EMBED_DIM**: 1280 (embedding dimension)
- **MAX_LAYERS**: 36 (number of transformer blocks)
- **MAX_HEADS**: 20 (number of attention heads)
- **MAX_VOCAB**: 50304 (vocabulary size)
- **MAX_SEQ_LEN**: 1024 (maximum sequence length)
- **MAX_FFN**: 5120 (FFN hidden dimension)

### Data Types

The code supports three floating-point formats:

- **FP32** (default): Maximum compatibility and stability. Use this unless you have specific memory constraints.
- **FP64**: Double precision. Requires `cl_khr_fp64` device extension. Slower but more numerically stable.
- **FP16**: Half precision. Requires `cl_khr_fp16` device extension. Halves VRAM usage but may lose accuracy on older hardware.

To compile for a specific format, define the corresponding macro:

```bash
# FP32 (default)
gcc -O3 -lOpenCL -lm main.c tokenizer.c -o gpt2-opencl11

# FP64 (double precision)
gcc -O3 -DUSE_FP64 -lOpenCL -lm main.c tokenizer.c -o gpt2-opencl11

# FP16 (half precision, if supported)
gcc -O3 -DUSE_FP16 -lOpenCL -lm main.c tokenizer.c -o gpt2-opencl11
```

At runtime, the program detects device capabilities and automatically benchmarks available formats. If you compile for a format your device doesn't support, the program will warn and may fall back to FP32.

### Device Tuning

Tuning results are cached in device-specific configuration files after the first run. If you upgrade drivers or move to a different GPU, delete the cache to force re-tuning.

## Performance Considerations

### Kernel Selection Strategy

| Scenario | Kernel | Why |
|----------|--------|-----|
| Prefill (M >> 1) | `matmul` | Tile-based GEMM maximizes reuse |
| Decode (M = 1, large N) | `matvec4` | SIMD-friendly with coalesced memory |
| Decode (M = 1, very large K) | `matvec4_sk` (split-K) | Reduces memory bandwidth pressure via parallel reduction |
| Fallback | `matvec` | When N not divisible by 4 or device doesn't support float4 |

### Memory Layout Choices

- **K-cache transposed** `[D, MAX_SEQ]` — enables coalesced reads for attention score computation
- **V-cache unchanged** `[MAX_SEQ, D]` — efficient row-wise access during output sum
- **Weight matrices** row-major — natural for matrix multiplication with OpenCL's row-major global memory

### Optimization Techniques

1. **Loop unrolling**: 4-way unrolling in matvec to hide memory latency
2. **Local work-group optimization**: Reduced softmax via tree reduction
3. **Fused epilogues**: Bias, residual, GELU applied in same kernel
4. **Barrier minimization**: Strategic synchronization in split-K reduce
5. **Padding**: Bank conflict avoidance in local arrays
6. **Vectorized I/O**: float4 SIMD loads for decode where supported

## Supported Models

Tested on:
- **DistilGPT2** (82M parameters)
- **GPT-2 Small** (124M parameters)
- **GPT-2 Medium** (355M parameters)
- **GPT-2 Large** (774M parameters)

Requires model weights in safetensors format with standard GPT-2 architecture.

## Device Support

Any hardware with **OpenCL 1.1 driver support** should/could be able to work with this provided drivers are installed.

**Confirmed working on:** GT 730 DDR3 128-bit 4GB — [TechPowerUp GPU Database](https://www.techpowerup.com/gpu-specs/geforce-gt-730.c2590)

## Disclaimer

**USE AT YOUR OWN RISK.** This is experimental, educational software. The author makes **no warranties** and assumes **no liability** for:

- System crashes, lockups, or data loss
- Graphics card overheating, throttling, or permanent damage
- Driver corruption or boot failures
- Incorrect model outputs or numerical errors
- Power supply strain or hardware failure

Test on non-critical hardware first. Monitor temperatures and power draw. If your device becomes unstable, stop using this code immediately and check your driver/hardware health.

## References

- [GPT-2](https://en.wikipedia.org/wiki/GPT-2)
- [OpenCL 1.1 Specification](https://khronos.org/registry/OpenCL/specs/1.1/pdf/OpenCL_1.1.pdf)
- [Safetensors Format](https://huggingface.co/docs/safetensors)
