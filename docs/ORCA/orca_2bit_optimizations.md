# ORCA 2-bit GPU Optimizations on Strix Halo (gfx1152)

**Model**: ORCA 2-bit (W2A8) — Llama-style, 40 layers, hidden=5120, GQA 40q/10kv,
bits=2 packed weights with per-tensor uint16 activation quantization.
**Hardware**: AMD Radeon 860M (gfx1152, Strix Halo iGPU), LPDDR5X shared memory.
**Stack**: onnx-hipdnn-ep (MorphiZen EP) via ONNX Runtime.

---

## Measured Results (warm autotune, 5 consecutive runs)

| Metric | Value |
|--------|-------|
| Decode throughput | **~7.0–7.4 tok/s** |
| Per decode step (warm) | **~130–145 ms** |
| Prefill (128 tokens) | **~19–22 s** |
| Effective streaming BW | **~27 GB/s** |

> **Warm vs cold**: The GEMV autotune is in-memory and resets on each process
> launch. The first run (cold) is ~4.7 tok/s as the autotuner explores 35
> block/tile configs. By run 2+ the best config is cached and throughput
> stabilises at 7.0–7.4 tok/s.

---

## Optimizations Implemented

### 1. Fix morphizen EP hang — kDynamic translation
**File**: `3rd-party/morphizen/mlir-constants.cpp`, line 79

**Problem**: ONNX uses `-1` for dynamic tensor dimensions. The morphizen MLIR
compiler was passing raw ONNX dims (including `-1`) directly to
`RankedTensorType::get()`. MLIR requires `ShapedType::kDynamic` (INT64_MIN) for
dynamic dims. Any ORCA model with a dynamic sequence dimension caused a hang
before a single kernel was launched.

**Fix**: In `onnxElementTypeToMlirType()`, translate every negative dim to
`ShapedType::kDynamic` before building the `RankedTensorType`.

**Impact**: **Unblocked all GPU inference.** Without this nothing ran at all.

---

### 2. bits=2 vectorized GEMV kernel
**File**: `lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip`
**Function**: `matmul_nbits_gemv_i2_kernel<BLOCK_SIZE, TILE_N, T>`

**Problem**: No kernel existed for 2-bit packed weights. The model fell back to a
naive scalar loop.

**What was built**:
- 128-bit `uint4` loads for B (16 bytes = 64 weights per load — one full block)
- Factored dequant: `(dot - a_sum * zp) * scale` — avoids per-element zp
  subtraction by precomputing the activation sum once per block
- Templated on `T = float | __half` so both fp32 (ORCA) and fp16 activations work
- Autotuner: benchmarks 35 `(BLOCK_SIZE, TILE_N)` configs per shape, caches best
- ZP packing: bits=2 ZP is 2-bit packed (4 per byte), read as
  `zp_byte = n*(k_blocks/4) + grp/4`, `crumb = grp & 3`

**Impact**: Core of all decode performance — replaces a naive O(M×N×K) fallback.

---

### 3. fp32 activation correctness fix
**File**: `lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip`

**Problem**: ORCA uses fp32 activations and fp32 scales (W2A8 format). The initial
kernel hard-cast the fp32 activation pointer as `__half*`, reading 4-byte floats
as 2-byte halves — producing garbage output with no error.

**Fix**: Templated the GEMV kernel on `T` (was hardcoded `__half`). `T=float`
uses fp32 loads throughout. Same fix applied to weight scales.

**Impact**: Correctness — ORCA produces correct tokens.

---

### 4. fp16 LM head model variant
**File**: `make_fp16_lmhead.py` → `orca_2bit_lm_head_fp16.onnx`

**Problem**: The LM head ONNX (`vocab=32000 × hidden=5120`) had fp32 input,
output, and scales. The fp32 matmul uses the naive scalar kernel (no vectorised
fp32 GEMV was implemented), taking ~235 ms per decode step just for the LM head.

**Fix**: Created `orca_2bit_lm_head_fp16.onnx` by converting the LM head input,
output, and scales to fp16. This routes through the optimised fp16 GEMV path with
128-bit loads and WMMA (vocab=32000 × 5120, M=1).

**Impact**: LM head: **235 ms → 7 ms per step** (33× speedup). Single largest
wall-clock win.

---

### 5. GQA seqlens_k value-based cache
**File**: `lib/Runtime/real/gqa.cpp`

**Problem**: The GQA attention dispatch reads `seqlens_k[0]` from the GPU (a D2H
copy + `hipStreamSynchronize`) to determine `total_seq` before choosing the fused
vs decomposed attention path. The cache was keyed on the *pointer* to `seqlens_k`.
ORCA creates a fresh numpy array each decode step, so the pointer changes on every
call — the cache missed every time, triggering **39 D2H syncs per decode step**
(one per attention layer).

**Fix**: Changed cache validity from pointer equality to a boolean flag
`seqlens_k_cached_valid` that is set on first read per `Compute()` call and
cleared at the start of each new `Compute()` via `begin_compute()`. The cached
value is reused for all 39 GQA layers within one decode step.

**Impact**: Eliminated 38 out of 39 unnecessary GPU→CPU synchronisation points
per decode step.

---

### 6. GQA seqlens_k convention fix (prefill correctness)
**File**: `lib/Runtime/real/gqa.cpp`

**Problem**: The runtime computed `total_seq = seqlens_k + 1` for all values of M
(both decode and prefill). For prefill with `sq=128` and `past_seq_len=0`, this
gave `total_seq = 0 + 1 = 1` instead of `total_seq = 128`. GQA then attended
over only 1 key/value position for a 128-token prefill — effectively zeroing most
of the attention output.

**Fix**: `total_seq = seqlens_k + sq` (past tokens plus current tokens).
The `+1` is only correct for decode where `sq=1`; for prefill `sq=128` must be
added.

**Impact**: Prefill correctness — attention now spans all tokens in the window.

---

### 7. Fused DQ + MatMulNBits + Q MLIR op (prefill M=128)
**Files**:
- `lib/Conversion/OnnxToHip/MsMatMulNBitsI2FusedConversion.cpp` — fusion pattern
- `include/hip/Dialect/IR/HipOps.td` — `Hip_MsMatMulNBitsI2FusedOp`
- `lib/Conversion/HipToLLVM/MsMatMulNBitsI2FusedLowering.cpp` — lowering
- `lib/Runtime/real/ms_matmul_nbits_i2_fused.cpp` — runtime wrapper
- `lib/Runtime/Kernels/hip/matmul_nbits_kernel.hip` — fused kernel

**Problem**: Each W2A8 matmul requires three separate ops:
`uint16 → [DQ] → fp32 → [MatMul] → fp32 → [Q] → uint16`. For prefill (M=128)
this means two extra global memory roundtrips per matmul — writing and re-reading
164 KB of fp32 intermediates per layer.

**What was built**: A new pre-lowering MLIR fusion pattern that matches the
`com.microsoft.DequantizeLinear → com.microsoft.MatMulNBits(bits=2) →
com.microsoft.QuantizeLinear` chain and replaces it with a single
`hip.ms_matmul_nbits_i2_fused` op. The fused kernel carries the DQ scale/zp and
Q scale/zp directly, computing all three ops in one pass over the weight matrix.

**Guard**: Disabled for `M=1` decode — ORCA's DQ has 2–3 consumers (q/k/v
projections share one activation DQ), so the DQ cannot be eliminated. The fused
matmul adds overhead without saving the DQ roundtrip, making M=1 fusion
net-negative.

**Impact**: ~3× prefill speedup for M=128 (eliminates 2 global memory passes per
fused matmul layer).

---

### 8. ms_quant/dequant kernels and reduce_sum fp32
**Files**:
- `lib/Runtime/Kernels/hip/ms_quant_linear_kernel.hip`
- `lib/Runtime/real/ms_dequantize_linear.cpp`
- `lib/Runtime/Kernels/hip/reduce_sum_kernel.hip`

**Problem**: ORCA uses `com.microsoft.QuantizeLinear` (fp32→uint16) and
`com.microsoft.DequantizeLinear` (uint16→fp32) for per-tensor activation
quantization, and `ReduceSum` over fp32 activations. None of these had GPU
implementations — all fell back to CPU with synchronous D2H/H2D copies.

**What was built**: GPU kernels for both directions of uint16 quantization,
fp32 reduce_sum, and MLIR dialect ops + lowerings to wire them into the pipeline.

**Impact**: Eliminates CPU fallback for the W2A8 quantization ops — keeps
activations on GPU throughout inference.

---

## Theoretical Maximum Decode Performance

### Weight bytes per decode step (M=1)

| Source | Bytes |
|--------|-------|
| bits=2 packed weights (all 40 layers) | 3,461 MB |
| fp32 scales (all 40 layers) | 853 MB |
| LM head (fp16) | 328 MB |
| **Total** | **~3,800 MB per token** |

### BW-bound ceiling on this hardware

The Radeon 860M (gfx1152) is an iGPU on Strix Halo sharing LPDDR5X with the CPU.
Peak theoretical DRAM BW is ~120 GB/s but practical GPU-accessible BW (after CPU
contention, memory controller overhead, iGPU share) is approximately **40–70 GB/s**
under real workloads.

| Effective GPU BW | Theoretical max tok/s |
|------------------|-----------------------|
| 17 GB/s (current) | 4.5 tok/s |
| 27 GB/s (warm autotune) | **7.1 tok/s** ← measured |
| 40 GB/s | 10.5 tok/s |
| 60 GB/s | 15.8 tok/s |
| 80 GB/s | 21.0 tok/s |

### Best achievable on this chip

With the current model format (fp32 activations, fp32 scales, bits=2 weights), the
realistic ceiling on a Radeon 860M iGPU is:

**~10–12 tok/s**

Reasoning:
- The chip can realistically sustain ~40–45 GB/s streaming to the GPU under a
  pure-GPU workload (no CPU contention)
- At 40 GB/s and 3,800 MB/token → **10.5 tok/s**
- Closing the gap from 7 to 10 tok/s requires raising BW utilisation from
  27 GB/s to ~40 GB/s, which needs higher kernel occupancy (more concurrent waves
  to hide the ~400-cycle DRAM latency)

### Why 20 tok/s is not achievable on this chip

20 tok/s requires **80 GB/s effective GPU BW** — that is the chip's theoretical
*total* DRAM bandwidth shared between CPU and GPU. Achieving 80 GB/s GPU-only is
not possible while the system is running normally.

20 tok/s would be realistic on a **discrete GPU** with dedicated VRAM (e.g.
RX 7900 XTX at 960 GB/s — ceiling would be 250+ tok/s), or on a higher-memory-BW
APU configuration.

### What would help most to close the 7 → 10 tok/s gap

1. **Convert activations to fp16** — ORCA currently uses fp32 activations. Halving
   activation size reduces register pressure, increases wave occupancy, and
   improves the memory pipeline. Requires a model-level change (re-export with
   fp16 activations).

2. **Persistent kernel / weight pre-fetch** — Load all 40 layers' weights once
   into a persistent L2-resident structure. Not feasible here (3.5 GB >> L2 cache
   size of ~4–8 MB).

3. **CPU+GPU concurrent execution** — Pipeline CPU ops (embedding, layernorm,
   RMS norm) to overlap with GPU weight streaming. Small gain on iGPU since they
   share the same memory bus.
