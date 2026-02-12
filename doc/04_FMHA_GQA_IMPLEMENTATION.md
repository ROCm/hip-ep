<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# FMHA / GQA Implementation Details

This document describes the Flash Multi-Head Attention (FMHA) kernel implementation
for Grouped Query Attention (GQA) in onnx-hipdnn-ep, the design decisions made,
and the accuracy issues encountered on RDNA3 (GFX11) hardware.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
  - [Custom Portable Kernel (Default)](#custom-portable-kernel-default)
  - [CK Tile naive\_attention\_fwd (Optional)](#ck-tile-naive_attention_fwd-optional)
- [Build Configuration](#build-configuration)
- [Runtime Configuration](#runtime-configuration)
- [RDNA3 Wave32 Accuracy Issues](#rdna3-wave32-accuracy-issues)
  - [Root Cause: cross\_wave\_reduce Bug](#root-cause-cross_wave_reduce-bug)
  - [Why wave64 Workarounds Failed](#why-wave64-workarounds-failed)
  - [CK Tile Prefill Failure: No Causal Masking](#ck-tile-prefill-failure-no-causal-masking)
- [Accuracy Test Results](#accuracy-test-results)
- [Approaches Tried and Discarded](#approaches-tried-and-discarded)
- [File Reference](#file-reference)

---

## Overview

The ONNX `com.microsoft.GroupQueryAttention` (GQA) operator implements multi-head
attention with grouped key/value heads, as used in models like Llama 3.1. The operation
has two phases:

- **Prefill** (`seqlen_q > 1`): Processes a full prompt sequence. Requires **causal
  masking** so that each query token only attends to current and previous key tokens.
- **Decode** (`seqlen_q == 1`): Generates one token at a time. The single query token
  attends to all past key/value tokens — no causal masking needed.

The kernel operates in **BHSD layout** `[Batch, Head, Seqlen, Hdim]` with FP16
data types and supports GQA head ratios (e.g., 32 query heads / 8 KV heads = 4:1).

---

## Architecture

```
                  launch_fmha_fwd_fp16_causal()    (Public C API)
                              │
                    ┌─────────┴──────────┐
                    │  use_ck_naive_impl()│
                    │  (env var check)    │
                    └─────────┬──────────┘
                              │
              FMHA_USE_CK_NAIVE=1?
                   /                \
                 YES                 NO (default)
                  │                   │
    ┌─────────────▼───────────┐  ┌───▼──────────────────────┐
    │  launch_fmha_ck_naive() │  │  launch_fmha_portable()  │
    │  (requires HAS_CK_TILE) │  │  (always available)      │
    │                         │  │                           │
    │  Uses CK Tile's         │  │  Custom HIP kernel with  │
    │  naive_attention_fwd    │  │  portable shared-memory   │
    │  host API               │  │  tree reductions          │
    │                         │  │  + causal masking          │
    │  ⚠ No causal masking    │  │                           │
    │  ⚠ wave64-hardcoded     │  │  ✓ wave32 + wave64 safe  │
    └─────────────────────────┘  └───────────────────────────┘
```

---

## Implementation Details

### Custom Portable Kernel (Default)

**Source**: `kernels/src/fmha_ck_tile_kernels.hip` — function `fmha_naive_causal_kernel`

This kernel implements online softmax attention with bottom-right causal masking.
It is the **default and recommended** implementation for all GPU architectures,
especially RDNA3.

**Key design decisions:**

1. **Portable shared-memory tree reductions**: Instead of using warp-level
   intrinsics (which differ between wave32 and wave64), the kernel uses
   `__syncthreads()`-based tree reductions in shared memory. This works correctly
   regardless of wavefront size.

   ```cpp
   // Portable block reduction — works on wave32 AND wave64
   __device__ AccType block_reduce_max(AccType val, AccType* smem) {
       smem[threadIdx.x] = val;
       __syncthreads();
       for (int s = kBlockSize / 2; s > 0; s >>= 1) {
           if (threadIdx.x < s)
               smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
           __syncthreads();
       }
       return smem[0];
   }
   ```

2. **Online softmax with rescaling**: The kernel processes K/V in chunks of
   `kBlockSize=256` tokens, maintaining running max and sum statistics. When
   a new chunk produces a larger max, previous accumulations are rescaled using
   `expf(old_max - new_max)` to maintain numerical stability.

3. **Bottom-right causal masking**: For prefill, each query row `i` can only
   attend to key positions `0..causal_limit` where
   `causal_limit = i + (seqlen_k - seqlen_q)`. For decode (`seqlen_q=1`),
   `causal_limit = seqlen_k - 1`, which allows attending to all keys.

4. **Grid layout**: `(ceil(hdim/256), seqlen_q, batch*nhead_q)` — each thread
   block computes one output element dimension for one query row.

**Accuracy**: Excellent on all architectures. Typical max diff vs CPU: ~0.001 for
prefill, ~0.00006 for decode (FP16 precision limited).

### CK Tile naive_attention_fwd (Optional)

**Source**: `kernels/src/fmha_ck_tile_kernels.hip` — function `launch_fmha_ck_naive`
(guarded by `#ifdef HAS_CK_TILE`)

This path uses CK Tile's reference `naive_attention_fwd` host API from
`ck_tile/ref/naive_attention.hpp`. It is available as an alternative for
testing and comparison when built with `CK_ROOT`.

**Limitations:**

- **No causal masking**: The `naive_attention_fwd` API does not support causal
  masks, making it incorrect for prefill.
- **wave64-hardcoded reductions**: The internal kernel uses `ds_bpermute`-based
  wave reductions and a `cross_wave_reduce` function that hardcodes `wave_size=64`
  and `waves=4`. On RDNA3 (native wave32), this causes incomplete reductions.

---

## Build Configuration

CK_ROOT is **optional**. The build system behavior:

| CK_ROOT | HAS_CK_TILE | Available Implementations |
|---------|-------------|--------------------------|
| Not set | OFF | Portable kernel only |
| Set (valid) | ON | Portable kernel + CK naive |

CMake automatically detects CK_ROOT and sets `HAS_CK_TILE`:

```cmake
# In root CMakeLists.txt:
set(CK_ROOT "" CACHE PATH "Path to Composable Kernel source tree (optional)")
# ...
if(CK_ROOT AND EXISTS "${CK_ROOT}/include/ck_tile/core.hpp")
    set(HAS_CK_TILE ON)
endif()

# In kernels/CMakeLists.txt: passes -DHAS_CK_TILE=1 to hipcc when ON
```

---

## Runtime Configuration

| Environment Variable | Values | Description |
|---------------------|--------|-------------|
| `FMHA_USE_CK_NAIVE` | `1` = CK naive, unset/other = portable | Selects FMHA implementation at runtime. Only effective when built with `HAS_CK_TILE`. |

The selection is logged to stderr on first invocation:
```
[FMHA] Using custom portable kernel (default). Set FMHA_USE_CK_NAIVE=1 to use CK Tile.
```
or
```
[FMHA] FMHA_USE_CK_NAIVE=1: using CK Tile naive_attention_fwd
```
or (when CK was not available at build time):
```
[FMHA] Using custom portable kernel (CK Tile not available at build time)
```

---

## RDNA3 Wave32 Accuracy Issues

### Root Cause: cross_wave_reduce Bug

The core issue is in CK Tile's `naive_attention_fwd_kernel`, specifically the
`cross_wave_reduce` function in
`ck_tile/ref/naive_attention.hpp`:

```cpp
template <typename T, typename F>
__device__ constexpr T cross_wave_reduce(T local, F reduce_f, T* smem)
{
    constexpr int waves     = 4;       // ← HARDCODED: assumes 4 waves
    constexpr int wave_size = 64;      // ← HARDCODED: assumes wave64
    int lane_id = threadIdx.x % wave_size;

    __syncthreads();
    smem[threadIdx.x] = local;
    __syncthreads();

    T v_local = smem[lane_id];
    for (int i_stage = 1; i_stage < waves; i_stage++) {
        T v_remote = smem[i_stage * wave_size + lane_id];
        v_local = reduce_f(v_local, v_remote);
    }
    return v_local;
}
```

**What goes wrong on RDNA3 (wave32):**

With a block size of 256 threads:
- **On wave64 (CDNA)**: 256 threads = 4 waves × 64 threads. The function
  correctly reduces across all 4 waves. ✓
- **On wave32 (RDNA3)**: 256 threads = 8 waves × 32 threads. But the function
  hardcodes `waves=4` and `wave_size=64`, so it only reads shared memory at
  offsets `0, 64, 128, 192` with stride 64. This means it only samples data
  from threads `0..31`, `64..95`, `128..159`, `192..223` — **skipping threads
  `32..63`, `96..127`, `160..191`, `224..255`** (4 out of 8 physical waves).

The result: the global max and sum for softmax normalization are computed from
only half the data, leading to incorrect attention weights.

Similarly, the `wave_reduce` function uses `__builtin_amdgcn_ds_bpermute` with
6 stages (for wave64: 2^6 = 64), which on wave32 hardware performs extra
unnecessary permutations but the result is still constrained to the first 32 lanes.

### Why wave64 Workarounds Failed

We attempted several workarounds to force wave64 mode on RDNA3:

| Approach | Result | Explanation |
|----------|--------|-------------|
| `GPU_ENABLE_WAVE32_MODE=0` (runtime env) | No effect on accuracy | This is a runtime hint that the driver may ignore. The ISA is already compiled for wave32. |
| `GPU_ENABLE_WAVE32_MODE=0` (build-time env) | No effect on accuracy | hipcc doesn't use this env var to select wavefront size at compile time. |
| `-mwavefrontsize64` (hipcc flag) | No effect on accuracy | The flag appeared to be silently ignored for gfx1151 targets, or the CK kernel's hardcoded constants override any ISA-level wavefront change. |
| `exp2f` vs `expf` test | No effect on accuracy | Confirmed that the exponential function choice is not the cause; both give identical accuracy in the custom kernel. |

**Conclusion**: The bug is in the **algorithm** (hardcoded constants), not in the
ISA instruction encoding. Even if wave64 mode could be forced, the hardcoded
`waves=4` and `wave_size=64` constants in C++ source code would still produce
incorrect indexing on any system where the actual wavefront size differs.

### CK Tile Prefill Failure: No Causal Masking

CK Tile's `naive_attention_fwd` API does not support causal masking. For prefill,
where causal masking is mandatory, this results in catastrophic errors:

- Each query token incorrectly attends to **all** key tokens (including future ones)
- Max diff: **1.67** (vs CPU), with **7.87%** of elements exceeding tolerance
- The output is essentially random compared to the correct causal result

---

## Accuracy Test Results

Tested with Llama 3.1 8B GQA models (batch=1, nhead_q=32, nhead_k=8, hdim=128)
on RDNA3 (gfx1151, Radeon 880M, native wave32):

### Prefill (seqlen_q=255, seqlen_k=255)

| Implementation | Max Diff | Mean Diff | Mismatches | Result |
|---------------|----------|-----------|------------|--------|
| Custom portable kernel | 0.000977 | 0.000009 | 0 (0%) | **PASS** |
| CK Tile naive_attention_fwd | 1.672913 | 0.037814 | 82,151 (7.87%) | **FAIL** |

### Decode (seqlen_q=1, seqlen_k=256)

| Implementation | Max Diff | Mean Diff | Mismatches | Result |
|---------------|----------|-----------|------------|--------|
| Custom portable kernel | 0.000061 | 0.000005 | 0 (0%) | **PASS** |
| CK Tile naive_attention_fwd | 0.048279 | 0.002695 | 0 (0%) | PASS* |

\* CK naive decode passes tolerance (0.1) but with ~800x larger max diff than the
portable kernel, due to the `cross_wave_reduce` bug.

### Error Scaling with Sequence Length

The decode error **scales with sequence length** because longer sequences have more
waves carrying valid data that get skipped:

| seqlen_k | Max Diff (CK naive) | Explanation |
|----------|---------------------|-------------|
| 129 | 0.021881 | ~half of wave groups have valid data |
| 256 | 0.048279 | All 8 wave groups have valid data; 4 are skipped |

This scaling behavior directly confirms the `cross_wave_reduce` root cause.

---

## Approaches Tried and Discarded

### 1. CK Tile FmhaFwdKernel (Flash Attention production kernel)

The first approach was to use CK Tile's production `FmhaFwdKernel` for Flash
Attention, which supports causal masking and is highly optimized for CDNA (MI-series)
GPUs.

**Result**: The kernel failed to compile for GFX11 (RDNA3) targets. The tile-based
pipeline and matrix instruction dispatching are designed for CDNA's MFMA
instructions which are not available on RDNA3.

### 2. Custom HIP kernel without CK Tile

A standalone FMHA kernel was implemented without any CK dependency, using hipRTC-style
kernels with manual FP16 handling.

**Result**: Worked correctly but was eventually replaced by the current approach
which leverages CK Tile's `half_t` type when available and provides cleaner code.

### 3. CK Tile naive_attention_fwd for decode only

Since CK naive doesn't support causal masking, we tried using it only for decode
(where masking isn't needed) while keeping the custom kernel for prefill.

**Result**: Worked but showed reduced accuracy on RDNA3 due to the
`cross_wave_reduce` bug. The custom portable kernel was found to be superior
for both paths.

### 4. Hybrid: Custom causal prefill + CK naive decode

This was the intermediate solution before the current unified approach.

**Result**: Functionally correct but unnecessarily complex. Since the custom kernel
handles both cases with excellent accuracy, having two different implementations
added complexity without benefit.

---

## File Reference

| File | Description |
|------|-------------|
| `kernels/src/fmha_ck_tile_kernels.hip` | FMHA kernel implementations (portable + CK naive) |
| `kernels/include/rocm_kernels.h` | C API declaration for `launch_fmha_fwd_fp16_causal` |
| `kernels/CMakeLists.txt` | Build configuration (conditional CK Tile support) |
| `CMakeLists.txt` | Root CMake (CK_ROOT detection, HAS_CK_TILE) |
| `custom-op-rocm/src/custom_op.cpp` | GQA custom op that calls the FMHA kernel |
| `level-2-pass-rocm-gqa/src/pass_main.cpp` | GQA pattern matching pass |
| `patterns/gqa.json` | ONNX subgraph pattern for GQA detection |
| `test/model_verifier.cpp` | CPU vs GPU comparison test tool |

### CK Tile Source References (for debugging)

| File (relative to CK_ROOT) | Key Functions |
|----------------------------|---------------|
| `include/ck_tile/ref/naive_attention.hpp` | `naive_attention_fwd_kernel`, `wave_reduce`, `cross_wave_reduce` |
| `include/ck_tile/core.hpp` | CK Tile core types and utilities |
