<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Per-Operator GPU Profiling

## Overview

Per-operator GPU profiling measures GPU and CPU time for each runtime wrapper function (matmul, GQA, conv, etc.). Enabled by `HIPDNN_EP_PERF=1`, it extends the existing phase-level profiling (H2D/Compute/D2H) with a per-operator breakdown grouped by operator name, sorted by GPU time descending.

## Design

### Zero-overhead when disabled

Each operator wrapper contains one macro call:

```cpp
int wrap_hipblasLtMatmul(RuntimeState *state, ..., int64_t M, int64_t N,
                         int64_t K_a, int64_t K_b, ...) {
  // The wrapper has already established K_a == K_b.
  int64_t K_equal = K_a;
  OP_PROFILE("matmul", [&] {
    char b[64];
    snprintf(b, sizeof(b), "%lldx%lldx%lld", (long long)M, (long long)N,
             (long long)K_equal);
    return std::string(b);
  }, state);
  // ... operator implementation ...
}
```

The macro expands to a `std::optional<OpProfileScope>` that is only populated when `hipdnn_ep_perf_enabled()` returns true. The second argument is a callable (typically a lambda) that returns the shape string — it is only invoked when profiling is active, so `snprintf` and string construction have zero overhead when disabled.

### Per-shape profiling

All operators pass a shape-producing lambda as the second parameter. The profiling table groups entries by op name, with indented sub-rows for each distinct shape.

### Deferred GPU synchronization

GPU events are recorded but not synchronized per-operator. Instead:

1. Each `OpProfileScope` records `hipEventRecord(start)` in its constructor and `hipEventRecord(stop)` in its destructor, both on the existing HIP stream.
2. Pending entries (name, shape, event pool index, CPU time) are stored in a vector inside `OpProfileState`.
3. After `hipStreamSynchronize` completes, all pending events are resolved in bulk via `hipEventElapsedTime` and accumulated into a two-level profile map (op → shapes). Resolve+print is invoked by the EP through `hipdnn_ep_runtime_flush_op_profile()` AFTER the `MlirCustomOp::Compute` wall-clock window closes — see "Resolve placement" below.

This avoids per-operator GPU synchronization overhead.

### Event pool

HIP events are pre-allocated in `OpProfileState::eventPool` and reused across inferences. `op_profile_acquire_event_pair()` returns the next available index; if the pool is exhausted, a new pair is created and appended. The pool grows on demand but never shrinks — after the first inference, all subsequent inferences reuse existing events with zero `hipEventCreate`/`hipEventDestroy` calls.

For Llama 8B (486 operator calls per inference), this eliminates 972 `hipEventCreate` + 972 `hipEventDestroy` per inference.

### `hipEventDisableSystemFence` on every recorded event

Every event in the pool — plus the four H2D/D2H phase events in `hipdnn_ep_runtime_tensor.cpp` and the two EP-side wall-clock events in `MlirCustomOp.cpp` — is created with `hipEventCreateWithFlags(..., hipEventDisableSystemFence)`. By default, each `hipEventRecord` issues a system-scope acquire/release fence (CPU-visible cache flush across all GPUs and the CPU). The fence is wasted work for events we only read via `hipEventElapsedTime` AFTER an explicit `hipStreamSynchronize` — that sync already establishes the required ordering. Dropping the per-record fence cuts ~1 ms / Compute / ~500 records on Llama-8B decode. The flag is set once at creation time and applies to every subsequent record on the same event, so the win compounds across the session.

Do NOT use this flag on events whose results are observed without a subsequent stream sync (e.g., cross-stream synchronization, multi-GPU coordination, CPU-side polling via `hipEventQuery`). All HIPDNN_EP_PERF events are read after sync, so all of them use the flag.

### Resolve placement (`hipdnn_ep_runtime_flush_op_profile`)

The resolve+print step (N × `hipEventElapsedTime` + `std::map` aggregation + ~hundreds of `fprintf` lines per Compute) was historically called from `hipdnn_ep_stream_sync` — which runs inside the generated `main_graph` function, which the EP times as part of `compute_cpu_ms` / `wall_ms`. That inflated the per-Compute latency reported in `[PERF SUMMARY]` and conflated kernel+dispatch time with profile-printing time.

The resolve is now exported as a separate runtime entry point — `hipdnn_ep_runtime_flush_op_profile(RuntimeState*)` — and called by the EP from `MlirCustomOp::Compute` AFTER the `hipDeviceSynchronize` fence and AFTER `perf_collector().record(s)`. Symbol resolution is cached at session creation (`InferenceState::create`) so the per-Compute dispatch is a single indirect call; older model.dlls without the export silently skip the flush (per-op `[PERF]` block missing, inference otherwise unaffected — same compatibility contract as `hipdnn_ep_runtime_begin_compute`).

This does NOT reduce OGA's measured TPS (the flush still runs inside `MlirCustomOp::Compute`, just after the EP's wall window) — its purpose is to make the `wall_ms` / `compute_cpu_ms` / `gpu_ms` numbers in `[PERF SUMMARY]` honest kernel+dispatch time, NOT kernel+dispatch+resolve time.

### State ownership

Profiling state is per-inference-session, not global:

```
RuntimeState
  └── void *op_profile  →  OpProfileState (two-level map + pending events)
```

- Created in `hipdnn_ep_state_init_with_fs()` (only when PERF enabled)
- Reset at the start of each inference (`prepare_input` with index==0)
- Resolved and printed by the EP via `hipdnn_ep_runtime_flush_op_profile()` AFTER the EP's wall-clock window closes (post-`hipDeviceSynchronize`, post-`perf_collector().record`) — see "Resolve placement" above
- Destroyed in `hipdnn_ep_state_cleanup()`

### Output format

```
[PERF] =========================================================================
[PERF]                                  calls  gpu (ms)  cpu (ms)  gpu %
[PERF]  matmul_nbits                      225      62.3       0.3  78.0%
[PERF]    m=1,n=14336,k=4096               64      27.6       0.1  34.6%
[PERF]    m=1,n=4096,k=14336               32      12.9       0.0  16.1%
[PERF]    m=1,n=4096,k=4096                64      11.7       0.1  14.6%
[PERF]    m=1,n=1024,k=4096                64       5.9       0.1   7.4%
[PERF]    m=1,n=128256,k=4096               1       4.3       0.0   5.3%
[PERF]  skip_layernorm                     64       5.2       0.3   6.5%
[PERF]    1x4096                           64       5.2       0.3   6.5%
[PERF]  rotary_emb                         64       4.2       0.1   5.3%
[PERF]    h=32,d=128                       32       2.3       0.0   2.9%
[PERF]    h=8,d=128                        32       1.9       0.0   2.4%
[PERF]  gqa                                32       3.9       0.1   4.9%
[PERF]    b=1,sq=1,skv=128,h=32,d=128      32       3.9       0.1   4.9%
[PERF]  activation                         32       2.2       0.1   2.7%
[PERF]    n=14336                          32       2.2       0.1   2.7%
[PERF]  elementwise                        64       2.1       0.1   2.6%
[PERF]    1x1x1x14336                      64       2.1       0.1   2.6%
[PERF]  TOTAL                                      79.9       0.9
[PERF] =========================================================================
```

Sorted by GPU time descending. The `gpu %` column shows each op's share of total GPU time. Shape sub-rows are indented under their parent op.

### Interpreting CPU time

CPU time measures wall-clock time from `OpProfileScope` construction to destruction — the host-side cost of each operator call. For correctly async GPU ops, this should be near-zero (~0.1ms) because `hipEventRecord` returns immediately.

**Non-zero CPU time on a GPU op signals an unexpected `hipStreamSynchronize` in its code path.** For example, GQA's decomposed path (separate past/present KV buffers) does a D2H copy + `hipStreamSynchronize` per layer to read `seqlens_k` on the host. With 32 layers (Llama 8B), this adds ~87ms of CPU stall. The fused decode path (used when `memory_type == TENSOR_MEMORY_GPU` enables aliasing so `past_key_gpu == present_key_gpu`) avoids this entirely — total CPU time drops to 0.1 ms for all 32 GQA layers.

Use the CPU column as a diagnostic: if an op that should be fully async shows >1ms CPU time, look for synchronization calls in its code path.

## File layout

| File | Role |
|------|------|
| `lib/Runtime/op_profile.h` | RAII scope guards (`OpProfileScope`, `OpProfileCpuScope`), `OP_PROFILE`/`OP_PROFILE_CPU` macros |
| `lib/Runtime/op_profile.cpp` | `OpProfileState` struct (event pool + two-level: OpEntry → ShapeEntry), create/destroy/reset/resolve/print. Pool events created with `hipEventDisableSystemFence`. |
| `lib/Runtime/debug_log.h` | `hipdnn_ep_perf_enabled()` — env var check (uses Win32 API on Windows) |
| `lib/Runtime/runtime_state_internal.h` | `void *op_profile` field in `RuntimeState` |
| `lib/Runtime/hipdnn_ep_runtime.h` | `hipdnn_ep_state_get_op_profile()` accessor; declarations for `hipdnn_ep_runtime_begin_compute` and `hipdnn_ep_runtime_flush_op_profile` |
| `lib/Runtime/hipdnn_ep_runtime_state.cpp` | Implementations of the two `__declspec(dllexport)` runtime hooks above |
| `lib/Compiler/CompilerDriver.cpp` | `export_symbols` list — both hooks must appear here so lld-link emits them in the model.dll export table |
| `backend-mlir-compiler/custom-op-mlir/src/InferenceState.{h,cpp}` | Cached symbol lookup for both hooks (`begin_compute_fn_`, `flush_op_profile_fn_`); null when the model.dll predates the export |
| `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.{h,cpp}` | EP-side wall-clock timing; session-scoped `ep_perf_ev_{start,stop}_` event pair (lazy-allocated with `hipEventDisableSystemFence`); calls `flush_op_profile()` after `t_after_fence` |
| `lib/Runtime/real/*.cpp` | One `OP_PROFILE`/`OP_PROFILE_CPU` call per wrapper function |

## Adding profiling to a new operator

Add two lines to the wrapper function:

```cpp
#include "../op_profile.h"

int wrap_new_op(RuntimeState *state, ..., int64_t M, int64_t N, ...) {
  OP_PROFILE("new_op", [&] {
    char b[64];
    snprintf(b, sizeof(b), "%lldx%lld", (long long)M, (long long)N);
    return std::string(b);
  }, state);
  // ... implementation ...
}
```

Use `OP_PROFILE_CPU` instead for operations that don't launch GPU kernels (e.g., `stream_sync`). The shape lambda is only called when profiling is active, so there is no `snprintf` overhead on the hot path.

## Measured overhead (Llama-3.1-8B AWQ INT4 g128, gfx1151, OGA `model_benchmark -l 128 -g 128 -r 3`)

After Fix 1 (`hipEventDisableSystemFence`) + Fix 2 (`flush_op_profile` hook) + Fix 3 (session-scoped EP-side events):

| Mode | decode tok/s | per-Compute median wall | gap vs `mode=none` |
|---|---|---|---|
| `HIPDNN_EP_PERF=0` (none) | 42.79 | n/a | baseline |
| `HIPDNN_EP_PERF=1` (perf) | 38.68 / 39.70 (two runs) | 24.05–24.76 ms | ~8% |

Before any of these fixes the same workload showed `HIPDNN_EP_PERF=1` decode = 37.85 tok/s and per-Compute median wall = 25.51 ms — a ~12.3% gap. Roughly a third of the original PERF overhead was driver-induced (per-record system fences); the rest is intrinsic to per-op profiling (the ~500 `hipEventRecord` calls themselves, the same number of `hipEventElapsedTime` queries during resolve, and the `fprintf` traffic for the per-op breakdown block). Closing more requires changes in the "Future optimization ideas" section below.

## Future optimization ideas

After the three fixes above, residual overhead on Llama 8B is ~8% of decode TPS, dominated by the bare cost of recording ~500 events per inference plus the `fprintf` traffic for the per-op block. Approaches to reduce this further:

### Two-tier profiling (recommended next step)

`HIPDNN_EP_PERF=1` → CPU-only timing (`steady_clock::now()`, no GPU events). `HIPDNN_EP_PERF=2` → full GPU events (current behavior). CPU-only mode has near-zero overhead and still catches the most important signal: unexpected `hipStreamSynchronize` stalls showing up as high CPU time on a GPU op. Per-op GPU breakdown is lost, but total GPU time is available from the phase-level H2D/Compute/D2H timing.

### Sampling

Only instrument every Nth inference (e.g., `HIPDNN_EP_PERF=10` → profile 1 in 10). The event pool and resolve logic stay the same, the `OP_PROFILE` emplace is just skipped most of the time. Amortized overhead drops to ~3.5 ms.

### Fuse adjacent events

Many ops share a stop→start boundary (op N's stop event immediately precedes op N+1's start event on the same stream). Instead of 2 events at the boundary, use 1 event as both — record a single "fence" event between ops and compute elapsed time as `fence[i+1] - fence[i]`. Cuts `hipEventRecord` calls roughly in half (from 972 to ~487).

## Build dependency: bitcode recompilation

The `compile_to_bitcode` macro in `lib/Runtime/CMakeLists.txt` has a `DEPENDS` list that controls when bitcode is recompiled. **All headers included by runtime `.cpp` files must appear in this list.** Missing a header means changes to it don't trigger recompilation, producing stale bitcode that gets linked into compiled model DLLs.

For profiling, both `op_profile.h` and `debug_log.h` must be in the `DEPENDS` list. The stale-DLL problem is compounded by MorphiZen's cache key being based on the ONNX graph hash, not the runtime bitcode version — so stale DLLs persist until manually deleted (`del %TEMP%\morphizen_mlir_*`).

## Windows static CRT caveat

Model DLLs are compiled with static CRT (`/MT`), which gives each DLL its own CRT instance. `std::getenv()` inside the DLL cannot see environment variables set by the host process. `debug_log.h` uses `GetEnvironmentVariableA()` (Win32 API) instead, which reads the shared process environment block.
