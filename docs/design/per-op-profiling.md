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
int wrap_hipblasLtMatmul(RuntimeState *state, ..., int64_t M, int64_t N, int64_t K, ...) {
  OP_PROFILE("matmul", [&] {
    char b[64];
    snprintf(b, sizeof(b), "%lldx%lldx%lld", (long long)M, (long long)N, (long long)K);
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
2. Events are stored in a `pending` vector inside `OpProfileState`.
3. After `hipStreamSynchronize` completes (in `hipdnn_ep_stream_sync`), all pending events are resolved in bulk via `hipEventElapsedTime` and accumulated into a two-level profile map (op → shapes).

This avoids per-operator GPU synchronization overhead.

### State ownership

Profiling state is per-inference-session, not global:

```
RuntimeState
  └── void *op_profile  →  OpProfileState (two-level map + pending events)
```

- Created in `hipdnn_ep_state_init_with_fs()` (only when PERF enabled)
- Reset at the start of each inference (`prepare_input` with index==0)
- Resolved and printed in `hipdnn_ep_stream_sync()` after GPU sync
- Destroyed in `hipdnn_ep_state_cleanup()`

### Output format

```
[PERF] =========================================================================
[PERF]                                  calls  gpu (ms)  cpu (ms)  gpu %
[PERF]  matmul_nbits                      225      66.3       0.4  77.7%
[PERF]    m=1,n=14336,k=4096               64      30.1       0.1  35.3%
[PERF]    m=1,n=4096,k=14336               32      13.6       0.1  15.9%
[PERF]    m=1,n=4096,k=4096                64      11.8       0.1  13.8%
[PERF]    m=1,n=1024,k=4096                64       5.5       0.1   6.4%
[PERF]    m=1,n=128256,k=4096               1       3.5       0.0   4.1%
[PERF]  skip_layernorm                     64       4.8       0.3   5.6%
[PERF]    1x4096                           64       4.8       0.3   5.6%
[PERF]  gqa                                32       4.3       0.1   5.0%
[PERF]    b=1,sq=1,skv=128,h=32,d=128      32       4.3       0.1   5.0%
[PERF]  rotary_emb                         64       3.9       0.1   4.6%
[PERF]    h=32,d=128                       32       2.1       0.1   2.5%
[PERF]    h=8,d=128                        32       1.8       0.1   2.1%
[PERF]  elementwise                        64       2.8       0.2   3.3%
[PERF]    1x1x1x14336                      64       2.8       0.2   3.3%
[PERF]  activation                         32       2.2       0.1   2.6%
[PERF]    n=14336                          32       2.2       0.1   2.6%
[PERF]  stream_sync                         1       n/a      85.0    n/a
[PERF]  TOTAL                                      85.3      87.6
[PERF] =========================================================================
```

Sorted by GPU time descending. CPU-only operations (e.g. `stream_sync`) are listed at the bottom with `n/a` for GPU columns. The `gpu %` column shows each op's share of total GPU time. Shape sub-rows are indented under their parent op.

### Interpreting CPU time

CPU time measures wall-clock time from `OpProfileScope` construction to destruction — the host-side cost of each operator call. For correctly async GPU ops, this should be near-zero (~0.1ms) because `hipEventRecord` returns immediately. Only `stream_sync` should show significant CPU time (the full GPU pipeline wait).

**Non-zero CPU time on a GPU op signals an unexpected `hipStreamSynchronize` in its code path.** For example, GQA's decomposed path (separate past/present KV buffers) does a D2H copy + `hipStreamSynchronize` per layer to read `seqlens_k` on the host. With 32 layers (Llama 8B), this adds ~95ms of CPU stall. The fused decode path avoids this entirely when shared buffer detection ensures `past_key_gpu == present_key_gpu`.

Use the CPU column as a diagnostic: if an op that should be fully async shows >1ms CPU time, look for synchronization calls in its code path.

## File layout

| File | Role |
|------|------|
| `lib/Runtime/op_profile.h` | RAII scope guards (`OpProfileScope`, `OpProfileCpuScope`), `OP_PROFILE`/`OP_PROFILE_CPU` macros |
| `lib/Runtime/op_profile.cpp` | `OpProfileState` struct (two-level: OpEntry → ShapeEntry), create/destroy/reset/resolve/print |
| `lib/Runtime/debug_log.h` | `hipdnn_ep_perf_enabled()` — env var check (uses Win32 API on Windows) |
| `lib/Runtime/runtime_state_internal.h` | `void *op_profile` field in `RuntimeState` |
| `lib/Runtime/hipdnn_ep_runtime.h` | `hipdnn_ep_state_get_op_profile()` accessor |
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

## Build dependency: bitcode recompilation

The `compile_to_bitcode` macro in `lib/Runtime/CMakeLists.txt` has a `DEPENDS` list that controls when bitcode is recompiled. **All headers included by runtime `.cpp` files must appear in this list.** Missing a header means changes to it don't trigger recompilation, producing stale bitcode that gets linked into compiled model DLLs.

For profiling, both `op_profile.h` and `debug_log.h` must be in the `DEPENDS` list. The stale-DLL problem is compounded by MorphiZen's cache key being based on the ONNX graph hash, not the runtime bitcode version — so stale DLLs persist until manually deleted (`del %TEMP%\morphizen_mlir_*`).

## Windows static CRT caveat

Model DLLs are compiled with static CRT (`/MT`), which gives each DLL its own CRT instance. `std::getenv()` inside the DLL cannot see environment variables set by the host process. `debug_log.h` uses `GetEnvironmentVariableA()` (Win32 API) instead, which reads the shared process environment block.
