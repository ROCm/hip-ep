# GPU Profiling Guide

This document describes the built-in GPU profiling infrastructure for the
hipDNN EP runtime. When enabled, it records per-kernel GPU timestamps using
`hipEvent` pairs and produces detailed timing reports.

## Quick Start

### 1. Enable at build time

```bash
cmake -DHIPDNN_EP_ENABLE_PROFILING=ON ...
```

This defines `HIPDNN_EP_ENABLE_PROFILING` and compiles the profiler into the
runtime bitcode. When OFF (default), all profiling macros expand to no-ops
with zero overhead.

### 2. Enable at runtime

```bash
export HIPDNN_EP_PROFILE=1
```

The profiler checks this environment variable during `hipdnn_ep_state_init`.
If unset or `"0"`, no events are recorded even when compiled in.

### 3. Run your model

```bash
./your_model_runner
```

### 4. Inspect output files

After cleanup (session teardown), three outputs are produced:

| File | Format | Description |
|------|--------|-------------|
| `hipdnn_ep_profile.json` | Chrome Tracing JSON | Open in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev) |
| `hipdnn_ep_profile.csv` | CSV | `name,category,start_us,duration_us,duration_ms` |
| `stderr` | Summary table | Sorted by total GPU time descending |

## Output Formats

### Chrome Tracing JSON (`hipdnn_ep_profile.json`)

```json
{"traceEvents":[
  {"name":"wrap_hipblasLtMatmul","cat":"compute","ph":"X","ts":1234,"dur":56.7,"pid":1,"tid":1},
  ...
]}
```

- `ph:"X"` = complete event (begin + duration)
- `ts` = host timestamp in microseconds (relative to profiler start)
- `dur` = GPU-measured duration in microseconds

### CSV (`hipdnn_ep_profile.csv`)

```csv
name,category,start_us,duration_us,duration_ms
wrap_hipblasLtMatmul,compute,1234,56.7,0.0567
tensor_h2d,h2d,500,12.3,0.0123
```

### Summary Table (stderr)

```
[PROFILER] ==================== GPU Profile Summary ====================
Function                                    Calls    Total (ms)     Avg (ms)       %
--------------------------------------------------------------------------------
wrap_hipblasLtMatmul                           24       12.345        0.514    45.2%
wrap_group_query_attention                     12        8.901        0.742    32.6%
...
--------------------------------------------------------------------------------
TOTAL                                                   27.300
================================================================================
```

## Instrumented Functions

| Function | Category | Description |
|----------|----------|-------------|
| `wrap_hipblasLtMatmul` | `compute` | Dense matrix multiply (hipBLASLt) |
| `wrap_group_query_attention` | `compute` | Group query attention (GQA) |
| `wrap_matmul_nbits` | `compute` | Quantized matrix multiply (N-bit) |
| `wrap_miopenT5LayerNormForward` | `compute` | Simplified layer norm (RMS) |
| `wrap_skip_simplified_layer_norm` | `compute` | Skip + simplified layer norm |
| `wrap_rotary_embedding` | `compute` | Rotary position embedding (RoPE) |
| `wrap_miopenOpTensor` | `compute` | Element-wise tensor operations |
| `tensor_h2d` | `h2d` | Host-to-device tensor transfer |
| `tensor_d2h_sync` | `d2h` | Device-to-host transfer + sync |

## Adding New Instrumentation

To profile a new runtime function:

1. Include the profiler header:
   ```cpp
   #include "../hipdnn_ep_profiler.h"   // from real/ subdirectory
   // or
   #include "hipdnn_ep_profiler.h"      // from Runtime/ directory
   ```

2. Add `HIPDNN_EP_PROFILE_BEGIN` / `HIPDNN_EP_PROFILE_END` around the
   GPU work:
   ```cpp
   int wrap_my_kernel(RuntimeState *state, ...) {
     // ... validation ...
     HIPDNN_EP_PROFILE_BEGIN(state, "wrap_my_kernel", "compute");
     // ... GPU kernel launch / library call ...
     HIPDNN_EP_PROFILE_END(state, "wrap_my_kernel", "compute");
     // ... error handling ...
     return 0;
   }
   ```

3. The `name` string must match between BEGIN and END (searched backwards).
4. Use `"compute"` for GPU kernels, `"h2d"` for host-to-device transfers,
   `"d2h"` for device-to-host transfers, or define a new category.

## Architecture

### hipEvent-Based Timing

The profiler uses `hipEventCreate` / `hipEventRecord` / `hipEventElapsedTime`
for GPU-accurate timing. Events are recorded on the runtime's HIP stream
(`state->stream`) so they measure actual GPU execution time, not host-side
dispatch time.

### Data Flow

```
PROFILE_BEGIN(state, name, cat)
  -> hipEventCreate + hipEventRecord(start)
  -> store in ProfilerState.events[]

PROFILE_END(state, name, cat)
  -> find matching begin (search backwards by name)
  -> hipEventCreate + hipEventRecord(end)
  -> mark has_end = true

profiler_dump(state)
  -> hipStreamSynchronize
  -> hipEventElapsedTime for each completed pair
  -> write JSON, CSV, summary
```

### Memory Layout

- `ProfilerState` is heap-allocated and stored in `RuntimeState.profiler`
  (void* field).
- Events array is pre-allocated to 16,384 entries (sufficient for typical
  inference sessions).
- All hipEvents are destroyed in `profiler_cleanup`.

### Compile-Time Gating

When `HIPDNN_EP_ENABLE_PROFILING` is not defined:
- All functions become inline no-ops
- `HIPDNN_EP_PROFILE_BEGIN` / `HIPDNN_EP_PROFILE_END` expand to `((void)0)`
- Zero runtime overhead in production builds
