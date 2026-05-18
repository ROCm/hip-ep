<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Qwen Vision Runtime Kernels — Design Notes

This document captures the design rationale, runtime ABI conventions, and
gotchas encountered while implementing 18 GPU runtime kernels for the
Qwen3.5-35B-A3B vision encoder (`vision.onnx`). All 18 kernels live in
`lib/Runtime/real/*.cpp` (host wrappers) and `3rd-party/custom_kernels/hip/*.hip`
(device code), and are dispatched from MLIR via `wrap_*` functions declared in
`lib/Runtime/hipdnn_ep_runtime.h`.

## 1. Scope

The kernels implemented in this round are:

| Group        | Ops                                                       | New `.hip` file                      |
|--------------|-----------------------------------------------------------|--------------------------------------|
| Unary EW     | Neg, Sign, Cos, Sin, Not                                  | `elementwise_unary_kernel.hip`       |
| Binary EW    | Div, Mod, Equal, Less                                     | `elementwise_binary_kernel.hip`      |
| Reduction    | ReduceMax, ReduceProd                                     | (extends `reduce_sum_kernel.hip`)    |
| Shape        | Tile, Expand, GatherND, **Slice**, **ScatterND**          | `tile_kernel.hip`, `expand_kernel.hip`, `gather_nd_kernel.hip`, **`slice_kernel.hip`**, **`scatter_nd_kernel.hip`** |
| Scan / Pad   | CumSum, Pad                                               | `cumsum_kernel.hip`, `pad_kernel.hip`|
| Norm         | LayerNormalization                                        | `layer_norm_kernel.hip`              |
| Constant     | ConstantOfShape (compile-time fold; no kernel)            | —                                    |

All kernels target FP16 / FP32 / INT32 / INT64 data types (a strict subset
of ORT's CUDA EP support). Indices for GatherND, ScatterND, and Slice are
INT64 only.

Reference implementations were ported from
`onnxruntime/core/providers/cuda/...` at tag **v1.22.2**. We deliberately
simplified or replaced ORT's design where the EP framework already gave us
information (host-side shapes from the MLIR lowering) that lets us skip ORT
machinery (TArray + fast_divmod, scratch buffers, broadcasting).

## 2. Library dependencies (ours vs ORT's CUDA / HIP EP)

**Short version**: 15 of the 16 ops are pure custom HIP kernels with **no
library dependency**, on both our side and ORT's. Only **ReduceMax /
ReduceProd** has a library-backed path in ORT (cuDNN -> MIOpen) and we
deliberately don't use it.

### How ORT's HIP EP is built

`cmake/onnxruntime_providers_rocm.cmake` does **not** ship a hand-written
HIP EP for these ops. It runs `hipify()` at build time over the entire
`onnxruntime/core/providers/cuda/` tree (`cudnn*` -> `miopen*`,
`cublas*` -> `hipblas*`, `cudaMalloc` -> `hipMalloc`, ...) and links the
result against:

```
roc::hipblas  MIOpen  hip::hipfft  rocm_smi  rccl  roctracer
                                              [optional: roc::hipblaslt]
```

So any library call you find in the CUDA EP's `.cu` / `.cc` for an op
carries over to the HIP EP as the hipified equivalent. The CUDA EP
source is therefore the authoritative reference for what the HIP EP
will use at runtime.

### Per-op breakdown (verified against ORT v1.22.2)

| Op group                     | CUDA EP source                                  | Library calls (CUDA -> ROCm)                                                                                                                              | Our impl uses    |
|------------------------------|-------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| Neg / Sign / Cos / Sin / Not | `math/unary_elementwise_ops.{cc,cu}`            | **none** -- pure custom kernels                                                                                                                           | pure custom HIP  |
| Div / Mod / Equal / Less     | `math/binary_elementwise_ops.{cc,cu}`           | **none** -- pure custom kernels                                                                                                                           | pure custom HIP  |
| ReduceMax / ReduceProd       | `reduction/reduction_ops.cc` + `reduction_functions.cu` | `cudnnReduceTensor` -> `miopenReduceTensor` for the general case, plus custom `reduce_matrix_rows` / `reduce_matrix_columns` for hot shapes        | pure custom HIP (see below) |
| Tile                         | `tensor/tile.{cc,_impl.cu}`                     | **none**                                                                                                                                                  | pure custom HIP  |
| Expand                       | `tensor/expand.{cc,_impl.cu}`                   | **none**                                                                                                                                                  | pure custom HIP  |
| GatherND                     | `tensor/gather_nd.{cc,_impl.cu}`                | **none**                                                                                                                                                  | pure custom HIP  |
| **Slice**                    | `tensor/slice.{cc,_impl.cu}`                    | **none**                                                                                                                                                  | pure custom HIP  |
| **ScatterND**                | `tensor/scatter_nd.{cc,_impl.cu}` + `atomic/common.cuh` | **none** -- pure custom kernel + native `atomicAdd` / `atomicMin` / `atomicMax` where available, CAS-emulated otherwise                          | pure custom HIP  |
| CumSum                       | `math/cumsum.{cc,_impl.cu}`                     | **none**                                                                                                                                                  | pure custom HIP  |
| Pad                          | `tensor/pad.{cc,_impl.cu}`                      | **none**                                                                                                                                                  | pure custom HIP  |
| LayerNormalization           | `nn/layer_norm.{cc,_impl.cu}`                   | **none** -- pure custom block-reduce; ORT does **not** call cuDNN / MIOpen LN here                                                                        | pure custom HIP  |

Verification method: each CUDA `.cc` was grepped for `cudnn`, `cublas`,
`miopen`, `hipblas`, `cufft`, `hipfft`. Only `reduction_ops.cc` matched
(`cudnn_common.h`, `cudnnReduceTensor*`, `CudnnReduceDescriptor`,
`CudnnTensor`). Everything else lives entirely in the EP's own `.cu`
files.

Note: `lib/Runtime/real/simplified_layer_norm.cpp` in this repo *does*
call `miopenT5LayerNormForward` -- that's a **MorphiZen-side choice**
for `SimplifiedLayerNormalization` (RMS-norm), **not** what ORT does
for plain `LayerNormalization`. Our `wrap_layer_normalization` in this
round matches ORT's pure-custom CUDA path.

### Why we don't use MIOpen for ReduceMax / ReduceProd

ORT's path is, conceptually:

```cpp
// reduction_ops.cc::ReduceComputeCore
CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(
    cudnn_handle, reduce_desc, indices, indices_bytes,
    workspace, workspace_bytes, &one, input_tensor, input_data,
    &zero, output_tensor, output_data));
```

After hipify this becomes a `miopenReduceTensor` call. To use it we
would need:

- A MIOpen tensor-descriptor cache keyed on
  `(input_shape, output_shape, data_type, reduce_op)` -- the same
  shape of cache as `T5NormCacheKey` in
  `lib/Runtime/real/simplified_layer_norm.cpp`.
- A per-shape `miopenGetReductionWorkspaceSize` query and integration
  with the shared `hipdnn_ep_state_ensure_workspace` buffer.
- A separate "indices workspace" allocation for ReduceMax (which is
  required by `cudnn` / `miopenReduceTensor` even when we don't return
  indices -- see ORT's `cudnnGetReductionIndicesSize` call).
- Handling for the fact that `miopenReduceTensor` and its CUDA
  counterpart require **at least 3-D** input (ORT left-pads to rank 3
  via `input_dims_cudnn.insert(..., pads.begin(), pads.end())`),
  adding host-side shape massaging that our current host wrappers
  don't need.

For a kernel as cheap as max / prod over a contiguous trailing axis,
the custom block-per-output kernel beats this complexity hands down.
ORT's hot path itself bypasses cuDNN for the common shapes via
`reduce_matrix_rows` / `reduce_matrix_columns` in
`reduction_functions.cu` -- our kernel is essentially that hot path
generalised, and we drop the cuDNN fallback entirely.

### When library deps WILL start to matter

The 16 ops in this round are all bandwidth-bound or
arithmetic-trivial enough that custom kernels are the right choice.
The next ops likely to be requested (softmax, conv, batch-norm, FFT,
all-reduce) **do** have meaningful library paths in ORT -- the
`ONNXRUNTIME_ROCM_LIBS` list above shows which library each class of
op pulls in:

| Op class    | ORT CUDA library  | After hipify (HIP EP) |
|-------------|-------------------|------------------------|
| Conv / Pool / BN / LRN | cuDNN  | **MIOpen**             |
| GEMM / MatMul          | cuBLAS | **hipBLAS** (or hipBLASLt) |
| Softmax (general)      | cuDNN  | **MIOpen**             |
| FFT                    | cuFFT  | **hipFFT**             |
| All-reduce             | NCCL   | **RCCL**               |
| Reduction (general)    | cuDNN  | **MIOpen**             |

When implementing any of those, the right first move is to grep the
matching CUDA `.cc` for `cudnn*` / `cublas*` symbols and follow the
existing `wrap_miopenT5LayerNormForward` pattern in
`lib/Runtime/real/simplified_layer_norm.cpp` (descriptor cache +
shared workspace + `MIOPEN_BETA_API` opt-in if needed).

## 3. Recurring design choices

### 3.1 Host-side shape arrays — no GPU shape D2H

Every `wrap_*` function for shape-aware ops (Tile, Expand, GatherND, Pad,
CumSum, ReduceMax/Prod) receives `input_shape` / `output_shape` as
**host-side `int64_t*` arrays** from the HipToLLVM lowering. The lowering
emits these as stack-allocated arrays built from the MLIR `MemRefType`'s
static dims (or from `extractStridedMetadata` for dynamic).

Consequence: the GPU `shape` / `repeats` input tensors (when present)
are **not** read by the runtime. We save a per-call D2H of the shape
tensor that the ORT CUDA EP performs. The same pattern works for Tile
(repeats[d] = output_shape[d] / input_shape[d]) and Expand (output_shape
is the broadcast result).

**Gotcha**: when a future op gets added, the lowering must pass shapes
through, otherwise the runtime is forced into a per-call D2H. The
template to follow is `lib/Conversion/HipToLLVM/TileLowering.cpp`'s
`emitShapeArray` helper.

### 3.2 D2H is sometimes unavoidable

Three ops still need synchronous D2H reads:

| Op       | What is D2H-read                          | When                                  |
|----------|-------------------------------------------|---------------------------------------|
| CumSum   | `axis` scalar (int32 or int64)            | Once per call (one stall)             |
| Pad      | `pads[]` (int64), optional `axes[]`, optional `constant_value` | Once per call (one stall, batched)    |

Each stall is one `hipStreamSynchronize`. This is acceptable because both
ops typically appear at most once or twice in a graph, but if either ever
becomes hot the right fix is to **fold the constant tensor into an
operator attribute at OnnxToHip time** — the way `Reshape`'s shape
tensor is already folded today. That moves the value into the compiled
DLL and eliminates the stall entirely.

### 3.3 Single fused kernel over multi-pass

ORT's GatherND splits into `_ComputeSliceOffsetsKernel` (writes per-slice
base offsets into a scratch buffer) + `_GatherNDKernel` (gathers using
those offsets). We **fuse into one kernel**: each output thread re-reads
the K = `indices.shape[-1]` int64 indices inline. K is small (≤ 2 for
all vision-graph GatherND nodes seen so far), so the extra global loads
are dwarfed by the scatter copy, and we avoid a scratch buffer + a
kernel launch.

The same fused-kernel approach applies to:
- **Pad**: ORT has a separate kernel per mode + an NCHW special case;
  we have one generic kernel with a `pad_mode` branch.
- **ReduceMax / ReduceProd**: a single templated
  `reduce_int_kernel<T, OP>` + `reduce_f16_kernel<OP>` covers both
  operators (and ReduceSum's existing path).
- **CumSum**: one generic per-slice serial-scan template covers
  forward / reverse and inclusive / exclusive via four kernel branches.

### 3.4 FP16 numerics: float accumulators everywhere

Every kernel that performs reduction or normalization on FP16 input
accumulates in **float**:

- ReduceMax / ReduceProd / ReduceSum: float accumulator, FP16 init via
  `-INF` (max) / `1.0f` (prod) / `0.0f` (sum).
- CumSum (FP16 variant): float accumulator across the axis.
- LayerNormalization: sum / sum² and final `(x - mean) * inv_std * s + b`
  all in float, narrow on store.
- Equal / Less (FP16 inputs): promote to float for the comparison; output
  is a 1-byte bool stream.

The pattern is `static_cast<float>(x)` on load and
`static_cast<T>(value)` (or `__float2half`) on store — see the existing
matmul_nbits and reduce_sum kernels for the convention.

### 3.5 Rank bounded to 8

All shape-aware kernels (`tile_kernel.hip`, `expand_kernel.hip`,
`gather_nd_kernel.hip`, `pad_kernel.hip`) hard-cap input rank at 8 via a
`kFooMaxRank` constant. This matches `TArray<int64_t, 8>` from ORT and
covers every op shape we observed in `vision.onnx` (max rank 6). A
higher rank would silently fail the host pre-check — error message
identifies the kernel + the requested rank.

### 3.6 Broadcasting is upstream's job

The binary elementwise kernels (Div, Mod, Equal, Less) **do not
broadcast** — they require identical layouts for the two inputs. The
ONNX-to-HIP conversion is expected to insert explicit `Expand` (or
similar) nodes upstream to produce equally-shaped operands. This matches
the existing Add/Mul/Sub family in `elementwise_kernel.hip`. If a graph
arrives with implicit broadcasting it will fail at the lowering stage,
not at runtime.

### 3.7 Slice and ScatterND — host-side indices

`Slice` and `ScatterND` both move tensor data based on small INT64
index/control tensors:

| Op        | Index-shaped inputs                                | Where they live       | What we do                                |
|-----------|----------------------------------------------------|-----------------------|-------------------------------------------|
| Slice     | `starts`, `ends`, optional `axes`, optional `steps` | GPU tensors (graph inputs in the non-folded case) | D2H + `hipStreamSynchronize` once per call, then resolve per ONNX clamping rules host-side and pass the per-axis `(start, step)` arrays into the kernel as host int64 vectors |
| ScatterND | `indices`                                           | GPU tensor             | **Stay on the device** — the kernel reads them inline. No D2H at all. |

Slice has to D2H because per-axis (start, step) needs ONNX clamping
(`step > 0`: `start ∈ [0, dim]`, `end ∈ [0, dim]`; `step < 0`:
`start ∈ [0, dim-1]`, `end ∈ [-1, dim-1]`) plus the optional `axes`
list to know which axis each entry applies to. Doing this on the GPU
would require either a launcher prepass or an oversized device kernel
with the same per-axis state-machine — neither is worth it for a 4×
int64 D2H. The matching pattern in `lib/Runtime/real/pad.cpp` and
`lib/Runtime/real/cumsum.cpp` does the same thing (one stall per call).

ScatterND keeps indices on the device because there is no clamping
state machine to run host-side — each thread does its own
out-of-range clamp inline (`idx >= dim ? dim-1 : idx`, etc.) and looks
up the stride from a compact host-built `ScatterNDParams` struct. The
index data itself is whatever shape the graph provides; we don't need
to inspect any individual value.

The compile-time `Slice` fold lives in `SliceConversion.cpp` and runs
when all four index tensors are graph-constant AND every step is
positive. It lowers to `tensor.extract_slice` (which becomes a
`memref.subview` after bufferization and is zero-cost at runtime). Any
graph that doesn't meet both conditions falls through to `wrap_slice`
and the runtime path described above. Test `test_slice_negative_step`
in `test/python/tests/test_shape_ops.py` covers both legs.

### 3.8 ConstantOfShape and the pre-fold ordering

`ConstantOfShape` is the only op in this round that produces NO
runtime kernel — `ConstantOfShapeConversion.cpp` folds it to an
`arith.constant` splat at compile time. The fold has two
non-obvious requirements:

1. **It must run BEFORE `lowerOnnxConstants`** (which externalises
   any `onnx.Constant >= 1 element` into `constants.bin` via a
   `memref.global` with `initial_value = nullptr`). After
   externalisation the shape data is on disk, not in the IR, and the
   fold can't reach it. `ConvertOnnxToHipPass::runOnOperation` runs
   ConstantOfShape patterns in a dedicated "pre-fold" greedy pass
   immediately before the externaliser loop.

2. **The fold accepts `onnx.Shape(static-tensor)` as a compile-time
   constant input**, not just `onnx.Constant`. This is what
   transformer graphs typically emit for zero-initialised KV / mask
   buffers: `output_shape = Shape(some_static_input);
   buf = ConstantOfShape(output_shape)`. `getCompileTimeConstantTensor`
   in `ConstantOfShapeConversion.cpp` handles this by reading the
   source tensor's static shape directly, honouring the optional
   `start` / `end` slicing attributes from ONNX-15 Shape.

The result is that every gpt-oss-style "alloc-zero" pattern collapses
to a single splat constant, which then flows through the normal
constant-externalisation path (small splats stay inline; larger ones
land in `constants.bin`). No runtime cost, no kernel.

### 3.9 No new `RuntimeState` fields

None of the 16 kernels add persistent per-session state — they're all
stateless or use the existing shared workspace via
`hipdnn_ep_state_ensure_workspace`. If any of them ever grows a cache
(e.g. LayerNorm wants to memoize a tuned block size per shape), follow
the **runtime module registry** convention in
`docs/design/runtime-module-registry.md` — do **not** add a new field
on `RuntimeState`.

## 4. Build / cache hygiene gotchas

Discovered the hard way during incremental commits. None of these are
new — they're listed in `CLAUDE.md` already — but they bit us multiple
times per kernel.

1. **Bitcode `DEPENDS` list in `lib/Runtime/CMakeLists.txt`.** Every
   header included by a runtime `.cpp` must appear in the bitcode
   target's `DEPENDS` list (`compile_to_bitcode(...)` macro). Without
   this, editing the header (e.g. `hip_custom_kernels.h`) leaves the
   bitcode stale and the compiled model DLL silently uses old code.

2. **Stale compiled-model DLLs in `%TEMP%`.** The MorphiZen cache key is
   the ONNX-graph hash, **not** the runtime version. Every kernel
   commit that changes runtime behaviour must be followed by
   `del %TEMP%\morphizen_mlir_*` (the build helper `_build.bat` does
   this automatically). Forgetting this is the #1 source of "my fix
   doesn't work" false alarms.

3. **`compile_to_bitcode` listed twice.** When adding a new
   `compile_to_bitcode(real/foo.cpp ...)` and `RUNTIME_BC_MODULES`
   entry, double-check the file isn't already present further down
   (often is, from a previous skeleton stub). A duplicate isn't fatal —
   CMake silently re-uses the same target — but it costs build time.

4. **New `.hip` file requires CMakeLists.txt update.** Each new
   `3rd-party/custom_kernels/hip/foo_kernel.hip` must be added to the
   `HIP_KERNEL_SOURCES` list in `3rd-party/custom_kernels/CMakeLists.txt`
   or it won't be linked into `custom_kernels.lib`.

5. **PowerShell + commit messages.** Use `git commit -F <file>` with a
   pre-written message file — `cmd /c` heredoc-style commits do not
   work reliably on Windows PowerShell.

## 5. Silent stubs are the worst kind of bug

Before this round, all 16 op `wrap_*` functions were `return 0` stubs.
The runtime returned success while doing nothing — the output tensor
was left at whatever uninitialised state the pool allocator gave it. The
E2E lit tests in `test/lit/` check **only** for NaN/Inf in the output,
not numerical correctness against a reference, so they **passed** for
every silent-stub op. The op was completely missing in production and
nobody noticed until model accuracy regressed at a higher level.

**Lesson**: when seeing a new "kernel didn't break the test" green
checkmark, verify the kernel was actually called. The host wrapper
should always log `[REAL] wrap_foo: ... -> hip_foo` at debug level so
`HIPDNN_EP_DEBUG=1` traces show whether the new code path ran. Every
kernel in this round added that line.

The follow-up structural fix (not in this commit) is to make the test
harness compare against a CPU reference, not against NaN/Inf. Until that
lands, prefer to validate new kernels with the actual production model
(`vision.onnx`) and inspect the final embedding rather than trusting
the lit test alone.

## 6. Reduce / binary signature loss

The MLIR lowering for `ReduceMax` / `ReduceProd` collapses all reduce
axes into the **trailing dims** of `data` before the call, so the
runtime can compute `reduce_size = data_num / output_num` without
inspecting `axes`. This is why `wrap_reduce_*` accepts only `data_num`
and `output_num` plus `axes_num_elements` (and a `noop_with_empty_axes`
flag for the short-circuit). The actual axes vector is gone by the
time we get the call — the lowering has consumed it.

Same story for the binary elementwise ops: `wrap_div` / `wrap_mod` /
`wrap_equal` / `wrap_less` take only a single `num_elements` and
require both inputs already broadcast to that shape. No per-input
shape, no broadcast factors. The MLIR side does the work, and the
runtime stays simple.

When adding a new reduction or binary op, **do not** request shape info
that the lowering didn't volunteer — that's a sign the op needs to be
expressed differently in the MLIR pipeline rather than worked around in
C++.

## 7. ONNX corner cases handled

A few small specification quirks worth recording so the next session
doesn't re-discover them:

- **`Mod` and the `fmod` attribute**: integer mod follows Python's
  sign-of-divisor convention. Float Mod with `fmod=0` is also
  Python-style; `fmod=1` is C99 `fmod()` (sign-of-dividend). Two code
  paths inside `hip_elementwise_mod`.

- **`Not` ignores `data_type`**: ONNX `Not` runs on `tensor<i1>` but
  the lowering passes `data_type=0` (FLOAT) because the MLIR i1 type
  has no HIPDNN_EP enum slot. `wrap_not` treats both input and output
  as raw uint8 byte streams unconditionally.

- **`GatherND`'s negative indices**: indices in `[-D, D-1]` are valid;
  the kernel normalises `idx += dim` when `idx < 0`. Out-of-range
  indices are NOT clamped — matches ORT's release-build behaviour
  (CUDA_KERNEL_ASSERT becomes a no-op).

- **`Pad` mode IDs from the lowering**: 0 = constant, 1 = reflect,
  2 = edge, 3 = wrap. The lowering maps the string attribute via
  `PadOpLowering::modeIdFromString`. Wrap mode is not in ORT's
  `pad_impl.cu` — we added it as a `%` of the axis length.

- **`CumSum` axis is a 0-D scalar input**, not an attribute. Can be
  int32 or int64; D2H + sync once per call to read it.

- **`LayerNormalization` stash_type is raw ONNX `TensorProto.DataType`**
  (1 = FLOAT, 10 = FLOAT16), **not** the HIPDNN_EP enum. The lowering
  passes `op.getStashType()` unchanged. Default is FLOAT.

- **`Slice` non-constant indices / negative steps**: the graph-constant
  + positive-stride case folds to `tensor.extract_slice` in
  `lib/Conversion/OnnxToHip/SliceConversion.cpp` and never reaches the
  runtime. The `hip_slice` kernel only services slices whose
  `starts`/`ends`/`axes`/`steps` are graph inputs (D2H + sync once per
  call) **or** that have at least one negative step. Indices are
  INT64 only — INT32 indices would need a stride-aware ABI bump.

- **`Slice` end-sentinel for "all the way down" with `step < 0`**: per
  ONNX-13+ spec, any `end < 0` is normalised via `end += dim` **before**
  clamping. To express "stop at index 0 inclusive" with a negative step
  you must pass `end = -(dim + 1)` (which post-normalisation becomes
  `-1`, the sentinel that step<0 clamping permits in `[-1, dim-1]`).
  Passing `end = -1` literally means "stop at index `dim-1`" — i.e.
  empty output when start ≥ dim-1. Matches ORT CPU `slice_helper.h`.

- **`ScatterND` out-of-range indices**: clamped into range, not
  rejected. `idx ≥ dim → dim-1`; `idx < -dim → 0`. This matches ORT's
  CUDA EP (`scatter_nd_impl.cu` lines 43-53); ORT CPU treats OOB as an
  error, but throwing from device kernels is impractical. CPU-vs-GPU
  divergence here is documented upstream as "consistent with other GPU
  backends".

- **`ScatterND` duplicate indices + `reduction="none"`**: per ONNX spec
  the result is undefined when multiple updates target the same
  position. Our kernel uses last-writer-wins via a non-atomic store;
  no thread-ordering guarantees. ORT CPU is the same.

- **`ScatterND` reduction dtype matrix**: native atomics are used where
  available (`atomicAdd` for f32/i32; `atomicAdd(unsigned long long*)`
  for i64; `atomicMin`/`atomicMax` for i32). Everything else
  (mul for all types, min/max for i64/f32/f16, add for f16) is
  packed-CAS emulated in `cas_atomic_apply<T>` / `atomic_apply_f16`.
  fp16 packed CAS rewrites the 32-bit-aligned word containing the
  target half — relies on the standard 2-byte alignment of fp16 ONNX
  tensors. Matches the ORT CUDA EP's `atomic_add` / `atomic_mul` /
  `atomic_min` / `atomic_max` helpers in `atomic/common.cuh`.

- **`ConstantOfShape` is folded at compile time** in
  `lib/Conversion/OnnxToHip/ConstantOfShapeConversion.cpp`. The shape
  input must be a recognised compile-time constant — currently:
  `arith.constant`, `onnx.Constant` with a dense `value` attribute,
  `onnx.Shape(static-tensor)` (with optional `start`/`end` slicing),
  or a `bufferization.to_tensor` of an externalised
  `memref.get_global`. The fold runs in `ConvertOnnxToHipPass`
  **before** `lowerOnnxConstants` externalises constants, so the
  shape's dense bytes are still reachable. After externalisation the
  shape would be `memref.global` with null `initial_value` and the
  fold could no longer recover the data. No runtime symbol exists —
  the op disappears from the IR entirely.

## 8. What this leaves unfinished for `vision.onnx`

The 16 ops fix the obvious silent-stub holes. They do **not** by
themselves make `vision.onnx` produce correct output end-to-end —
known remaining blockers:

- **`GroupQueryAttention` packed-QKV variants** beyond the
  `(HPG, d) in {(4, 64), (4, 128), (8, 64)}` cluster aren't
  instantiated for flash_decode. Qwen2.5-VL uses HPG=5; gemma3 uses
  d=256. Neither has a flash_decode instantiation. The fused/legacy
  fallback paths exist but cap at `total_seq=256`.

- **`castlike_model` E2E test** still fails — pre-existing,
  unrelated to this work.

- **Dynamic shape support.** `vision.onnx` is fixed-shape (the perf
  test harness runs `fix_shapes()` first), but the production OGA
  pipeline may want dynamic image-patch counts. The compiler does
  not yet support symbolic dims.

- **`SkipLayerNormalization` and `BiasAdd` fusions** that ORT applies
  to LN-heavy paths are not implemented here. The
  `hip_layer_norm` kernel handles the bias but won't fuse a residual
  add.

- **Per-op profiling.** Every kernel in this round added an
  `OP_PROFILE("opname", ...)` scope, so `HIPDNN_EP_PERF=1` will
  surface their GPU/CPU times. Use this **before** chasing the next
  bottleneck rather than after.

## 9. Future-development checklist

When adding the next op:

1. Read `CLAUDE.md`'s "Adding a New Operator" section — it's the
   authoritative checklist.
2. Decide if the lowering can hand you everything you need (host
   shape arrays, constant attributes). If yes, write a stateless
   `wrap_foo` + a `.hip` file. If you need anything cross-call, use
   `HIPDNN_OP_MODULE` (`docs/design/runtime-module-registry.md`).
3. Always add an `OP_PROFILE` scope with a meaningful shape string —
   it costs nothing when profiling is off and is invaluable when on.
4. Always log `[REAL] wrap_foo: ... -> hip_foo` so
   `HIPDNN_EP_DEBUG=1` traces show the kernel ran.
5. Add the `.cpp` to **both** `compile_to_bitcode(...)` and
   `RUNTIME_BC_MODULES` in `lib/Runtime/CMakeLists.txt`. Add the
   `.hip` to `HIP_KERNEL_SOURCES` in
   `3rd-party/custom_kernels/CMakeLists.txt`.
6. Add the kernel header's prototype to
   `3rd-party/custom_kernels/include/hip_custom_kernels.h`, **inside
   the existing `extern "C"` block**.
7. After building, `del %TEMP%\morphizen_mlir_*` before testing — the
   build helper does this, manual cmake runs don't.
8. Run the matching `E2E_Execute_test_<op>_model` lit test, but
   remember (Section 5) that it only checks NaN/Inf — verify against
   the production model too.

## 10. References

- `CLAUDE.md` — the canonical session knowledge base. Read first.
- `docs/design/custom_kernel_design.md` — the bitcode-link-into-DLL
  pipeline these kernels plug into.
- `docs/design/runtime-module-registry.md` — for ops that need
  per-session state.
- `docs/design/compiler-runtime-contract.md` — the ABI rules the
  `wrap_*` functions must obey (extern "C", listed in
  `getRuntimeFuncSpecs`).
- `onnxruntime/core/providers/cuda/...` @ tag `v1.22.2` — the
  reference implementations cited per-op in each kernel commit.
