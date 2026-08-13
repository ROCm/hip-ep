<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Quick Start Guide (AMD Instinct MI350X / gfx950)

This guide covers running hip-ep on **AMD Instinct MI350X** (`gfx950`, CDNA4).
It is a **delta** on top of [quick_start_linux.md](quick_start_linux.md) — build
entry paths, `$ROOT` setup, and the shared *Testing & Benchmarking* tooling are
identical and are not repeated here. Read that guide first; come back here for
what MI350X does differently.

**Linux only.** MI350X is now only supported on the Linux platform.


## Why MI350X needs its own notes

The custom kernels were originally written for RDNA3/RDNA4 (`gfx11xx`/`gfx12xx`),
which run **wave32** and expose the **WMMA** matrix intrinsics. CDNA4 is a
different architecture family:

| | RDNA3 / RDNA4 (e.g. gfx1151) | CDNA4 (gfx950 / MI350X) |
|---|---|---|
| Wavefront size | 32 | **64** |
| Matrix intrinsics | WMMA (`__builtin_amdgcn_wmma_*`) | **MFMA** — no WMMA |
| Mixed-sign dot4 | `__builtin_amdgcn_sudot4` | signed `__builtin_amdgcn_sdot4` only |

The shared kernel sources compile for both families through
[`lib/Runtime/Kernels/include/hip_arch_compat.h`](../lib/Runtime/Kernels/include/hip_arch_compat.h),
which supplies a compile-time `HIPDNN_WAVE_SIZE` / `HIPDNN_HAS_WMMA` and a
portable `hipdnn_sudot4()`. Host-side dispatch decisions are made at **runtime**
from the device's `warpSize`, not from the architecture string, so the same
binary behaves correctly on a mixed-GPU host.

## Host prerequisites

Same as the Linux guide, with these differences:

| Tool | Purpose |
|------|---------|
| **AMD Instinct MI350X** + `/dev/kfd` + `/dev/dri/renderD*` | HIP runtime (replaces the gfx1151 requirement) |
| **ROCm with hipBLASLt** | **Hard requirement** on MI350X — see [hipBLASLt is required](#hipblaslt-is-required) |
| cmake ≥ 3.29, ninja, git, a C++ compiler, python3 | Native build path |
| Docker 26+ | Docker build path (optional) |

## Build

**You must build from source.** The published `linux-gpu-test-package` is built
for `gfx1151` only (`HIP_ARCHITECTURES: gfx1151` in
[`.github/workflows/linux-build.yml`](../.github/workflows/linux-build.yml)), so
it does not contain the gfx950 kernel library and will not run on MI350X.

`build.py` auto-detects the architecture from
`/sys/class/kfd/kfd/topology/nodes/*/properties`, where MI350X reports
`gfx_target_version 90500` → `gfx950`. On an MI350X host no flag is needed:

```bash
git clone https://github.com/ROCm/hip-ep.git
cd hip-ep
python3 build.py                    # auto-detects gfx950
python3 build.py --hip_arch gfx950  # or state it explicitly
```

The Docker path detects the arch host-side the same way:

```bash
./docker/run.sh image        # first time only
./docker/run.sh build        # auto-detects gfx950
#   HIP_ARCHITECTURES=gfx950 ./docker/run.sh build   # explicit
```

There is no gfx950 allow-list anywhere in the tree — the arch string is passed
through to the HIP compiler, so nothing needs to be registered to add it.

### Artifacts

```
<workspace>/install/lib/libhipgpu.so                  # the EP
<workspace>/install/lib/libcustom_kernels_gfx950.so   # gfx950 kernels
```

`libcustom_kernels_<arch>.so` **must sit next to `libhipgpu.so`**. At session
init the EP probes `hipGetDeviceProperties(0).gcnArchName`, builds the filename
from it, and `dlopen`s it from its own directory (the EP is linked with
`RPATH=$ORIGIN`, so it does not need to be on `LD_LIBRARY_PATH`). A missing or
wrong-arch kernel library is a **hard failure** at init, not a fallback.

## Runtime setup and benchmarking

Identical to the Linux guide — see
[Open a container shell and set `$ROOT`](quick_start_linux.md#open-a-container-shell-and-set-root)
and [Testing & Benchmarking](quick_start_linux.md#testing--benchmarking). Nothing
in `hip-onnx-runner`, `onnxruntime_perf_test`, or `model_benchmark` is
MI350X-specific.

A worked end-to-end example, batch 1, INT4 RTN block-32:

```bash
export THEROCK_DIST=/opt/rocm     # or the auto-downloaded build/hip-ep/_therock
export LD_LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib:<ort-prefix>/lib"

$ROOT/bin/model_benchmark \
  -i /path/to/oga-model-dir \
  -l 2048 -g 128 -r 3 -w 1 -b 1
```

`THEROCK_DIST` is not optional even when the ROCm libs are already on
`LD_LIBRARY_PATH`: `hip-compiler` reads it to build the `-L` paths it hands to
`ld.lld` when linking a per-model native artifact. Without it that link fails
with `undefined symbol: hipGetDeviceCount`.

`<ort-prefix>` must be the **same ONNX Runtime the EP was compiled against**.
`cmake/deps.cmake` downloads a specific ORT release into the build tree
(`build/<repo>/_deps/onnxruntime-src`), and the EP requests that release's API
version at load time — see [ORT version skew](#ort-version-skew) below.

The EP is selected by the model's `genai_config.json` `provider_options` and
auto-discovered next to the OGA runtime lib, exactly as on RDNA. (If your OGA
build needs to be pointed at a specific EP `.so`, that is an onnxruntime-genai
umbrella-EP override such as `AMDGPU_EP_PATH` — hip-ep itself does not read it.)

## What is not supported on MI350X

Everything below keys off the **runtime wave size** (`warpSize >= 64`), so it
applies to any CDNA part, not just gfx950.

| Feature | Behaviour on MI350X | Where |
|---|---|---|
| **Quantized / INT8 KV cache for GQA** | **Hard error, returns −1.** Use an fp16 KV cache. | `lib/Runtime/real/gqa.cpp` |
| WMMA fused GQA (the `hip_gqa_flash_prefill_*` / `_decode_*` kernels, including the attention-sink and sliding-window prefill) | Silently routed to the decomposed hipBLASLt pipeline | `gqa.cpp` (`fused_supported`) |
| Legacy `flash_decode` launcher | Silently disabled; `HIPDNN_EP_GQA_FLASH_DECODE=1` has no effect | `gqa.cpp` (`gqa_flash_decode_enabled`) |
| WMMA MatMulNBits prefill | Replaced by dequant-once + hipBLASLt GEMM for INT4 M≥16 | `matmul_nbits_kernel.hip`, `matmul_nbits.cpp` |
| WMMA MultiHeadAttention flash prefill | Silently routed to the decomposed path | `multi_head_attention.cpp` |

Only the first is user-visible. If you point an INT8-KV-cache model at MI350X
you get:

```
wrap_group_query_attention: quantized KV cache is not supported on wave64
devices (CDNA, e.g. MI350) -- its kernels are on the WMMA fused path, which is
RDNA-only. Use an fp16 KV cache.
```

This fails loudly on purpose rather than misreading quantized bytes as fp16.
Supporting it on CDNA is follow-up work.

### hipBLASLt is required

On MI350X, hipBLASLt is not an optimization — it is the execution path:

- **GQA** always uses `gqa_forward_hipblaslt()`, because the fused WMMA path is
  off. A null handle fails the op.
- **INT4 MatMulNBits prefill** (batch 1, `K % 32 == 0`, M ≥ 16, fp16) dequantizes
  the weight to fp16 once, caches it per weight pointer, and runs the GEMM
  through hipBLASLt on the MFMA matrix cores.
- Session creation calls `hipblasLtCreate()` and fails if it does not succeed.

The first call for each new `(M, N, K)` shape **autotunes** the hipBLASLt
algorithm by timing the ranked heuristic candidates and caching the winner. This
is independent of `HIPDNN_EP_AUTOTUNE` (which gates a different code path) and
has two consequences worth knowing: the first request of a shape is slower, and
dense-prefill timings vary run to run because a cold autotune under ramping GPU
clocks can elect a different algorithm.

### Environment variables that behave differently

The GQA env knobs documented for RDNA still parse on MI350X, but several are
inert because the code path they control is disabled:

| Variable | On MI350X |
|---|---|
| `HIPDNN_EP_GQA_FLASH_DECODE` | **Ignored** — flash_decode is off on wave64 regardless |
| `HIPDNN_GQA_DECODE_WMMA`, `HIPDNN_GQA_DECODE_SCALAR`, `HIPDNN_GQA_DECODE_SPLITS` | **No effect** — they tune the fused decode path, never entered on wave64 |
| `HIPDNN_EP_GQA_DISABLE_FUSED_DECODE` | Redundant — the decomposed path is already the only one |
| `HIPDNN_EP_MATMUL_DP4A` | Active; the DP4A GEMV uses the portable `hipdnn_sudot4()` |

## Measured performance

MI350X (gfx950, 8 GPUs present, pinned to device 0), EPYC 9965, ROCm 7.2.0,
ONNX Runtime 1.25.1. OGA `model_benchmark`, batch 1, generate 128, 1 warmup +
3 reps, INT4 RTN block-32.

| Model | Prompt | TTFT | Decode | Peak working set |
|---|---:|---:|---:|---:|
| phi-4 (14B dense) | 128 | 45.5 ms | 54.0 tok/s | 4.2 GB |
| phi-4 | 512 | 173.6 ms | 106.5 tok/s | 4.5 GB |
| phi-4 | 2048 | 550.1 ms | 64.6 tok/s | 6.2 GB |
| gpt-oss-20b (MoE) | 128 | 683.9 ms | 238.8 tok/s | 4.3 GB |
| gpt-oss-20b | 512 | 2945.1 ms | 210.5 tok/s | 4.6 GB |
| gpt-oss-20b | 2048 | 11 153.8 ms | 107.2 tok/s | 6.9 GB |

Two things to read from this. Dense INT4 prefill is healthy (~2.8–3.7k tok/s)
because it runs the hipBLASLt MFMA path. **MoE prefill is the known weak spot**:
gpt-oss-20b sits near 180 tok/s regardless of prompt length, because the expert
FFNs go through the `qmoe` kernel, which was ported for correctness on wave64 but
still uses the naive, un-tiled prefill route — exactly where MatMulNBits was
before the dense fast path was added. MoE *decode* is strong.

The TTFT column above is a single sweep, and **dense-prefill TTFT is far noisier
than that presentation suggests** — the hipBLASLt autotune described above elects
a different algorithm from run to run. Measured over five separate process
invocations on one build, phi-4 spans 43.3–55.9 ms at L=128 and 536.7–711.5 ms at
L=2048, a coefficient of variation near 12 % in both cases.

So do not read a single dense-TTFT sample as a measurement: a 10 % swing between
two builds is inside the noise. Average several *process* invocations (repeats
within one process share the elected algorithm and understate the spread) before
concluding anything about a change. Decode throughput and peak memory are stable
to ~1 %, so they are the reliable signals for A/B comparisons.

## Troubleshooting

**`LlvmIrJit: Load(.../libcustom_kernels_gfx950.so) ... failed`**

The gfx950 kernel library is missing from the EP's own directory. Confirm it was
built and co-located:

```bash
ls -l "$ROOT/lib/libcustom_kernels_"*.so   # expect libcustom_kernels_gfx950.so
```

If only `libcustom_kernels_gfx1151.so` is present you are running the prebuilt CI
package, which does not support MI350X — build from source (see [Build](#build)).

**Segfault at session creation, with `The requested API version [N] is not
available, only API versions [1, M] are supported in this build`**

<a id="ort-version-skew"></a>ONNX Runtime version skew. The EP was compiled
against the ORT release `cmake/deps.cmake` downloaded into the build tree, but a
different (older) `libonnxruntime.so` is ahead of it on `LD_LIBRARY_PATH` —
commonly one belonging to a separately built onnxruntime-genai. The API-version
line is the only real clue; the process then dies with SIGSEGV.

Point `<ort-prefix>` at the ORT that came with your build tree:

```bash
ls <workspace>/build/hip-ep/_deps/onnxruntime-src/VERSION_NUMBER
export LD_LIBRARY_PATH="$ROOT/lib:$THEROCK_DIST/lib:<workspace>/build/hip-ep/_deps/onnxruntime-src/lib"
```

**`Failed to create hipBLASLt handle`**

hipBLASLt is missing or not on the loader path. It is required on MI350X (see
[hipBLASLt is required](#hipblaslt-is-required)); make sure the ROCm libs are on
`LD_LIBRARY_PATH`.

**`quantized KV cache is not supported on wave64 devices`**

Expected — the model uses an INT8 KV cache. Use an fp16-KV-cache build of the
model.

**Kernels build but produce wrong results after editing a kernel**

Anything reducing across a wavefront must span the whole hardware wave. Use
`HIPDNN_WAVE_SIZE` (compile-time, inside kernels) and `hipdnn_device_wave_size()`
(host-side, for launch geometry) rather than a literal 32 or 64 — a block that is
a partial wave folds inactive lanes into `__shfl_xor` reductions. See
[`hip_arch_compat.h`](../lib/Runtime/Kernels/include/hip_arch_compat.h).

For everything else, see
[Troubleshooting in the Linux guide](quick_start_linux.md#troubleshooting) —
those entries apply unchanged.
