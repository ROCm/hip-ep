<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**MANDATORY:** When a session produces new critical observations (build gotchas, API patterns, model quirks, test conventions), update this file and relevant `docs/*` files before finishing. **Any code change that affects behavior documented in `docs/` or `docs/design/` must update those docs in the same session — never leave them stale.** Always check `docs/design/*` for design documents that reference renamed symbols, changed APIs, updated metrics, or modified architecture. This is the single source of truth — do not use `.claude/memory` or external memory files.

## Project Overview

ONNX HIP DNN Execution Provider — an MLIR compiler that converts ONNX models into AMD GPU-accelerated DLLs via a HIP dialect, targeting MIOpen, hipBLASLt, and custom HIP kernels. Integrates into ONNX Runtime through the MorphiZen Execution Provider framework.

## Build Commands

### Quick start (recommended)

`build.py` automates everything: downloads dependencies (TheRock, LLVM/MLIR, Protobuf, FlatBuffers, ONNX Runtime), detects VS2022 and GPU, configures, builds, and installs. All artifacts go under `install/`.

```bash
# One-time: create conda environment
conda env create -f environment.yml
conda activate hipdnn-ep

# Build (from any shell — VS Developer Prompt not required)
python build.py                # full build
python build.py --skip-build   # download deps only
python build.py --clean        # wipe install/ and start fresh
```

`build.py` auto-detects VS2022 via `vswhere.exe`, injects MSVC environment, detects GPU architecture via `amdgpu-arch.exe`, and selects real vs mock runtime accordingly. Idempotent — skips already-completed steps.

Output layout:
- `install/therock/` — TheRock ROCm SDK
- `install/deps/` — prebuilt LLVM/MLIR/Protobuf/FlatBuffers
- `install/onnxruntime/` — ONNX Runtime binaries
- `install/build/` — CMake build directory
- `install/dist/` — final installed binaries

### Run tests

```bash
# All tests
ctest --test-dir install/build -C RelWithDebInfo

# LIT tests only (MLIR pass verification)
ctest --test-dir install/build -C RelWithDebInfo -R MorphizenMLIRLitTests

# Python tests (performance benchmarks, integration)
pytest test/python -v -s
```

### Lint / format

```bash
lintrunner -a                    # format changed files
lintrunner -a --all-files        # format all files
pre-commit run --all-files       # all hooks (whitespace, license, clang-format, ruff)
```

### Manual CMake (advanced)

For manual builds without `build.py`, see the cmake invocations in `build.py`'s `configure_and_build()`. Key variables:

| Variable | Purpose |
|----------|---------|
| `CMAKE_PREFIX_PATH` | Semicolon-separated: deps dir + ORT dir |
| `THEROCK_DIST` | Path to TheRock ROCm SDK |
| `HIP_ARCHITECTURES` | GPU target (e.g. `gfx1150`). **Mandatory** for real runtime |
| `BUILD_HIP_TOOLS` | Build MLIR compiler tools |
| `BUILD_EP` | Build MorphiZen Execution Provider |
| `BUILD_MOCK_RUNTIME` | Use CPU stubs instead of GPU runtime |

## Critical Build Gotchas

- **Conda env is a prerequisite.** `build.py` expects tools (cmake, ninja, sccache, lit) from the `hipdnn-ep` conda env to be on PATH.
- **HIP_ARCHITECTURES must match the GPU.** Omitting or mismatching causes silent build success but runtime crashes (`0xC0000005` in `hipLaunchKernel`). `build.py` auto-detects via `amdgpu-arch.exe`.
- **CRT must be Release /MT.** Pre-built LLVM/MLIR use static Release CRT. Debug (`/MTd`) or dynamic (`/MD`) produces linker errors.
- **Static CRT means separate CRT per DLL.** Model DLLs compiled with `/MT` have their own CRT instance — `std::getenv()` cannot see env vars set by the host process. Use `GetEnvironmentVariableA()` (Win32 API) instead. See `debug_log.h` for the pattern.
- **DIA SDK junction** may be needed: prebuilt LLVM hardcodes `C:\msvsn2022` for DIA SDK path. `build.py` creates this automatically.
- **sccache + RelWithDebInfo**: CMakeLists.txt swaps `/Zi` → `/Z7` and disables incremental linking to prevent PDB file contention during parallel builds.
- **TheRock DLLs on PATH at runtime.** Compiled model DLLs link against `amdhip64_7.dll`, `MIOpen.dll`, etc. from `install/therock/bin/`. Add that directory to PATH before running `hip-onnx-runner` or any compiled model.
- **Models must have static shapes.** The MLIR compiler does not support dynamic/symbolic tensor dimensions. Convert models to fixed shapes before use (see Test Models section).
- **ORT version must match pip.** `build.py` pins `ORT_VERSION` to match the pip `onnxruntime-directml` package. The MorphiZen EP DLL checks the ORT API version at registration; a mismatch (e.g. EP built against API v25 vs pip API v24) causes segfault. When upgrading, update `ORT_VERSION` in `build.py` and `onnxruntime-directml` in pip in lockstep.
- **Runtime functions called from generated code need `extern "C"` linkage.** Declare them in `hipdnn_ep_runtime.h` (which wraps everything in `extern "C"`). Without this, Clang produces C++-mangled names in the bitcode but `GenerateInterface.cpp` emits unmangled references — causing link failures.
- **Stale compiled-model DLLs after runtime changes.** Compiled model DLLs embed the runtime bitcode from build time. The MorphiZen cache key is based on the ONNX graph hash, not the runtime version — so changing runtime `.cpp` files (or custom kernels) and rebuilding does NOT invalidate cached models. After rebuilding, delete stale DLLs manually: `rm "$TEMP"/morphizen_mlir_*` (bash) or `del %TEMP%\morphizen_mlir_*` (cmd).
- **Bitcode DEPENDS list must include all headers.** The `compile_to_bitcode` macro in `lib/Runtime/CMakeLists.txt` uses a `DEPENDS` list to track when recompilation is needed. Every header that a runtime `.cpp` file `#include`s must be listed there — otherwise, editing the header doesn't rebuild the bitcode, and compiled models silently use stale runtime code. When adding a new `#include` to a runtime `.cpp` file, always update the `DEPENDS` list in the macro.
- **Linker byproducts not cleaned up.** `CompilerDriver::cleanupIntermediates()` removes `.ll` and `.obj` files after linking, but LLD also creates `.lib`, `.pdb`, and `.exp` byproducts alongside each `.dll`. These accumulate in `%TEMP%` (hundreds of files over time). Known issue — fix requires extending `cleanupIntermediates()` in `CompilerDriver.cpp`.

## Architecture

### Compilation Pipeline

```
ONNX (.onnx) → [onnx-to-hip-pipeline] → HIP dialect → [bufferize + optimize] → [hip-to-llvm-pipeline] → LLVM IR → link → model.dll + constants.bin
```

Key passes in order:
1. `hip-add-context-arg` — inject `!hip.context` argument
2. `convert-onnx-to-hip` — ONNX ops to HIP dialect ops (externalizes large constants to `.constants.bin`)
3. `one-shot-bufferize` — tensor semantics to memref (buffer) semantics
4. `buffer-deallocation` + `hip-optimize-memrefs` — liveness-based buffer reuse
5. `hip-pool-allocs` — pack all allocations into a single grow-on-demand GPU buffer
6. `hip-lower-allocs` — `memref.alloc` to `hip.alloc`/`hip.free`
7. `convert-hip-to-llvm` — HIP ops to runtime C API calls
8. `generate-interface` — emit `inference_init`/`inference_compute`/`inference_cleanup` entry points

### Three-layer lowering

| Layer | Location | What it does |
|-------|----------|--------------|
| ONNX → HIP | `lib/Conversion/OnnxToHip/` | Pattern-matches ONNX ops by name (no onnx-mlir dependency) and emits HIP dialect ops |
| HIP dialect | `lib/Dialect/` | Custom MLIR dialect with ops for MIOpen, hipBLASLt, and custom kernels; TableGen-defined |
| HIP → LLVM | `lib/Conversion/HipToLLVM/` | Lowers HIP ops to C API calls into the runtime library |

### Runtime

- **Real** (`lib/Runtime/real/`): GPU execution via MIOpen, hipBLASLt, custom HIP kernels. Linked as bitcode into the compiled DLL.
- **Mock** (`lib/Runtime/mock/`): CPU stubs for GPU-free development/testing. Enabled with `BUILD_MOCK_RUNTIME=ON`.

### Generated DLL entry points

`GenerateInterface.cpp` emits three functions into each compiled model DLL:
- `inference_init` — allocates RuntimeState, loads constants
- `inference_compute` — the hot path: prepare_input → prepare_output → main_graph → finalize_output → stream_sync → free_input
- `inference_cleanup` — destroys RuntimeState

Runtime functions called from generated code must be declared in `hipdnn_ep_runtime.h` (extern "C") and listed in `getRuntimeFuncSpecs()` in `GenerateInterface.cpp`.

### Backend integration (MorphiZen EP)

`backend-mlir-compiler/` bridges this compiler to ONNX Runtime:
- **Level-1 pass** dispatches to Level-2 passes for per-op pattern matching
- **Custom Op** executes compiled DLLs at inference time with a shared HIP stream
- All ops on the same stream — no explicit synchronization needed between MIOpen/hipBLASLt/kernel calls

### Tools

| Tool | Purpose |
|------|---------|
| `hip-mlir-opt` | MLIR optimizer with all HIP passes registered |
| `hip-compiler` | End-to-end ONNX → DLL compiler CLI |
| `hip-test-dll` | Load and execute a compiled DLL |
| `hip-inspect-dll` | Inspect DLL metadata (FlatBuffers) |
| `hip-onnx-runner` | Run ONNX models via HIP EP |

## Adding a New Operator

Each operator spans three layers — follow existing patterns:

1. **ONNX → HIP conversion**: add `lib/Conversion/OnnxToHip/<Op>Conversion.cpp`, register in `OnnxToHip.cpp`
2. **HIP dialect op**: define in `include/hip/Dialect/HipOps.td` (TableGen), add C++ verification in `lib/Dialect/IR/`
3. **HIP → LLVM lowering**: add `lib/Conversion/HipToLLVM/<Op>Lowering.cpp`, register in `HipToLLVM.cpp`
4. **Runtime function**: implement in `lib/Runtime/real/<op>.cpp` and `lib/Runtime/mock/mock_gpu.cpp`
5. **Custom kernel** (if needed): add `.hip` file in `3rd-party/custom_kernels/`, declare in `hip_custom_kernels.h`
6. **LIT test**: add `.mlir` test in `test/lit/`

Zero-cost shape ops (Reshape, Squeeze, Unsqueeze) lower through standard MLIR `tensor.expand_shape`/`tensor.collapse_shape` and need no runtime support.

## Test Models

ONNX models live under `models/<model-name>/` (gitignored). Python perf tests auto-download models on first run.

**Fixed-shape requirement:** The compiler requires all tensor dimensions to be static. Models with dynamic dimensions (e.g. `batch_size`, `sequence_length`) must be converted to fixed shapes before use. The perf test fixtures handle this automatically via `fix_shapes()` in `test/python/conftest.py`.

| Model | Quant | Layers | KV Heads | Head Dim | Size | Source |
|-------|-------|--------|----------|----------|------|--------|
| Llama-3.2-1B-Instruct | q4f16 | 16 | 8 | 64 | ~1.1 GB | [onnx-community/Llama-3.2-1B-Instruct-ONNX](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX) |
| Meta-Llama-3.1-8B-Instruct | INT4 | 32 | 8 | 128 | ~5.3 GB | [onnx-community/Meta-Llama-3.1-8B-Instruct-ONNX-DirectML-GenAI-INT4](https://huggingface.co/onnx-community/Meta-Llama-3.1-8B-Instruct-ONNX-DirectML-GenAI-INT4) |

**Note:** The 8B model has `position_ids` as an additional input (not present in the 1B model).

### Running with hip-onnx-runner

```cmd
rem From the repo root — TheRock ROCm DLLs must be on PATH for compiled model DLLs to load
set PATH=%CD%\install\therock\bin;%PATH%
install\dist\bin\hip-onnx-runner.exe -m models\Llama-3.2-1B-Instruct\model_q4f16_fixed.onnx
```

**Important:** `hip-onnx-runner` fills inputs with random data by default. For models with GQA, this produces garbage `seqlens_k` values and GQA failures. Use `-i <dir>` to provide valid binary input files (see `test/python/` for how the perf tests generate correct inputs).

## Python Performance Tests

Tests in `test/python/` compare single-token decode latency across three EPs using the **same** ORT Python API:

```bash
pytest test/python/test_llama1b.py -v -s   # 1B model
pytest test/python/test_llama8b.py -v -s   # 8B model
```

**Structure:** `conftest.py` has shared utilities (`download`, `fix_shapes`, `run_timed`, `report`, `compare_outputs`, `register_morphizen_ep`, `LlamaModelConfig`, `make_llama_inputs`, `run_timed_iobinding`). Each test file owns its model config (HF URLs, filenames, `LlamaModelConfig`). Adding a new model = new test file following the same pattern.

**IOBinding for MorphiZen EP tests:** MorphiZen EP perf tests use `run_timed_iobinding` which binds the same `OrtValue` to both the past KV input and present KV output. The runtime detects matching host pointers and reuses the same GPU allocation, so `past_key_gpu == present_key_gpu`. This makes GQA skip the per-layer D2H + `hipStreamSynchronize` stall (the `past_key != present_key` condition is false).

**KV cache shape convention:** Both `past_sequence_length` and `total_sequence_length` are set to `max_seq_len` (128) in the dim map. This makes past and present KV tensors the same shape, enabling shared buffer detection in the runtime.

### MorphiZen EP from Python

The MorphiZen EP is loaded via ORT's dynamic EP registration API (not `get_available_providers()`):

```python
import onnxruntime as ort
ort.register_execution_provider_library("MorphiZenExecutionProvider", str(ep_dll))
from onnxruntime.capi._pybind_state import get_ep_devices
devices = [d for d in get_ep_devices() if d.ep_name == "MorphiZenExecutionProvider"]
so = ort.SessionOptions()
so.add_provider_for_devices(devices, {})
sess = ort.InferenceSession(model_path, sess_options=so)
```

**Both `install/dist/bin` and `install/therock/bin` must be on PATH** for the EP DLL to find `hip-compiler.dll` and ROCm runtime DLLs.

## Performance Profiling

Set `HIPDNN_EP_PERF=1` to enable two levels of profiling output (to stderr):

1. **Phase-level timing** — H2D, Compute, D2H, Sync breakdown per inference (existing)
2. **Per-operator GPU profiling** — GPU and CPU time per operator, grouped by op name with shape sub-rows, sorted by GPU time descending

```bash
HIPDNN_EP_PERF=1 pytest test/python/test_llama8b.py -v -s
```

When disabled (default), profiling is zero-overhead: `std::optional` guards prevent `hipEvent` creation.

### Architecture

- Per-op profiling uses RAII scope guards (`OpProfileScope`, `OpProfileCpuScope`) that record `hipEvent_t` start/stop pairs on the HIP stream. Events are resolved in bulk after `hipStreamSynchronize` — no per-op sync.
- Profiling state lives in `RuntimeState->op_profile` (opaque `void*`), not globals — each inference session has its own state. Two-level data model: `OpEntry` (per op name) → `ShapeEntry` (per shape string).
- Key files: `op_profile.h` (RAII scopes + macros), `op_profile.cpp` (state + printing), `debug_log.h` (env var check)
- Each operator wrapper adds one line: `OP_PROFILE("opname", shape_lambda, state)` or `OP_PROFILE_CPU("opname", state)`. The shape lambda is only invoked when profiling is active — zero `snprintf` overhead on the hot path.

### Llama 8B baseline (gfx1150, single-token decode, April 2026)

CPU time tracks host-side overhead — GPU ops should show near-zero CPU time (all async), with `stream_sync` capturing the full CPU wait. Non-zero CPU on a GPU op indicates an unexpected `hipStreamSynchronize` in its code path.

| Operator | Calls | GPU (ms) | CPU (ms) | GPU % |
|----------|------:|---------:|---------:|------:|
| matmul_nbits | 225 | 66.3 | 0.4 | 77.7% |
|   m=1,n=14336,k=4096 | 64 | 30.1 | 0.1 | 35.3% |
|   m=1,n=4096,k=14336 | 32 | 13.6 | 0.1 | 15.9% |
|   m=1,n=4096,k=4096 | 64 | 11.8 | 0.1 | 13.8% |
|   m=1,n=1024,k=4096 | 64 | 5.5 | 0.1 | 6.4% |
|   m=1,n=128256,k=4096 | 1 | 3.5 | 0.0 | 4.1% |
| skip_layernorm (1x4096) | 64 | 4.8 | 0.3 | 5.6% |
| gqa (b=1,sq=1,skv=128,h=32,d=128) | 32 | 4.3 | 0.1 | 5.0% |
| rotary_emb | 64 | 3.9 | 0.1 | 4.6% |
|   h=32,d=128 | 32 | 2.1 | 0.1 | 2.5% |
|   h=8,d=128 | 32 | 1.8 | 0.1 | 2.1% |
| elementwise (1x1x1x14336) | 64 | 2.8 | 0.2 | 3.3% |
| activation (n=14336) | 32 | 2.2 | 0.1 | 2.6% |
| stream_sync | 1 | n/a | 85.0 | n/a |
| **TOTAL** | | **85.3** | **87.6** | |

End-to-end: **~67 ms avg** (14.9 tok/s)

**GQA fused decode path:** For single-token decode (`sq==1`) with supported head dimensions (`d∈{64,128,256}`), GQA uses a fused custom HIP kernel path (rope + KV append + fused attention decode). This path is fully async — no D2H copies, no per-layer sync. The `local_window_size` attribute must be `<= 0` (ONNX uses `-1` for no windowing). The decomposed hipBLASLt path (used for prefill or unsupported configs) falls back to D2H + sync per layer.

**Shared buffer detection:** The runtime (`hipdnn_ep_runtime_tensor.cpp`) detects when an output's host pointer matches a previously-prepared input's host pointer (IOBinding with shared OrtValue for past/present KV cache). It reuses the same GPU allocation so `past_key_gpu == present_key_gpu`, which makes GQA skip the per-layer D2H `seqlens_k` copy + `hipStreamSynchronize` in the decomposed path.

**GQA path selection:** With shared buffers, `past_key == present_key` at the GPU level → the `need_host_past_len` condition (`past_key && past_key != present_key`) is false → no D2H copy, no `hipStreamSynchronize` per layer. The concat path (which requires host-side `past_len`) is only triggered when `past_key != present_key`.

**Runtime state:** Buffer pool, shared-buffer detection map, and GQA GEMM descriptor cache are all per-session (stored in `RuntimeState` as opaque pointers), not globals. This supports concurrent inference sessions.

## Crash Reporting (cpptrace)

All executables and DLLs register crash handlers via `hip::install_crash_handlers()` (`lib/Support/CrashHandler.h`). On crash, a stack trace is printed to stderr.

Three handlers are installed (guarded by `std::call_once`):
1. `cpptrace::register_terminate_handler()` — uncaught C++ exceptions
2. `std::signal(SIGABRT, ...)` — `abort()` calls
3. `AddVectoredExceptionHandler()` (Windows) — SEH exceptions (`0xC0000000` prefix, excludes invalid-handle and stack-overflow)

Always enabled — no environment variable needed. The `CrashHandler.h` header is private (lives in `lib/Support/`, not installed).

**Integration points:**
- **Compiler tools** (`BUILD_HIP_TOOLS`): link `HipSupport` library, call at top of `main()` or in `DllMain`
- **EP DLL** (`BUILD_EP`): `CrashHandler.cpp` compiled into level-1-pass static lib (whole-archived into EP DLL), called in `CreateEpFactories()`
- **hip-onnx-runner**: `CrashHandler.cpp` compiled directly into the executable

**Dependency:** cpptrace v0.8.3 fetched via FetchContent (in `CMakeLists.txt` for `BUILD_HIP_TOOLS`, in `cmake/deps.cmake` for `BUILD_EP`).

## Code Conventions

- C++ 17, formatted with clang-format 16. Python formatted with ruff.
- MIT license headers enforced by pre-commit hook (template: `LICENSES/license.txt`).
- `3rd-party/` is excluded from all linting.
- Design documents live in `docs/design/`.
- Use MorphiZen C++ wrappers (`morphizen_cxx::NodeConstRef`, etc.) for graph/node APIs — do not use raw ONNX protobuf methods.
- The ONNX-to-HIP conversion uses MLIR's generic `Operation` API to match ops by name — no onnx-mlir headers required.
- Always use `python build.py` to build — never suggest manual cmake invocations unless specifically asked.
- Use `onnxruntime-genai-directml` (not plain or CUDA variant) when installing genai tools.
- **MANDATORY:** Do not use `.claude/memory`. All persistent knowledge belongs in this file or `docs/`.
- **Comments on non-obvious code are mandatory.** When adding or fixing code whose behavior isn't self-explanatory — especially workarounds, spec quirks, or subtle correctness invariants — add a short comment explaining *why*. Examples: ONNX convention differences (`local_window_size=-1` meaning "disabled"), shared-buffer detection rationale, or why a condition uses `<=` instead of `==`. Don't comment obvious code; do comment anything a reader might question.
- When an approach fails, revert immediately and completely — no partial experimental code left in the tree. Prefer runtime-only fixes (`lib/Runtime/real/`) over cross-cutting changes spanning compiler + interface + runtime. If a multi-layer fix doesn't work after one attempt, revert and reassess.
