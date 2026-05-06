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
python build.py --build-oga    # full build + OGA fork (onnxruntime-genai)
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
- **Static CRT means separate CRT per DLL.** Model DLLs compiled with `/MT` have their own CRT instance — `std::getenv()` cannot see env vars set by the host process. Use `GetEnvironmentVariableA()` (Win32 API) instead. See `debug_log.h` for the pattern. **This applies to the EP DLL too**: `lib/Compiler/CompilerDriver.cpp` and `include/hip/debug_log.h` use the `hip_get_env()` helper (defined in `include/hip/debug_log.h`) for `THEROCK_DIST`, `HIP_CUSTOM_KERNELS_DIR`, `HIPDNN_EP_IR_DUMP_PATH`. A regression here is silent: `std::getenv("THEROCK_DIST")` returns NULL → `library_paths` empty → lld-link fails to find `amdhip64.lib` → EP falls back to CPU and tests still pass cosine=1.0 because they compare CPU-vs-CPU. Always re-verify with `HIPDNN_EP_DEBUG=1` and look for `[REAL] wrap_*` / `[custom_kernels]` lines (or `[PERF]` op-level rows when `HIPDNN_EP_PERF=1`) that prove GPU dispatches are happening.
- **cmd.exe `set X=value && next` captures trailing whitespace into the value.** When invoking `model_benchmark.exe` (or anything else that reads `THEROCK_DIST`) from a chained cmd line, **always quote the assignment**: `set "THEROCK_DIST=C:\...\install\therock" && model_benchmark.exe ...`. Without quotes, the value becomes `C:\...\install\therock ` (trailing space before `&&`), and `CompilerDriver` builds the search path as `C:\...\install\therock /lib` — lld-link fails to open `amdhip64.lib`/`MIOpen.lib` and the EP silently falls back to CPU. Symptom in the log: `Adding library path: C:\...\install\therock /lib` with a stray space. Same trap applies to `set "PATH=...;%PATH%"`. (PowerShell does not have this issue.)
- **DIA SDK junction** may be needed: prebuilt LLVM hardcodes `C:\msvsn2022` for DIA SDK path. `build.py` creates this automatically.
- **sccache + RelWithDebInfo**: CMakeLists.txt swaps `/Zi` → `/Z7` and disables incremental linking to prevent PDB file contention during parallel builds.
- **TheRock DLLs on PATH at runtime.** Compiled model DLLs link against `amdhip64_7.dll`, `MIOpen.dll`, etc. from `install/therock/bin/`. Add that directory to PATH before running `hip-onnx-runner` or any compiled model.
- **Dynamic shapes are supported for batch/sequence dimensions.** The MLIR compiler handles ONNX models with symbolic dimensions (e.g., `batch_size`, `sequence_length`). Architecture-constant dimensions (hidden_size, num_heads, vocab_size) must remain static. The `fix_shapes()` utility in `test/python/conftest.py` is still available for backward compatibility but is no longer required — unmodified ONNX models compile to a single DLL that works for any shape.
- **gfx1151 dynseqlen host-scalar SEGV — host-mapped scratch buffer design.** Bufferized `tensor.from_elements` for shape arithmetic (e.g. `Shape→Sub→Cast→seqlens_k` for GQA) emits `memref.alloc + memref.store` for tiny rank-0 / 1xi64 / small i32 scalars on the host side. Once `hip-pool-allocs` absorbs the alloc into the GPU pool, the host `memref.store` writes into real device memory on gfx1151 and SEGVs in `inference_compute`. gfx1150 silently worked because `hipMalloc` there returned UMA-mapped host-accessible memory; gfx1151's hipMalloc returns true device memory. Fix is a separate runtime-owned **host-accessible scratch buffer** distinct from the GPU pool:
  - **Pass:** `lib/Dialect/Transforms/MaterializeHostScalars.cpp` (`--hip-materialize-host-scalars`). Pipeline position: after `PromoteStridedHipOperands`, **before** `hip-pool-allocs` so candidates never enter the GPU pool. Pass def in `include/hip/Dialect/Transforms/Passes.td`.
  - **Candidate filter** (`isHostScalarCandidate`): static shape, ≤16 elements, integer or index element type, has at least one `memref.store`/`memref.load` user (the host I/O — that is the SEGV trigger). All other users must be `memref.dim`/`memref.dealloc` or **any hip-dialect op**. Float-typed allocs are excluded (almost always GPU-consumed in flight, not host-staged). `hip` users are intentionally allowed: the canonical regression is `memref.alloc<i64>; memref.store<HOST>; hip.cast<GPU>` for `seqlens_k`, and `hipHostMalloc(hipHostMallocMapped)` memory is GPU-readable on UMA so the bare-ptr ABI used by `--convert-hip-to-llvm` works whether backing is `hipMalloc`'d or `hipHostMalloc`'d. The earlier "no-hip-users" filter rejected this exact pattern → pass was a no-op → SEGV persisted.
  - **Replacement:** all candidates in a function share **one** `hip.get_host_scratch(%ctx, %total) : memref<?xi8>` (op defined in `include/hip/Dialect/IR/HipOps.td`) emitted at the entry block. Each candidate is rewritten to `memref.view %scratch[%offset]` at its original alloc site. Offsets are 64-byte aligned (matches the GPU pool alignment, gives every candidate its own cache line); element sizes derived from `getIntOrFloatBitWidth()` (index = 64 bits). Original `memref.dealloc`s are erased (scratch is runtime-owned).
  - **Op lowering:** `lib/Conversion/HipToLLVM/MemoryLowering.cpp::GetHostScratchOpLowering` lowers `hip.get_host_scratch(%ctx, %size)` to `llvm.call @hipdnn_ep_get_host_scratch_base(state, size)` and synthesizes a `memref<?xi8>` descriptor over the returned pointer.
  - **Runtime:** `hipdnn_ep_get_host_scratch_base(state, needed_size)` in `lib/Runtime/hipdnn_ep_runtime_state.cpp` (declared in `lib/Runtime/hipdnn_ep_runtime.h`, state fields `host_scratch_base`/`host_scratch_size` in `lib/Runtime/runtime_state_internal.h`). Allocates with `hipHostMalloc(&p, n, hipHostMallocMapped)` — host-writable AND GPU-readable via the device pointer mapping at the same VA on UMA. **Per-session** (lives on `RuntimeState`, not global). **Grow-on-demand, never shrinks**: when `needed_size > host_scratch_size`, syncs the stream (so any in-flight kernel reading the old buffer finishes), `hipHostFree`s the old buffer, `hipHostMalloc`s the new size, and returns the new base. Freed in `hipdnn_ep_state_cleanup` via `hipHostFree`.
  - **Memory hygiene guarantees** (these match the durable user requirement "no host memory allocations in the middle of model run; host memory must be reused from run to run and might grow when shape changes"): (1) zero per-inference host allocations on the steady-state shape — the pass picks one `total` per function at compile time, the runtime call is a single pointer return after first inference; (2) the buffer is reused across runs (lives on `RuntimeState` for the session lifetime); (3) it grows only when a shape change pushes `needed_size` above the current capacity — exactly when the GPU pool also grows. The grow path stream-syncs first, mirroring `hipdnn_ep_get_pool_base`.
  - **LIT coverage:** `test/lit/Dialect/hip-materialize-host-scalars.mlir` — rank-0 i64 redirected, 1xi64 redirected, two scalars share one scratch, float left alone, large (>16 elem) int left alone, pure-GPU (no host I/O) left alone, **host-store-then-hip-consumer redirected** (the regression pattern).
- **ORT version must match pip.** `build.py` pins `ORT_VERSION` to match the pip `onnxruntime-directml` package. The MorphiZen EP DLL checks the ORT API version at registration; a mismatch (e.g. EP built against API v25 vs pip API v24) causes segfault. When upgrading, update `ORT_VERSION` in `build.py` and `onnxruntime-directml` in pip in lockstep.
- **Runtime functions called from generated code need `extern "C"` linkage.** Declare them in `hipdnn_ep_runtime.h` (which wraps everything in `extern "C"`). Without this, Clang produces C++-mangled names in the bitcode but `GenerateInterface.cpp` emits unmangled references — causing link failures.
- **Stale compiled-model DLLs after runtime changes.** Compiled model DLLs embed the runtime bitcode from build time. The MorphiZen cache key is based on the ONNX graph hash, not the runtime version — so changing runtime `.cpp` files (or custom kernels) and rebuilding does NOT invalidate cached models. After rebuilding, delete stale DLLs manually: `rm "$TEMP"/morphizen_mlir_*` (bash) or `del %TEMP%\morphizen_mlir_*` (cmd).
- **Bitcode DEPENDS list must include all headers.** The `compile_to_bitcode` macro in `lib/Runtime/CMakeLists.txt` uses a `DEPENDS` list to track when recompilation is needed. Every header that a runtime `.cpp` file `#include`s must be listed there — otherwise, editing the header doesn't rebuild the bitcode, and compiled models silently use stale runtime code. When adding a new `#include` to a runtime `.cpp` file, always update the `DEPENDS` list in the macro.
- **OGA gives tight attention_mask, pure ORT gives padded.** OGA sets `attention_mask.shape[1]` to the actual token count (e.g., `[1,7]` for a 7-token prompt, `[1,8]` after one decode step). Pure ORT tests typically pad to `max_seq_len` (e.g., `[1,128]`). Since DimSource maps `total_sequence_length` to `attention_mask` (first input defining that dim_param), this difference causes DimSource to resolve different shapes for present KV outputs. The shape override in `marshal_output_tensors` handles this correctly — it only triggers when needed (OGA case) and is a no-op when shapes already match (padded ORT case).
- **OGA `generate_next_token()` syncs the PREVIOUS step.** The call is async for the current step but synchronizes the previous dispatch before starting new work. So the 1st call dispatches prefill (returns immediately), the 2nd call syncs prefill + dispatches decode 1 (wall time = TTFT), and calls 3+ each sync one decode step (steady-state tps). Neither `get_next_tokens()` nor `get_sequence()` provides a GPU sync point.
- **Linker byproducts not cleaned up.** `CompilerDriver::cleanupIntermediates()` removes `.ll` and `.obj` files after linking, but LLD also creates `.lib`, `.pdb`, and `.exp` byproducts alongside each `.dll`. These accumulate in `%TEMP%` (hundreds of files over time). Known issue — fix requires extending `cleanupIntermediates()` in `CompilerDriver.cpp`.
- **OGA chunked prefill: set `search.chunk_size = 1024` for dynamic-shape Llama on MorphiZenEP.** OGA reads `search.chunk_size` directly from `genai_config.json` (`config.cpp:1196-1202`) — the older claim that you must use `og.Config.overlay()` is wrong; both paths work. With chunking enabled, OGA splits prompts longer than `chunk_size` into multiple EP calls (e.g., a 2048-token prompt with chunk_size=1024 produces two `input_ids` shapes of `[1,1024]`). 8B sweep on gfx1151 (2026-05-06, `-r 5 -w 1`) found **1024 is Pareto-optimal**: at L=2048 TTFT 1973 ms (-1.4% vs 512), at L=4096 TTFT 4332 ms (-3.6% vs 512). Peak working set grows only +0.16 GB at L=4096 (1.24 → 1.40 GB) — far below the unchunked 2.87 GB baseline. At L≤1024 chunk_size=1024 is identical to chunk_size=512 (single chunk in both). Decode tok/s is invariant to chunk_size across all L. The earlier 1B sweep at -l 3072 -g 1024 -ml 4096 (2026-05-01) found 512 was the sweet spot for that smaller model where activation cost per-chunk dominated; the 8B sweep with flash_decode kernels and gfx1151 changes the balance — fewer chunks win on TTFT, and the activation cost of a 1024-token chunk is bounded. **Default to 1024.** Drop to 512 (or smaller: 256→3.43 GB+6.6% TTFT, 128→2.95 GB+34% TTFT on 1B) only when memory is the binding constraint. Going larger (no chunking) loses on both TTFT (+11% at L=2048, +19% at L=4096) and peak memory (2× at L=4096 on 8B).
- **OGA requires `MorphiZenEP` (short name) everywhere — not `MorphiZenExecutionProvider`.** The OGA dispatch table (`session_options.cpp`) maps the short name `MorphiZenEP` to `MorphiZenEPExecutionProvider::AppendExecutionProvider`, which sets `DeviceType::MorphiZenEP`. Using the long name `MorphiZenExecutionProvider` makes OGA fall through to `AppendExecutionProviderV2` with `DeviceType::CPU` — the model runs correctly but with CPU memory semantics, causing ~40% TPS degradation (no GPU memory aliasing). This applies to `genai_config.json` `provider_options`, `og.register_execution_provider_library()` calls, and `patch_genai_config_for_morphizen()`. Note: pure ORT (not OGA) uses the long name `MorphiZenExecutionProvider` for `ort.register_execution_provider_library()` — these are different APIs.
- **`model_benchmark.exe` requires `genai_config.json` with `MorphiZenEP` in `provider_options`.** The EP DLL is auto-discovered next to `onnxruntime-genai.dll` or `model_benchmark.exe` — do NOT pass `--ep_library`, which causes a double registration and a crash (`STATUS_STACK_BUFFER_OVERRUN` / `0xC0000409`) during OGA shutdown. Required env vars: `PATH` must include `install/dist/bin` and `install/therock/bin`; `THEROCK_DIST` must point to `install/therock`. No other env vars (`LIB`, `HIP_CUSTOM_KERNELS_DIR`) are needed — compile-time defaults cover them. Example: `PATH=install/dist/bin:install/therock/bin:$PATH THEROCK_DIST=install/therock model_benchmark.exe -i models/<model> -l 128 -g 128 -r 3 -w 1 -v`. **For fixed-shape pipeline directories** (e.g. `Llama-3.2-1B-Instruct-Pipeline-p128m4096`) you MUST also pass `-ml <KV_LEN>` (e.g. `-ml 4096`) to override model_benchmark's default `max_length = prompt_length + generation_length` — otherwise the KV-cache buffers OGA pre-allocates won't match the ONNX's static `total_sequence_length` and prefill bind fails with `Got invalid dimensions for input: past_key_values.0.key Got: 256 Expected: 4096`. Same root cause as the "OGA static-mask sizing" gotcha below.
- **OGA `decoder-pipeline` for fixed-shape bi-model (prefill + decode).** OGA supports a two-model pipeline via `model.type = "decoder-pipeline"`. Each sub-model has `run_on_prompt` / `run_on_token_gen` flags. Pipeline entries are written as a single object whose **keys become `model_id`** (e.g., `{"prefill": {...}, "decode": {...}}`), matching OGA's `test/test_models/pipeline-model/genai_config.json` reference. The top-level `decoder.session_options.providers` field is auto-populated post-parse from `provider_options` (`config.cpp:1571`) — do NOT add it manually (schema rejects it). Put `provider_options` (with `session.disable_cpu_ep_fallback: "1"`) at the decoder level only and let sub-models inherit; per-sub-model `session_options` is optional. KV cache OrtValues are shared across sub-models by `GeneratorState`, preserving pointer identity for `past_present_share_buffer=true`. The canonical config builder is `make_pipeline_sliding_genai_config(...)` in `test/python/conftest.py` (kwargs-driven, with `has_position_ids` switching between 1B and 8B input layouts); `ensure_pipeline_sliding_oga_files(...)` materializes the directory and `run_oga_static_kv_pipeline(...)` drives generation. See `TestLlama1BOGA::test_oga_pipeline_sliding_window_160tok` (prefill `[1,128]`, KV=4096) and `TestLlama8BOGA::test_oga_pipeline_sliding_window_600tok` (prefill `[1,512]`, KV=16384) for end-to-end usage.
- **OGA `fixed_prompt_length` is a single-Run pad-up path, not a chunker — use `sliding_window` for long prompts.** Setting `decoder.fixed_prompt_length = N` routes prompts of length `2..N-1` through `DefaultInputIDs` with right-padding to N (single `session.Run` to a model fixed at seq=N). At length exactly N, no padding occurs. **For prompts > N, the prompt is sent at native length** (`input_ids.cpp:82-84` guard `sequence_length < fixed_prompt_length`) → bind failure against the static prefill. To accept prompts longer than the static prefill input shape, use `decoder.sliding_window` instead: `{"window_size": N, "alignment": "left", "slide_inputs": true, "slide_key_value_cache": false}`. OGA's `WindowedInputIDs` (`input_ids.cpp`) splits the prompt into `ceil(prompt_len / window_size)` chunks and the dispatch loop in `decoder_only_pipeline.cpp:382-394` calls the prefill sub-model once per chunk inside a single `Generator::Run()`. `WindowedInputIDs` requires `p_device_inputs_->GetType()` ∈ {QNN, CPU}; MorphiZenEP is not in OGA's GPU-input-device list (`model.cpp:592-597`) so it falls through to CPU input device — sliding_window works. Keep `slide_key_value_cache=false` to preserve the static KV buffer + `past_present_share_buffer=true` fast path. `fixed_prompt_length` and `sliding_window` are mutually exclusive (validated at config load, `config.cpp:1560`).
- **OGA static-mask sizing: `max_length` MUST equal the model's static `total_sequence_length`.** When `ShouldUseStaticMaskHandling()` returns true (which fires for MorphiZenEP + `past_present_share_buffer=true`), `DefaultPositionInputs::InitializeStaticMask` sizes `attention_mask.shape[1]` to `params.search.max_length` — NOT to the model's static input shape. If `max_length` differs from the ONNX model's fixed `total_sequence_length`, OGA produces a mask shape that the static-shape model rejects (`Got invalid dimensions for input: attention_mask`). For fixed-shape pipeline models, set `params.set_search_options(max_length=KV_LEN, min_length=desired_token_count, do_sample=False)` — never set `max_length=len(prompt)+max_new` like the dynamic helpers `oga_generate` / `oga_generate_timed` do. Bound generation length via `min_length` instead.
- **Pipeline directory layout for fixed-shape bi-model on the EP.** The `ensure_pipeline_dir` helper (`test/python/conftest.py`) materializes a self-contained OGA model dir from the dynamic parent ONNX: it hardlinks (or copies) the `.onnx_data` weights, copies tokenizer files, and runs `fix_shapes()` twice to emit static-shape prefill+decode ONNX files. Filename convention (shared across 1B/8B via `pipeline_sliding_prefill_filename` / `pipeline_sliding_decode_filename` in conftest): `prefill_p{window_size}m{kv_len}.onnx` (depends on both window_size and kv_len) and `decode_m{kv_len}.onnx` (decode input is always `[1,1]`, so it depends only on kv_len and can be reused across pipeline configs with different window sizes). Each fresh ONNX hash → its own MorphiZen-compiled DLL in `%TEMP%`, so the bi-model dir produces two distinct compiled DLLs on first run.
- **Counting per-sub-model Compute calls.** Each pipeline sub-model gets its own `MlirCustomOp` / `InferenceState`, and `HIPDNN_EP_PERF=1` prints a `[PERF] inference #N` block per Compute call where `#N` is per-sub-model (each sub-model counts independently from its own counter). For a 160-token prompt + window_size=128 + 9 decode steps, expect 11 total `[PERF] inference` blocks (2 prefill + 9 decode). `MORPHIZEN_DEBUG_MLIR_BACKEND` controls glog `MY_LOG()` verbosity in `MlirCustomOp.cpp` and `pass_main.cpp`, but glog output is routed to log files (not stderr) via the morphizen logger adapter (`logger_adapter.cpp:29`), so `GLOG_logtostderr=1` does NOT capture these — use `HIPDNN_EP_PERF=1` for stderr-visible per-Compute counting instead.

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
- **Level-1 pass** (`pass_main.cpp`): compiles MLIR bytecode, builds metadata (including DimSource entries for dynamic output dims), fuses graph into a single custom op
- **Custom Op** (`MlirCustomOp.cpp`): executes compiled DLLs at inference time — marshals inputs/outputs, resolves dynamic shapes via DimSource, manages the compiled DLL lifecycle
- All ops on the same stream — no explicit synchronization needed between MIOpen/hipBLASLt/kernel calls

**DimSource resolution (dynamic shapes):** For dynamic output dimensions (shape == -1 in metadata), `marshal_output_tensors` resolves the actual size at runtime. Each output dim has a `DimSource{input_idx, dim_idx}` entry built by `pass_main.cpp::build_metadata_json()` from the model's `dim_params_map` (symbolic dimension names like `total_sequence_length`). The dim_param is mapped to whichever input tensor *first* defines that symbolic name. For transformer models, `total_sequence_length` maps to `attention_mask` (which appears before `past_key_values` in input order).

**OGA past_present_share_buffer shape override:** OGA binds the same OrtValue to both `past_key_values.N.key` (input) and `present.N.key` (output) for zero-copy KV cache reuse. DimSource resolves `present.N.key`'s sequence dim from `attention_mask.shape[1]`, which OGA sets to the tight token count (e.g., 7 for a 7-token prompt) rather than the pre-allocated buffer size (e.g., 128). If `ctx.GetOutput()` is called with the tight shape, ORT allocates a *new* buffer instead of returning the pre-allocated one — breaking pointer identity (`past_key != present_key`) and corrupting multi-token generation (GQA writes to the new buffer, but OGA reads from the old one next step). The fix in `marshal_output_tensors` overrides dynamic dims of `present.N.{key,value}` outputs from the corresponding `past_key_values.N.{key,value}` input's actual runtime shape *before* calling `GetOutput`. The override uses a `>` guard (not `!=`) so it only fires for shared-buffer mode (past is `max_length` > DimSource's tight count). For `past_present_share_buffer=false`, past is `prev_total` which is smaller than DimSource-resolved `curr_total` — the override is skipped and OGA's growing allocations work correctly. Only dims marked dynamic (-1) in compiled metadata are overridden — static dims (batch, num_heads, head_dim) are never touched. See `find_past_input_for_present()` helper.

**`past_present_share_buffer=false` is supported.** When OGA uses separate past/present buffers, the GQA runtime handles `past_key != present_key` via the concat path (copies past into present, then appends new tokens). Performance is lower than shared-buffer mode (D2H copy + `hipStreamSynchronize` per GQA layer for the concat path) but correctness is maintained.

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

**Dynamic shapes:** The compiler supports ONNX models with dynamic batch and sequence dimensions. Architecture-constant dimensions (hidden_size, num_heads, head_dim, vocab_size) must remain static in the model. A single compiled DLL handles any input shape at runtime. The `fix_shapes()` utility in `test/python/conftest.py` is still available for backward compatibility.

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

**Never run GPU benchmarks in parallel.** GPU contention between concurrent benchmarks produces unreliable numbers. Always run benchmark approaches (DML, ORT EP, OGA Python, model_benchmark) sequentially — one at a time. Practical rules when driving benchmarks from Claude Code: (1) **never** launch a benchmark with `run_in_background: true` — always foreground with a long enough timeout; (2) before starting any GPU benchmark, verify the GPU is idle: `tasklist | grep -i model_benchmark` (Windows) and kill stragglers with `taskkill //F //PID <pid>` if anything appears; (3) `TaskStop` on a backgrounded bash task does NOT free the GPU — it signals the bash wrapper but the child `model_benchmark.exe` keeps running on the GPU. Always kill the child process explicitly and re-verify `tasklist` is clean before launching the next benchmark; (4) **TPS/TTFT runs MUST have `HIPDNN_EP_PERF` unset** — profiling adds ~58% per-Compute overhead and saturates CPU-side per-op recording, which can invert the ranking of two configurations (see "Performance Profiling" section).

Two test files with identical coverage per model. **Prefer 1B tests when debugging or iterating on fixes** — they run significantly faster (~4 min vs ~15 min) due to the smaller model:

```bash
pytest test/python/test_llama1b.py -v -s   # 1B: DML, EP (fixed/dynamic, prefill/decode, accuracy, per-step logits), OGA
pytest test/python/test_llama8b.py -v -s   # 8B: DML, EP (fixed/dynamic, prefill/decode, accuracy, per-step logits), OGA
```

**Structure:** `conftest.py` has shared utilities — session helpers (`create_cpu_session`, `create_ep_session`, `create_dml_session`, `cleanup`), input builders (`make_prefill_inputs`, `make_decode_inputs`, `extract_kv_cache`, `get_next_token`, `compare_logits`), model download (`ensure_model`, `ensure_fixed_model`), OGA helpers (`setup_oga_ep`, `oga_generate`, `oga_generate_timed`, `patch_genai_config_for_morphizen`, `restore_genai_config`), plus low-level utilities (`download`, `fix_shapes`, `run_timed`, `report`, `compare_outputs`, `register_morphizen_ep`, `LlamaModelConfig`, `make_llama_inputs`, `run_timed_iobinding`, `run_iobinding_once`). Each test file owns its model config. Adding a new model = new test file following the same pattern.

Both test files have three test classes ordered to avoid GPU memory contention on iGPU:

| Class | Tests | What it covers |
|-------|-------|----------------|
| `TestLlama{1B,8B}DML` | `test_dml_decode` | DML latency baseline (runs before EP registration) |
| `TestLlama{1B,8B}ORT` | `test_ort_fixed_decode`, `test_ort_fixed_prefill_128`, `test_ort_dynamic_prefill_128`, `test_ort_dynamic_decode`, `test_ort_dynamic_vs_fixed`, `test_ort_per_step_logits` | MorphiZen EP accuracy vs CPU + latency (IOBinding + device memory) |
| `TestLlama{1B,8B}OGA` | `test_oga_ep_generation`, `test_oga_ep_shape_switching`, `test_oga_ep_chunked_prefill` | OGA+EP generation, shape switching, chunked prefill accuracy vs CPU |

**1B OGA config:** The 1B HuggingFace repo lacks `genai_config.json`. The test generates it programmatically with hardcoded 1B model parameters (16 layers, head_size=64, hidden_size=2048, no position_ids). Tokenizer files are downloaded from the HF repo.

**Memory management:** Sessions are explicitly deleted and `gc.collect()` called between tests to free GPU memory. DML and EP sessions must never coexist — DML tests run first in their own class. OGA tests create/destroy `og.Model` within each test.

**Accuracy vs latency:** Accuracy tests use `run_iobinding_once(use_device_memory=False)` with zero-initialized KV cache matching CPU `sess.run()`. Latency tests use `run_timed_iobinding(use_device_memory=True)` with the production `hipHostMalloc` path. OGA latency includes a warmup generation before the timed run.

**IOBinding with device memory for MorphiZen EP tests:** MorphiZen EP perf tests use `run_timed_iobinding(..., use_device_memory=True)` which allocates KV cache OrtValues via the EP's `hipHostMalloc` GPU allocator (`device_type="gpu", vendor_id=0x1002`). The runtime sees `memory_type == TENSOR_MEMORY_GPU` and aliases the buffer directly (zero-copy). The same OrtValue is bound to both past KV input and present KV output, so `past_key_gpu == present_key_gpu` — GQA skips the per-layer D2H + `hipStreamSynchronize` stall.

**KV cache shape convention:** Both `past_sequence_length` and `total_sequence_length` are set to `max_seq_len` (256) in the dim map — matching OGA's `max_length = prompt_len + generation_len = 128 + 128`. This makes past and present KV tensors the same shape, enabling the memory_type aliasing fast path in the runtime.

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

When disabled (default), profiling is zero-overhead: `hipdnn_ep_perf_enabled()` is a `static const bool` checked once, and `std::optional` guards prevent any GPU event or string operations.

**NEVER measure TPS / TTFT with `HIPDNN_EP_PERF=1` set.** The event pool adds ~58% wall-clock overhead per inference (~34 ms per Compute call on 8B) AND CPU-saturates per-op recording, which compresses the gap between fast and slow configurations and can even invert it. Concrete observed inversion (8B, L=128, May 2026): under HIPDNN_EP_PERF=1 the fixed-pipeline EP looked *faster* per Compute (37 ms) than the dynamic EP (42 ms); without profiling the truth was the opposite (FIX 39.2 ms/tok vs DYN 24.4 ms/tok), because OGA's per-step orchestration cost (decoder-pipeline + 16384 static mask) lives *between* Compute calls and is invisible to per-op profiling. Use HIPDNN_EP_PERF only for (a) per-op GPU-time breakdown, and (b) checking that the CPU column is near zero on GPU ops (non-zero CPU on a GPU op flags an unintended `hipStreamSynchronize`). For TPS/TTFT/E2E numbers, profile must be OFF.

### Architecture

- Per-op profiling uses RAII scope guards (`OpProfileScope`, `OpProfileCpuScope`) that record `hipEvent_t` start/stop pairs on the HIP stream. Events are resolved in bulk after `hipStreamSynchronize` — no per-op sync.
- **Event pool**: HIP events are pre-allocated in `OpProfileState` and reused across inferences — no `hipEventCreate/Destroy` per operator call. The pool grows on demand but never shrinks.
- Profiling state lives in `RuntimeState->op_profile` (opaque `void*`), not globals — each inference session has its own state. Two-level data model: `OpEntry` (per op name) → `ShapeEntry` (per shape string).
- Key files: `op_profile.h` (RAII scopes + macros), `op_profile.cpp` (state + event pool + printing), `debug_log.h` (env var check)
- Each operator wrapper adds one line: `OP_PROFILE("opname", shape_lambda, state)` or `OP_PROFILE_CPU("opname", state)`. The shape lambda is only invoked when profiling is active — zero `snprintf` overhead on the hot path.

### Llama 8B baseline (gfx1150, single-token decode, April 2026)

CPU time tracks host-side overhead — GPU ops should show near-zero CPU time (all async), with `stream_sync` capturing the full CPU wait. Non-zero CPU on a GPU op indicates an unexpected `hipStreamSynchronize` in its code path.

| Operator | Calls | GPU (ms) | CPU (ms) | GPU % |
|----------|------:|---------:|---------:|------:|
| matmul_nbits | 225 | 62.3 | 0.3 | 78.0% |
|   m=1,n=14336,k=4096 | 64 | 27.6 | 0.1 | 34.6% |
|   m=1,n=4096,k=14336 | 32 | 12.9 | 0.0 | 16.1% |
|   m=1,n=4096,k=4096 | 64 | 11.7 | 0.1 | 14.6% |
|   m=1,n=1024,k=4096 | 64 | 5.9 | 0.1 | 7.4% |
|   m=1,n=128256,k=4096 | 1 | 4.3 | 0.0 | 5.3% |
| skip_layernorm (1x4096) | 64 | 5.2 | 0.3 | 6.5% |
| rotary_emb | 64 | 4.2 | 0.1 | 5.3% |
|   h=32,d=128 | 32 | 2.3 | 0.0 | 2.9% |
|   h=8,d=128 | 32 | 1.9 | 0.0 | 2.4% |
| gqa (b=1,sq=1,skv=128,h=32,d=128) | 32 | 3.9 | 0.1 | 4.9% |
| activation (n=14336) | 32 | 2.2 | 0.1 | 2.7% |
| elementwise (1x1x1x14336) | 64 | 2.1 | 0.1 | 2.6% |
| **TOTAL** | | **79.9** | **0.9** | |

End-to-end: **~59 ms avg** (17.0 tok/s)

### Llama 8B at L=1024 with GQA flash_decode (gfx1150, May 2026)

Long-context decode previously suffered from the original `gqa_fused_decode` kernel scaling linearly with `skv` (each query head re-read the full K/V cache → 4× bandwidth waste at HPG=4). The new `gqa_flash_decode` kernel (`3rd-party/custom_kernels/hip/gqa_kernel.hip`) uses a GQA-aware split-K Flash Attention 2 design: one block per (batch, kv-head, K_SPLIT) loads K/V tiles into LDS once and reuses them across all HPG=4 query heads, then a small reduction kernel merges partials. Templated for `D ∈ {64, 128}` with `K_SPLITS=8`. Per-step result at L=1024:

| Operator | Calls | GPU (ms) | GPU % |
|----------|------:|---------:|------:|
| matmul_nbits | 225 | 61.0 | 76.5% |
| **gqa (b=1,sq=1,skv=1056,h=32,d=128)** | **32** | **5.0** | **6.3%** |
| skip_layernorm (1x4096) | 64 | 4.7 | 5.9% |
| rotary_emb | 64 | 4.0 | 5.0% |
| elementwise (1x1x1x14336) | 64 | 2.6 | 3.3% |
| activation (n=14336) | 32 | 2.1 | 2.6% |
| **TOTAL** | | **79.7** | |

End-to-end at L=1024: **75.7 ms/tok (13.21 tok/s)** — vs. pre-flash_decode baseline ~95 ms/tok (10.5 tok/s) → **+26% TPS**. GQA per-layer at depth 1056 dropped from ~720 µs (linear-scaling fused_decode) to **156 µs** (~4.6× faster). At L=2048 (skv≈2080) decode is 80.4 ms/tok — only 6% slower than L=1024, confirming flash_decode flattened the depth scaling. At L=128 (skv<256) the dispatcher correctly falls back to the original fused_decode (17.46 tok/s, no regression vs short-context baseline).

**Dispatch gate** (`lib/Runtime/real/gqa.cpp::gqa_flash_decode_enabled`): flash_decode runs when `sq==1`, `d ∈ {64,128}`, `H == G * 4` (HPG=4), and `skv >= HIPDNN_EP_GQA_FLASH_DECODE_MIN_SKV` (default 256). At small skv the original fused_decode wins because the new kernel's K_SPLITS=8 reduction overhead exceeds its bandwidth savings. Disable entirely with `HIPDNN_EP_GQA_FLASH_DECODE=0`. Workspace partials for the reduction step share the GQA workspace (placed after rope temps) — `hipdnn_ep_state_ensure_workspace` is called once with the combined size, since growing the workspace does NOT preserve data.

**Short-context decode (LDS-tiled fused_decode).** When `gqa_flash_decode` does not dispatch (skv < 256, HPG != 4, or d ∉ {64,128}), the path is `gqa_fused_decode` — which now uses LDS-tiled K/V prefetch. The decode block grid is tiny (B=1, H=32 → ~32 blocks / ~128 waves on RDNA3 wave32), far below what is needed to hide ~400-cycle global-load latency by wave switching. The kernel cooperatively prefetches `TILE=8` rows of K and V into LDS per outer iteration, then crunches 8 softmax steps from fast LDS — driving memory-level parallelism via TILE (16 outstanding loads per thread per tile) instead of wave count. LDS use per block: `2 * TILE * D * sizeof(_Float16)` = 8 KB at D=128 (well under RDNA3's 64 KB/CU). No `__syncthreads` between prefetch and inner loop because each thread writes/reads only its own column. TILE=8 is sweep-validated. Measured impact (5 trials + warmup, L=128): **8B short-context decode 16.99 → 17.66 tok/s (+3.9%)**; 1B (HPG=1, fused_decode all depths) 66.02 → 66.37 tok/s (within noise — at H=G=8 the kernel is already small enough that single-row latency isn't the bottleneck). TTFT unchanged on both (fused_decode is `sq==1` only).

**L=256+ TPS cliff (8B, open issue).** Decode TPS drops sharply at the flash_decode dispatch boundary: L=128 ≈ 17 tok/s → L=256 ≈ 12 tok/s (also reproduced at higher depths — measured 11.77 / 12.47 tok/s at L=256, σ ~0.5). Per-op TOTAL is essentially identical at L=128 vs L=256 (74.4 vs 74.2 ms with profiling), so the gap is launch/scheduling overhead invisible to per-op profiling. Disabling flash_decode at L=256 makes things WORSE (12.84 tok/s) — flash_decode IS the right path, just suboptimally tuned. Vulkan source analysis (`/c/local/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp:9088-9122`) shows their adaptive split_k formula is `shader_core_count * 2 / total_workgroups_no_split`, which for 8B Llama on the 890M (16 CUs, B=1, G=8, HPG=4) gives **split_k=4, not 8** — our hardcoded `kFlashDecodeKSplits = 8` over-splits by 2×. Each split processes only 32 KV positions × HPG=4 heads at skv=256 — too small to amortize launch + reduce overhead. The K_SPLITS=8 hardcode is in `3rd-party/custom_kernels/hip/gqa_kernel.hip` (launcher hard-checks `K_SPLITS != 8`, only `<D, 8>` template instantiations exist) and `lib/Runtime/real/gqa.cpp` (`kFlashDecodeKSplits = 8`). Fix would: relax the launcher check, add `<128,4>` and `<64,4>` instantiations, and either drop the constant to 4 or compute it adaptively via Vulkan's formula.

Profiling overhead (event pool): ~34 ms (+58%), from 972 `hipEventRecord` + 486 `hipEventElapsedTime` calls. This is the inherent cost of GPU timing instrumentation. With profiling on: ~93 ms avg (10.7 tok/s).

**GQA fused decode path:** For single-token decode (`sq==1`) with supported head dimensions (`d∈{64,128,256}`), GQA uses a fused custom HIP kernel path (rope + KV append + fused attention decode). This path is fully async — no D2H copies, no per-layer sync. The `local_window_size` attribute must be `<= 0` (ONNX uses `-1` for no windowing). The decomposed hipBLASLt path (used for prefill or unsupported configs) falls back to D2H + sync per layer.

**GPU memory aliasing:** When the EP's `hipHostMalloc` allocator is used (OGA or IOBinding with `device_type="gpu"`), KV cache tensors arrive with `memory_type == TENSOR_MEMORY_GPU`. The runtime aliases them directly — zero H2D/D2H. This makes `past_key_gpu == present_key_gpu`, so GQA skips the per-layer D2H `seqlens_k` copy + `hipStreamSynchronize` in the decomposed path.

**GQA path selection:** With aliased GPU buffers, `past_key == present_key` at the GPU level → the `need_host_past_len` condition (`past_key && past_key != present_key`) is false → no D2H copy, no `hipStreamSynchronize` per layer. The concat path (which requires host-side `past_len`) is only triggered when `past_key != present_key`.

**Runtime state:** Buffer pool and GQA GEMM descriptor cache are per-session (stored in `RuntimeState` as opaque pointers), not globals. This supports concurrent inference sessions.

**Pool grow-on-demand:** The GPU memory pool is initially allocated at `inference_init` with the static pool size (sum of compile-time-known buffer sizes). When dynamic shapes require additional space (e.g., batch-dependent intermediate tensors), `hipdnn_ep_get_pool_base()` grows the pool via `hipFree` + `hipMalloc` at `inference_compute` time. The pool never shrinks. Dynamic buffer offsets are computed at runtime by `PoolAllocs`-emitted arithmetic in the `main_graph` function.

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

## Vulkan baseline (llama.cpp)

`build-vulkan.py` sets up a llama.cpp + Vulkan SDK installation under `install/` for collecting Vulkan baseline numbers comparable to the local `C:\local\llama.cpp` reference setup. Pinned versions: llama.cpp commit `683c5acb`, Vulkan SDK `1.4.341.1`. Idempotent — `.ok` sentinels, shares `install/_cache/` with `build.py`.

```bash
python build-vulkan.py             # full setup + build
python build-vulkan.py --skip-build # fetch SDK + sources only
python build-vulkan.py --clean      # wipe install/vulkan-sdk + install/llama*
```

Outputs (under `install/`): `vulkan-sdk/` (silent install, user-writable, no admin), `llama.cpp/` (pinned source), `llama.cpp-build/` (Ninja build dir), `llama-vulkan/bin/` (`llama-bench.exe`, `llama-cli.exe`, `llama-server.exe`, `ggml-vulkan.dll`).

**LunarG download URL gotcha.** The installer is `vulkansdk-windows-X64-{ver}.exe` (lowercase `vulkansdk`, `X64` token), NOT `VulkanSDK-{ver}-Installer.exe` — the latter returns 404. Append `?Human=true` to bypass LunarG's download-token throttling. The CDN also rejects the default Python `urllib` User-Agent — `build-vulkan.py` installs a `Mozilla/5.0` opener for the download.

### Fair Vulkan benchmarking with llama-bench

`-p N` runs prefill at length N; `-n M` runs decode of M tokens **starting from an empty KV cache regardless of `-p`**. So `llama-bench -p 4096 -n 128` reports pp4096 and tg128 as **two independent benchmarks** — the decode number is NOT "decode after a 4096-token prompt". Decode tps is essentially flat across `-p` values (44.7 t/s at both `-p 128` and `-p 4096` on the 8B Q4_K_M).

To measure decode at realistic depth, use `-d N` / `--n-depth N` which pre-populates the KV cache with N tokens before the test runs. Multi-value works: `-n 128 -d 0,4096` reports tg128 @ d=0 and tg128 @ d=4096 in one invocation. Alternative: `-pg pp,tg` runs prefill-then-generation back-to-back as a single combined timing.

### Llama 3.1 8B Q4_K_M Vulkan baseline (gfx1151, AMD Radeon 8060S Graphics, 2026-05-05)

llama.cpp build `683c5acb`, AMD proprietary Vulkan driver, KHR_coopmat matrix cores, fp16+bf16, UMA. Model: `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4.58 GiB, 8.03 B params).

| Test | t/s |
|------|----:|
| pp128 | 903.04 ± 4.08 |
| pp4096 | 748.93 ± 0.30 |
| tg128 @ d=0 | 44.71 ± 0.02 |
| tg128 @ d=4096 | 30.87 ± 0.01 |

Decode at d=4096 is ~31% slower than at d=0 due to attention cost scaling with KV cache depth.

## Code Conventions

- C++ 17, formatted with clang-format 16. Python formatted with ruff.
- MIT license headers enforced by pre-commit hook (template: `LICENSES/license.txt`).
- `3rd-party/` is excluded from all linting.
- Design documents live in `docs/design/`.
- Use MorphiZen C++ wrappers (`morphizen_cxx::NodeConstRef`, etc.) for graph/node APIs — do not use raw ONNX protobuf methods.
- The ONNX-to-HIP conversion uses MLIR's generic `Operation` API to match ops by name — no onnx-mlir headers required.
- Always use `python build.py` to build — never suggest manual cmake invocations unless specifically asked.
- **OGA fork required for EP tests.** Use `python build.py --build-oga` to build the `onnxruntime-genai` fork with `DeviceType::MorphiZenEP`. The repo/ref are read from `.github/workflows/windows-build.yml` (single source of truth). Do not install the stock `onnxruntime-genai-directml` pip package — it lacks the MorphiZen device type.
- **MANDATORY:** Do not use `.claude/memory`. All persistent knowledge belongs in this file or `docs/`.
- **Comments on non-obvious code are mandatory.** When adding or fixing code whose behavior isn't self-explanatory — especially workarounds, spec quirks, or subtle correctness invariants — add a short comment explaining *why*. Examples: ONNX convention differences (`local_window_size=-1` meaning "disabled"), shared-buffer detection rationale, or why a condition uses `<=` instead of `==`. Don't comment obvious code; do comment anything a reader might question.
- When an approach fails, revert immediately and completely — no partial experimental code left in the tree. Prefer runtime-only fixes (`lib/Runtime/real/`) over cross-cutting changes spanning compiler + interface + runtime. If a multi-layer fix doesn't work after one attempt, revert and reassess.
