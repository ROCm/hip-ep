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
- **Static CRT means separate CRT per DLL.** Model DLLs compiled with `/MT` have their own CRT instance — `std::getenv()` cannot see env vars set by the host process. Use `GetEnvironmentVariableA()` (Win32 API) instead. See `debug_log.h` for the pattern.
- **DIA SDK junction** may be needed: prebuilt LLVM hardcodes `C:\msvsn2022` for DIA SDK path. `build.py` creates this automatically.
- **sccache + RelWithDebInfo**: CMakeLists.txt swaps `/Zi` → `/Z7` and disables incremental linking to prevent PDB file contention during parallel builds.
- **TheRock DLLs on PATH at runtime.** Compiled model DLLs link against `amdhip64_7.dll`, `MIOpen.dll`, etc. from `install/therock/bin/`. Add that directory to PATH before running `hip-onnx-runner` or any compiled model.
- **Dynamic shapes are supported for batch/sequence dimensions.** The MLIR compiler handles ONNX models with symbolic dimensions (e.g., `batch_size`, `sequence_length`). Architecture-constant dimensions (hidden_size, num_heads, vocab_size) must remain static. The `fix_shapes()` utility in `test/python/conftest.py` is still available for backward compatibility but is no longer required — unmodified ONNX models compile to a single DLL that works for any shape.
- **ORT version must match pip.** `build.py` pins `ORT_VERSION` to match the pip `onnxruntime-directml` package. The MorphiZen EP DLL checks the ORT API version at registration; a mismatch (e.g. EP built against API v25 vs pip API v24) causes segfault. When upgrading, update `ORT_VERSION` in `build.py` and `onnxruntime-directml` in pip in lockstep.
- **Runtime functions called from generated code need `extern "C"` linkage.** Declare them in `hipdnn_ep_runtime.h` (which wraps everything in `extern "C"`). Without this, Clang produces C++-mangled names in the bitcode but `GenerateInterface.cpp` emits unmangled references — causing link failures.
- **Stale compiled-model DLLs after runtime changes.** Compiled model DLLs embed the runtime bitcode from build time. The MorphiZen cache key is based on the ONNX graph hash, not the runtime version — so changing runtime `.cpp` files (or custom kernels) and rebuilding does NOT invalidate cached models. After rebuilding, delete stale DLLs manually: `rm "$TEMP"/morphizen_mlir_*` (bash) or `del %TEMP%\morphizen_mlir_*` (cmd).
- **Bitcode DEPENDS list must include all headers.** The `compile_to_bitcode` macro in `lib/Runtime/CMakeLists.txt` uses a `DEPENDS` list to track when recompilation is needed. Every header that a runtime `.cpp` file `#include`s must be listed there — otherwise, editing the header doesn't rebuild the bitcode, and compiled models silently use stale runtime code. When adding a new `#include` to a runtime `.cpp` file, always update the `DEPENDS` list in the macro.
- **OGA gives tight attention_mask, pure ORT gives padded.** OGA sets `attention_mask.shape[1]` to the actual token count (e.g., `[1,7]` for a 7-token prompt, `[1,8]` after one decode step). Pure ORT tests typically pad to `max_seq_len` (e.g., `[1,128]`). Since DimSource maps `total_sequence_length` to `attention_mask` (first input defining that dim_param), this difference causes DimSource to resolve different shapes for present KV outputs. The shape override in `marshal_output_tensors` handles this correctly — it only triggers when needed (OGA case) and is a no-op when shapes already match (padded ORT case).
- **OGA `generate_next_token()` syncs the PREVIOUS step.** The call is async for the current step but synchronizes the previous dispatch before starting new work. So the 1st call dispatches prefill (returns immediately), the 2nd call syncs prefill + dispatches decode 1 (wall time = TTFT), and calls 3+ each sync one decode step (steady-state tps). Neither `get_next_tokens()` nor `get_sequence()` provides a GPU sync point.
- **Linker byproducts not cleaned up.** `CompilerDriver::cleanupIntermediates()` removes `.ll` and `.obj` files after linking, but LLD also creates `.lib`, `.pdb`, and `.exp` byproducts alongside each `.dll`. These accumulate in `%TEMP%` (hundreds of files over time). Known issue — fix requires extending `cleanupIntermediates()` in `CompilerDriver.cpp`.
- **OGA chunked prefill requires `og.Config.overlay()`.** Setting `chunk_size` directly in `genai_config.json`'s `search` section is ignored — it must be applied at runtime via `config = og.Config(model_dir); config.overlay(json.dumps({"search": {"chunk_size": 128}})); model = og.Model(config)`. This is the same mechanism OGA's official `model-generate.py` example uses. With chunking enabled, OGA splits prompts longer than `chunk_size` into multiple EP calls (e.g., a 200-token prompt with chunk_size=128 produces calls with `input_ids` shapes `[1,128]` + `[1,72]`).
- **OGA requires `MorphiZenEP` (short name) everywhere — not `MorphiZenExecutionProvider`.** The OGA dispatch table (`session_options.cpp`) maps the short name `MorphiZenEP` to `MorphiZenEPExecutionProvider::AppendExecutionProvider`, which sets `DeviceType::MorphiZenEP`. Using the long name `MorphiZenExecutionProvider` makes OGA fall through to `AppendExecutionProviderV2` with `DeviceType::CPU` — the model runs correctly but with CPU memory semantics, causing ~40% TPS degradation (no GPU memory aliasing). This applies to `genai_config.json` `provider_options`, `og.register_execution_provider_library()` calls, and `patch_genai_config_for_morphizen()`. Note: pure ORT (not OGA) uses the long name `MorphiZenExecutionProvider` for `ort.register_execution_provider_library()` — these are different APIs.
- **`model_benchmark.exe` requires `genai_config.json` with `MorphiZenEP` in `provider_options`.** The EP DLL is auto-discovered next to `onnxruntime-genai.dll` or `model_benchmark.exe` — do NOT pass `--ep_library`, which causes a double registration and a crash (`STATUS_STACK_BUFFER_OVERRUN` / `0xC0000409`) during OGA shutdown. Required env vars: `PATH` must include `install/dist/bin` and `install/therock/bin`; `THEROCK_DIST` must point to `install/therock`. No other env vars (`LIB`, `HIP_CUSTOM_KERNELS_DIR`) are needed — compile-time defaults cover them. Example: `PATH=install/dist/bin:install/therock/bin:$PATH THEROCK_DIST=install/therock model_benchmark.exe -i models/<model> -l 128 -g 128 -r 3 -w 1 -v`.
- **MORPHIZEN_DEBUG_MLIR_BACKEND shape logging.** Set `MORPHIZEN_DEBUG_MLIR_BACKEND=3` (env var, must be set before process starts) to see per-tensor input shapes on stderr from `MlirCustomOp::marshal_input_tensors`. Uses `fprintf(stderr, ...)` — not glog — because the EP DLL's static CRT (`/MT`) has its own stderr FILE* that glog's `LOG(INFO)` routes to files rather than fd 2. The `ENV_PARAM()` value is cached at DLL load time via static initialization.

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

**Never run GPU benchmarks in parallel.** GPU contention between concurrent benchmarks produces unreliable numbers. Always run benchmark approaches (DML, ORT EP, OGA Python, model_benchmark) sequentially — one at a time.

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
