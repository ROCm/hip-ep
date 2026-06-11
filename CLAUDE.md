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

# Numeric tests (per-op correctness vs ORT CPU; in-tree). Windows cmd shown;
# bash equivalent is in test/numeric/README.md "Example: MorphiZen EP".
# (set THEROCK_DIST + PATH first; see same section for the env setup.)
pytest test/numeric --backend ort_ep ^
       --ep-name   MorphiZenExecutionProvider ^
       --ep-dll    install\dist\bin\onnxruntime_morphizen_ep.dll ^
       --ep-option config_file=install\dist\bin\morphizen_config.json -s
pytest test/numeric -v -s --no-cache               # manual (skip the disk cache)
```

The in-tree `test/numeric/` suite replaces the external `onnx-numeric-tests` reference for everyday correctness checks. Each test builds a single-op ONNX model, runs it on the MorphiZen EP, and compares against an ORT CPU reference. Expensive CPU references (e.g. Llama 4096x4096 / 4096x14336 MatMuls) are cached on disk keyed by sanitised pytest node id — `manifest.json` stores a sha256 of `(model_bytes + inputs)` as a drift tripwire so any edit to seeds / shapes / scales auto-invalidates the entry. Flags: `--no-cache` (always CPU), `--refresh-cache` (rebuild then cache), `--keep-artifacts`. See [test/numeric/README.md](test/numeric/README.md) for the full backend / "bring your own ONNX" / new-op recipes, including a copy-pasteable "Example: MorphiZen EP" recipe.

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

## Remote development

This project's authoritative build + GPU test loop runs on a **gfx1151 Windows host** driven over SSH. Local-machine LIT/pytest passes on a non-gfx1151 box are not authoritative — the dynamic-sequence-length runtime path SEGVs only on gfx1151 (other arches' `hipMalloc` accidentally returns UMA-mapped host-accessible memory and silently masks the bug — see "gfx1151 dynseqlen host-scalar SEGV" gotcha below for the fix detail).

See **[docs/remote-dev-workflow.md](docs/remote-dev-workflow.md)** for: SSH+`cmd.exe` quoting/chaining gotchas, the SMB-share + SSH split (edit through the share, run through SSH), the cygwin `git` contamination warning, conda env Python disambiguation, and the OGA build-defense reference table (what each `build_oga()` flag protects against).

## Critical Build Gotchas

- **`HIPDNN_EP_STRICT=1` aborts on MLIR pass failure** (opt-in, applies across all build configs including Release) so the cpptrace SIGABRT handler prints a backtrace pinpointing the failing pass — instead of returning `false` to ORT and silently falling back to CPU (which masks the bug behind a "passing" CPU-vs-CPU accuracy test). Default behaviour in every config is soft-fail + ORT's CPU fallback, so multi-session pipelines that register MorphiZenEP only for `HipDataTransferImpl` visibility (e.g. OGA's gemma3 embedding/vision sub-sessions where MorphiZen does no compute and the graph claim is expected to fail) work without crashing. **Turn `HIPDNN_EP_STRICT=1` on whenever validating that a model is fully offloaded** — any graph MorphiZen claims but cannot compile is then a hard crash. Symptom when set: process aborts with `[CompilerDriver] aborting on pass failure (HIPDNN_EP_STRICT=1).` followed by a stack trace through `runMLIRPasses → compileImpl → compile → hip_compile_with_fs → MlirCompiler::compileFromBytecode → ... → MorphiZenEP::GetCapability`. CI / accuracy tests on models that should be fully offloaded should set it so silent CPU fallback can't pass cosine=1.0 against a CPU baseline.
- **`HIPDNN_EP_IR_DUMP_PATH` writes one file PER compile invocation** (counter suffix `.0`, `.1`, …). ORT typically issues 3 compiles per dynamic-shape session (shape-determining sub-graph + prefill specialization + decode specialization); a single sink file gets overwritten on each call and masks divergence between invocations. Set `HIPDNN_EP_IR_DUMP_SINGLE=1` to opt back into legacy single-file behaviour. Implementation: `lib/Compiler/CompilerDriver.cpp` static atomic counter appended to the path.
- **FastGelu fusion (`lib/Conversion/OnnxToHip/FastGeluFusion.cpp`) MUST peek through `onnx.CastLike`, not just `onnx.Cast`.** Some ORT export paths (notably the inlined `Gelu(approximate="tanh")` chain in Gemma-3-4B-IT) wrap every literal — `sqrt(2/π)`, `0.044715`, `0.5`, `1.0`, exponent `3` — in `CastLike(<f32 const>, <f16 activation>)` so the constant is promoted to the surrounding tensor's dtype at runtime. `onnx.Cast` and `onnx.CastLike` are value-preserving for fp→fp downconversion (modulo rounding absorbed by `kConstantTolerance`). `getScalarFloatConstant` peeks through both. Symptom of regression: `[FastGeluFusion] @main_graph: fused 0 of 34 Tanh chains; failure step histogram: 34x tanh.in.mul_sqrt2pi :: onnx.Mul` printed to stderr, followed by `error: op was not bufferized` from OneShotBufferizePass (the inlined Pow/Sum/Tanh primitives have no MorphiZen converters).
- **Same-rank dynamic Reshape (Gemma-3 q_norm/k_norm).** `ReshapeConversion.cpp` decomposes `tensor<?x?xH*D> ↔ tensor<?x?xD>` (same rank, both with at least one dynamic dim, last dim differs by an integer factor K) into `tensor.expand_shape + tensor.collapse_shape` — pure descriptor edits, no kernel needed. Split direction (in.static > out.static): expand input static dim into `(K, smaller)`, then collapse adjacent `(dyn, K)` into one. Combine direction: expand input dyn into `(dyn/K, K)` via an `arith.divui` in the expand `output_shape`, then collapse `(K, smaller)` into the larger static dim. Gemma-3 has 136 such Reshapes per dynamic-shape decoder graph (34 layers × 4: q_norm Reshape_1/2 + k_norm Reshape_1/2). Requires THREE collaborating fixes — DO NOT regress any one without considering the others:
  1. The decomposition itself in `lib/Conversion/OnnxToHip/ReshapeConversion.cpp` (same-rank dynamic case).
  2. `PoolAllocs.cpp::foldDimOfReshape` runs at the start of `PoolAllocsPass::runOnOperation` and recursively folds `memref.dim(memref.collapse_shape | memref.expand_shape | memref.alloc, i)` into pure arith on `memref.dim(<chain-root>, i)`. Without this, the BFS hoist worklist can't ascend through reshape descriptor ops (they're not in `isHoistable`) and PoolAllocs leaves dim ops at the top referencing a reshape op in the body — SSA dominance error.
  3. `PoolAllocs.cpp` Phase-4 hoist uses DFS post-order topological sort (not the original reverse-BFS-iteration), because shared producers reachable from multiple consumers via different paths can land in the wrong relative position with a naive iteration. Each `moveBefore(firstPooledAlloc)` pushes earlier moves up by one slot, so move order must visit operands BEFORE uses (= forward iteration of post-order = uses-first move order → final layout has operands at the top dominating uses).
  Also: `PoolAllocs::isHoistable` whitelist now includes `arith::DivUIOp` and `arith::DivSIOp` for the combine-direction `dyn/K` computation. These are pure index/integer arithmetic, safe to hoist.
- **Every surviving `hip.alloc` MUST be pooled — `hip_device_malloc`/`hip_device_free` have NO runtime definition.** `MemoryLowering.cpp` (`AllocOpLowering`, `MemRefAllocOpLowering`) lowers any leftover alloc to `hip_device_malloc`/`hip_device_free` (`kHipMalloc`/`kHipFree`), but those symbols are never implemented anywhere — the pipeline's contract is that every transient is either (a) absorbed into the GPU pool by `PoolAllocs` (→ `memref.view` over `hip.get_pool`) or (b) written through to an out-param by `BufferResultsToOutParams`. A surviving raw alloc is therefore a silent lld-link `undefined symbol: hip_device_malloc` at model-compile time (EP then can't claim the node → ORT "nodes assigned to CPU EP, fallback disabled" failure). Consequence: `PoolAllocs` must pool even a **single** alloc — it bails only when `allInfos.empty()` (was `< 2`). The canonical single-alloc trigger is a one-op graph whose op needs a destination temp it can't write through, e.g. a lone rank-3 `Conv` (the `collapse_shape` before the return blocks write-through; see conv gotcha below). The synthetic single-op numeric tests (`test/numeric/tests/*.py`) are exactly this shape, so any op that introduces a non-write-through temp will surface here first.
- **Rank-3 (1D) `onnx.Conv` reuses the shared 2D `hip.conv` via a unit-H reshape — there is NO dedicated `hip.conv1d` op/kernel.** `ConvConversion.cpp` rewrites a rank-3 Conv to: `tensor.expand_shape` input `[N,Cin,L]→[N,Cin,1,L]` and weights `[Cout,Cin,K]→[Cout,Cin,1,K]`, build the destination `init` at the rank-3 result shape then `expand_shape` it to `[N,Cout,1,L']`, run `hip.conv` (2D attrs: `kernel_shape=[1,K]`, `strides=[1,s]`, `pads=[0,b,0,e]`, `dilations=[1,1]`, `group=1`), then `collapse_shape` the result back to rank-3. **Why expand the `init` instead of just collapsing the conv result:** `collapse_shape(expand_shape(init))` folds to `init`, so the returned value aliases the destination buffer and bufferization write-throughs it to the out-param exactly like the native rank-4 path — leaving no transient alloc. Collapsing a freshly-`tensor.empty`'d rank-4 result instead would break write-through and leave a lone temp (→ the `hip_device_malloc` failure above). The runtime side is the shared `wrap_miopenConvolutionForward` (`lib/Runtime/real/miopen.cpp`), now dtype-parameterized (`element_size_bytes`: 2=fp16/`miopenHalf`, 4=fp32/`miopenFloat`), with per-channel bias via `miopenOpTensor`/`miopenTensorOpAdd` (NOT `miopenConvolutionForwardBias`, which is fixed at alpha=1/beta=0 and would compute `y=bias`), and workspace drawn from the per-`RuntimeState` `conv_scratch` pool (no per-call `hipMalloc`/`hipFree`). GPU-dispatch tripwire string is `[REAL] wrap_miopenConvolutionForward` (RUNTIME_DEBUG_LOG, gated on `HIPDNN_EP_DEBUG=1`). Guards: dilation≠1 and group≠1 on rank-3 bail to CPU (the H=1 reinterpretation doesn't honor them).
- **SHARED_CONSTANTS cache key MUST include a content fingerprint, not just `total_size`** (`lib/Runtime/hipdnn_ep_runtime_state.cpp::compute_constants_fingerprint`). The named-shared-memory key `Local\hipdnn_const_{pid}_{size}` is publisher/consumer-shared across all model.dlls in the process; OGA's prefill+decode sub-models intentionally exploit this to skip the 2 GB hipMalloc+upload on the second model. Risk: two distinct compiled DLLs of the SAME ONNX (e.g. the fixed-shape and dynamic-shape variants of one model, where the MLIR compiler folds shape-dependent constants differently) frequently produce constants blobs of the **same total_size** but **different content**. With a size-only key the second-to-load model silently attaches the first one's blob → reads wrong weights → wrong K/V at the first layer whose folded constants diverged → cosine ~0.7 cascading into NaN at downstream layers. The fingerprint is a 64-bit FNV-1a over per-constant descriptor metadata only (offset + size + source_type + 8-byte source-specific scalar: splat elem_bytes / file_offset / sidecar_offset), packed into a fixed 32-byte buffer per constant. **It does NOT read the multi-GB blob** — cost is O(num_constants) descriptor scans, ~22 KB total for a gemma3 (687 constants). The per-constant FileRefSource path is intentionally skipped because every FileRefSource in a single model points to `meta->constants_filename` (one external-data file per model), which is hashed once at the top; if a future EP design ever produces per-constant FileRefSource paths that differ from constants_filename, this fingerprint would alias models that share blob layout but read from different files — revisit then. Symptom of a regression (cache key reverts to size-only): pytest pair `test_gemma3_4b.py::TestGemma3_4BORT::test_ort_fixed_decode test_gemma3_4b.py::TestGemma3_4BORT::test_ort_dynamic_prefill_128` fails with `present.20.key cosine=0.728621 max_abs=26.35` (the canonical signature; verified 2026-05-19 by reverting the fingerprint and re-running).
- **`present_key`/`present_value` tail beyond `past_len + sq` is semantically undefined; tests MUST pass `valid_seq` to `compare_outputs`** (`test/python/conftest.py::compare_outputs`). ONNX `GroupQueryAttention` with `past_present_share_buffer=true` pre-allocates `present_key`/`present_value` to max_seq_len but only the first `past_len + sq` positions are spec-defined output. The remainder is undefined: ORT CPU returns a zero-initialized OrtValue (so its tail is zero by Python allocator coincidence), but MorphiZenEP's separate-buffer concat path (`kv_cache_concat_kernel` in `3rd-party/custom_kernels/hip/gqa_kernel.hip`) intentionally only writes the valid `[0, past_len+sq)` region and leaves the tail untouched — and the present buffer comes from the EP's `g_gpu_buffer_pool` (`lib/Runtime/hipdnn_ep_runtime_tensor.cpp`) which **does NOT zero on alloc or release**, so the tail leaks bytes from the prior pool consumer (most often the prior call's prefill `present_key`). A full-buffer comparator then sees a layer-specific divergence (whichever layer's pool slot was recycled — usually layer-0 K by allocation order) with signature `cosine=0.129 max_abs=23.58` even though the EP is correct per spec. **Fix is comparator-side**: every BaseORTTests call to `compare_outputs` for KV-cache outputs passes `valid_seq=past_len+sq` (decode: `position+1`; prefill: `seq_len`); the comparator slices `present.*.{key,value}` and `past_key_values.*.{key,value}` along axis 2 to `[0, valid_seq)` before computing cosine/L2. **Do NOT "fix" this by zeroing the kernel tail** — that pays µs/decode/layer of pointless GPU writes to bytes the model never reads back, and couples EP to ORT CPU's "happens to be zero" implementation detail. Append path (in-place, `past_key == present_key`, OGA hot path) is also untouched: in OGA the buffer is self-consistent across decode steps (past_len grows monotonically; unused tail was previously zeroed at initialization and never touched). Symptom of regression: pytest pair `test_ort_dynamic_prefill_128 test_ort_dynamic_decode` on **any** model with `reuse_ep_session=True` produces the `present.0.key cosine=0.129 max_abs=23.58` signature above — almost certainly because a new test was added without `valid_seq=` or `compare_outputs`'s slicing logic was regressed.
- **Conda env is a prerequisite.** `build.py` expects tools (cmake, ninja, sccache, lit) from the `hipdnn-ep` conda env to be on PATH.
- **HIP_ARCHITECTURES must match the GPU.** Omitting or mismatching causes silent build success but runtime crashes (`0xC0000005` in `hipLaunchKernel`). `build.py` auto-detects via `amdgpu-arch.exe`.
- **CRT must be Release /MT.** Pre-built LLVM/MLIR use static Release CRT. Debug (`/MTd`) or dynamic (`/MD`) produces linker errors.
- **Static CRT means separate CRT per DLL.** Model DLLs compiled with `/MT` have their own CRT instance — `std::getenv()` cannot see env vars set by the host process. Use `GetEnvironmentVariableA()` (Win32 API) instead. See `debug_log.h` for the pattern. **This applies to the EP DLL too**: `lib/Compiler/CompilerDriver.cpp` and `include/hip/debug_log.h` use the `hip_get_env()` helper (defined in `include/hip/debug_log.h`) for `THEROCK_DIST`, `HIP_CUSTOM_KERNELS_DIR`, `HIPDNN_EP_IR_DUMP_PATH`. A regression here is silent: `std::getenv("THEROCK_DIST")` returns NULL → `library_paths` empty → lld-link fails to find `amdhip64.lib` → EP falls back to CPU and tests still pass cosine=1.0 because they compare CPU-vs-CPU. Always re-verify with `HIPDNN_EP_DEBUG=1` and look for `[REAL] wrap_*` / `[custom_kernels]` lines (or `[PERF]` op-level rows when `HIPDNN_EP_PERF=1`) that prove GPU dispatches are happening.
- **cmd.exe `set X=value && next` captures trailing whitespace into the value.** When invoking `model_benchmark.exe` (or anything else that reads `THEROCK_DIST`) from a chained cmd line, **always quote the assignment**: `set "THEROCK_DIST=C:\...\install\therock" && model_benchmark.exe ...`. Without quotes, the value becomes `C:\...\install\therock ` (trailing space before `&&`), and `CompilerDriver` builds the search path as `C:\...\install\therock /lib` — lld-link fails to open `amdhip64.lib`/`MIOpen.lib` and the EP silently falls back to CPU. Symptom in the log: `Adding library path: C:\...\install\therock /lib` with a stray space. Same trap applies to `set "PATH=...;%PATH%"`. (PowerShell does not have this issue.)
- **DIA SDK junction** may be needed: prebuilt LLVM hardcodes `C:\msvsn2022` for DIA SDK path. `build.py` creates this automatically.
- **sccache + RelWithDebInfo**: CMakeLists.txt swaps `/Zi` → `/Z7` and disables incremental linking to prevent PDB file contention during parallel builds.
- **`HipDialect.cpp` needs `/bigobj` (MSVC).** `lib/Dialect/IR/HipDialect.cpp` includes `GET_OP_CLASSES` (every TableGen-generated op class — builders, parsers, printers, verifiers, accessors — for the whole HIP dialect in one translation unit). As the dialect grows, this single TU eventually exceeds the MSVC COFF section limit and fails with `fatal error C1128: number of sections exceeded object file format limit`. Fix already applied: `lib/Dialect/IR/CMakeLists.txt` adds `target_compile_options(HipDialectIR PRIVATE /bigobj)` under `if(MSVC)`. Symptom of regression (e.g. someone removes the flag while adding an op): C1128 on `HipDialect.cpp` only — other TUs are fine. Adding a new `Hip_*Op` to `HipOps.td` is the usual trigger.
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
- **WMMA matmul_nbits swizzle must cover every (bx,by) tile exactly once.** `GemmFp16U4Impl` in `matmul_nbits_kernel.hip` maps `block_id → (bx,by)` with a column-swizzle for L2 locality. The mapping MUST be a bijection over `[0, n_tiles*m_tiles)`. The original implementation fell back to row-major only when `bx >= n_tiles`, which produces collisions AND missing tiles whenever `n_tiles % sw != 0` (e.g. n_tiles=45, sw=8: tiles `(bx=43..44, by=1)` uncomputed; `(bx=40..41, by=1)` double-computed). Uncomputed tiles inherit stale values from prior kernel launches; `matmul_nbits_add_bias_rowmajor` then RMW-adds bias to those stale values, producing run-to-run non-determinism that cascades through QMoE (different fc1/fc2 outputs → different routing M counts → autotune re-triggers every prefill chunk → 11× TTFT regression on GPT-OSS-120B/20B). Fix: split the grid into a "main" region of `(n_tiles/sw)*sw` cols (full swizzle) and a "tail" of `n_tiles%sw` cols (row-major). When modifying tile-mapping logic, verify bijection exhaustively.
- **`convert-onnx-to-hip` must DCE unregistered `onnx.*` ops with no uses** (`lib/Conversion/OnnxToHip/OnnxToHip.cpp`, post-conversion cleanup loop in `runOnOperation`). Some HF ONNX exports ship dead shape-arithmetic subgraphs that produce a value consumed only by `onnx.Reshape` — e.g. Phi-4's `pos_ids_reformat/{Shape,Gather,Unsqueeze,Concat}` chain feeding a `Reshape(position_ids, computed_shape)`. The Reshape converter (`ReshapeConversion.cpp`) lowers to `tensor.expand_shape` using static type info from input/output ranks, so the computed-shape operand is never read — leaving e.g. `onnx.Concat` with no uses but still in the IR. Bufferize then trips on the unregistered op (no bufferization interface) and aborts the entire pipeline with the terse `error: op was not bufferized`. Same silent-CPU-fallback class as the affine.apply gotcha — accuracy tests pass cosine=1.0 because they compare CPU-vs-CPU. Fix: in the post-conversion cleanup loop, walk the module and erase any `onnx.*` op with `use_empty()`. Conservative — only touches dead ops, leaves live unconverted ops where they were so the bufferize error still surfaces if the model genuinely uses an unsupported op. Verify with `HIPDNN_EP_DEBUG=1` and look for `[REAL] wrap_*` lines (and KV cosine NOT exactly 1.0 — fp16 GPU should give 0.99999x).
- **`affine.apply` from `expand-strided-metadata` MUST be lowered before `convert-hip-to-llvm`** (`lib/Dialect/Transforms/Pipelines.cpp::buildHipToLLVMPipeline`). `memref::createExpandStridedMetadataPass` rewrites `memref.collapse_shape` / `memref.expand_shape` into `memref.reinterpret_cast` plus stride/offset arithmetic — and for collapse-shape stride products on multi-dim memrefs (canonical trigger: an MoE expert-major 3D buffer flattened to 2D for hipBLASLt) it emits `affine.apply` with a map like `()[s0, s1] -> (s0 * s1)`. `ConvertHipToLLVM` does NOT include affine→arith conversion patterns. Any surviving `affine.apply` leaves a `builtin.unrealized_conversion_cast` in the final LLVM IR and the translator aborts with `LLVM Translation failed for operation: builtin.unrealized_conversion_cast`. Fix: insert `createLowerAffinePass()` between `ExpandStridedMetadata` and `ConvertHipToLLVM` (and add `MLIRAffineToStandard` to `lib/Dialect/Transforms/CMakeLists.txt`'s `target_link_libraries`). **This is silent in production** — when compilation fails, MorphiZen falls back to CPU and any test that compares CPU-vs-CPU (e.g. accuracy tests using `compare_logits` against a CPU baseline) will pass `cosine == 1.0` and look healthy. Symptom is wall-clock: gpt-oss-20b decode at sq=1/kv=128 went from ~70 s (CPU fallback) → ~2 s (GPU) once the pass was added. Verify GPU dispatch with `HIPDNN_EP_DEBUG=1` (look for `[REAL] wrap_*` lines) BEFORE trusting accuracy numbers. Models that don't trigger collapse-shape on multi-dim memrefs (Llama 1B/8B as of 2026-05) will not exercise this path — expect new MoE / multi-expert / per-head-flatten models to surface this if the pass regresses.
- **Asym MatMulNBits zero_points unpack cache assumes stable pointers across inferences** (`lib/Runtime/real/matmul_nbits.cpp` + `lib/Runtime/real/zp_unpack_cache.h`). For `bits=4, zp_elem_size=1` (asym AWQ), `wrap_matmul_nbits` and `wrap_qmoe` look up the unpacked uint8 buffer (and, when WMMA/col-major-GEMV-M>1 dispatches, the converted fp16 buffer) from a per-`RuntimeState` `unordered_map<const void*, ...>` keyed on the input `zero_points` GPU pointer. Cache miss → `hipMalloc` + one launch of `hip_matmul_nbits_unpack_zp_u8` / `hip_matmul_nbits_convert_zp_fp16`; hit → no kernel launch. **Invariant**: the `zero_points` pointer must be stable across inferences — true today because it points into the `model.dll` constants blob (lifetime = session). If a future codepath ever passes a transient zp pointer (e.g. computed on-the-fly per inference), it MUST bypass this cache (or cache by a content hash, not pointer) — otherwise stale unpacked data is returned. The cache lives on `RuntimeState->zp_unpack_cache` and is freed in `hipdnn_ep_state_cleanup` via `hipdnn_ep_zp_unpack_cache_destroy`.
- **MatMulNBits GEMV autotune key MUST include `has_zp`** (`3rd-party/custom_kernels/hip/matmul_nbits_kernel.hip` `gemvCacheKey`). Sym (no zp) and asym (with zp) GEMV kernels follow different code paths inside the same generic launcher and have different optimal block dimensions; keying the autotune cache on `M_N_K_bs_col_major` only would silently apply sym-tuned configs to asym (or vice versa) when a single process loads both variants. The WMMA-path autotune already includes `has_zp` — the GEMV path now matches that contract. When adding any future kernel-shape or branch-affecting input (e.g. dtype variant), append it to the key.
- **wrap_qmoe scratch + pinned host staging — pageable hipMemcpyAsync silently sync-stages on Windows.** `lib/Runtime/real/qmoe.cpp` previously did per-call `hipMalloc`/`hipFree` of 8 transient device buffers AND used `std::vector` heap-allocated host buffers as the source/dest for `hipMemcpyAsync` D2H/H2D of the routing decision and per-expert (id, weight) staging. Two independent fixes (must apply together to see the win):
  - **Device side:** route the transient device buffers through a per-`RuntimeState` `qmoe_scratch` (`runtime_state_internal.h`). Layout is one contiguous buffer with 64-byte aligned sub-buffers; total size grows on demand via `hipdnn_ep_state_ensure_qmoe_scratch` (sync-then-realloc, never shrinks); freed in state cleanup. This eliminates 8 `hipMalloc` + 8 `hipFree` per call (192 of each per token on 24-layer gpt-oss-20b).
  - **Host side (the perf-critical fix):** route the D2H readback (`h_indices`, `h_weights`) and per-expert H2D staging through a per-`RuntimeState` pinned host buffer allocated with `hipHostMalloc(hipHostMallocDefault)` (`qmoe_host_scratch`). Pageable memory passed to `hipMemcpyAsync` on Windows silently falls back to a synchronous staging copy, blocking the GPU pipeline; pinned memory enables true async DMA that overlaps with prior in-flight kernels.
  - **Memory hygiene:** zero per-call `hipMalloc`, `hipFree`, `hipHostMalloc`, or `hipHostFree` at steady state. Both buffers are reused across runs and only grow when a larger `num_tokens` is seen.
  - **Bitcode rebuild:** when editing this code, also run `del %TEMP%\morphizen_mlir_*` after rebuild (cached compiled-model DLLs embed the runtime bitcode — the cache key is the ONNX hash, not the runtime version).
- **wrap_qmoe Phase 2: GPU-side bucketing eliminates per-expert host build + per-expert H2D round-trips.** Builds on the scratch + pinned-host fix above. Old flow per MoE layer: D2H of `expert_indices` + `expert_weights` → `hipStreamSynchronize` → host loop bucketing tokens by expert → for each active expert: build `h_ids`/`h_wts_e` on host → 2× `hipMemcpyAsync` H2D → launch gather/matmul/scatter chain. New flow: `hip_qmoe_bucket_tokens` kernel on the device (one block × `num_experts` threads, three passes — atomicAdd count, exclusive prefix sum into `expert_offsets`, scatter `(token_id, weight)` pairs into `sorted_token_ids`/`sorted_weights` ordered by expert) → D2H of just `num_experts` int32 counts → `hipStreamSynchronize` → host computes `h_offsets` from prefix sum → for each active expert: pointer arithmetic into the on-device sorted buffers handed straight to gather/scatter_add — no per-expert H2D. Sync still happens once per layer, but the D2H volume drops from `num_tokens·k·(4 + elem_size)` bytes to `num_experts·4` bytes (~30× smaller at L=128 decode), and 2× per-expert H2D launches per layer × 4 active experts = 8 launches × 24 layers = 192 launches/token are eliminated. Kernel: `3rd-party/custom_kernels/hip/qmoe_kernel.hip::bucket_tokens_kernel` (fp16-only, num_experts ≤ 1024, block_dim = `((num_experts+31)/32)*32`).
- **wrap_qmoe Phase 3: fully fused decode kernel (num_tokens == 1).** For decode the multi-pass path is replaced by **one early-return branch** in `wrap_qmoe` that calls a new `hip_qmoe_decode_fused` launcher right after `hip_qmoe_topk_routing`. The fused launcher issues exactly **three back-to-back kernel launches** with **zero `hipStreamSynchronize` and zero D2H per layer**:
  - `qmoe_decode_fc1_kernel<BS=64, TN=8>`: grid `(2*inter/TN, k)`. Each block reads `expert_indices[slot]` to pick its expert, dequants a `[TN, hidden]` tile of `fc1_weights` per-expert (factored AWQ INT4 dequant `(dot - a_sum*zp) * scale` in fp32, vectorized uint4 K-tile loads identical to matmul_nbits_gemv), warp-shuffle reduces, **fuses pair-wise SwiGLU** (`G * sigmoid(α·G) * (L + β)` with `swiglu_limit` clamp) into the write phase, and stores `[k, inter]` activations to `act_out`. BS=64 chosen so K-tile count (`hidden/32 = 90`) gives every thread 1-2 iterations (BS=256 left ~65% of threads idle).
  - `qmoe_decode_fc2_kernel<BS=64, TN=8>`: grid `(hidden/TN, k)`. Same dequant pattern over `fc2_weights`; weights the per-expert output by `expert_weights[slot]` and writes `[k, hidden]` partials to `slot_buf`.
  - `qmoe_decode_reduce_kernel`: grid `(hidden/256)`. Sums the k partials of `slot_buf` element-wise into `output[hidden]`.
  - **Per-layer cost drops from ~22 launches + 1 sync + 1 D2H to 4 launches (incl. topk) + 0 sync + 0 D2H.** Per-token (24 layers): 528 → 96 launches, 24 → 0 syncs.
  - **Memory hygiene:** zero per-call host or device allocations. Reuses existing `qmoe_scratch` sub-buffers with bumped sizing (`sz_act_buf`/`sz_fc2_buf` use `max(num_tokens, k)` so the k=4 slots fit when num_tokens=1). Multi-pass path (num_tokens > 1, prefill) is unchanged and still used.
  - **Files:** kernel launchers + the three new device kernels in `3rd-party/custom_kernels/hip/qmoe_kernel.hip` (search for `qmoe_decode_fc1_kernel` / `qmoe_decode_fc2_kernel` / `qmoe_decode_reduce_kernel` / `hip_qmoe_decode_fused`). Decl in `3rd-party/custom_kernels/include/hip_custom_kernels.h`. Dispatch in `lib/Runtime/real/qmoe.cpp::wrap_qmoe` — early-return branch immediately after `hip_qmoe_topk_routing` gated on `num_tokens == 1`.
- **OGA per-token IoBinding rebind cost scales with `max_length` (KV buffer size), not actual sequence length.** OGA's `State::Run` (`install/oga-source/src/models/model.cpp`) caches an `OrtIoBinding`, then **every call** does `ClearBoundInputs() + ClearBoundOutputs() + BindInput(name, ortvalue) × N + BindOutput(name, ortvalue) × N` for *all* names. In `past_present_share_buffer=true` mode the 48 KV `OrtValue*` (24 layers × 2 × past as input + present as output, both pointing to the same OrtValue per layer/kind) are set once in `DefaultKeyValueCache::Add()` and **never change pointer** for the entire generation; the bind calls are wasted work but cost **real wall time proportional to the bound OrtValue's byte size**. Likely root cause inside ORT IoBinding: `BindInput` of a `device_type="gpu"` OrtValue allocated by the EP's `hipHostMalloc` allocator triggers some allocator-side bookkeeping (page-table touch / device-pointer translation) on every call, scaling with the buffer's VA range rather than O(1). **Implications for measurement and citation:** (1) **Always cite gpt-oss decode tok/s with the `max_length` (or `-ml`) value** — comparing 1B at ml=160 vs 20B at ml=2080 mixes two regimes. (2) Pure-ORT `IoBinding` benches (e.g. `test/python/conftest.py::run_timed_iobinding`) keep the bind cycle outside the timed window, so they report **kernel-only** tok/s — they do NOT predict OGA-path TPS at large `max_length`. (3) When optimizing kernels for gpt-oss-20b, validate against the EP-direct bench, not OGA, otherwise OGA framework overhead masks kernel wins. **Decision: do NOT modify OGA** — this gotcha exists to (a) prevent future sessions from re-investigating the same gap, and (b) keep gpt-oss-20b L=2048 tok/s claims honest.
- **OGA gives tight attention_mask, pure ORT gives padded.** OGA sets `attention_mask.shape[1]` to the actual token count (e.g., `[1,7]` for a 7-token prompt, `[1,8]` after one decode step). Pure ORT tests typically pad to `max_seq_len` (e.g., `[1,128]`). Since DimSource maps `total_sequence_length` to `attention_mask` (first input defining that dim_param), this difference causes DimSource to resolve different shapes for present KV outputs. The shape override in `marshal_output_tensors` handles this correctly — it only triggers when needed (OGA case) and is a no-op when shapes already match (padded ORT case).
- **OGA `generate_next_token()` syncs the PREVIOUS step.** The call is async for the current step but synchronizes the previous dispatch before starting new work. So the 1st call dispatches prefill (returns immediately), the 2nd call syncs prefill + dispatches decode 1 (wall time = TTFT), and calls 3+ each sync one decode step (steady-state tps). Neither `get_next_tokens()` nor `get_sequence()` provides a GPU sync point.
- **`model_benchmark.exe` for fixed-shape pipeline directories needs `-ml <KV_LEN>`.** For fixed-shape pipelines (e.g. `Llama-3.1-8B-Instruct-awq-g128-int4-Pipeline-p512m16384`) you MUST pass `-ml <KV_LEN>` (e.g. `-ml 16384`) to override `model_benchmark`'s default `max_length = prompt_length + generation_length` — otherwise the KV-cache buffers OGA pre-allocates won't match the ONNX's static `total_sequence_length` and prefill bind fails with `Got invalid dimensions for input: past_key_values.0.key Got: 256 Expected: 16384`.
- **cmd.exe `set X=value && next` captures trailing whitespace into the value.** When invoking `model_benchmark.exe` (or anything else that reads `THEROCK_DIST`) from a chained cmd line, **always quote the assignment**: `set "THEROCK_DIST=C:\...\install\therock" && model_benchmark.exe ...`. Without quotes, the value becomes `C:\...\install\therock ` (trailing space before `&&`), and `CompilerDriver` builds the search path as `C:\...\install\therock /lib` — lld-link fails to open `amdhip64.lib`/`MIOpen.lib` and the EP silently falls back to CPU. Same trap applies to `set "PATH=...;%PATH%"`. (PowerShell does not have this issue.)
- **Linker byproducts not cleaned up.** `CompilerDriver::cleanupIntermediates()` removes `.ll` and `.obj` files after linking, but LLD also creates `.lib`, `.pdb`, and `.exp` byproducts alongside each `.dll`. These accumulate in `%TEMP%` (hundreds of files over time). Known issue — fix requires extending `cleanupIntermediates()` in `CompilerDriver.cpp`.
- **OGA chunked prefill: set `search.chunk_size = 1024` for dynamic-shape Llama on MorphiZenEP/gfx1151.** OGA reads `search.chunk_size` directly from `genai_config.json`. With chunking enabled, OGA splits prompts longer than `chunk_size` into multiple EP calls (e.g., a 200-token prompt with chunk_size=128 produces `input_ids` shapes `[1,128]` + `[1,72]`). 8B sweep on gfx1151 (2026-05-06): chunk=1024 beats chunk=512 on long prompts (−1.4% TTFT at L=2048, −3.6% at L=4096) at +0.16 GB peak WS at the worst L. At L≤1024 it is a no-op (single chunk). Decode is invariant to chunk_size. Earlier 1B sweep on gfx1150 found chunk=512 Pareto-optimal — that result does not transfer to gfx1151 (true device memory vs UMA) or the 8B model. Skip for fixed-shape configs (decoder-pipeline / sliding_window / fixed_prompt_length already chunk via their own mechanisms). `patch_genai_config_for_morphizen()` in `test/python/conftest.py` sets this automatically.
- **OGA requires `MorphiZenEP` (short name) everywhere — not `MorphiZenExecutionProvider`.** The OGA dispatch table (`session_options.cpp`) maps the short name `MorphiZenEP` to `MorphiZenEPExecutionProvider::AppendExecutionProvider`, which sets `DeviceType::MorphiZenEP`. Using the long name `MorphiZenExecutionProvider` makes OGA fall through to `AppendExecutionProviderV2` with `DeviceType::CPU` — the model runs correctly but with CPU memory semantics, causing ~40% TPS degradation (no GPU memory aliasing). This applies to `genai_config.json` `provider_options`, `og.register_execution_provider_library()` calls, and `patch_genai_config_for_morphizen()`. Note: pure ORT (not OGA) uses the long name `MorphiZenExecutionProvider` for `ort.register_execution_provider_library()` — these are different APIs.
- **`model_benchmark.exe` requires `genai_config.json` with `MorphiZenEP` in `provider_options`.** The EP DLL is auto-discovered next to `onnxruntime-genai.dll` or `model_benchmark.exe` — do NOT pass `--ep_library`, which causes a double registration and a crash (`STATUS_STACK_BUFFER_OVERRUN` / `0xC0000409`) during OGA shutdown. Required env vars: `PATH` must include `install/dist/bin` and `install/therock/bin`; `THEROCK_DIST` must point to `install/therock`. No other env vars (`LIB`, `HIP_CUSTOM_KERNELS_DIR`) are needed — compile-time defaults cover them. Example: `PATH=install/dist/bin:install/therock/bin:$PATH THEROCK_DIST=install/therock model_benchmark.exe -i models/<model> -l 128 -g 128 -r 3 -w 1 -v`. **For fixed-shape pipeline directories** (e.g. `Llama-3.2-1B-Instruct-Pipeline-p128m4096`) you MUST also pass `-ml <KV_LEN>` (e.g. `-ml 4096`) to override model_benchmark's default `max_length = prompt_length + generation_length` — otherwise the KV-cache buffers OGA pre-allocates won't match the ONNX's static `total_sequence_length` and prefill bind fails with `Got invalid dimensions for input: past_key_values.0.key Got: 256 Expected: 4096`. Same root cause as the "OGA static-mask sizing" gotcha below. **For dynamic-shape models, do NOT pass `-ml`** — TTFT/TPS are unaffected (verified at L=1024 8B g128: 40.12 vs 40.88 tok/s within σ; TTFT 808.7 vs 810.0 ms), but OGA pre-allocates KV buffers to `max_length`, so a stray `-ml 16384` inflates peak working set ~2.3× (1.33 GB → 3.10 GB at L=1024) for no benefit. Let model_benchmark default `max_length = L + G`.
- **OGA `fixed_prompt_length` is a single-Run pad-up path, not a chunker — use `sliding_window` for long prompts.** Setting `decoder.fixed_prompt_length = N` routes prompts of length `2..N-1` through `DefaultInputIDs` with right-padding to N (single `session.Run` to a model fixed at seq=N). At length exactly N, no padding occurs. **For prompts > N, the prompt is sent at native length** (`input_ids.cpp:82-84` guard `sequence_length < fixed_prompt_length`) → bind failure against the static prefill. To accept prompts longer than the static prefill input shape, use `decoder.sliding_window` instead: `{"window_size": N, "alignment": "left", "slide_inputs": true, "slide_key_value_cache": false}`. OGA's `WindowedInputIDs` (`input_ids.cpp`) splits the prompt into `ceil(prompt_len / window_size)` chunks and the dispatch loop in `decoder_only_pipeline.cpp:382-394` calls the prefill sub-model once per chunk inside a single `Generator::Run()`. `WindowedInputIDs` requires `p_device_inputs_->GetType()` ∈ {QNN, CPU}; MorphiZenEP is not in OGA's GPU-input-device list (`model.cpp:592-597`) so it falls through to CPU input device — sliding_window works. Keep `slide_key_value_cache=false` to preserve the static KV buffer + `past_present_share_buffer=true` fast path. `fixed_prompt_length` and `sliding_window` are mutually exclusive (validated at config load, `config.cpp:1560`).
- **OGA static-mask sizing: `max_length` MUST equal the model's static `total_sequence_length`.** When `ShouldUseStaticMaskHandling()` returns true (which fires for MorphiZenEP + `past_present_share_buffer=true`), `DefaultPositionInputs::InitializeStaticMask` sizes `attention_mask.shape[1]` to `params.search.max_length` — NOT to the model's static input shape. If `max_length` differs from the ONNX model's fixed `total_sequence_length`, OGA produces a mask shape that the static-shape model rejects (`Got invalid dimensions for input: attention_mask`). For fixed-shape models, set `params.set_search_options(max_length=KV_LEN, min_length=desired_token_count, do_sample=False)` — never set `max_length=len(prompt)+max_new` like the dynamic helpers `oga_generate` / `oga_generate_timed` do. Bound generation length via `min_length` instead.

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
| Llama-3.1-8B-Instruct-awq-g128-int4 | AWQ INT4 (block_size=128, symmetric / no zero_points) | 32 | 8 | 128 | ~5.0 GB | [amd/Llama-3.1-8B-Instruct-awq-g128-int4-onnx-directml](https://huggingface.co/amd/Llama-3.1-8B-Instruct-awq-g128-int4-onnx-directml) |
| Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml | AWQ INT4 (block_size=128, **asymmetric** / with `zero_points` packed uint8 nibbles) | 32 | 8 | 128 | ~5.0 GB | [amd/Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml](https://huggingface.co/amd/Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml) |
| gpt-oss-20b (webgpu int4-rtn-block-32) | MXFP4 MoE (32 experts × 4 active per token) + fp16 attention | 24 (alternating sliding/full attention, sliding_window=128) | 8 (HPG=8) | 64 | ~12.5 GB | [onnxruntime/gpt-oss-20b-onnx](https://huggingface.co/onnxruntime/gpt-oss-20b-onnx/tree/main/webgpu/webgpu-int4-rtn-block-32) |
| gpt-oss-120b-pergroup-asym-awq | AWQ INT4 pergroup-asym MoE (128 experts × 4 active per token, group_size=32) + fp16 attention | 36 (alternating sliding/full attention, sliding_window=128) | 8 (HPG=8) | 64 | ~64.5 GB | [amd/gpt-oss-120b-w-uint4-pergroup-asym-awq-onnx-fp16](https://huggingface.co/amd/gpt-oss-120b-w-uint4-pergroup-asym-awq-onnx-fp16) (gated AMD repo — `hf auth login` required) |

**Note:** The 8B model has `position_ids` as an additional input (not present in the 1B model). The sym 8B MatMulNBits ops have `block_size=128` and **no zero_points input** (symmetric AWQ quantization) — the kernel uses default zp=8.0 in that path. Both block_size values (32 and 128) are covered by the runtime autotune (`BLOCK_SIZE ∈ {32,64,128,256,512,1024}`); switching block_size triggers a fresh autotune on first call but is otherwise transparent.

**gpt-oss-20b structural notes (distinct from the Llama family).** Mixture-of-Experts backbone (32 experts, 4 active per token, MXFP4-quantized expert weights — **not** MatMulNBits/AWQ INT4), `attention_bias=true`, alternating sliding/full attention layers (sliding_window=128 baked into the graph), YaRN rope scaling, vocab=201088, hidden=2880. **No `position_ids` input** — rope is driven internally by the graph. GPU dispatch was previously failing silently (CPU fallback) until the `lower-affine` pipeline gotcha (above) was fixed — the MoE expert-major 3D→2D flatten in qmoe is what triggered the bad `affine.apply` survivors.

**gpt-oss-120b structural notes + ORT CPU baseline limitation.** Same MoE backbone as 20B (alternating sliding/full attention, packed QKV, head_sink, sliding_window=128, no `position_ids`, type="gptoss") scaled to 36 layers and 128 experts (4 active per token). Expert quantization is **AWQ INT4 pergroup-asym with group_size=32**, not MXFP4 — `fc1/fc2_experts_scales` ship with shape `[num_experts, 2*inter, hidden/group_size] = [128, 5760, 90]`. **ORT's CPU EP `QMoE` op only supports per-channel scales** (last-dim=1), so every CPU baseline run on this model fails immediately with `Input 'fc1_experts_scales' is expected to have shape {128,5760,1}, got {128,5760,90}`. `test/python/test_gptoss120b.py` maps that failure to `pytest.skip` via a module-level cache (first failure flips the flag, subsequent CPU-dependent tests skip without re-attempting the 64 GB CPU model load — the repeat loads also bad_alloc the Python heap on 96 GB Strix Halo). EP coverage is preserved via the OGA suite — `test_oga_ep_generation`, `test_oga_ep_no_share_buffer`, `test_oga_ep_shape_switching` all run end-to-end on MorphiZenEP. `test_ort_dynamic_vs_fixed` is also skipped because each EP session pins a ~69 GB SHARED_CONSTANTS blob and two distinct ONNX graphs (fixed + dynamic) cannot both be live in 96 GB. **None of these skips indicate an EP bug** — they are environment limits (CPU op coverage, system memory budget) and lift automatically when those limits do.

**Sym vs asym 8B model identity (AVOID PHANTOM-BUG CONFUSION).** The sym dyn 8B (`Llama-3.1-8B-Instruct-awq-g128-int4`) is the **Instruct** fine-tune; the asym dyn 8B (`Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml`) is the **base** Llama-3.1-8B. Different post-training. They produce **different greedy generations** for the same prompt — and the base model in particular tends to echo / repeat / wander on chat-template-style prompts (e.g. "Write a paragraph about Paris" → may emit garbled tail). This is **NOT a kernel bug**; it's normal base-model behavior on instruction-formatted prompts. Verification protocol when an output looks "broken": (a) run a per-step CPU-vs-EP logit comparison test — if it passes, the kernel is correct; (b) compare CPU-only vs CPU-only output of the same model with the same prompt — if CPU also produces "broken" English, the model is the cause.

### Running with hip-onnx-runner

```cmd
rem From the repo root — TheRock ROCm DLLs must be on PATH for compiled model DLLs to load
set PATH=%CD%\install\therock\bin;%PATH%
install\dist\bin\hip-onnx-runner.exe -m models\Llama-3.2-1B-Instruct\model_q4f16_fixed.onnx
```

**Important:** `hip-onnx-runner` fills inputs with random data by default. For models with GQA, this produces garbage `seqlens_k` values and GQA failures. Use `-i <dir>` to provide valid binary input files (see `test/python/` for how the perf tests generate correct inputs).

## Python Performance Tests

**Never run GPU benchmarks in parallel.** GPU contention between concurrent benchmarks produces unreliable numbers. Always run benchmark approaches (ORT EP, OGA Python, model_benchmark) sequentially — one at a time.

Nine test files, one per model. **Prefer 1B tests when debugging or iterating on fixes** — they run significantly faster (~2 min vs ~15 min) due to the smaller model:

```bash
pytest test/python/test_llama1b.py        -v -s   # 1B q4f16
pytest test/python/test_llama8b.py        -v -s   # 8B AWQ INT4 g128 (sym)
pytest test/python/test_llama8b_asym.py   -v -s   # 8B AWQ INT4 g128 (asym, zero_points uint8) — exercises zp_unpack_cache
pytest test/python/test_mistral7b_v3.py   -v -s   # Mistral 7B v0.3 INT4
pytest test/python/test_phi4_14b.py       -v -s   # Phi-4 14B RTN block_size=32
pytest test/python/test_qwen2_5_14b.py    -v -s   # Qwen2.5 14B (+ Coder variant smoke)
pytest test/python/test_gemma3_4b.py      -v -s   # Gemma3 4B VLM (inputs_embeds path)
pytest test/python/test_gptoss20b.py      -v -s   # gpt-oss-20b MoE
pytest test/python/test_gptoss120b.py     -v -s   # gpt-oss-120b MoE (gated AMD repo) — 3 pass + 7 skip on this hardware
pytest test/python/test_deepseek_r1_70b.py -v -s  # DeepSeek-R1-Distill-Llama-70B (gated AMD repo)
```

**Spec-driven layout (refactored 2026-05).** `conftest.py` owns the entire test framework. Each per-model file is a thin declaration (~30 lines for canonical cases, ~80-200 lines for specials):

```python
# test_llama8b.py
LLAMA8B = ModelSpec(
    name="llama8b",
    model_dir=REPO_ROOT / "models" / "Llama-3.1-8B-Instruct-awq-g128-int4",
    onnx_file="model.onnx",
    data_files=["model.onnx.data"],
    hf_base="https://huggingface.co/amd/...",
    num_layers=32, num_kv_heads=8, head_dim=128, has_position_ids=True,
    bos_token=128000, filler_tokens=[...],
    oga_files=[...],
)

dynamic_model_path, fixed_decode_path, fixed_prefill_128_path = (
    register_model_fixtures(LLAMA8B)
)

class TestLlama8BORT(BaseORTTests):  spec = LLAMA8B
class TestLlama8BOGA(BaseOGATests):  spec = LLAMA8B
```

`BaseORTTests` (6 methods) and `BaseOGATests` (4 methods) own the 10 canonical tests. The base classes call helpers in conftest (`create_ep_session`, `run_iobinding_once`, `compare_outputs`, `oga_generate_timed`, etc.).

**`ModelSpec` knobs** (defaults cover the canonical case):

| Field | Purpose |
|-------|---------|
| `hf_base` / `hf_repo` | Mutually exclusive. `hf_base` → urllib (public repos); `hf_repo` → huggingface_hub (gated AMD repos). |
| `hf_subdir` | Flatten downloads from `<subdir>/<file>` into model_dir/<file> (Qwen, Gemma3). |
| `data_files`, `extra_data_files` | ONNX external-data blobs; `extra_data_files` = auxiliary ONNX files (gemma3 embedding/vision). |
| `genai_config_template` + `hf_root_for_tokenizer` | For models whose HF repo lacks a genai_config (1B): we write the template and pull tokenizer files from the root. |
| `chunk_size` | `search.chunk_size` migration value (default 1024; matches CLAUDE.md OGA chunked-prefill gotcha). |
| `normalize_onnx_hook` | Per-spec post-fetch fixup. `normalize_drop_inputs_embeds` (in conftest) handles the Qwen/DeepSeek/120B dangling-input quirk. |
| `cpu_skip_predicate` + `cpu_skip_reason` | When CPU baseline raises (e.g. 120B QMoE), the predicate matches the exception → `pytest.skip` with the reason. |
| `markers` | `{test_method_name: pytest.MarkDecorator}`. Applied at collection via `pytest_collection_modifyitems`. Used for 120B's memory-budget skip and 20B/120B's MoE-precision xfails. |

**CPU-reference golden cache (`install/golden/`).** Each accuracy test routes its CPU `sess.run()` through a `GoldenStore`. The cache holds full output tensors (logits + present KV) under `install/golden/<spec.name>/<key>.npz` plus a sibling `<key>.meta.json` recording `{onnx_size, onnx_mtime, inputs_sha256, output_names}`. On hit, the test loads the saved arrays and runs EP only — typically 20-30% wall-clock savings on warm reruns (more on heavier models). Auto-invalidated when `onnx_mtime` / `onnx_size` / `inputs_sha256` change.

To force regeneration:
- All models: `rm -rf install/golden`
- One model:  `rm -rf install/golden/llama8b`
- `install/golden/` is gitignored (under `install/`) — cache is local-only; CI and devs regenerate on first run.

There is no CLI flag and no env var. `make_llama_inputs(seed=0)` is deterministic so the cache key (`inputs_sha256`) is stable.

**`BaseORTTests` methods** (all six inherited unchanged unless overridden):

| Method | What it covers |
|---|---|
| `test_ort_fixed_decode` | Fixed-shape decode (seq=1, kv=max_seq_len): EP-vs-golden accuracy + IOBinding latency |
| `test_ort_fixed_prefill_128` | Fixed-shape prefill (seq=128): same |
| `test_ort_dynamic_prefill_128` | Dynamic-shape prefill at seq=128 |
| `test_ort_dynamic_decode` | Dynamic-shape decode (seq=1) |
| `test_ort_dynamic_vs_fixed` | Pure EP-vs-EP — guards the dynseqlen path. KV cosine must be 1.0 (bit-identical). No CPU baseline. |
| `test_ort_per_step_logits` | Prefill + 9 decode steps; per-step logits compared against cached CPU step logits. Independent EP chain (its own KV); after token-divergence both chains stay internally consistent. |

**`BaseOGATests` methods:**

| Method | What it covers |
|---|---|
| `test_oga_ep_generation` | OGA+MorphiZen EP latency at prompt_len=128 |
| `test_oga_ep_no_share_buffer` | `past_present_share_buffer=false` → GQA concat fallback works |
| `test_oga_ep_shape_switching` | Alternates prompt lengths via `rewind_to(0)` on one Generator |
| `test_oga_ep_chunked_prefill` | OGA chunked-prefill match vs cached CPU reference tokens |

**Special-case files:**
- `test_gemma3_4b.py` — VLM: text decoder takes `inputs_embeds` (not `input_ids`); the spec sets `prefill_input_fn` / `decode_input_fn` to plug those builders into the standard `BaseORTTests` flow, so the file is a thin spec declaration like the others. Decode tests use zero past KV (`make_zero_kv_cache` in conftest) — same as every other model.
- `test_gptoss120b.py` — `cpu_skip_predicate` matches QMoE-per-group and bad_alloc; `markers` skip dynamic_vs_fixed (memory) and xfail per-step-logits / chunked-prefill (MoE precision).
- `test_gptoss20b.py` — `markers` xfail per-step-logits + chunked-prefill.
- `test_qwen2_5_14b.py` — Two specs (base + Coder); Coder variant has its own `TestQwen2_5_Coder_14BOGA` class with a single smoke test.
- `test_llama1b.py` — `genai_config_template` literal (HF repo lacks `genai_config.json`).

**Adding a new model:** copy one of the canonical files (e.g. `test_llama8b.py`), update the `ModelSpec` literal, done. The two test classes are inherited and need no body.

**1B OGA config:** The 1B HuggingFace repo lacks `genai_config.json`. `LLAMA1B.genai_config_template` carries the dict; conftest writes it to disk on first OGA test. Tokenizer files come from `LLAMA1B.hf_root_for_tokenizer` (the repo root, one level above the `/onnx` subdir holding the ONNX).

**Memory management:** Sessions are explicitly deleted and `gc.collect()` called between tests to free GPU memory. OGA tests create/destroy `og.Model` within each test.

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

**NEVER measure TPS / TTFT with `HIPDNN_EP_PERF=1` set.** The event pool adds ~58% wall-clock overhead per inference (~34 ms per Compute call on 8B) AND CPU-saturates per-op recording, which compresses the gap between fast and slow configurations and can even invert it. Use `HIPDNN_EP_PERF` only for (a) per-op GPU-time breakdown, and (b) checking that the CPU column is near zero on GPU ops (non-zero CPU on a GPU op flags an unintended `hipStreamSynchronize`). For TPS/TTFT/E2E numbers, profile must be OFF.

**Before chasing a perf-regression hypothesis with new instrumentation, RE-MEASURE without `HIPDNN_EP_PERF` first.** A full session was once spent adding phase markers and per-layer GPU markers to localize a 1-3 ms/Compute "penalty" — the penalty was the profiler itself, and the suspected slow path was actually faster than the baseline once profiling was off. If a perf gap is small (≤ 5% of per-Compute total) and points the wrong way, do the clean re-measure before writing one line of instrumentation.

**GPU benchmark hygiene.** (1) **never** launch a benchmark with `run_in_background: true` from Claude Code — always foreground with a long enough timeout; (2) before starting any GPU benchmark, verify the GPU is idle (`tasklist | findstr /I model_benchmark` on Windows) and kill stragglers with `taskkill /F /IM model_benchmark.exe` if anything appears; (3) `TaskStop` on a backgrounded bash task does NOT free the GPU — it signals the bash wrapper but the child `model_benchmark.exe` keeps running on the GPU.

### Architecture

- Per-op profiling uses RAII scope guards (`OpProfileScope`, `OpProfileCpuScope`) that record `hipEvent_t` start/stop pairs on the HIP stream. Events are resolved in bulk after `hipStreamSynchronize` — no per-op sync.
- **Event pool**: HIP events are pre-allocated in `OpProfileState` and reused across inferences — no `hipEventCreate/Destroy` per operator call. The pool grows on demand but never shrinks.
- Profiling state lives in `RuntimeState->op_profile` (opaque `void*`), not globals — each inference session has its own state. Two-level data model: `OpEntry` (per op name) → `ShapeEntry` (per shape string).
- Key files: `op_profile.h` (RAII scopes + macros), `op_profile.cpp` (state + event pool + printing), `debug_log.h` (env var check)
- Each operator wrapper adds one line: `OP_PROFILE("opname", shape_lambda, state)` or `OP_PROFILE_CPU("opname", state)`. The shape lambda is only invoked when profiling is active — zero `snprintf` overhead on the hot path.

### Llama 8B per-op profile shape (architectural)

For Llama 8B decode at single-token (`sq==1`), the per-op GPU-time breakdown is dominated by `matmul_nbits` (~78% of GPU time across 225 calls per inference: 64 at n=14336/k=4096, 32 at n=4096/k=14336, 64 at n=4096/k=4096, 64 at n=1024/k=4096, 1 at n=128256/k=4096 for the LM head). Other ops (skip_layernorm, rotary_emb, gqa, activation, elementwise) each take low single-digit %. Op call counts and shapes are graph-structural and don't change with quant layout (g32 vs g128) or context length — only the per-op ms shifts.

CPU time per op should be near zero for GPU ops (host-side hipMemcpy/hipLaunchKernel return immediately); `stream_sync` captures the full GPU wait. **Non-zero CPU on a GPU op flags an unexpected `hipStreamSynchronize` in its code path** — this is the canonical diagnostic for the GQA decomposed-path D2H stall (fixed for the fused-decode path via `seqlens_k` cache + GPU-buffer aliasing; see "GQA seqlens_k caching" below).

For specific gpu-ms numbers, run `HIPDNN_EP_PERF=1 install/dist/bin/model_benchmark.exe ...` and read the `[PERF]` block from stderr — they'll be current to the model + kernels in use.

### GQA flash_decode (long-context decode unlock)

Long-context decode previously suffered from the original `gqa_fused_decode` kernel scaling linearly with `skv` (each query head re-read the full K/V cache → 4× bandwidth waste at HPG=4). The `gqa_flash_decode` kernel (`3rd-party/custom_kernels/hip/gqa_kernel.hip`) uses a GQA-aware split-K Flash Attention 2 design: one block per (batch, kv-head, K_SPLIT) loads K/V tiles into LDS once and reuses them across all HPG query heads, then a small reduction kernel merges partials. Templated as `<D, K_SPLITS, HPG>` with `K_SPLITS=8`; instantiated for **(D=64, HPG=4)**, **(D=128, HPG=4)** (Llama family), and **(D=64, HPG=8)** (gpt-oss-20b). HPG=8 with D=128 is rejected at the launcher (would exceed thread/LDS budget — `THREADS = HPG * WAVE_SIZE = 256` already at HPG=8). At L=1024 GQA per-layer (skv≈1056) drops from ~720 µs (linear-scaling fused_decode) to ~156 µs (~4.6× faster), and decode at L=2048 is only ~6% slower than L=1024 — flash_decode has flattened the depth scaling.

**Sliding window + attention sinks (gpt-oss-20b unlock).** The reduce kernel (`gqa_flash_decode_reduce_kernel`) supports two extensions needed for gpt-oss-20b:
  - **Sliding window** (`local_window_size > 0`): each split-K block clamps its KV range to `[max(0, eff_skv - local_window_size), eff_skv)` — only the last `local_window_size` positions contribute. ONNX uses `local_window_size = -1` (or `<= 0`) for "no windowing"; positive values are honored. gpt-oss-20b alternates 12 sliding (`window=128`) + 12 full-attention layers; both take flash_decode now.
  - **Smooth softmax / head sink** (`head_sink` ptr or `use_smooth_softmax` flag): adds an extra term to the softmax denominator only (numerator unchanged). With `head_sink`, contributes `exp2((sink[head] - global_m) * LOG2E)`; with `use_smooth_softmax`, contributes `exp2(-global_m)` (sink ≡ 0). Implemented in the reduce step where the global max is known.

**Packed-QKV support in the fused branch (gpt-oss-20b unlock).** gpt-oss-20b's `GroupQueryAttention` nodes ship with packed QKV in `query` (in[0]) and **null `key`/`value`** (in[1]=in[2]=`""`) — the QKV split happens inside the op. Previously `fused_predicate` required `key && value`, so all 24 layers fell through to the linear-skv decomposed hipBLASLt path. The fused branch now accepts `(key && value) || (!key && !value)` and, in the packed case, runs `hip_gqa_split_qkv` (Step 0) into a workspace region `[Qsplit, Ksplit, Vsplit | Qroped, Kroped | flash_partials]` before rope/append/decode. Workspace is one `hipdnn_ep_state_ensure_workspace` call sized for the combined layout; growing the workspace does NOT preserve data so it must be sized once up-front.

**Dispatch gate** (`lib/Runtime/real/gqa.cpp::flash_decode_geometry_ok` + `fused_predicate`): flash_decode runs when `sq==1`, **geometry is `(HPG=4, d∈{64,128})` or `(HPG=8, d=64)`**, and `skv >= HIPDNN_EP_GQA_FLASH_DECODE_MIN_SKV` (default 256). The fused predicate also checks `sliding_ok_for_fused` (sliding only allowed when flash_decode is eligible — the legacy LDS-tiled fused_decode does not support windowing) and `sink_ok_for_fused` (head_sink / smooth_softmax only allowed when flash_decode is eligible). At small skv the original LDS-tiled fused_decode wins because the new kernel's K_SPLITS=8 reduction overhead exceeds its bandwidth savings. Disable entirely with `HIPDNN_EP_GQA_FLASH_DECODE=0`. Workspace partials for the reduction step share the GQA workspace (placed after Q/K split + rope temps) — `hipdnn_ep_state_ensure_workspace` is called once with the combined size.

**GQA smart-dispatch threshold + flash_decode exemption** (`lib/Runtime/real/gqa.cpp`). The legacy `gqa_fused_decode` kernel scales linearly in `total_seq` (skv). At very long contexts it loses ~12× to a GEMM-based decomposed path. The dispatcher caps legacy fused_decode at `HIPDNN_EP_GQA_FUSED_DECODE_MAX_T` (default 256) — beyond that it falls back to the decomposed (GEMM + softmax + GEMM) path. **flash_decode is exempt from this cap**: when `flash_decode_geometry_ok` is satisfied and skv ≥ 256, the smart-dispatch check passes regardless of `total_seq`, because flash_decode's split-K design already kills the linear scaling. The actual gating expression is `size_ok_for_fused = (total_seq_pre < 0) || (total_seq_pre <= MAX_T) || flash_decode_eligible`.

**GQA seqlens_k caching across layers in one Compute()** (`lib/Runtime/real/gqa.cpp`). The dispatcher needs to read `seqlens_k[0]` (a single int32 on the GPU) to know `total_seq` before deciding fused vs decomposed. Naively this is one D2H + `hipStreamSynchronize` per GQA layer per Compute() — 32 stalls per inference on 8B. The runtime caches the value across the N GQA layers within a single Compute() call: first layer pays the D2H, layers 2..N reuse the cached value. The cache is keyed on the seqlens_k device pointer (one cache slot per pointer) and invalidated at the start of every Compute() via the `inference_state_->begin_compute()` runtime hook. Disable with `HIPDNN_EP_GQA_CACHE_SEQLENS=0` (default ON); the cache sentinel is `kSeqlensKNotRead = -2` so a real read of -1 (no past tokens) is distinguishable from "not yet read".

**`inference_state_->begin_compute()` runtime contract** (`backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp`). Per-Compute() hook into the model.dll runtime that invalidates per-Compute caches (currently only the GQA seqlens_k cache). Called by the EP at the top of every `Compute()` before any input marshaling. **Backwards compatibility:** older model.dll snapshots predate this export — they don't crash, the hook is a cached indirect dispatch that no-ops on missing symbol, but `HIPDNN_EP_GQA_CACHE_SEQLENS` MUST be set to `0` for old DLLs or the cache will return stale total_seq across forward passes. Cost on the fast path is ~1 ns (one cached indirect call).

**Short-context decode (LDS-tiled fused_decode).** When `gqa_flash_decode` does not dispatch (skv < 256, or geometry not in {`HPG=4 d∈{64,128}`, `HPG=8 d=64`}, or sliding/sink requested but flash_decode ineligible), the path is `gqa_fused_decode` — which now uses LDS-tiled K/V prefetch. The decode block grid is tiny (B=1, H=32 → ~32 blocks / ~128 waves on RDNA3 wave32), far below what is needed to hide ~400-cycle global-load latency by wave switching. The kernel cooperatively prefetches `TILE=8` rows of K and V into LDS per outer iteration, then crunches 8 softmax steps from fast LDS — driving memory-level parallelism via TILE (16 outstanding loads per thread per tile) instead of wave count. LDS use per block: `2 * TILE * D * sizeof(_Float16)` = 8 KB at D=128 (well under RDNA3's 64 KB/CU). No `__syncthreads` between prefetch and inner loop because each thread writes/reads only its own column. TILE=8 is sweep-validated.

**L=256+ TPS cliff (8B, open issue).** Decode TPS drops at the flash_decode dispatch boundary: L=128 ≈ 17 tok/s → L=256 ≈ 12 tok/s. Per-op TOTAL is ~identical at L=128 vs L=256 with profiling, so the gap is launch/scheduling overhead invisible to per-op profiling. Disabling flash_decode at L=256 makes things WORSE — flash_decode IS the right path, just suboptimally tuned. The `kFlashDecodeKSplits = 8` constant in `lib/Runtime/real/gqa.cpp` is right for L=2048 (verified) but suspected wrong at L=256–1024; a static reduction in K_SPLITS regresses long-context, so a real fix likely needs per-shape autotune or a fused single-pass online-softmax kernel that eliminates the reduce launch entirely.

Profiling overhead (event pool): ~34 ms (+58%) per inference on 8B, from ~972 `hipEventRecord` + ~486 `hipEventElapsedTime` calls. This is the inherent cost of GPU timing instrumentation.

**GQA fused decode path:** For single-token decode (`sq==1`) with supported head dimensions (`d∈{64,128,256}`), GQA uses a fused custom HIP kernel path (rope + KV append + fused attention decode). This path is fully async — no D2H copies, no per-layer sync. The `local_window_size` attribute must be `<= 0` (ONNX uses `-1` for no windowing). The decomposed hipBLASLt path (used for prefill or unsupported configs) falls back to D2H + sync per layer.

**GPU memory aliasing:** When the EP's `hipHostMalloc` allocator is used (OGA or IOBinding with `device_type="gpu"`), KV cache tensors arrive with `memory_type == TENSOR_MEMORY_GPU`. The runtime aliases them directly — zero H2D/D2H. This makes `past_key_gpu == present_key_gpu`, so GQA skips the per-layer D2H `seqlens_k` copy + `hipStreamSynchronize` in the decomposed path.

**EP allocator uses `hipHostMallocMapped|NonCoherent`, not `Coherent`.** `morphizen-hip-gpu-allocator.cpp::HipGpuAllocator::AllocImpl` allocates the aliased pinned buffers (KV cache, activations) with `hipHostMallocMapped | hipHostMallocNonCoherent` (coarse-grained). Coherent forces every GPU load/store on the buffer to bypass the GPU's MALL/L2 caches so CPU cache lines stay snoopable in real time — wasted on this workload, where OGA touches the buffer once on the host (zero-init / IOBinding setup) and the GPU is the sole reader+writer for the rest of the session. Synchronization happens at OrtRun / `hipStreamSynchronize` boundaries, which is exactly the contract NonCoherent requires. Measured win on gfx1151 (8B Llama AWQ INT4 g128, `model_benchmark -r 5 -w 1`): dynamic-shape decode +1.6% @ L=128, +3.9% @ L=2048; fixed pipeline (KV pre-allocated to 16384) decode +2.4% @ L=128, +4.4% @ L=2048; TTFT -1.8% to -3.6%; peak working set unchanged; output bytes bit-identical. Fixed pipeline gains more because more bytes pass through the GPU cache hierarchy per layer. **Do not switch back to Coherent without re-measuring** — gfx1151's small MALL (~32 MB) makes every Coherent miss visible. Discrete-GPU DEFAULT memory should still go through `hipMalloc` (true VRAM) — see the TODO in the allocator file; that path needs OGA-side surgery (`MorphiZenEP::Memory` p_cpu_ ≠ p_device_) before it can be wired up.

**GQA path selection:** With aliased GPU buffers, `past_key == present_key` at the GPU level → the `need_host_past_len` condition (`past_key && past_key != present_key`) is false → no D2H copy, no `hipStreamSynchronize` per layer. The concat path (which requires host-side `past_len`) is only triggered when `past_key != present_key`.

**Runtime state:** Buffer pool and GQA GEMM descriptor cache are per-session (stored in `RuntimeState` as opaque pointers), not globals. This supports concurrent inference sessions.

### GQA flash_decode coverage by CI model

Flash_decode template instantiations: `(D=64, HPG=4)`, `(D=128, HPG=4)`, `(D=64, HPG=8)`. Models outside this set fall through to `gqa_fused_decode` (LDS-tiled, short-context-tuned) for `sq==1` and to the decomposed hipBLASLt path otherwise. Use `_inspect_geom.py` to print H/G/HPG/d for every model in `models/CI/`.

| Model family | H | G | HPG | d | flash_decode? |
|---|---|---|---|---|---|
| Llama-3.1-8B (sym + asym) | 32 | 8 | 4 | 128 | YES |
| Mistral-7B-Instruct-v0.3 | 32 | 8 | 4 | 128 | YES |
| Llama-3.2-1B | 32 | 8 | 4 | 64 | YES |
| Phi-4-14B | 40 | 10 | 4 | 128 | YES |
| gpt-oss-20b / gpt-oss-120b | 64 | 8 | 8 | 64 | YES (with sliding/sink/packed-QKV) |
| DeepSeek-R1-70B | 64 | 8 | 8 | **128** | NO — would need `<128, K, 8>` (rejected today: HPG·WAVE=256 already maxes out the block) |
| Qwen2.5-14B / Qwen2.5-Coder-14B | 40 | 8 | **5** | 128 | NO — would need `<128, K, 5>` instantiation |
| gemma3-4b | 8 | 4 | 2 | **256** | NO — D=256 not instantiated |

### Benchmark hygiene additions (2026-05)

- **OGA chunked prefill warmup count:** sliding_window=512 models (gpt-oss-20b/120b, Llama-3.1-8B-Pipeline-p512m16384, etc.) chunk prefill into 512-token slices. `model_benchmark -w N` runs `N` end-to-end warmups, so for matmul_nbits GEMV autotune to converge (needs ~4 calls per shape) total warmup invocations of each prefill chunk must be ≥ 4. Rule: `-w max(1, ceil(4 / chunks_per_prefill))`. L=128 → -w 4; L=512 → -w 4; L=1024 → -w 2; L=2048+ → -w 1 OK.
- **WMMA disk-persistent autotune cache:** `_results.json` next to `model_benchmark.exe` — a stale or missing entry can make a cold-cache run report 5× lower decode TPS than a primed one. Always discard rep-1 numbers when comparing kernels; quote rep-3+ steady-state.
- **Always cite gpt-oss decode tok/s with `-ml`.** OGA's per-token IoBinding rebind cost scales with KV buffer size (see gotcha above), not actual sequence length. Comparing 1B at -ml 160 vs 20B at -ml 2080 mixes two regimes.

### Verified perf snapshot — gfx1151 (Strix Halo, 16 CUs), 2026-05-08 (gpt-oss-120b row re-measured 2026-05-19)

Best observed `model_benchmark.exe` numbers (`-g 32 -ml 16384 -r 5`, `-w 1` for L=2048 / `-w 4` for L=128, fp16 KV). Use these as the regression baseline; future changes that drop any cell by >5% on the same hardware should be investigated.

| Model | L=128 prefill / decode (tok/s) | L=2048 prefill / decode (tok/s) |
|---|---|---|
| Mistral-7B-Instruct-v0.3 (AWQ b128) | 317.7 / **48.2** | 865.3 / **42.6** |
| Llama-3.1-8B-Instruct (AWQ g128 sym) | 301.3 / **44.5** | 1078.5 / **39.6** |
| Llama-3.1-8B (AWQ g128 asym) | 295.0 / **40.5** | 1055.7 / **36.4** |
| Phi-4-14B (RTN b32) | 207.8 / **24.2** | 719.2 / **22.3** |
| Qwen2.5-14B-instruct (RTN g128) | 182.3 / **23.7** | 634.3 / **22.6** |
| Qwen2.5-Coder-14B (RTN g128) | 182.7 / **23.6** | 632.7 / **22.8** |
| gemma3-4b-it (RTN g128) | 572.6 / **65.1** | 9020.0 / **62.2** |
| gpt-oss-20b (webgpu int4-rtn b32) | 324.4 / **76.4** | 1122.2 / **72.4** |
| gpt-oss-120b (uint4 pergroup-asym AWQ) | 333.4 / **37.9** | 664.3 / **35.9** |
| DeepSeek-R1-Distill-Llama-70B (AWQ b128) | 25.8 / **5.5** | 104.6 / **5.4** |

### Known limitations (open / accepted)

- **L=256+ TPS cliff on Llama-8B family** — see "L=256+ TPS cliff" gotcha. Vulkan-style adaptive K_SPLITS does not help. Likely needs a fused single-pass online-softmax kernel or per-shape K autotune.
- **DeepSeek-R1-70B, Qwen2.5-14B, gemma3-4b** fall off flash_decode (table above). Decode at long context will use the legacy fused_decode (capped at total_seq=256 — beyond that, decomposed hipBLASLt). Adding the missing instantiations is "low-hanging fruit" but unverified.
- **OGA per-token IoBinding rebind overhead** scales with `max_length`. Not fixable from the EP side; documented gotcha. Pure-ORT IoBinding benches do not see this.
- **Linker byproducts** (`.lib`, `.pdb`, `.exp`) accumulate in `%TEMP%` — `CompilerDriver::cleanupIntermediates()` only removes `.ll`/`.obj`. Cosmetic.
- **Stale model.dll cache** is keyed on the ONNX hash, not the runtime version. Any runtime/.cpp or kernel/.hip change MUST be followed by `del %TEMP%\morphizen_mlir_*` or compiled DLLs will silently use the old bitcode.

### Future improvements (ranked by expected impact)

1. **Fused single-pass attention decode** that eliminates the reduce kernel — would unblock the L=256+ cliff and remove the K_SPLITS guesswork entirely. High effort, high impact.
2. **Add `<128, K, 5>` flash_decode instantiations** for Qwen2.5-14B / Coder-14B. Medium effort (HPG=5 is unusual — verify thread/LDS budget).
3. **Per-shape K_SPLITS autotune** keyed on `(D, HPG, B*G, skv_bucket)` — lower-risk than (1).
4. **Profile-OFF investigation of flash_decode at L=256–1024 on Llama-8B** — the cliff is launch/scheduling overhead, not per-op cost.
5. **Lower `kFlashDecodeMinSkv`** (currently 256) and benchmark — short-context flash_decode might already win on gpt-oss-20b's 12 sliding-window layers (window=128 → effective skv ≤ 128).
6. **Extend `cleanupIntermediates()`** to remove `.lib/.pdb/.exp` byproducts. Trivial.

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

`build.py --build-vulkan` fetches the LunarG Vulkan SDK and builds llama.cpp with `GGML_VULKAN=ON` for collecting Vulkan baseline numbers. Pinned versions for reproducibility: llama.cpp commit `683c5acb`, Vulkan SDK `1.4.341.1`. Idempotent — `.ok` sentinels, shares `install/_cache/` with the rest of `build.py`.

```bash
python build.py --build-vulkan   # full project build + Vulkan baseline
```

Outputs (under `install/`): `vulkan-sdk/` (silent install, user-writable, no admin), `llama.cpp/` (pinned source), `llama.cpp-build/` (Ninja build dir), `llama-vulkan/bin/` (`llama-bench.exe`, `llama-cli.exe`, `llama-server.exe`, `ggml-vulkan.dll`).

**LunarG download URL gotcha.** The installer is `vulkansdk-windows-X64-{ver}.exe` (lowercase `vulkansdk`, `X64` token), NOT `VulkanSDK-{ver}-Installer.exe` — the latter returns 404. Append `?Human=true` to bypass LunarG's download-token throttling. The CDN also rejects the default Python `urllib` User-Agent — `_download_with_browser_ua()` in `build.py` installs a `Mozilla/5.0` opener for the Vulkan download only (other downloads keep the default UA).

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
- **MANDATORY (compiler code only): Include an IR before/after snippet next to non-trivial transformations.** Any MLIR pass, RewritePattern, or fold (in `lib/Conversion/`, `lib/Dialect/`, `lib/Compiler/`, `include/hip/`, `backend-mlir-compiler/`) that rewrites the IR in a non-obvious way must carry a short `Before:` / `After:` MLIR snippet in the comment block describing it — either at the file header (for a whole pass) or immediately above the matcher/rewrite function (for a single pattern). The snippets should be the smallest reproducer of the canonical case, edited for readability (drop attribute clutter, shorten SSA names, elide irrelevant ops). Reason: reviewers and future maintainers cannot reconstruct the IR shape from C++ alone — the snippet is the only place that shows what the transformation actually does to the IR. Skip for trivial DCE / constant-fold / typo-fix rewrites where the source code IS the IR description. Exempt: runtime, custom kernels, tests. Applies to new transformations AND to any non-trivial transformation touched in a PR.
  - **Keep the snippets in sync with the code.** When the transformation logic changes — input/output op kinds, operand order, attribute names, the shape of the rewritten subgraph, or which constants are still live after the rewrite — update the `Before:` / `After:` snippet in the SAME commit / PR. A stale snippet is worse than no snippet: it silently misleads the next reader into believing the code does something it no longer does, and reviewers will trust it instead of re-deriving from C++. Treat the snippet as part of the function/pass contract: if you would not ship the code without updating the docstring on a public API, do not ship a transformation change without updating its IR snippet.
- **MANDATORY (compiler code only): Prefer `llvm::seq` (or `llvm::seq_inclusive`) over vanilla index-counting `for` loops** when iterating an integer range. Use `for (int64_t i : llvm::seq<int64_t>(0, n))` instead of `for (int64_t i = 0; i < n; ++i)`. Scope: code that already lives inside the LLVM/MLIR ecosystem — `lib/Conversion/`, `lib/Dialect/`, `lib/Compiler/`, `include/hip/`, `backend-mlir-compiler/`. Does NOT apply to `lib/Runtime/`, `3rd-party/custom_kernels/`, `.hip` kernels, or `tools/`, where the LLVM ADT headers are not in scope and vanilla loops are the local idiom. Vanilla loops are still fine in compiler code when the loop body mutates the index (skip-ahead, early break with index reuse outside the loop, non-unit stride that isn't naturally a seq). Applies to new code AND to any vanilla index loops touched in a PR.
- **MANDATORY: No hardware-specific identifiers (e.g. `gfx1151`, `gfx1150`, `Strix Halo`, specific SKU names) in code comments.** Code comments must describe the behavior in generic terms (e.g. "on some GPUs hipMalloc returns true device memory; on others it returns UMA-mapped host-accessible memory"). Hardware-specific facts, perf snapshots, and reproduction details belong in `CLAUDE.md`, `docs/`, or commit messages — NOT in `.cpp`/`.h`/`.hip`/`.td`/`.mlir` comments. Rationale: arch names rot as new GPUs ship and the same code paths are exercised on more SKUs; comments that name one arch mislead readers on every other. This applies to new code AND to any existing comments touched in a PR.
- **MANDATORY: Code comments in generic compiler/dialect/pass code must not be model-specific.** Files under `lib/Conversion/`, `lib/Dialect/`, `lib/Compiler/`, `include/hip/`, and `backend-mlir-compiler/` implement generic MLIR passes / dialect ops / lowerings — their comments should describe the IR pattern, the invariant, or the spec quirk being handled, NOT the model that happens to trigger it today. Bad: "≤ 2 (Gemma-3 q/k_norm)", "this fires for Llama-8B's GQA path". Good: "≤ 2 in practice (typical case: same-rank dynamic Reshape pair around a norm op)", "fires when same-rank dynamic Reshape sees an integer-factor split on the last dim". Model-frequency notes / perf-impact-per-model / historical regressions belong in `CLAUDE.md` or commit messages.
  - **Exception**: it's OK to name a model when the comment is documenting a model-specific exporter quirk that genuinely cannot be described generically — e.g. "HF Gemma-3 export wraps every literal in `CastLike(<f32 const>, <f16 act>)`" is a real exporter behavior that future readers WILL need the model name to recognize. Rule of thumb: if removing the model name leaves the comment less clear to the next maintainer, keep it; if removing the name only loses a "this is what hit it for me" anecdote, drop it.
  - Tests (`test/lit/`, `test/python/`, `test/numeric/`) are exempt — naming the model under test is part of the test's identity.
- When an approach fails, revert immediately and completely — no partial experimental code left in the tree. Prefer runtime-only fixes (`lib/Runtime/real/`) over cross-cutting changes spanning compiler + interface + runtime. If a multi-layer fix doesn't work after one attempt, revert and reassess.
