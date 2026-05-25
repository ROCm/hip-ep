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
- **WMMA matmul_nbits swizzle must cover every (bx,by) tile exactly once.** `GemmFp16U4Impl` in `matmul_nbits_kernel.hip` maps `block_id → (bx,by)` with a column-swizzle for L2 locality. The mapping MUST be a bijection over `[0, n_tiles*m_tiles)`. The original implementation fell back to row-major only when `bx >= n_tiles`, which produces collisions AND missing tiles whenever `n_tiles % sw != 0` (e.g. n_tiles=45, sw=8: tiles `(bx=43..44, by=1)` uncomputed; `(bx=40..41, by=1)` double-computed). Uncomputed tiles inherit stale values from prior kernel launches; `matmul_nbits_add_bias_rowmajor` then RMW-adds bias to those stale values, producing run-to-run non-determinism that cascades through QMoE (different fc1/fc2 outputs → different routing M counts → autotune re-triggers every prefill chunk → 11× TTFT regression on GPT-OSS-120B/20B). Fix: split the grid into a "main" region of `(n_tiles/sw)*sw` cols (full swizzle) and a "tail" of `n_tiles%sw` cols (row-major). When modifying tile-mapping logic, verify bijection exhaustively.
- **`affine.apply` from `expand-strided-metadata` MUST be lowered before `convert-hip-to-llvm`** (`lib/Dialect/Transforms/Pipelines.cpp::buildHipToLLVMPipeline`). `memref::createExpandStridedMetadataPass` rewrites `memref.collapse_shape` / `memref.expand_shape` into `memref.reinterpret_cast` plus stride/offset arithmetic — and for collapse-shape stride products on multi-dim memrefs (canonical trigger: an MoE expert-major 3D buffer flattened to 2D for hipBLASLt) it emits `affine.apply` with a map like `()[s0, s1] -> (s0 * s1)`. `ConvertHipToLLVM` does NOT include affine→arith conversion patterns. Any surviving `affine.apply` leaves a `builtin.unrealized_conversion_cast` in the final LLVM IR and the translator aborts with `LLVM Translation failed for operation: builtin.unrealized_conversion_cast`. Fix: insert `createLowerAffinePass()` between `ExpandStridedMetadata` and `ConvertHipToLLVM` (and add `MLIRAffineToStandard` to `lib/Dialect/Transforms/CMakeLists.txt`'s `target_link_libraries`). **This is silent in production** — when compilation fails, MorphiZen falls back to CPU and any test that compares CPU-vs-CPU (e.g. accuracy tests using `compare_logits` against a CPU baseline) will pass `cosine == 1.0` and look healthy. Symptom is wall-clock: gpt-oss-20b decode at sq=1/kv=128 went from ~70 s (CPU fallback) → ~2 s (GPU) once the pass was added. Verify GPU dispatch with `HIPDNN_EP_DEBUG=1` (look for `[REAL] wrap_*` lines) BEFORE trusting accuracy numbers. Models that don't trigger collapse-shape on multi-dim memrefs (Llama 1B/8B as of 2026-05) will not exercise this path — expect new MoE / multi-expert / per-head-flatten models to surface this if the pass regresses.
- **`arith.ceildivsi` / `arith.floordivsi` MUST be expanded before `convert-hip-to-llvm`** (`lib/Dialect/Transforms/Pipelines.cpp::buildHipToLLVMPipeline`). MLIR's `populateArithToLLVMConversionPatterns` (used by `ConvertHipToLLVMPass` via `populateFinalizeMemRefToLLVMConversionPatterns`) does NOT include patterns for these ops — they survive the conversion as raw `arith.*` ops which then fail LLVM translation: `LLVM Translation failed for operation: arith.ceildivsi`. The dynamic-shape `wrap_range` fallback in `OnnxToHip` emits `arith.ceildivsi` for the Range output-length computation (`(limit - start) / delta` rounded toward +∞ when delta>0, similar with sign games for delta<0), so this trips on every Cat-C Range op. Fix: insert `arith::createArithExpandOpsPass()` between `LowerAffinePass` and `ConvertHipToLLVMPass`. The expand pass rewrites `ceildivsi` into basic `sdiv` + `mul` + `cmp` + `select` that the LLVM converter handles natively. Add `MLIRArithTransforms` to `lib/Dialect/Transforms/CMakeLists.txt`'s `target_link_libraries`. Same silent-fallback-to-CPU caveat as the `affine.apply` gotcha above.
- **`hipdnn_ep_get_pool_base` signature carries a required-size parameter and grows the pool on demand** (`lib/Runtime/hipdnn_ep_runtime_state.cpp` + `lib/Runtime/hipdnn_ep_runtime.h`). The MLIR lowering of `hip.get_pool(%ctx, %pool_size_index)` (in `lib/Conversion/HipToLLVM/MemoryLowering.cpp::GetPoolOpLowering`) generates `llvm.call @hipdnn_ep_get_pool_base(%state, %size) : (!llvm.ptr, i64) -> !llvm.ptr`. The runtime function MUST take this size and reallocate the pool when `required_size > state->pool_size`. The legacy 1-arg signature (returning `state->pool_base` populated by `hipdnn_ep_pool_init` at `inference_init` time) only worked when `hipdnn.pool_size` was a compile-time constant from `hip-pool-allocs`. The dynamic-shape Cat-C path emits `hip.get_pool` with a runtime-computed size for the pooled view over a `wrap_*_dyn`-allocated buffer, and at `inference_init` time `hipdnn.pool_size = 0` (no static allocs). Without grow-on-demand the call returns `nullptr` and any downstream `memref.view` over the pool yields a null `alignedPtr` — passed to `wrap_constant_of_shape_dyn` / etc. as `shape=NULL`, which dies in the early null-check. The base pointer can change across calls (the function `hipFree`s the old buffer when growing), so consumers must re-query on every entry; this is the contract the lowering already follows (no SSA value of pool base survives across `Compute()` boundaries).
- **`hip-pool-allocs` now pools every alloc (no min-count short-circuit)** (`lib/Dialect/Transforms/PoolAllocs.cpp::runOnOperation`). The pass used to fall through when the function had fewer than 2 `memref.alloc`s — but that left those single allocs to flow through `LowerAllocs` → `hip.alloc` → `MemoryLowering`, which lowers to a global `hip_device_malloc` symbol the runtime does NOT export. Pooling unconditionally (even a single alloc) ensures the dyn-pool path is the only allocation path compiled models use. The LIT test `test/lit/Dialect/hip-pool-allocs.mlir::single_alloc_noop` was updated to check for `hip.get_pool` + `memref.view` instead of the previous `CHECK-NOT: hip.get_pool`.
- **`ComposeDimSpecs` must scan the function body for slot publishers, not just output `RuntimeSlot` leaves** (`lib/Dialect/Transforms/ComposeDimSpecs.cpp`). The `hipdnn.dyn_dim_slots_count` module attribute is read by both `GenerateInterface` (to size the per-state `dyn_dim_slots` table at `inference_init`) and the runtime's `hipdnn_ep_state_publish_dim` (to range-check the slot id and `LOG(FATAL)` on overflow). The original implementation only counted slots referenced from function-output `RuntimeSlot` leaves under `hipdnn.output_dim_specs`. This breaks for compositions where a Category-C op claims a slot but its dim is consumed *internally* by a downstream op that converts it to a static-typed value (canonical example: `Shape(NonZero(X))` — `NonZero` publishes slot 0, but `Shape`'s output is a static `[2]xi64` so no `RuntimeSlot` reaches the function output). Symptom: `[hipdnn_ep_state_publish_dim] slot_id 0 out of range [0, 0)` abort at the first `wrap_nonzero` call. Fix: walk the entire body with `mainFunc.getBody().walk([&](Operation *op){...})` collecting `slot_id` (IntegerAttr) and `slot_ids` (DenseI32ArrayAttr) attributes BEFORE the output-spec loop runs, so `maxSlotIdSeen` covers every publisher whether or not its slot is plumbed through to a function output.
- **`onnx.Shape` has two distinct lowering paths — keep them both** (`lib/Conversion/OnnxToHip/ShapeConversion.cpp`). The static-input fast path (`ShapeToConstant`, benefit=2) rewrites the op to a single `arith.constant` of i64 dims — every `fix_shapes()`-converted test model hits this. The dynamic-input path (`ShapeToHip`, benefit=1) emits `hip.shape` with one DimSpec per output element; `ShapeLowering.cpp` then materialises each DimSpec via `DimSpecMaterializer` (constants for `Static` leaves, `MemRefDescriptor::size` reads for `InputDim`, `hipdnn_ep_state_read_dim` calls for `RuntimeSlot`), packs the i64s onto the host stack via `llvm.alloca`, and hands the buffer to `wrap_shape` which does a single `hipMemcpyAsync` H2D into the destination GPU memref. DimSpec `IntegerAttr` reads MUST use `getValue().getSExtValue()` not `getInt()` — the ONNX importer emits `si64` (signed) attributes and `IntegerAttr::getInt()` asserts on non-signless types (`assertion failed: getType().isIndex() || getType().isSignlessInteger() && "must be signless integer"`). Same fix applied in `ConstantOfShapeConversion.cpp` for the `start`/`end` reads. `Shape` is the only HIP op today that consumes a DimSpec via the `element_dim_specs` ArrayAttr — adding more "DimSpec-as-runtime-value" ops should follow the same pattern (DimSpec serialised to ArrayAttr at conversion, materialised to i64 SSA at HipToLLVM via the shared `materializeDimSpec` helper).
- **Subprocess pytest that spawns the EP MUST drain stdout asynchronously — Windows pipe buffer = 64 KB, EP emits multi-MB per inference.** Any test that uses `subprocess.Popen(..., stdout=subprocess.PIPE, stderr=subprocess.STDOUT)` to run a pytest-in-pytest harness against the MorphiZen EP (canonical examples: `test/numeric/tests/test_debug_surface.py`, `test/numeric/tests/test_backcompat.py`) MUST continuously drain `proc.stdout` from a daemon thread; calling `proc.communicate()` only after the polling loop deadlocks the subprocess on the first write that fills the pipe. We measured a 4 s Sigmoid EP test serialising at **119 s in the wrapped subprocess** (30× slowdown) because of this exact bug. Fix pattern: spawn a `threading.Thread(target=lambda: out.extend(proc.stdout), daemon=True)` immediately after `Popen`, kill+join the thread on exit. Do NOT use `subprocess.run(capture_output=True)` either — it works on small outputs (its internal drain threads are correct) but if the surrounding test needs to interleave any work (e.g. polling `%TEMP%` for emitted DLLs in `test_backcompat.py`'s `_TempDllSnapshot`), the `subprocess.run` blocks entirely until exit. Pattern: explicit `Popen` + daemon-thread drain + polling loop in parallel.
- **`getResultDimSpec` consults a per-op DimSpec builder registry (2026-05) so `Shape(hip.transpose(?))`, `Shape(hip.add(?,?))`, etc. lower without every conversion having to pre-attach an `output_dim_specs` attribute.** Resolution order in `shape_interface::getResultDimSpec` (`lib/Dialect/IR/HipShapeInterface.cpp`): (1) explicit `output_dim_specs` attribute, (2) registered per-op builder, (3) static MLIR dim. Builders are registered from `HipDialect::initialize()` via `populateBuiltinDimSpecBuilders` (idempotent `std::call_once`). Two builders cover most "shape behavior" classes today: `buildTransposeDimSpec` (rank-preserving permutation: output dim `d` ← `resolveDimFromValue(operand, perm[d])` — registered for `hip.transpose`) and `buildBroadcastDimSpec` (NumPy-style broadcast: walks data operands, skips statically-broadcast-1 dims, returns the first resolvable DimSpec — registered for `hip.add/sub/mul/div/min/equal/less/and/mod/where/miopen.add/not/neg/cos/sin/sign/silu/sigmoid/softplus/gelu/reciprocal/sqrt/cast`). **Adding a new shape-behavior op is one registration call** — see the two builders for the pattern. **Before**: `ShapeToHip` notify-match-failed for any dynamic input dim whose producer was a rank-preserving HIP op without an explicit attribute; the `onnx.Shape` op survived into bufferize with `error: op was not bufferized` and the EP refused partition. **After**: the Qwen embedding subgraph (`NonZero → Transpose → Shape → Gather → Unsqueeze → Slice → ScatterND`) compiles, links, and dispatches every op on the GPU. LIT coverage: `test/lit/Conversion/onnx-to-hip/test_shape_dyn_via_passthrough.mlir`; end-to-end coverage: `test/numeric/tests/test_nonzero_qwen_embedding.py` (all 3 scenarios — N=0, N=4096 / K=1, N=16384 / K=4 — now pass bit-exact against ORT-CPU, see "Category-C consumer slot wiring" and "NonZero must produce row-major output" gotchas below). When extending the registry, prefer registering an existing builder for a new op name (`buildBroadcastDimSpec` already covers any pure-elementwise op regardless of arity) rather than writing a new builder; only write a builder when the shape semantics genuinely differ (e.g. transpose's permutation, future shape-transforming ops like reduce/concat).
- **`resolveDimFromValue` MUST walk back through DPS writers, not just the defining op, after bufferization** (`lib/Dialect/IR/HipShapeInterface.cpp`). After `BufferResultsToOutParams` + `hip-pool-allocs`, every HIP op operand is a `memref` whose `getDefiningOp()` is `memref.alloc` / `memref.view` (neither carries a DimSpec). The DimSpec walk would silently break at the first bufferized hop: `Transpose(NonZero(...))` queries `resolveDimFromValue(transpose_input, srcDim)` → producer = `memref.alloc` → empty spec → caller falls back to "no slot" and reads the upper-bound from the descriptor. Fix: when `v` is a memref, prefer the DPS-init writer (the unique HIP-dialect op that uses `v` as a DPS output) over the defining op, so the `RuntimeSlot` leaf attached to e.g. `hip.nonzero`'s `output_dim_specs` reaches downstream consumers' lowerings transitively. Helper is the static `findDpsWriter` in the same file; only triggers for memref operands and only when the DPS-writer-side spec is non-empty, so it is a no-op for pre-bufferize IR and any case the old behavior already handled.
- **Category-C consumer wiring: `hip-annotate-input-dim-slots` + slot-aware HipToLLVM lowerings** (`lib/Dialect/Transforms/AnnotateInputDimSlots.cpp`, `lib/Conversion/HipToLLVM/HipToLLVMUtils.h::extractContiguousMemRefPtrWithSlot` + `getMemRefDimSizeWithSlot`). Category-C producers (canonically `hip.nonzero`) publish their data-dependent dim AND a freshly-allocated exact-size buffer to a runtime slot — they do NOT write into their DPS-init upper-bound buffer at all. Without the consumer-side fix, every downstream op would (a) read garbage from the uninitialised upper-bound buffer and (b) use the upper-bound dim sizes from the descriptor, dispatching kernels with the wrong element count (e.g. Transpose writing `upper_bound * R` elements instead of the actual `N * R`). The pass runs once between `ComposeDimSpecsPass` and `ConvertHipToLLVMPass` in `buildHipToLLVMPipeline`; it walks `@main_graph`, finds the DPS writer of every operand (via `DestinationStyleOpInterface::getDpsInits`), and attaches **two** attributes on each consumer that has at least one slot-resolved dim:
 - `hipdnn.input_dim_slots`: per-operand list of `[dim_idx, slot_id]` pairs — consumers' SHAPE arithmetic uses `hipdnn_ep_state_read_dim(slot)` instead of `extractvalue %desc[3, d]` for those dims.
 - `hipdnn.input_slot_buffers`: per-operand i32 (-1 = use descriptor pointer) — consumers' POINTER reads use `hipdnn_ep_state_peek_buffer(slot)` instead of `extractvalue %desc[1]` for direct-consumer-of-publisher operands (i.e. when the operand's *immediate* writer is the slot publisher itself, not a translucent propagator).
 - The split matters: a propagator like Transpose writes into its OWN upper-bound DPS-init (in its first-N prefix), so its consumers need shape rewiring but NOT pointer rewiring. The annotation pass distinguishes by checking `producer->getAttrOfType<IntegerAttr>("slot_id")` — only direct producers get a non-negative entry in `hipdnn.input_slot_buffers`. The pointer helper uses the *peek* (non-aborting) `hipdnn_ep_state_peek_buffer` so N=0 cases (where the publisher legitimately publishes a null buffer because there's nothing to allocate) don't crash; the kernel sees a null pointer + 0 element count and short-circuits at the dispatcher.
 - Lowerings updated to use these utilities: `TransposeLowering`, `ScatterNDLowering`. `ShapeLowering` is already slot-aware via `DimSpecMaterializer` (it lowers from `element_dim_specs`, not from operand descriptors). `SliceLowering` did NOT need changes for the Qwen embedding case because the Slice output bound matches the dynamic shape exactly there — re-evaluate if a future model has Slice consuming a slot publisher.
 - LIT coverage: `test/lit/Dialect/hip-annotate-input-dim-slots.mlir`. Numeric coverage: `test/numeric/tests/test_nonzero_qwen_embedding.py` (3 scenarios, bit-exact).
- **NonZero kernel MUST produce row-major-ordered output to match `numpy.nonzero` / ONNX spec** (`3rd-party/custom_kernels/hip/nonzero_kernel.hip`). The original kernel used `atomicAdd` for slot assignment which produced non-deterministic per-warp ordering. Per-element VALUES are still correct (each non-zero element gets one slot with its correct per-axis coords), but the SET-to-POSITION mapping is scrambled. Downstream consumers where the (indices, updates) pairing is order-sensitive then write the right VALUES to the wrong POSITIONS — passes `cosine ≈ 0.99` and `max/min/mean stat-match` but element-wise diff is large. Canonical trigger: the Qwen embedding subgraph `NonZero → Transpose → ScatterND(data, indices, updates)` where `updates` comes from `Slice(Reshape(features))` in deterministic flat order while `indices` comes from the scrambled NonZero output → scatter writes `features[i]` at the wrong row position for every `i`. Fix: replace the atomic-allocate fill with a three-pass deterministic scan: (1) `hip_nonzero_blockscan_kernel<T>` does per-block exclusive scan over the mask (BS=64 = RDNA wave size, in-block shared-memory Hillis–Steele), (2) `hip_nonzero_blocksum_scan_kernel` does a single-thread sequential exclusive scan over the per-block sums, (3) `hip_nonzero_fill_kernel<T>` writes `output[a * N + (block_offset[bx] + thread_off[tid])]`. The wrapper (`lib/Runtime/real/nonzero.cpp`) now allocates two dyn-pool scratch buffers: `thread_off_dev` (`num_elements * sizeof(int32_t)`) and `block_sums_dev` (`ceil(num_elements/64) * sizeof(int32_t)`). The legacy `atomic_idx` param is retained in the kernel ABI for source compatibility but unused. Tests: `test/numeric/tests/test_nonzero.py` (all 13 scenarios), `test_nonzero_composition.py`, `test_nonzero_qwen_embedding.py` — all pass bit-exact.
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
- **OGA gives tight attention_mask, pure ORT gives padded.** OGA sets `attention_mask.shape[1]` to the actual token count (e.g., `[1,7]` for a 7-token prompt, `[1,8]` after one decode step). Pure ORT tests typically pad to `max_seq_len`. Since DimSource maps `total_sequence_length` to `attention_mask` (first input defining that dim_param), this difference causes DimSource to resolve different shapes for present KV outputs.
- **OGA `generate_next_token()` syncs the PREVIOUS step.** The call is async for the current step but synchronizes the previous dispatch before starting new work. So the 1st call dispatches prefill (returns immediately), the 2nd call syncs prefill + dispatches decode 1 (wall time = TTFT), and calls 3+ each sync one decode step (steady-state tps). Neither `get_next_tokens()` nor `get_sequence()` provides a GPU sync point.
- **`model_benchmark.exe` for fixed-shape pipeline directories needs `-ml <KV_LEN>`.** For fixed-shape pipelines (e.g. `Llama-3.1-8B-Instruct-awq-g128-int4-Pipeline-p512m16384`) you MUST pass `-ml <KV_LEN>` (e.g. `-ml 16384`) to override `model_benchmark`'s default `max_length = prompt_length + generation_length` — otherwise the KV-cache buffers OGA pre-allocates won't match the ONNX's static `total_sequence_length` and prefill bind fails with `Got invalid dimensions for input: past_key_values.0.key Got: 256 Expected: 16384`.
- **cmd.exe `set X=value && next` captures trailing whitespace into the value.** When invoking `model_benchmark.exe` (or anything else that reads `THEROCK_DIST`) from a chained cmd line, **always quote the assignment**: `set "THEROCK_DIST=C:\...\install\therock" && model_benchmark.exe ...`. Without quotes, the value becomes `C:\...\install\therock ` (trailing space before `&&`), and `CompilerDriver` builds the search path as `C:\...\install\therock /lib` — lld-link fails to open `amdhip64.lib`/`MIOpen.lib` and the EP silently falls back to CPU. Same trap applies to `set "PATH=...;%PATH%"`. (PowerShell does not have this issue.)
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
| Llama-3.1-8B-Instruct-awq-g128-int4 | AWQ INT4 (block_size=128, symmetric / no zero_points) | 32 | 8 | 128 | ~5.0 GB | [amd/Llama-3.1-8B-Instruct-awq-g128-int4-onnx-directml](https://huggingface.co/amd/Llama-3.1-8B-Instruct-awq-g128-int4-onnx-directml) |
| Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml | AWQ INT4 (block_size=128, **asymmetric** / with `zero_points` packed uint8 nibbles) | 32 | 8 | 128 | ~5.0 GB | [amd/Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml](https://huggingface.co/amd/Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml) |
| gpt-oss-20b (webgpu int4-rtn-block-32) | MXFP4 MoE (32 experts × 4 active per token) + fp16 attention | 24 (alternating sliding/full attention, sliding_window=128) | 8 (HPG=8) | 64 | ~12.5 GB | [onnxruntime/gpt-oss-20b-onnx](https://huggingface.co/onnxruntime/gpt-oss-20b-onnx/tree/main/webgpu/webgpu-int4-rtn-block-32) |

**Note:** The 8B model has `position_ids` as an additional input (not present in the 1B model). The sym 8B MatMulNBits ops have `block_size=128` and **no zero_points input** (symmetric AWQ quantization) — the kernel uses default zp=8.0 in that path. Both block_size values (32 and 128) are covered by the runtime autotune (`BLOCK_SIZE ∈ {32,64,128,256,512,1024}`); switching block_size triggers a fresh autotune on first call but is otherwise transparent.

**gpt-oss-20b structural notes (distinct from the Llama family).** Mixture-of-Experts backbone (32 experts, 4 active per token, MXFP4-quantized expert weights — **not** MatMulNBits/AWQ INT4), `attention_bias=true`, alternating sliding/full attention layers (sliding_window=128 baked into the graph), YaRN rope scaling, vocab=201088, hidden=2880. **No `position_ids` input** — rope is driven internally by the graph. GPU dispatch was previously failing silently (CPU fallback) until the `lower-affine` pipeline gotcha (above) was fixed — the MoE expert-major 3D→2D flatten in qmoe is what triggered the bad `affine.apply` survivors.

**Sym vs asym 8B model identity (AVOID PHANTOM-BUG CONFUSION).** The sym dyn 8B (`Llama-3.1-8B-Instruct-awq-g128-int4`) is the **Instruct** fine-tune; the asym dyn 8B (`Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml`) is the **base** Llama-3.1-8B. Different post-training. They produce **different greedy generations** for the same prompt — and the base model in particular tends to echo / repeat / wander on chat-template-style prompts (e.g. "Write a paragraph about Paris" → may emit garbled tail). This is **NOT a kernel bug**; it's normal base-model behavior on instruction-formatted prompts. Verification protocol when an output looks "broken": (a) run a per-step CPU-vs-EP logit comparison test — if it passes, the kernel is correct; (b) compare CPU-only vs CPU-only output of the same model with the same prompt — if CPU also produces "broken" English, the model is the cause.

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

**IOBinding with device memory for MorphiZen EP tests:** MorphiZen EP perf tests use `run_timed_iobinding(..., use_device_memory=True)` which allocates KV cache OrtValues via the EP's `hipHostMalloc` GPU allocator (`device_type="gpu", vendor_id=0x1002`). The runtime sees `memory_type == TENSOR_MEMORY_GPU` and aliases the buffer directly (zero-copy). The same OrtValue is bound to both past KV input and present KV output, so `past_key_gpu == present_key_gpu` — GQA skips the per-layer D2H + `hipStreamSynchronize` stall.

**KV cache shape convention:** Both `past_sequence_length` and `total_sequence_length` are set to `max_seq_len` (128) in the dim map. This makes past and present KV tensors the same shape, enabling the memory_type aliasing fast path in the runtime.

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

### Verified perf snapshot — gfx1151 (Strix Halo, 16 CUs), 2026-05-08

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
| gpt-oss-20b (webgpu int4-rtn b32) | 324.4 / **76.4** | 1116.9 / **72.4** |
| gpt-oss-120b (uint4 pergroup-asym AWQ) | 5.6 / **35.7** | 128.2 / **35.1** |
| DeepSeek-R1-Distill-Llama-70B (AWQ b128) | 22.6 / **5.5** | 102.0 / **5.2** |

### Known limitations (open / accepted)

- **L=256+ TPS cliff on Llama-8B family** — see "L=256+ TPS cliff" gotcha. Vulkan-style adaptive K_SPLITS does not help. Likely needs a fused single-pass online-softmax kernel or per-shape K autotune.
- **DeepSeek-R1-70B, Qwen2.5-14B, gemma3-4b** fall off flash_decode (table above). Decode at long context will use the legacy fused_decode (capped at total_seq=256 — beyond that, decomposed hipBLASLt). Adding the missing instantiations is "low-hanging fruit" but unverified.
- **OGA per-token IoBinding rebind overhead** scales with `max_length`. Not fixable from the EP side; documented gotcha. Pure-ORT IoBinding benches do not see this.
- **Compiler requires fully static shapes.** No `batch_size` / `sequence_length` symbols. `fix_shapes()` in `test/python/conftest.py` handles this for the perf tests; the dynseqlen branch (off this stage-1) is the long-term solution.
- **Linker byproducts** (`.lib`, `.pdb`, `.exp`) accumulate in `%TEMP%` — `CompilerDriver::cleanupIntermediates()` only removes `.ll`/`.obj`. Cosmetic.
- **Stale model.dll cache** is keyed on the ONNX hash, not the runtime version. Any runtime/.cpp or kernel/.hip change MUST be followed by `del %TEMP%\morphizen_mlir_*` or compiled DLLs will silently use the old bitcode.

### Future improvements (ranked by expected impact)

1. **Fused single-pass attention decode** that eliminates the reduce kernel — would unblock the L=256+ cliff and remove the K_SPLITS guesswork entirely. High effort, high impact.
2. **Add `<128, K, 5>` flash_decode instantiations** for Qwen2.5-14B / Coder-14B. Medium effort (HPG=5 is unusual — verify thread/LDS budget).
3. **Per-shape K_SPLITS autotune** keyed on `(D, HPG, B*G, skv_bucket)` — lower-risk than (1).
4. **Profile-OFF investigation of flash_decode at L=256–1024 on Llama-8B** — the cliff is launch/scheduling overhead, not per-op cost.
5. **Lower `kFlashDecodeMinSkv`** (currently 256) and benchmark — short-context flash_decode might already win on gpt-oss-20b's 12 sliding-window layers (window=128 → effective skv ≤ 128).
6. **Extend `cleanupIntermediates()`** to remove `.lib/.pdb/.exp` byproducts. Trivial.

## Dynamic-shape Debugging

Three env vars (and one always-on sanity-check pair) cover the data-dependent dynamic output shape feature. Each one is zero-overhead when off (`static const bool` cache + branch-prediction).

| Env var | What it does | Where to look | Typical use |
|---|---|---|---|
| `HIPDNN_EP_DEBUG_SHAPES=1` (alias: `HIPDNN_EP_TRACE_SHAPES`) | At EP session open, dumps every output's DimSpec tree (`[CTor] Output[i]: …`). Per `Compute()`, dumps the resolved shape both pre- and post-Compute (`[EP] Output[i] pre-/post-compute resolved shape=[…]`). The resolver also traces each DimSpec leaf (`[Resolver] InputValueI64(…)` / `[Resolver] Add(…)`). | EP side (`MlirCustomOp.cpp`, `DimSpecResolver.cpp`) | When an output's dim resolves to a surprising value — gives you `(declared spec) -> (per-leaf resolution) -> (final dim)` in one log. |
| `HIPDNN_EP_TRACE_SLOTS=1` | Traces every dynamic-output slot ABI call inside the model.dll (`[Slots] publish_dim(N) = V`, `read_dim(N) = V`, same for `publish_buffer` / `read_buffer`). | Runtime side (`hipdnn_ep_runtime_state.cpp`) | Confirm that a producer wrap (e.g. `wrap_nonzero`) publishes BEFORE a consumer reads, or localize a read-before-publish abort by name. |
| `HIPDNN_EP_VALIDATE_SHAPES=1` | **Gate exists, intentionally not wired.** `debug_log.h::hipdnn_ep_validate_shapes_enabled()` returns true when set but no call site reads it — setting it today is a no-op. Designed to do post-Compute() shape-only comparison against an ORT CPU reference run of the same fused subgraph; deferred because the existing numeric suite already catches DimSpec / resolver bugs as numeric or fatal divergence, and the framework cost is high relative to the residual coverage gain. See [docs/design/dynamic-shape-debug-surface.md § Deferred](docs/design/dynamic-shape-debug-surface.md#deferred-hipdnn_ep_validate_shapes) for what would trigger reviving it. | n/a yet | Reserved. |

**Always-on sanity checks (no env var; cannot be disabled).** The runtime's `hipdnn_ep_state_read_dim` and `_read_buffer` `LOG(FATAL)` (via `fprintf(stderr) + std::abort`) on:
- slot id out of range — model metadata is inconsistent with the wrapper that called it; almost always a stale `morphizen_mlir_*` cache after editing the runtime / a new wrapper without `getRuntimeFuncSpecs()` entry / hand-rolled bitcode skew. **Always `del %TEMP%\morphizen_mlir_*` after a runtime change.**
- read-before-publish — the producing Category-C wrap either didn't run, or ran but bailed out before `publish_dim` / `publish_buffer`, or the lowering ordered the read ahead of the write (ComposeDimSpecs ordering bug). The abort message names the offending slot id; the crash-handler stack trace names the consumer.

**`_read_*` vs `_peek_*` split** (`hipdnn_ep_runtime.h`). The runtime exposes two flavors of the slot read ABI:
- **`hipdnn_ep_state_read_*`** is the aborting flavor used by in-DLL Category-C consumer wraps (alloc-site `read_dim` materialised by `DimSpecMaterializer`).
- **`hipdnn_ep_state_peek_*`** is the silent flavor (returns `kDynSlotUnpublishedSize` / nullptr on miss). Only the EP-side post-compute resolver shim should use it, via the `inference_dyn_slot_get_dim` / `_get_buffer` exports — `GenerateInterface` re-binds those shims to the `_peek_*` symbols. This lets `resolveOutputShapePostCompute` print a model-level `LOG(FATAL)` ("Output i, dim d, slot s never published") before any in-DLL abort fires.

When adding a new dynamic-output ABI exporter, default to `_read_*` (loud); only call `_peek_*` from EP-side probing code.

### Offline DimSpec inspection (compiler-side + DLL-side)

Three pretty-printers / analyses share a single textual form (the same one produced by `DimSpec::toString()` and by the EP-side `HIPDNN_EP_DEBUG_SHAPES` tracer): a tree spelled root-first, e.g. `mul(arg[0].shape[0], slot[3])`, `floordiv(arg[1].i64[0], 8)`, or a bare leaf `slot[0]` / `arg[0].shape[2]` / `16`. Three layers, one language — diff a compiler dump against an EP resolver trace without reformatting.

| Tool | Inspects | Use when |
|---|---|---|
| `hip-mlir-opt --hip-dump-dim-specs` | Every per-op `output_dim_specs` / `element_dim_specs` attribute AND the module-level `hipdnn.output_dim_specs` produced by `--hip-compose-dim-specs`. Writes to stderr; does not modify IR. | Auditing a new `OnnxToHip` conversion that produces dynamic-shape results, or diagnosing why `ComposeDimSpecs` resolved a tree differently than expected. Combine with `--split-input-file` to dump multiple test cases in one run. |
| `hip-mlir-opt --hip-verify-dim-specs` | Same scope as the dumper, but as an assertion: every tree must satisfy `DimSpec::verify()` (well-formed, no cycles, leaf/binary arity correct), the module-level outer rank must match `hipdnn.output_shapes`, every dynamic (`-1`) output dim must have a non-empty entry (catches silent composition failures), and every `RuntimeSlot.slot_id` must be `< hipdnn.dyn_dim_slots_count`. Signals pass failure on the first violation. Prints `hip-verify-dim-specs: OK` on success. | As a final pre-emit gate in a custom pipeline, or as a regression check after editing `ComposeDimSpecs.cpp`. |
| `hip-inspect-dll --dim-specs <model.dll>` | Walks the FlatBuffers metadata blob baked into a compiled model DLL via `inference_get_metadata_json`, then renders each output's per-dim DimSpec using the same textual form. Also surfaces `dyn_dim_slots_count`. Operates on JSON only — no HipDialect linkage required. | Sanity-checking a freshly cached `%TEMP%\morphizen_mlir_*.dll`: confirm dynamic dims are present, RuntimeSlot ids are in range, and the dim tree matches what the EP resolver expects. |

LIT coverage for the two MLIR passes lives in `test/lit/Transforms/dim-specs-debug.mlir`. Both passes are analysis-only (`runOnOperation` never mutates IR) so they're safe to chain anywhere in a debug pipeline without changing downstream behaviour.

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
- Use `onnxruntime-genai-directml` (not plain or CUDA variant) when installing genai tools.
- **MANDATORY:** Do not use `.claude/memory`. All persistent knowledge belongs in this file or `docs/`.
- **Comments on non-obvious code are mandatory.** When adding or fixing code whose behavior isn't self-explanatory — especially workarounds, spec quirks, or subtle correctness invariants — add a short comment explaining *why*. Examples: ONNX convention differences (`local_window_size=-1` meaning "disabled"), shared-buffer detection rationale, or why a condition uses `<=` instead of `==`. Don't comment obvious code; do comment anything a reader might question.
- When an approach fails, revert immediately and completely — no partial experimental code left in the tree. Prefer runtime-only fixes (`lib/Runtime/real/`) over cross-cutting changes spanning compiler + interface + runtime. If a multi-layer fix doesn't work after one attempt, revert and reassess.
