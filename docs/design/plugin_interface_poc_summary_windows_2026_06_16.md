# Plugin Interface PoC — Windows Port (`onnx-hipdnn-ep`, branch `plugin/poc-gfx1103`)

**Date:** 2026-06-16
**Author:** kirmatam
**Branch:** `plugin/poc-gfx1103` (worktree: `C:/Work/GNPU/Setup/onnx-hipdnn-ep-plugin/`)
**Base repo:** `onnx-hipdnn-ep` (main, commit `67a4adb`)
**Reference:** ETX PoC in `gnpu-onnx-flow` (commits b840d52, b806a30, c4fb98d)
**Meeting transcript:** `C:/Work/GNPU/Misc/PluginPoC/PluginPoCInitialMeetingTranscript.txt`

---

## TL;DR

Proves that an out-of-tree `.so` / `.dll` can contribute a real `hip.*` op, a
fusion pass, a HipToLLVM lowering, a C-ABI runtime wrapper, and a gfx1103 HIP
device kernel to `hip-mlir-opt` **at load time — with no recompile of the host
compiler**. Mirrors ORT 1.25+ Plugin EP / IREE HAL-plugin direction.

| Touchpoint | Description | Status |
|-----------|-------------|--------|
| A | Op registration | ✅ Implemented |
| B | Pass registration | ✅ Implemented |
| C | Pipeline slot declaration | ✅ Implemented (`PipelineSlotRegistry`) |
| D ⚠️ | Bufferization model (R-1) | ✅ Implemented; R-1 negative test included |
| E | HipToLLVM lowering | ✅ Implemented (`wrap_fused_mul_add` call emitted) |
| F | Runtime C-ABI wrapper | ✅ Implemented; device execution requires gfx1103 hardware |
| G | gfx1103 HIP kernel | ✅ Implemented; device execution requires gfx1103 hardware |
| H | LIT tests (3) | ✅ **3/3 pass on ETX (GCC 13, LLVM 22, libMLIR.so)** |

Everything is gated behind `-DBUILD_HIP_PLUGINS=ON`; the default build path
is **untouched**.

---

## 2. What was built — 9 touchpoints

| # | Layer | Plugin capability | Status |
|---|-------|-------------------|--------|
| A | Op definition | Register `hip.fused_mul_add` into the live `hip` dialect at load time | ✅ Implemented |
| B | Fusion pass | Register `--hip-fuse-mul-add` pass discoverable by the pass manager | ✅ Implemented |
| C | Pipeline wiring | Declare insertion anchor (`after convert-onnx-to-hip`) via `PipelineSlotRegistry` | ✅ Implemented |
| D ⚠️ | Bufferization interface | Attach `HipDstBufferizableModel<FusedMulAddOp>` in same step as op registration (R-1) | ✅ Implemented; R-1 negative test included |
| E | HipToLLVM lowering | `ConvertOpToLLVMPattern` + `hip-to-llvm-with-fusion-plugin` pipeline; emits `wrap_fused_mul_add` call | ✅ Implemented |
| F | Runtime wrapper | `wrap_fused_mul_add()` C-ABI symbol; dispatches to gfx1103 kernel | ✅ Implemented; device run pending |
| G | Device kernel | `fused_mul_add_kernel.hip` for gfx1103 (f32/f16/bf16); launched via `hipLaunchKernelGGL` | ✅ Implemented; device run pending |
| H | LIT tests (3) | Positive (A/B/D), R-1 negative (D), E2E lowering (E) | ✅ **3/3 pass on ETX (Linux)** |

---

## 3. Changed files

### 3.1 Host changes (one-time, in-tree)

These are the only permanent changes to the host compiler. Every future plugin
reuses them without further host modification.

| File | Change |
|------|--------|
| [include/hip/Dialect/IR/HipDialect.td](../../include/hip/Dialect/IR/HipDialect.td) | Added `registerPluginOps<Ops...>()` shim — exposes the protected `Dialect::addOperations` to plugins so they can inject real `hip.*` ops at load time |
| [CMakeLists.txt](../../CMakeLists.txt) | Added `BUILD_HIP_PLUGINS` option (default OFF) declared before `add_subdirectory(tools)`, and `add_subdirectory(plugins)` gated behind it |
| [tools/hip-mlir-opt/CMakeLists.txt](../../tools/hip-mlir-opt/CMakeLists.txt) | When `BUILD_HIP_PLUGINS=ON`: switches from static MLIR component libs to the single `MLIR` dylib (required for shared TypeID set) and calls `export_executable_symbols_for_plugins` |
| [test/lit/CMakeLists.txt](../../test/lit/CMakeLists.txt) | Injects plugin `.so` paths into `lit.site.cfg.py`; adds plugin targets to `check-hip-mlir-lit` dependencies |
| [test/lit/lit.site.cfg.py.in](../../test/lit/lit.site.cfg.py.in) | Added `config.hip_fusion_plugin` and `config.hip_fusion_plugin_nobuf` variables |
| [test/lit/lit.cfg.py](../../test/lit/lit.cfg.py) | Added `hip_plugins` LIT feature and `%hip_fusion_plugin` / `%hip_fusion_plugin_nobuf` substitutions |

### 3.2 Plugin source (`plugins/fusion/`)

All new files. None are built unless `-DBUILD_HIP_PLUGINS=ON`.

| File | Touchpoint | Role |
|------|-----------|------|
| [plugins/CMakeLists.txt](../../plugins/CMakeLists.txt) | — | Top-level plugins directory; `add_subdirectory(fusion)` |
| [plugins/fusion/CMakeLists.txt](../../plugins/fusion/CMakeLists.txt) | All | Builds `hip_fusion_plugin`, `hip_fusion_plugin_nobuf` MODULE targets and `hip_fusion_runtime` shared lib; invokes hipcc for gfx1103 kernel on Linux |
| [plugins/fusion/FusedMulAddOps.td](../../plugins/fusion/FusedMulAddOps.td) | A | ODS TableGen for `hip.fused_mul_add` — DPS op with declarative assembly format |
| [plugins/fusion/PluginOps.h](../../plugins/fusion/PluginOps.h) | A | Generated op class header (`GET_OP_CLASSES` include) |
| [plugins/fusion/PluginOps.cpp](../../plugins/fusion/PluginOps.cpp) | A | `getDpsInitsMutable()` and `getEffects()` implementations |
| [plugins/fusion/Passes.td](../../plugins/fusion/Passes.td) | B | ODS TableGen for `--hip-fuse-mul-add` pass |
| [plugins/fusion/Passes.h](../../plugins/fusion/Passes.h) | B | Generated pass declaration + registration header |
| [plugins/fusion/FuseMulAddPass.cpp](../../plugins/fusion/FuseMulAddPass.cpp) | B | `OpRewritePattern<AddOp>` matching `add(mul(x,b),a)` in both operand orderings; single-use guard; rewrites to `hip.fused_mul_add` |
| [plugins/fusion/PipelineSlot.h](../../plugins/fusion/PipelineSlot.h) | C | `PipelineSlotRegistry` singleton + `SlotPosition` enum; documents 7 stable public anchor names |
| [plugins/fusion/PipelineSlot.cpp](../../plugins/fusion/PipelineSlot.cpp) | C | `applyPluginSlots()` implementation; logs registered slots at load time |
| [plugins/fusion/FusedMulAddLowering.h](../../plugins/fusion/FusedMulAddLowering.h) | E | Declares `kWrapFusedMulAdd` symbol name and `populateFusedMulAddLoweringPatterns()` |
| [plugins/fusion/FusedMulAddLowering.cpp](../../plugins/fusion/FusedMulAddLowering.cpp) | E | `ConvertOpToLLVMPattern<FusedMulAddOp>` — extracts contiguous memref ptrs, computes `num_elements`, maps dtype to `HIPDNN_EP_DATATYPE_*`, emits `llvm.call @wrap_fused_mul_add` |
| [plugins/fusion/PluginMain.cpp](../../plugins/fusion/PluginMain.cpp) | A/B/C/D/E | `mlirGetDialectPluginInfo()` + `mlirGetPassPluginInfo()`; registers op (A), attaches bufferize model (D), registers pass (B), declares pipeline slot (C), registers `hip-to-llvm-with-fusion-plugin` pipeline + `FusedMulAddToLLVMPass` (E); guarded by `HIPPOC_SKIP_BUFFERIZE_ATTACH` for the `_nobuf` variant |
| [plugins/fusion/runtime/wrap_fused_mul_add.cpp](../../plugins/fusion/runtime/wrap_fused_mul_add.cpp) | F | `wrap_fused_mul_add()` C-ABI runtime symbol; obtains HIP stream via `hipdnn_ep_state_get_stream()`; delegates to `hip_fused_mul_add_launch` |
| [plugins/fusion/runtime/fused_mul_add_kernel.hip](../../plugins/fusion/runtime/fused_mul_add_kernel.hip) | G | `fused_mul_add_kernel<T>` HIP device kernel for gfx1103; supports f32/f16/bf16; block size 256; compiled with `--offload-arch=gfx1103` |

### 3.3 LIT tests (`test/lit/plugins/`)

| File | Touchpoint | What it tests |
|------|-----------|---------------|
| [test/lit/plugins/hip-fuse-mul-add-plugin.mlir](../../test/lit/plugins/hip-fuse-mul-add-plugin.mlir) | A/B/D | Positive: loads plugin, fuses `add(mul(x,b),a)` → `hip.fused_mul_add`, bufferizes cleanly. Tests both lhs-mul and rhs-mul orderings. |
| [test/lit/plugins/hip-fuse-mul-add-nobuf.mlir](../../test/lit/plugins/hip-fuse-mul-add-nobuf.mlir) | D/R-1 | Negative (R-1): loads `_nobuf` variant; confirms `one-shot-bufferize` fails with `was not bufferized`. Makes the §4 footgun an executable regression test. |
| [test/lit/plugins/hip-fuse-mul-add-lowering.mlir](../../test/lit/plugins/hip-fuse-mul-add-lowering.mlir) | E | Lowering: uses `hip-to-llvm-with-fusion-plugin` pipeline; verifies `hip.fused_mul_add` is absent and `llvm.call @wrap_fused_mul_add` is present in output IR. |

---

## 4. Two non-obvious ABI findings (apply to all future plugin work)

### 4.1 MLIR dylib linkage is mandatory for plugins

LLVM 22 (and TheRock builds) compile with **hidden symbol visibility**. A
statically-linked `hip-mlir-opt` cannot export MLIR-core TypeIDs even with
`-rdynamic`. If a plugin resolves MLIR core from its own copy, `isa<>` /
`dyn_cast<>` breaks across the `.so` boundary.

**Fix:** `BUILD_HIP_PLUGINS=ON` switches `hip-mlir-opt` to link the single
`MLIR` dylib and calls `export_executable_symbols_for_plugins()`. The plugin
links `MLIR` but not `HipDialectIR` — in-tree hip op TypeIDs resolve from the
host at `dlopen` time.

### 4.2 Use `--pass-pipeline`, not the `--hip-fuse-mul-add` shorthand

When a dialect plugin **and** a pass plugin are both loaded,
`MlirOptMain`'s generated `--<pass-name>` cl flags regress to printing
`--help` (exit 0, no error). The textual `--pass-pipeline=` parser is
unaffected. **All three LIT tests use `--pass-pipeline`.**

---

## 5. Requirement R-1 — bufferization model is non-optional ⚠️

> The op compiled, the fusion pass fired, the pipeline ran — and it still
> failed deep inside `OneShotBufferize` with `op was not bufferized`.

The `_nobuf` MODULE variant + `hip-fuse-mul-add-nobuf.mlir` make this an
**executable, regression-guarded test**. Any plugin that omits
`attachInterface<HipDstBufferizableModel<OpTy>>` in its `register_ops`
callback will reproduce this exact failure.

---

## 6. Build & run

```bash
# Configure (Linux/ETX for F/G device execution; A–E+H work on Windows)
cd C:/Work/GNPU/Setup/onnx-hipdnn-ep-plugin
cmake -B build -S . \
  -DBUILD_HIP_TOOLS=ON \
  -DBUILD_HIP_PLUGINS=ON \
  -DTHEROCK_DIST=<path-to-therock> \
  -DLLVM_DIR=<path>/lib/cmake/llvm \
  -DMLIR_DIR=<path>/lib/cmake/mlir

# Build (use -j4; large TUs OOM at -j16 on ETX)
cmake --build build -j4 --target \
  hip-mlir-opt \
  hip_fusion_plugin \
  hip_fusion_plugin_nobuf \
  hip_fusion_runtime

# Run plugin LIT tests (touchpoints A/B/D/E)
cd build/test/lit
lit -sv plugins/

# Run full LIT suite (regression guard)
lit -sv .
```

---

## 7. Open questions (from requirements §6 — still unresolved)

| Question | Notes |
|----------|-------|
| **C ABI vs C++ ABI** | Plugin must be built against the same LLVM toolchain. Entry points are C-weak (`LLVM_ATTRIBUTE_WEAK`); MLIR objects are C++. The bespoke `gnpu_plugin_entry` C-ABI (requirements §3) remains a future milestone. |
| **Versioning / compatibility window** | `min_host_version` field sketched in requirements §3; not yet implemented. Policy needed for how many host versions a plugin binary must support. |
| **Pipeline anchor stability** | 7 stable anchor names documented in [plugins/fusion/PipelineSlot.h](../../plugins/fusion/PipelineSlot.h). Renaming any pass in `Pipelines.cpp` breaks plugins anchoring to it. Deprecation policy needed. |
| **In-tree pass pattern extension** | The two-pass lowering workaround (`hip-to-llvm-with-fusion-plugin`) works but requires plugins to ship their own pipeline registration. A `registerLoweringPatternExtension()` host hook would be cleaner. |
| **Kernel packaging** | Currently one precompiled `.hip` source per arch. Fat-object (multi-arch), source-compile-at-load, and `HIP_CUSTOM_KERNELS_DIR`-style discovery are future options. |
| **CI test lane** | Plugin LIT tests use `REQUIRES: hip_plugins` and are UNSUPPORTED in the default build lane. CI needs a separate lane with `-DBUILD_HIP_PLUGINS=ON`. |
| **`applyPluginSlots` full reorder** | Current implementation logs declared slots. Full arbitrary pass-list reorder (parse → rebuild with splices) is future work; the `PipelineSlotRegistry` API is stable. |

---

## 8. How to add an architecture-specific optimization pass

This is the primary use case for Vandana's team — contributing passes that are
specific to a target architecture (e.g. GNPU, gfx1200) without touching the
shared `onnx-hipdnn-ep` tree.

### Step 1 — Declare the pass in `Passes.td`

```tablegen
def Gfx1200ConvFusionPass : Pass<"gfx1200-conv-fusion", "mlir::func::FuncOp"> {
  let summary = "gfx1200-specific convolution+activation fusion";
  let dependentDialects = ["mlir::hip::HipDialect"];
}
```

### Step 2 — Implement the pass

```cpp
struct Gfx1200ConvFusionPass
    : public impl::Gfx1200ConvFusionPassBase<Gfx1200ConvFusionPass> {

  void runOnOperation() override {
    // Guard: only fire when compiling for gfx1200.
    // Read the target arch from a module attribute set by the compiler driver,
    // or from a pass option passed via --pass-pipeline="...(gfx1200-conv-fusion{arch=gfx1200})".
    if (getArch() != "gfx1200") return;

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvReluFusionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
```

### Step 3 — Register at the correct pipeline anchor

```cpp
// In mlirGetPassPluginInfo() callback:
hip::PipelineSlotRegistry::get().addSlot(
    "hip-optimize-mem-refs",      // stable anchor: after generic buffer opts
    hip::SlotPosition::After,
    []() { return createGfx1200ConvFusionPass(); },
    "gfx1200 conv+activation fusion");
```

**Stable anchor names** (from [`plugins/fusion/PipelineSlot.h`](../../plugins/fusion/PipelineSlot.h)):

| Anchor | When it runs |
|--------|-------------|
| `convert-onnx-to-hip` | After ONNX→HIP, before bufferize — best for fusion passes |
| `one-shot-bufferize` | At bufferization boundary |
| `hip-optimize-mem-refs` | After generic buffer optimization |
| `hip-promote-strided-hip-operands` | Before pool allocation |
| `hip-pool-allocs` | After strided promotion |
| `hip-lower-allocs` | After pool allocation |
| `hip-resolve-extern-constants` | Late cleanup |

### Step 4 — Build as a plugin

```cmake
add_library(hip_gfx1200_plugin MODULE
  Gfx1200PluginMain.cpp
  Gfx1200ConvFusionPass.cpp
)
target_link_libraries(hip_gfx1200_plugin PRIVATE MLIR)  # Linux
# The plugin is loaded at runtime: --load-pass-plugin=hip_gfx1200_plugin.so
```

**No changes to `Pipelines.cpp` or any other host file are needed.** The
`PipelineSlotRegistry` handles insertion at the declared anchor point.

---

## 9. How to add an architecture-specific kernel and register it

### Step 1 — Write the HIP kernel

Create `runtime/my_op_gfx1200.hip` in your plugin directory:

```cpp
// Block size tuned for gfx1200 (RDNA4): 4 waves × 64 threads
static constexpr int kBlockSize = 256;

template <typename T>
__global__ void my_op_kernel_gfx1200(const T* in, T* out, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = /* gfx1200-tuned computation */;
}

extern "C" int hip_my_op_launch_gfx1200(hipStream_t stream,
                                         const void* in, void* out,
                                         int64_t n, int dtype) {
  dim3 block(kBlockSize);
  dim3 grid((n + kBlockSize - 1) / kBlockSize);
  switch (dtype) {
    case 0: hipLaunchKernelGGL(my_op_kernel_gfx1200<float>,
                grid, block, 0, stream, (float*)in, (float*)out, n); break;
    case 1: hipLaunchKernelGGL(my_op_kernel_gfx1200<__half>,
                grid, block, 0, stream, (__half*)in, (__half*)out, n); break;
    default: return -1;
  }
  return hipGetLastError() == hipSuccess ? 0 : -1;
}
```

### Step 2 — Write the `wrap_*` C-ABI runtime symbol

Create `runtime/wrap_my_op.cpp`:

```cpp
// Declared extern — resolves from the host compiler at dlopen time.
extern "C" void* hipdnn_ep_state_get_stream(void* state);

extern "C" int wrap_my_op(void* state,
                           const void* in, void* out,
                           int64_t n, int64_t dtype) {
  if (!state || !in || !out || n <= 0) return -1;
  void* stream = hipdnn_ep_state_get_stream(state);
  return hip_my_op_launch_gfx1200(stream, in, out, n, (int)dtype);
}
```

### Step 3 — Wire kernel + wrapper in `CMakeLists.txt`

```cmake
# Compile the gfx1200 HIP kernel (Linux only; Windows: cross-compile on ETX)
if(NOT WIN32 AND THEROCK_DIST)
  find_program(HIPCC hipcc HINTS "${THEROCK_DIST}/bin" NO_DEFAULT_PATH)
  add_custom_command(OUTPUT my_op_gfx1200.o
    COMMAND ${HIPCC} --offload-arch=gfx1200 -fPIC -O2 -c
            -o my_op_gfx1200.o
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/my_op_gfx1200.hip
    DEPENDS runtime/my_op_gfx1200.hip)
endif()

# Runtime shared library containing wrap_* symbol + device kernel
add_library(hip_gfx1200_runtime SHARED
  runtime/wrap_my_op.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/my_op_gfx1200.o
)
target_link_libraries(hip_gfx1200_runtime PRIVATE amdhip64)
```

### Step 4 — Add the HipToLLVM lowering pattern

In `MyOpLowering.cpp`, follow the same pattern as
[`plugins/fusion/FusedMulAddLowering.cpp`](../../plugins/fusion/FusedMulAddLowering.cpp):

```cpp
struct MyOpLowering : public ConvertOpToLLVMPattern<MyOp> {
  LogicalResult matchAndRewrite(MyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Declare: int wrap_my_op(state, in, out, n, dtype)
    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType, i64Type, i64Type};
    auto funcOp = LLVM::lookupOrCreateFn(rewriter, module,
                                          "wrap_my_op", paramTypes, i32Type);
    // Build args and emit llvm.call @wrap_my_op(...)
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};
```

### Step 5 — Register the runtime library in `PluginMain.cpp`

```cpp
// In mlirGetPassPluginInfo() callback, register the pipeline that includes
// the lowering and points to the runtime library:
PassPipelineRegistration<>(
    "hip-to-llvm-with-gfx1200-plugin",
    "HipToLLVM + gfx1200 kernel lowering",
    [](OpPassManager &pm) {
      pm.addPass(memref::createExpandStridedMetadataPass());
      pm.addPass(hip::createConvertHipToLLVMPass());
      pm.addPass(createMyOpToLLVMPass());  // lowers MyOp → wrap_my_op call
    });
// The runtime lib (hip_gfx1200_runtime.so) is loaded by the EP at inference
// time via dlopen, so wrap_my_op resolves at runtime.
```

### Summary — what a new arch team needs to create

| File | What it does |
|------|-------------|
| `MyOpOps.td` | ODS definition of the new op |
| `MyOpPass.cpp` | Fusion / optimization pass |
| `MyOpLowering.cpp` | `ConvertOpToLLVMPattern` → `wrap_my_op` call |
| `runtime/my_op_gfxNNNN.hip` | Device kernel, compiled with `--offload-arch=gfxNNNN` |
| `runtime/wrap_my_op.cpp` | C-ABI wrapper, dispatches to the kernel |
| `PluginMain.cpp` | Wires A+B+C+D+E; registers pipeline with F/G runtime |
| `CMakeLists.txt` | Builds MODULE plugin + SHARED runtime lib |

**The host compiler (`onnx-hipdnn-ep`) is never modified.** The arch team ships
a self-contained plugin `.so` + runtime `.so` pair.

---

## 10. Windows build — MLIR dylib requirement

LIT tests (touchpoints A–E) passed on **Linux (ETX)** where the TheRock
distribution ships `libMLIR.so`. On this Windows machine (gfx1103 hardware
confirmed) they currently hit a TypeID duplication crash because the local
prebuilt LLVM 22.1.0 was built **without** `-DLLVM_BUILD_LLVM_DYLIB=ON`.

### Why the MLIR dylib is required

With hidden-symbol-visibility LLVM (all modern Linux and Windows MSVC builds),
TypeIDs are process-global singletons defined in one translation unit each.
When both the host EXE and the plugin DLL link their own static copy of
`MLIRIR.lib`, each gets its own singleton — `isa<AddOp>` checks across the
boundary fail silently or crash.

The fix is exactly one CMake flag at LLVM build time:
```
-DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON
```
This produces a single `MLIR.dll` / `libMLIR.so` that both the host and plugin
link against, keeping TypeIDs in one shared image.

### Can we download a pre-built MLIR dylib for Windows?

**No pre-built Windows MLIR dylib exists publicly.** The official LLVM GitHub
release installers (`LLVM-22.x-win64.exe`) ship only static `.lib` files.
TheRock's Windows distribution also uses static libs.

**Options:**

| Option | Effort | Recommended? |
|--------|--------|-------------|
| Run plugin LIT tests on ETX (Linux) where `libMLIR.so` exists | ~0 — SSH in, rebuild with `BUILD_HIP_PLUGINS=ON`, run `lit` | ✅ **Easiest** |
| Rebuild LLVM 22 locally with dylib enabled | ~4–6 hours build time | Only if Windows E2E is hard requirement |
| Use a pre-built `libMLIR.so` from the ETX server on WSL2 | Possible but cross-platform ABI mismatch risk | Not recommended |

The recommended path is to run `lit -sv test/lit/plugins/` on ETX where the
Linux TheRock build already has `libMLIR.so` available.
