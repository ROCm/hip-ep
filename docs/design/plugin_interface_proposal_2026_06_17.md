# Plugin Interface Proposal — `onnx-hipdnn-ep`

**Date:** 2026-06-17
**Author:** kirmatam
**Branch:** `plugin/poc-gfx1103` (draft PR pending review)
**Status:** PoC complete; proposing merge after team review

---

## 1. Problem

Today, adding a custom op + fusion to the `hip` dialect requires editing
**9 in-tree locations across 7 layers** and rebuilding the whole compiler.
There is no way for a downstream team (GNPU, NPI, partner) to contribute an
op, a fusion pass, a lowering, and a kernel without forking the repo or
syncing against upstream every time main moves.

---

## 2. Proposed solution — one permanent host hook, everything else out-of-tree

Add **one small, op-agnostic shim** to `HipDialect`:

```tablegen
// HipDialect.td — the only permanent in-tree change
template <typename... Ops>
void registerPluginOps() { addOperations<Ops...>(); }
```

This exposes the otherwise-protected `addOperations` to plugins.
Every future team reuses this same shim — no further host changes needed.

A plugin is a `.so` / `.dll` that implements two stock MLIR entry points:

```cpp
mlirGetDialectPluginInfo()  // loaded via --load-dialect-plugin
mlirGetPassPluginInfo()     // loaded via --load-pass-plugin
```

No bespoke plugin loader, no new `PluginAPI.h` — we reuse the MLIR
plugin infrastructure that LLVM itself uses for out-of-tree targets.

---

## 3. What the PoC proves (all 9 touchpoints)

The PoC uses `hip.fused_mul_add` (`b*x + a`) as the example op.
It is deliberately **distinct** from the in-tree `hip.fused_add_mul`
so there is zero risk of collision.

| Touchpoint | What the plugin contributes | How |
|---|---|---|
| **A** — Op registration | `hip.fused_mul_add` as a real `hip.*` op | `HipDialect::registerPluginOps<>()` shim |
| **B** — Pass registration | `--hip-fuse-mul-add` fusion pass | `mlirGetPassPluginInfo` → `registerHipFuseMulAddPass()` |
| **C** — Pipeline wiring | Declares anchor `after convert-onnx-to-hip` | `PipelineSlotRegistry` (7 named stable anchors) |
| **D ⚠️** — Bufferization model | Attaches `HipDstBufferizableModel<FusedMulAddOp>` at load time | Same `register_ops` call — non-optional (R-1) |
| **E** — HipToLLVM lowering | `ConvertOpToLLVMPattern` → `llvm.call @wrap_fused_mul_add` | `hip-to-llvm-with-fusion-plugin` pipeline extension |
| **F** — Runtime wrapper | `wrap_fused_mul_add()` C-ABI symbol | `plugins/fusion/runtime/wrap_fused_mul_add.cpp` |
| **G** — Device kernel | gfx1103 HIP kernel (f32/f16/bf16) | `plugins/fusion/runtime/fused_mul_add_kernel.hip` |
| **H** — LIT tests | 3 tests: positive, R-1 negative, E2E lowering | `test/lit/plugins/` — **3/3 pass on ETX (Linux)** |

**Everything is gated behind `-DBUILD_HIP_PLUGINS=ON`.
The default build path is completely untouched.**

---

## 4. What a new arch team needs to ship a plugin

The plugin model supports several contribution patterns depending on what a
team needs. In all cases, **the host compiler is never modified.**

### Case 1 — Arch-specific op + kernel (full stack)

A team adding a new op with a custom device kernel (e.g. a fused op
specific to gfx1200) writes:

| File | Role |
|---|---|
| `MyOps.td` | ODS op definition (DPS traits, assembly format) |
| `MyPass.cpp` | Fusion pass that produces the new op (touchpoint B) |
| `MyLowering.cpp` | `ConvertOpToLLVMPattern` → `wrap_my_op` call (touchpoint E) |
| `runtime/my_kernel_gfxNNNN.hip` | Device kernel, compiled `--offload-arch=gfxNNNN` (touchpoint G) |
| `runtime/wrap_my_op.cpp` | C-ABI wrapper, dispatches to the kernel (touchpoint F) |
| `PluginMain.cpp` | Wires A + B + C + D + E; registers runtime pipeline |
| `CMakeLists.txt` | Builds MODULE plugin + SHARED runtime lib |

The team ships `hip_arch_plugin.so` (compiler plugin) + `hip_arch_runtime.so`
(runtime symbol provider).

### Case 2 — Arch-specific optimization pass only

A team adding an optimization or transformation pass without a new op
(e.g. an arch-specific memory layout pass, a cost-model-gated fusion, or a
quantization-specific rewrite) writes only:

| File | Role |
|---|---|
| `MyPass.cpp` | `OpRewritePattern` or `OpConversionPattern` targeting in-tree ops |
| `PluginMain.cpp` | Registers the pass (touchpoint B) + declares pipeline anchor (touchpoint C) |
| `CMakeLists.txt` | Builds MODULE plugin (no runtime lib needed) |

The pass runs at the declared anchor point (e.g. `after convert-onnx-to-hip`)
via the `PipelineSlotRegistry`. No new op, no kernel, no runtime symbol needed.

### Case 3 — Arch-specific lowering pattern only

A team that already has an in-tree op but wants to contribute an alternative
lowering for a specific arch (e.g. a different `ConvOp` lowering for a new
target) writes:

| File | Role |
|---|---|
| `MyLowering.cpp` | `ConvertOpToLLVMPattern` for the target op, emitting an arch-specific `wrap_*` symbol |
| `runtime/wrap_my_op.cpp` | Arch-specific runtime wrapper + dispatch |
| `runtime/my_kernel_gfxNNNN.hip` | Arch-tuned device kernel |
| `PluginMain.cpp` | Registers the `hip-to-llvm-with-<arch>-plugin` pipeline extension (touchpoint E) |
| `CMakeLists.txt` | Builds MODULE plugin + SHARED runtime lib |

The plugin pipeline extension replaces the standard lowering only for the
targeted op, leaving everything else untouched.

---

## 5. Open questions

| Question | Options |
|---|---|
| **Host hook for lowering patterns (E)** | PoC uses a plugin-registered pipeline extension. A cleaner option: host exposes `registerLoweringPatternExtension()` so plugins don't need to register their own pipeline. |
| **`PipelineSlotRegistry` (C)** | Currently lives in the plugin. Could move to the host so all plugins share one registry and `applyPluginSlots()` is called automatically by the pipeline builder. |
| **C ABI vs C++ ABI** | Entry points are C-weak; MLIR objects are C++. Plugin must be built against the same LLVM toolchain. A versioned `gnpu_plugin_entry` C-ABI is a future milestone. |
| **Pipeline anchor stability** | Which pass names become stable public anchors? Renaming any breaks plugins. 7 candidates listed in `PipelineSlot.h`. |
| **Kernel packaging** | Currently one precompiled `.hip` source per arch. Fat-object (multi-arch) and `HIP_CUSTOM_KERNELS_DIR`-style discovery are future options. |

---

## 6. Key design findings (apply to all future plugins)

### 6.1 Bufferization model is non-optional ⚠️

> The op compiled, the pass fired, the pipeline ran — and it still crashed
> deep inside `OneShotBufferize` with `op was not bufferized`.

The cause: a missing `attachInterface<HipDstBufferizableModel<OpTy>>` call.
The PoC includes a `_nobuf` plugin variant that reproduces this failure
exactly, as an **executable LIT regression test**. Any future plugin that
forgets the bufferization model will see this test fail immediately.

### 6.2 MLIR dylib linkage is mandatory on Linux

With hidden-symbol-visibility LLVM (all modern builds), TypeIDs must live
in a single shared object. On Linux, both `hip-mlir-opt` and the plugin
must link `libMLIR.so` (not static component libs). The PoC gates this
via `BUILD_HIP_PLUGINS=ON` → switch to dylib + `export_executable_symbols_for_plugins`.

### 6.3 Use `--pass-pipeline`, not generated flag shortcuts

When a dialect plugin and a pass plugin are loaded together,
`--hip-fuse-mul-add` shorthand regresses to printing `--help`.
The `--pass-pipeline=` textual form is unaffected. All LIT tests use it.

---

## 7. Build & test

```bash
# Linux (ETX — verified)
cmake -S . -B build \
  -DBUILD_HIP_TOOLS=ON \
  -DBUILD_HIP_PLUGINS=ON \
  -DLLVM_DIR=/usr/lib/llvm-22/cmake \
  -DMLIR_DIR=/usr/lib/llvm-22/lib/cmake/mlir \
  -DTHEROCK_DIST=<therock-dist> \
  -Dflatbuffers_DIR=<prebuilt>/lib/cmake/flatbuffers \
  -G Ninja

ninja -j4 hip-mlir-opt hip_fusion_plugin hip_fusion_plugin_nobuf

cd build/test/lit && lit -sv plugins/
# Expected: 3/3 Passed
```

---

## 8. Files changed (summary)

**In-tree (permanent):**
- `include/hip/Dialect/IR/HipDialect.td` — `registerPluginOps<>()` shim
- `CMakeLists.txt` — `BUILD_HIP_PLUGINS` option
- `tools/hip-mlir-opt/CMakeLists.txt` — MLIR dylib switch
- `test/lit/` — plugin feature + substitutions

**New (plugin, all gated):**
- `plugins/fusion/` — 15 source files (op, pass, lowering, runtime, kernel)
- `test/lit/plugins/` — 3 LIT tests

**Pre-existing bug fixes (GCC 13 compat, independent of plugin work):**
- 35 source files: remove redundant `mlir::hip::` qualification inside namespace bodies
- 3 files: add missing `<cstddef>` / `<cstdint>` includes
