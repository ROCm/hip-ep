<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Plugin interface

How an out-of-tree shared library contributes compiler passes, runtime
bitcode, and link libraries to `hip-compiler` at load time -- with no
recompile of the host compiler -- and what a downstream team has to write
to ship one.

## Table of contents

- [Motivation](#motivation)
- [Overview](#overview)
- [The plugin ABI](#the-plugin-abi)
  - [Entry point and info struct](#entry-point-and-info-struct)
  - [The registry](#the-registry)
  - [Pipeline slots](#pipeline-slots)
- [How the host loads and dispatches plugins](#how-the-host-loads-and-dispatches-plugins)
- [Linkage requirement](#linkage-requirement)
- [What a downstream team does](#what-a-downstream-team-does)
- [Component layout](#component-layout)
- [Custom ops: a plugin-owned dialect end-to-end](#custom-ops-a-plugin-owned-dialect-end-to-end)
- [Planned extensions](#planned-extensions)

## Motivation

Adding a custom op, fusion, or kernel to the compiler today means editing
several in-tree locations across the ONNX-to-HIP, HIP dialect, HIP-to-LLVM,
and runtime layers, then rebuilding the whole compiler. That is the right
workflow for changes that belong in the public tree, but it does not work
for a downstream team that needs to:

1. Ship **architecture-specific or out-of-tree kernels** and the lowerings
   that target them, without adding them to the public repo.
2. Add or reorder **compiler passes** for a specific target without forking
   `lib/Dialect/Transforms/Pipelines.cpp`.
3. Track the public repo as a **clean upstream dependency** -- linking
   against a released build -- instead of carrying a long-lived fork and
   re-merging on every change to `main`.

The plugin interface provides a stable boundary for exactly that subset. A
downstream team builds a shared library that the compiler loads at compile
time; the library registers passes, asks for them to run at named points in
the pipeline, and contributes the runtime bitcode and link libraries its
ops need. The public repo is consumed as a binary dependency, and the only
footprint a deployment carries is one environment variable.

A downstream team may use the plugin interface for any reason -- an
architecture-specific kernel, a target-specific pass, faster local
iteration, or code that is simply not ready to upstream or does not belong
in the shared tree. It complements upstream contribution rather than
replacing it: a generic op, bug fix, or non-target-specific pass is still
best contributed upstream through the normal flow. The plugin exists to let
downstream code live outside the public tree, not to discourage day-to-day
improvements to it.

## Overview

A plugin is a shared library (`.so` / `.dll`) listed in the
`HIP_EP_PLUGINS` environment variable. When the compiler starts, it loads
each library, validates its API version, and invokes a single registration
callback. Through the registry handed to that callback, the plugin makes its
contributions, each consumed at a fixed point in the existing compile flow:

```text
HIP_EP_PLUGINS=vendor.so
  |
  v   (each plugin library is loaded once, at compiler start-up)
Plugin loader  -->  HipEpPluginRegistry
  |
  |   the registry routes each contribution to a fixed point in the
  |   compile flow; all of them feed the final model.dll:
  |
  +--  registerPass + requestPipelineSlot  -->  ONNX->HIP->LLVM pipeline
  |
  +--  addDialectRegistration              -->  pipeline MLIRContext (loadAllDialects)
  |
  +--  addRuntimeBitcode                    -->  LLVM backend (link)
  |
  +--  addLibraryPath / addLibrary          -->  native link (CompilerDriver)
```

| Contribution | Registry call | Consumed by |
|---|---|---|
| MLIR pass | `registerPass<T>()` | `lib/Dialect/Transforms/Pipelines.cpp` |
| Pass placement | `requestPipelineSlot(slot, name)` | same |
| Dialect + op + interface models | `addDialectRegistration(fn)` | `hip::compiler::loadAllDialects` (`include/hip/InitAllPasses.h`) |
| Runtime bitcode | `addRuntimeBitcode(data, size)` | `lib/Target/LLVM/LLVMBackend.cpp` |
| Link path / library | `addLibraryPath` / `addLibrary` | `lib/Compiler/CompilerDriver.cpp` |

When no plugin is configured, every hook is a no-op and the default build
path is unchanged.

## The plugin ABI

The public surface a plugin compiles against is two headers:
[include/hip/Compiler/PluginAPI.h](../../include/hip/Compiler/PluginAPI.h)
(the C entry point and info struct) and
[include/hip/Compiler/PluginRegistry.h](../../include/hip/Compiler/PluginRegistry.h)
(the registry the callback uses).

### Entry point and info struct

The host looks up one symbol, `hipEpGetPluginInfo`, by name and calls it to
obtain a small info struct returned by value:

```cpp
#define HIP_EP_PLUGIN_API_VERSION 2

extern "C" {
struct HipEpPluginLibraryInfo {
  uint32_t    APIVersion;       // == HIP_EP_PLUGIN_API_VERSION at build time
  const char *PluginName;       // logged on load
  const char *PluginVersion;    // vendor's own string; not parsed
  void (*RegisterCallbacks)(HipEpPluginRegistry &);  // called once on load
};
}

extern "C" ::hip::compiler::HipEpPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
hipEpGetPluginInfo();
```

`APIVersion` is incremented for any ABI-breaking change to the struct; the
loader rejects any version it does not equal. `RegisterCallbacks` may be
`nullptr` for a placeholder plugin; the load still succeeds.

### The registry

`HipEpPluginRegistry` is passed by reference to `RegisterCallbacks` and
exposes the contributions:

```cpp
class HipEpPluginRegistry {
public:
  // Register a pass with MLIR's pass registry so the pipeline can
  // instantiate it by name at a requested slot.
  template <typename PassT> void registerPass();

  // Run a registered pass (by its MLIR command-line name) at a named slot.
  void requestPipelineSlot(PipelineSlot slot, llvm::StringRef passName);

  // Contribute a dialect-registration callback. The host runs it against the
  // DialectRegistry it builds the pipeline's MLIRContext from, so the callback
  // can registry.insert<VendorDialect>() and registry.addExtension(...) to
  // attach the op's bufferization and HIP->LLVM-lowering interface models --
  // the same thing the upstream mlirGetDialectPluginInfo callback does.
  void addDialectRegistration(void (*registerFn)(mlir::DialectRegistry &));

  // Contribute LLVM bitcode linked into the model module after the in-tree
  // runtime bitcode (e.g. vendor wrap_* implementations).
  void addRuntimeBitcode(const void *data, std::size_t sizeBytes);

  // Contribute a library search path and a library to the native link.
  void addLibraryPath(llvm::StringRef path);
  void addLibrary(llvm::StringRef nameOrFullPath);
};
```

Capabilities live on this class rather than as extra fields in the C
struct, so a future capability is added by extending the class -- the C
entry point stays at a single callback and older plugins keep loading.

Internally the registry is a value handle over a function-pointer table the
host fills in (see
[lib/Compiler/PluginRegistry.cpp](../../lib/Compiler/PluginRegistry.cpp)).
`hip-compiler` is linked as a static library into a host process (the EP
DLL, `hip-mlir-opt`, etc.) rather than shipped as its own shared library,
so a plugin has no `hip-compiler` import library to resolve method symbols
against; the table-of-function-pointers indirection bridges that boundary.
A plugin therefore depends only on these headers plus MLIR (for the
definition of `registerPass<T>()`'s `mlir::PassRegistration`).

### Pipeline slots

`requestPipelineSlot` places a registered pass at one of a fixed set of
named anchors in the ONNX-to-HIP-to-LLVM pipeline:

```cpp
enum class PipelineSlot {
  AfterSimplifyOnnx,
  AfterOnnxLoopOutline,
  AfterConvertOnnxToHip,    // most common: lower a custom op here
  BeforeBufferization,
  AfterPoolAllocs,
  BeforeConvertHipToLLVM,
  AfterGenerateInterface,
};
```

The enum is append-only: new anchors are added at the end, and removing or
renaming one is an ABI break that bumps `HIP_EP_PLUGIN_API_VERSION`. Each
anchor corresponds to a single point in
[lib/Dialect/Transforms/Pipelines.cpp](../../lib/Dialect/Transforms/Pipelines.cpp),
where the pipeline builder consults the registry and adds any
plugin-requested passes:

```cpp
// In Pipelines.cpp, at each anchor:
addPluginPassesForSlot(pm, ::hip::compiler::PipelineSlot::AfterConvertOnnxToHip);
```

`addPluginPassesForSlot` resolves each requested pass by name through MLIR's
pass registry (the same lookup a `--pass=name` flag uses) and adds it to the
active pass manager. A name that does not resolve produces a one-line
warning rather than a silent miss.

## How the host loads and dispatches plugins

The loader lives in
[include/hip/Compiler/PluginLoader.h](../../include/hip/Compiler/PluginLoader.h)
and [lib/Compiler/PluginLoader.cpp](../../lib/Compiler/PluginLoader.cpp).
`dispatchPluginRegistrationsOnce()` is the production entry point; it is
idempotent (guarded by `std::call_once`) and called from each host that
builds a pipeline (`hip-compiler`, `hip-mlir-opt`).

```
HIP_EP_PLUGINS=/opt/vendor/lib/vendor.so;/opt/other/plugin.so
```

The list is semicolon-separated on every platform (a colon would clash with
Windows drive letters). For each entry the loader:

1. Loads the library permanently into the process.
2. Looks up `hipEpGetPluginInfo` and calls it.
3. Rejects the plugin if `APIVersion` does not match, or if the name /
   version pointers are null -- with a single-line `[plugin-loader]
   WARNING:` to stderr, then continues with the next entry.
4. Invokes `RegisterCallbacks` against the per-process registry, inside a
   `try/catch` so an exception escaping a plugin built against a different
   C++ runtime is contained rather than crashing the host.

After dispatch, the host reads back what the plugins contributed through the
accessors in `PluginRegistry.h` -- `pluginPassesForSlot`,
`pluginDialectRegistrations`, `pluginBitcodeBuffers`, `pluginLibraryPaths`,
`pluginLibraries` -- at the four hook sites.

### Multiple plugins

`HIP_EP_PLUGINS` may list several libraries (semicolon-separated); they load
in listed order, with duplicates removed, and each `RegisterCallbacks` runs
once. Contributions compose by load order:

- **Slots:** passes requested for the same slot run in plugin load order.
- **Pass names:** a pass is resolved by its registered name, so two plugins
  must not register the same name -- prefix plugin pass names to keep them
  unique.
- **Bitcode / library symbols:** a symbol defined by more than one source
  resolves to the last contributor in load order (bitcode is linked with
  `OverrideFromSrc`; libraries follow link search order). Prefix vendor
  symbols so cross-plugin collisions cannot happen by accident.

## Linkage requirement

`registerPass<T>()` writes MLIR's process-global pass registry. For a
plugin-registered pass to be visible to the host, the plugin's registration
must land in **the host's** copy of that registry. The host (`hip-compiler` /
`hip-mlir-opt`) links MLIR statically, so this is achieved exactly the way
upstream LLVM tools do it -- the host **exports its statically-linked MLIR
symbols**, and the plugin leaves its MLIR symbols **undefined** so they bind to
the host at `dlopen` time:

- **Host side:** built with `HIPDNN_ENABLE_PLUGINS` (default ON), which calls
  `export_executable_symbols_for_plugins()` on the tool executables -- the same
  call `mlir-opt` makes (`mlir/tools/mlir-opt/CMakeLists.txt`). On Linux this
  takes the `ENABLE_EXPORTS` / `-rdynamic` path. LLVM is already linked as the
  shared `libLLVM`, so only the MLIR symbols are exported.
- **Plugin side:** links MLIR **headers only** (no MLIR libraries), so its
  registry/op references are undefined and resolve to the host's single copy.
  A plugin that statically links MLIR gets its own registry; the host's lookup
  then misses and the slot dispatch warns.

This is the model the public build ships. (An alternative -- routing MLIR
through a shared `libMLIR.so` dylib so host and plugin both link it -- also
works and is what a non-executable host, e.g. an EP DLL, would need; it is not
how the CLI tools are built.) ROCm's `libamd_comgr.so` references no MLIR
symbols and resolves its LLVM from the same shared `libLLVM`, so exporting the
host's MLIR symbols does not perturb comgr.

The bitcode and library contributions do not cross MLIR's global state, so
they work regardless of this -- the symbol-export requirement applies
specifically to the pass (and, for the planned dialect/op extension, the
dialect and type registries).

## What a downstream team does

A downstream team builds its plugin in a separate repo that consumes a
released `hip-compiler` install tree. No fork of the public repo is
required. The files a full-stack plugin (custom op + kernel) writes:

| File | Role |
|---|---|
| `MyPass.cpp` | Fusion / lowering pass that produces or rewrites the op |
| `MyLowering.cpp` | Lowers the op to a `wrap_*` runtime call |
| `runtime/wrap_my_op.cpp` | C-ABI runtime wrapper, compiled to bitcode and embedded |
| `runtime/my_kernel.hip` | Device kernel(s), built into a vendor library |
| `plugin_main.cpp` | `hipEpGetPluginInfo` + `RegisterCallbacks` |
| `CMakeLists.txt` | Builds the plugin shared library + vendor kernel library |

A pass-only or lowering-only plugin writes a subset (no kernel / no runtime
wrapper).

> Prerequisite for pass-contributing plugins: a plugin pass only takes effect
> when the host (`hip-compiler` / `hip-mlir-opt`) was built with
> `HIPDNN_ENABLE_PLUGINS` (default ON) so it exports its MLIR symbols, and the
> plugin links MLIR headers only (see
> [Linkage requirement](#linkage-requirement)). The bitcode and library
> contributions do not depend on this and work regardless.

### 1. Consume the public install tree

The public build installs the plugin headers, the compiler binaries, and a
CMake package config under a prefix:

```
<prefix>/include/hip/Compiler/PluginAPI.h
<prefix>/include/hip/Compiler/PluginRegistry.h
<prefix>/include/hip/Compiler/PluginLoader.h
<prefix>/bin/hip-compiler, hip-mlir-opt
<prefix>/lib/cmake/HipDnnEp/HipDnnEpConfig.cmake
```

The plugin's CMake does `find_package(HipDnnEp CONFIG REQUIRED)` (with
`-DHipDnnEp_DIR=<prefix>/lib/cmake/HipDnnEp`), which provides the
`HipDnnEp::plugin_headers` INTERFACE target (the include path) and the
`HipDnnEp::hip-compiler` / `HipDnnEp::hip-mlir-opt` IMPORTED executables. It
also `find_package(MLIR CONFIG)`s the same MLIR build and links MLIR
**headers only** (see [Linkage requirement](#linkage-requirement)). The plugin
links no symbol from `hip-compiler` itself.

### 2. Write the pass

The pass is a normal MLIR pass. Its `getArgument()` name is the string the
entry point passes to `requestPipelineSlot`, and the slot decides where it
runs:

```cpp
// MyLoweringPass.cpp -- scheduled at the AfterConvertOnnxToHip slot.
struct MyLoweringPass
    : mlir::PassWrapper<MyLoweringPass,
                        mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyLoweringPass)

  // The name requestPipelineSlot() refers to.
  llvm::StringRef getArgument() const final { return "my-vendor-lowering"; }

  void runOnOperation() override {
    // Match the vendor op (e.g. onnx.Custom("myvendor.FusedOp")) and rewrite
    // it to a call into the vendor runtime wrapper below. Vendor-specific
    // matching / rewriting logic goes here.
  }
};
```

### 3. Write the entry point

`RegisterCallbacks` is the entire compiler-facing surface. It registers the
pass, asks for a slot, and contributes the runtime bitcode and kernel
library:

```cpp
// plugin_main.cpp
extern "C" const unsigned char my_wrap_bc[];   // embedded at build time
extern "C" const std::size_t   my_wrap_bc_size;

extern "C" ::hip::compiler::HipEpPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
hipEpGetPluginInfo() {
  return {
      HIP_EP_PLUGIN_API_VERSION,
      "MyVendorPlugin",
      "1.0.0",
      [](::hip::compiler::HipEpPluginRegistry &R) {
        R.registerPass<MyLoweringPass>();
        R.requestPipelineSlot(
            ::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
            "my-vendor-lowering");        // == MyLoweringPass::getArgument()
        if (my_wrap_bc_size != 0)
          R.addRuntimeBitcode(my_wrap_bc, my_wrap_bc_size);
        R.addLibraryPath("/opt/vendor/lib");
        R.addLibrary("vendor_kernels");
      }};
}
```

On Windows, `LLVM_ATTRIBUTE_WEAK` is a no-op: export the definition with
`__declspec(dllexport)`, or set `WINDOWS_EXPORT_ALL_SYMBOLS` on the plugin
target.

### 4. Write the runtime wrapper and kernel

The lowering in step 2 emits a call to a C-ABI wrapper. That wrapper is a
plain C function -- compiled separately to LLVM bitcode (not part of the
plugin's C++ objects) and embedded into the plugin as a byte array, then
handed to `addRuntimeBitcode`. Prefix the symbol so it cannot collide with
an in-tree `wrap_*`:

```cpp
// runtime/wrap_my_op.cpp -- compiled to LLVM bitcode, embedded in the plugin.
extern "C" void vendor_wrap_my_op(void *stream, const void *in, void *out,
                                  int64_t n) {
  launch_my_kernel(stream, in, out, n);   // dispatch to the device kernel
}
```

```cpp
// runtime/my_kernel.hip -- built into vendor_kernels.{a,lib}.
__global__ void my_kernel(const float *in, float *out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = /* vendor math */ in[i];
}

void launch_my_kernel(void *stream, const void *in, void *out, int64_t n) {
  /* hipLaunchKernelGGL(my_kernel, ...); */
}
```

The wrapper bitcode is linked into the generated `model.dll` after the
in-tree runtime; the kernel library is added to the native link.

### 5. Build the plugin

```cmake
# CMakeLists.txt (downstream repo). Consume the public install tree's package
# config; MLIR must be the same build the host was compiled against.
#   -DHipDnnEp_DIR=<prefix>/lib/cmake/HipDnnEp
find_package(HipDnnEp CONFIG REQUIRED)
find_package(MLIR CONFIG REQUIRED)

# (a) Compile the runtime wrapper to LLVM bitcode and embed it as a byte
#     array (my_wrap_bc / my_wrap_bc_size), e.g. clang -emit-llvm then an
#     xxd-style codegen step -> ${EMBEDDED_BC_SRC}.
# (b) Build the device kernels into a vendor library.
add_library(vendor_kernels STATIC runtime/my_kernel.hip)

# (c) Build the plugin shared library. Link the plugin ABI headers + MLIR
#     HEADERS ONLY (no MLIR libraries) so the plugin's MLIR symbols bind to the
#     symbol-exporting host at load (see Linkage requirement); link no
#     hip-compiler symbol.
add_library(my_vendor_plugin SHARED
  plugin_main.cpp MyLoweringPass.cpp ${EMBEDDED_BC_SRC})
target_include_directories(my_vendor_plugin PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(my_vendor_plugin PRIVATE HipDnnEp::plugin_headers)
set_target_properties(my_vendor_plugin PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS ON                          # export hipEpGetPluginInfo
  PREFIX "")
```

### 6. Deploy

Point the compiler at the plugin and run as usual:

```
HIP_EP_PLUGINS=/opt/vendor/lib/my_vendor_plugin.so \
THEROCK_DIST=<therock> \
hip-compiler model.mlir -o model.dll
```

The same variable is read by `hip-mlir-opt` for pass-level testing.

### 7. Test the plugin

A plugin is testable with the same tools the in-tree sample uses, and the
compile-side checks need no GPU:

- A **unit test** that loads the plugin and asserts its name, version, and
  contributions round-trip across the ABI boundary. Use
  [test/plugin/test_plugin_loader.cpp](../../test/plugin/test_plugin_loader.cpp)
  as the template.
- A **LIT test** that runs the pass at a pipeline point. Use the textual
  pipeline form (`--pass-pipeline=`) rather than the generated `--<pass>`
  flag shorthand, which is unreliable for a plugin-contributed pass:

```
// RUN: env HIP_EP_PLUGINS=%my-plugin \
// RUN:   hip-mlir-opt \
// RUN:     --pass-pipeline='builtin.module(func.func(my-vendor-lowering))' \
// RUN:     %s | FileCheck %s
```

The pass half of both tests requires the shared-MLIR host build (see the
prerequisite above); the loader / bitcode / library round-trip in the unit
test does not.

## Component layout

In-tree, the mechanism is:

| File | Role |
|---|---|
| [include/hip/Compiler/PluginAPI.h](../../include/hip/Compiler/PluginAPI.h) | C entry point + info struct |
| [include/hip/Compiler/PluginRegistry.h](../../include/hip/Compiler/PluginRegistry.h) | Registry class + `PipelineSlot` enum + accessors |
| [include/hip/Compiler/PluginLoader.h](../../include/hip/Compiler/PluginLoader.h) | Loader + `dispatchPluginRegistrationsOnce` |
| [lib/Compiler/PluginLoader.cpp](../../lib/Compiler/PluginLoader.cpp) | `HIP_EP_PLUGINS` parsing, load, version check, dispatch |
| [lib/Compiler/PluginRegistry.cpp](../../lib/Compiler/PluginRegistry.cpp) | Host-side function-pointer table + contribution storage |
| [lib/Dialect/Transforms/Pipelines.cpp](../../lib/Dialect/Transforms/Pipelines.cpp) | Slot hook (`addPluginPassesForSlot`) |
| [include/hip/InitAllPasses.h](../../include/hip/InitAllPasses.h) | Dialect hook (`loadAllDialects` applies plugin dialect registrations) |
| [lib/Target/LLVM/LLVMBackend.cpp](../../lib/Target/LLVM/LLVMBackend.cpp) | Links plugin bitcode into the model module |
| [lib/Compiler/CompilerDriver.cpp](../../lib/Compiler/CompilerDriver.cpp) | Appends plugin link paths / libraries |
| [test/plugin/sample_plugin/](../../test/plugin/sample_plugin/) | Worked example exercised in CI |
| [test/plugin/test_plugin_loader.cpp](../../test/plugin/test_plugin_loader.cpp) | Loader / round-trip unit test |

The sample plugin is built behind a CMake option and exercised by the unit
test and a LIT test, so the ABI stays exercised in CI even before any
downstream plugin exists.

## Custom ops: a plugin-owned dialect end-to-end

A plugin can contribute its **own dialect op** that lives across the whole
pipeline -- introduced from a model op, bufferized like an in-tree op, and
lowered to a vendor kernel -- entirely from out-of-tree code, via three
idiomatic MLIR seams that ride the same [linkage](#linkage-requirement) as a
plugin pass (no shared `libMLIR` dylib):

1. **Dialect + op + interface models** -- `addDialectRegistration(fn)` hands the
   host a callback it runs against the `DialectRegistry` the pipeline's
   `MLIRContext` is built from (`hip::compiler::loadAllDialects`). The callback
   does `registry.insert<VendorDialect>()` and, via `registry.addExtension(...)`
   DialectExtensions, attaches the op's `BufferizableOpInterface` model and its
   `ConvertToLLVMPatternInterface`.
2. **Bufferization** -- one-shot-bufferize finds the attached
   `BufferizableOpInterface` model and turns the tensor-form op into its
   buffer form, exactly like an in-tree HIP op. (A custom op with memref
   operands must also carry `MemoryEffectOpInterface`, or the ownership-based
   buffer-deallocation pass rejects it with "unknown memory side effects".)
3. **HIP -> LLVM lowering** -- `convert-hip-to-llvm` calls
   `populateConversionTargetFromOperation`, which walks the module, collects
   every dialect implementing `ConvertToLLVMPatternInterface`, and lets each
   add its lowering patterns and mark its ops illegal. The vendor op lowers to
   a call into the vendor runtime wrapper (contributed via `addRuntimeBitcode`),
   which launches the vendor kernel (contributed via `addLibrary`).

A vendor pass (registered + slotted as above, typically at
`AfterSimplifyOnnx`) introduces the op from a model op -- e.g. rewriting
`onnx.Add` into `vendor.add`. Because the in-tree `convert-onnx-to-hip` only
matches `onnx.*` ops, the vendor op passes through untouched to bufferization.

This is exercised end-to-end (compile + GPU run + numeric check) by the
`rocm-ep-plugin` example's `vendor.add` op (`out = a + b`): `onnx.Add` ->
`vendor.add` -> bufferized -> `wrap_rocm_ep_vendor_add` -> vendor HIP kernel.

## Planned extensions

### Pipeline composition

Beyond inserting at a fixed slot, a downstream may want to see the full pass
set and compose -- or fully replace -- the pass order. The design has four
parts; the first two **ship today**, the third is planned.

- **Expose all passes by name** *(shipped)*. Every production pipeline pass is
  registered with a stable command-line name (`getArgument()`) in the registry
  the driver resolves the override against -- one comprehensive registrar,
  `include/hip/InitAllPasses.h::registerAllPasses()`, is called by both the EP
  / `hip-compiler` driver and the `hip-mlir-opt` tool, so the set of nameable
  passes can never drift between them. The composable pipeline names
  (`onnx-to-hip-pipeline`, `hip-to-llvm-pipeline`, `hipdnn-pipeline`) are
  registered alongside. A published "pass menu" lists the names, anchor ops,
  and `PipelineSlot` anchors: [`docs/pipeline_pass_menu.md`](../pipeline_pass_menu.md).
  A few load-bearing passes are intentionally *not* individually nameable
  (`generate-interface` needs a `CompilationOptionsT`; `compile-hipdnn-graphs`
  needs a runtime handle; a few MLIR utility passes are added only inside the
  pipeline builders) -- a hand-listed override that needs the C-ABI entry point
  composes the registered *pipeline names*, which include them.
- **Textual pipeline override** *(shipped)*. The driver consults the
  `HIPDNN_EP_PIPELINE` env var (`CompilerDriver::runMLIRPasses`). When set, it
  strips an optional `builtin.module(...)` wrapper and runs `parsePassPipeline`
  on the inner text (composing from in-tree passes by name); when unset, the
  in-tree default runs unchanged. A parse failure is reported up front; the
  override resolves from the host's own registry, so this part does **not**
  require shared MLIR.
- **Registered pipeline builder** *(planned)*. For passes that need
  runtime-bound state a textual string cannot carry (e.g. the
  in-memory-filesystem variant of `convert-onnx-to-hip`), a plugin would
  register a pipeline *builder* that receives a context (the filesystem and
  pipeline options) and composes passes -- including in-tree stage builders --
  in C++. This is the full-fidelity path; it would share the
  [linkage requirement](#linkage-requirement).
- **Guardrails** *(shipped, partial)*. The load-bearing ordering invariants
  (host-scalar materialization before pool allocation; output-allocator after
  buffer deallocation; affine lowering after strided-metadata expansion; the
  terminal `convert-hip-to-llvm` + `generate-interface` stages) are owned by
  whoever replaces the order. A post-pipeline check hard-fails with a clear
  message when the produced module lacks an `inference_compute` entry point
  after an override.

**Stability:** pass and pipeline names are version-pinned compiler internals,
not a frozen cross-release contract -- a plugin that composes by name pins to
a release. (Renaming a pass is a documented breaking change for such plugins.)

The custom-op path and the pipeline-composition extension share the
[linkage requirement](#linkage-requirement): the dialect, type, and pass
registries are process-global MLIR state, so any plugin-registered op, pass,
or pipeline needs the host and plugin to share one MLIR instance.
