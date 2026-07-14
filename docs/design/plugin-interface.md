<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Plugin interface

## Table of contents

- [Motivation](#motivation)
- [Overview](#overview)
- [The plugin ABI](#the-plugin-abi)
  - [Entry point](#entry-point)
  - [The registry](#the-registry)
  - [Pipeline slots](#pipeline-slots)
- [How the host dispatches plugins](#how-the-host-dispatches-plugins)
- [Linkage model](#linkage-model)
- [What a downstream team does](#what-a-downstream-team-does)
- [Component layout](#component-layout)
- [Custom ops: a plugin-owned dialect end-to-end](#custom-ops-a-plugin-owned-dialect-end-to-end)
- [Pipeline composition](#pipeline-composition)

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
downstream team builds a static library and selects it into its own build of
the host (which it already builds to target its GPU arch); the library
registers passes, asks for them to run at named points in the pipeline, and
contributes the runtime bitcode and link libraries its ops need. The public
repo is consumed as a source dependency built alongside the plugin, and the only
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

A plugin is a **static library** selected into the host at configure time with
`-DHIPDNN_EP_COMPILER_PLUGINS=<id>`. It defines one registration entry point,
`hipEpRegisterPlugin_<id>`; a CMake-generated registrar calls it once per
process. Through the registry handed to that entry point, the plugin makes its
contributions, each consumed at a fixed point in the existing compile flow:

```text
-DHIPDNN_EP_COMPILER_PLUGINS=vendor   (statically linked into the host)
  |
  v   (registrar calls each selected plugin's hipEpRegisterPlugin_<id> once)
Static registrar  -->  HipEpPluginRegistry
  |
  |   the registry routes each contribution to a fixed point in the
  |   compile flow; all of them feed the compiled model module:
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
(the `extern "C"` entry-point contract and the `HIP_EP_DEFINE_PLUGIN` macro) and
[include/hip/Compiler/PluginRegistry.h](../../include/hip/Compiler/PluginRegistry.h)
(the registry the entry point uses).

### Entry point

A plugin defines one `extern "C"` registration function, named after the
plugin's id:

```cpp
// PluginAPI.h provides the convenience macro; `myvendor` is the plugin id.
HIP_EP_DEFINE_PLUGIN(myvendor) {   // extern "C" void hipEpRegisterPlugin_myvendor(HipEpPluginRegistry &R)
  R.registerPass<MyPass>();
  R.requestPipelineSlot(::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
                        "my-pass");
  R.addRuntimeBitcode(my_bc, my_bc_size);
  R.addLibrary("vendor_kernels");
}
```

The plugin is a **static library** selected into the host at configure time
(`-DHIPDNN_EP_COMPILER_PLUGINS=myvendor`). CMake generates a registrar
([lib/Compiler/StaticPlugins.cpp](../../lib/Compiler/StaticPlugins.cpp)) that
calls each selected plugin's `hipEpRegisterPlugin_<id>` exactly once per process.
There is no runtime version handshake: plugin and host are built and linked
together, so an ABI mismatch is a build error, not a load-time surprise.
`HIP_EP_PLUGIN_API_VERSION` remains a source-level marker for the registry
surface only. (See "Linkage model" below for why a plugin is statically linked
rather than loaded from a shared library at runtime.)

### The registry

`HipEpPluginRegistry` is passed by reference to the plugin's
`hipEpRegisterPlugin_<id>` entry and exposes the contributions:

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
  // attach the op's bufferization and HIP->LLVM-lowering interface models.
  void addDialectRegistration(void (*registerFn)(mlir::DialectRegistry &));

  // Contribute LLVM bitcode linked into the model module after the in-tree
  // runtime bitcode (e.g. vendor wrap_* implementations).
  void addRuntimeBitcode(const void *data, std::size_t sizeBytes);

  // Contribute a library search path and a library to the native link.
  void addLibraryPath(llvm::StringRef path);
  void addLibrary(llvm::StringRef nameOrFullPath);
};
```

Capabilities live on this class rather than as free function pointers, so a
future capability is added by extending the class. The registry is a value
handle over a function-pointer table the host fills in (see
[lib/Compiler/PluginRegistry.cpp](../../lib/Compiler/PluginRegistry.cpp)); the
indirection is retained (harmless under static linking) so a plugin TU depends
only on these headers plus MLIR (for `registerPass<T>()`'s
`mlir::PassRegistration`), never on a hip-compiler symbol.

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

## How the host dispatches plugins

Selected plugins are linked statically into the host; the CMake-generated
registrar [lib/Compiler/StaticPlugins.cpp](../../lib/Compiler/StaticPlugins.cpp)
calls each one's `hipEpRegisterPlugin_<id>` against the per-process registry.
`dispatchPluginRegistrationsOnce()` (declared in `PluginRegistry.h`) is the
production entry point; it is idempotent (`std::call_once`) and called from each
host that builds a pipeline (`hip-compiler`, `hip-mlir-opt`, the EP DLL). A
plugin whose registration throws is contained (logged + skipped) rather than
crashing the host. When no plugins were selected, dispatch is a no-op.

After dispatch, the host reads back what the plugins contributed through the
accessors in `PluginRegistry.h` -- `pluginPassesForSlot`,
`pluginDialectRegistrations`, `pluginBitcodeBuffers`, `pluginLibraryPaths`,
`pluginLibraries` -- at the hook sites.

### Multiple plugins

`HIPDNN_EP_COMPILER_PLUGINS` may list several ids; each plugin's
`hipEpRegisterPlugin_<id>` runs once, in list order. Contributions compose by
that order:

- **Slots:** passes requested for the same slot run in registration order.
- **Pass names:** a pass is resolved by its registered name, so two plugins
  must not register the same name -- prefix plugin pass names to keep them
  unique.
- **Bitcode / library symbols:** a symbol defined by more than one source
  resolves to the last contributor (bitcode is linked with `OverrideFromSrc`;
  libraries follow link search order). Prefix vendor symbols so cross-plugin
  collisions cannot happen by accident.

## Linkage model

Plugins are linked **statically** into the host at configure time, selected via
`-DHIPDNN_EP_COMPILER_PLUGINS=<id;...>` (default empty). The machinery is in
[cmake/HipEpPlugins.cmake](../../cmake/HipEpPlugins.cmake): a plugin package
calls `hipdnn_ep_compiler_plugin_register(PLUGIN_ID <id> TARGET <static-lib>)`
to make itself available; `hipdnn_ep_finalize_static_plugins()` then generates
the registrar include for the selected ids and links their static libs into
`LibHipCompiler` (which every host links).

Why static rather than a runtime-loaded shared library: an MLIR-contributing
plugin (a pass, a dialect/op, an interface model) must share the host's
process-global MLIR state -- one pass registry, one set of op `TypeID`s. Static
linking makes the plugin and host **one binary**, so they share that state by
construction, with **no symbol export and no dynamic loader** -- identical on
every platform.

The dynamic alternative does not hold up. Sharing MLIR state across a
dynamic-load boundary would require the host to export its entire `mlir::`
symbol surface -- on the order of a hundred thousand symbols, which exceeds the
export-table capacity some object formats impose, so it cannot be built on every
platform. A per-plugin scoped export is possible but inverts the build
dependency (the host link would depend on each plugin's objects) and adds
per-plugin export-list machinery. Static linking avoids all of it.

**Correctness note:** the registrar calls each `hipEpRegisterPlugin_<id>`
**explicitly** -- registration must never rely on a static initializer in the
plugin lib, or `--gc-sections` / `/OPT:REF` would drop the unreferenced object
and silently discard the registration.

The bitcode and library contributions (`addRuntimeBitcode`, `addLibrary`) are
pure C-ABI and cross no MLIR state; they are unaffected by linkage and are the
one class of contribution that also works dynamically into a prebuilt host.

### Deploying plugins with a prebuilt host

Because static plugins are chosen at the host's configure time, a downstream
builds `hip-ep` itself (which it already does to target its GPU arch and
per-target kernels) and adds its plugin id to `HIPDNN_EP_COMPILER_PLUGINS`.
There is no runtime drop-in of an MLIR plugin into an already-shipped host: that
would require the dynamic-load boundary this design deliberately avoids.
Kernel-level customization (a custom `wrap_*` for an op) does not need it -- it
is already possible against a prebuilt host through the bitcode / library
contributions, which cross a pure C boundary.

## What a downstream team does

A downstream team builds its plugin as a static library alongside its own
build of `hip-ep` (no fork of the public repo is required). The files a
full-stack plugin (custom op + kernel) writes:

| File | Role |
|---|---|
| `MyPass.cpp` | Fusion / lowering pass that produces or rewrites the op |
| `MyLowering.cpp` | Lowers the op to a `wrap_*` runtime call |
| `runtime/wrap_my_op.cpp` | C-ABI runtime wrapper, compiled to bitcode and embedded |
| `runtime/my_kernel.hip` | Device kernel(s), built into a vendor library |
| `plugin_main.cpp` | `HIP_EP_DEFINE_PLUGIN(<id>)` registration entry |
| `CMakeLists.txt` | Builds the plugin static library + vendor kernel library; calls `hipdnn_ep_compiler_plugin_register` |

A pass-only or lowering-only plugin writes a subset (no kernel / no runtime
wrapper).

> A plugin pass only takes effect because the plugin is statically linked into
> the host (one binary, one MLIR pass registry — see
> [Linkage model](#linkage-model)). The bitcode and library contributions work
> regardless of linkage.

### 1. Build alongside the host

A plugin is built in the same CMake build as `hip-ep` and uses its
public headers:

```
include/hip/Compiler/PluginAPI.h        # HIP_EP_DEFINE_PLUGIN, entry contract
include/hip/Compiler/PluginRegistry.h   # the registry the entry uses
cmake/HipEpPlugins.cmake                 # HIPDNN_EP_COMPILER_PLUGINS + register fn
```

The plugin's CMake `find_package(MLIR CONFIG)`s the same MLIR build, compiles
against the plugin surface + MLIR headers, and declares the MLIR libraries it
references (so static link order resolves — see [Linkage model](#linkage-model)).
It calls `hipdnn_ep_compiler_plugin_register(PLUGIN_ID <id> TARGET <lib>)` and is
selected with `-DHIPDNN_EP_COMPILER_PLUGINS=<id>`. The plugin links no symbol
from `hip-compiler` itself.

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

The `hipEpRegisterPlugin_<id>` function is the entire compiler-facing surface.
It registers the pass, asks for a slot, and contributes the runtime bitcode and
kernel library. `HIP_EP_DEFINE_PLUGIN(<id>)` (from `PluginAPI.h`) expands to the
correctly-named `extern "C"` signature:

```cpp
// plugin_main.cpp
extern "C" const unsigned char my_wrap_bc[];   // embedded at build time
extern "C" const std::size_t   my_wrap_bc_size;

// `myvendor` must match the id in hipdnn_ep_compiler_plugin_register(...) and
// -DHIPDNN_EP_COMPILER_PLUGINS=myvendor.
HIP_EP_DEFINE_PLUGIN(myvendor) {
  R.registerPass<MyLoweringPass>();
  R.requestPipelineSlot(
      ::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
      "my-vendor-lowering");        // == MyLoweringPass::getArgument()
  if (my_wrap_bc_size != 0)
    R.addRuntimeBitcode(my_wrap_bc, my_wrap_bc_size);
  R.addLibraryPath("/opt/vendor/lib");
  R.addLibrary("vendor_kernels");
}
```

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

The wrapper bitcode is linked into the generated model module after the
in-tree runtime (so it rides along whether that module is JIT-loaded or emitted
as a native DLL); the kernel library is added to the native link.

### 5. Build the plugin (a static library selected into the host build)

The plugin is built as part of the same CMake build as `hip-ep` (a
downstream adds its plugin directory to that build, or builds the host as a
subproject). It builds a **static** library, registers it as available, and is
selected with `-DHIPDNN_EP_COMPILER_PLUGINS=myvendor`.

```cmake
# CMakeLists.txt (in the downstream's build of hip-ep).
find_package(MLIR CONFIG REQUIRED)   # the SAME MLIR the host is built against

# (a) Compile the runtime wrapper to LLVM bitcode and embed it as a byte array
#     (my_wrap_bc / my_wrap_bc_size), e.g. clang -emit-llvm + an xxd-style step.
# (b) Build the device kernels into a vendor library.
add_library(vendor_kernels STATIC runtime/my_kernel.hip)

# (c) Build the plugin STATIC library. It uses the plugin surface headers +
#     MLIR headers, and declares the MLIR libraries it references (so they
#     follow it in static link order). It is linked into the host only when
#     selected; it links no hip-compiler symbol.
add_library(my_vendor_plugin STATIC
  plugin_main.cpp MyLoweringPass.cpp ${EMBEDDED_BC_SRC})
target_include_directories(my_vendor_plugin PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(my_vendor_plugin PUBLIC MLIRPass MLIRIR MLIRSupport)
set_target_properties(my_vendor_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Make it available to the static registrar; selected via
# -DHIPDNN_EP_COMPILER_PLUGINS=myvendor.
hipdnn_ep_compiler_plugin_register(PLUGIN_ID myvendor TARGET my_vendor_plugin)
```

### 6. Deploy

Configure the host build with the plugin's directory on the plugin path and its
id selected, then build and run as usual. `HIPDNN_EP_COMPILER_PLUGIN_PATHS` is a
semicolon-separated list of out-of-tree plugin source dirs — each is
`add_subdirectory()`d into the host build so its `CMakeLists.txt` can register
its id; `HIPDNN_EP_COMPILER_PLUGINS` then selects which registered ids to link:

```
cmake -S hip-ep -B build \
  -DBUILD_HIP_TOOLS=ON \
  -DHIPDNN_EP_COMPILER_PLUGIN_PATHS=<your plugin repo> \
  -DHIPDNN_EP_COMPILER_PLUGINS=myvendor ...
cmake --build build
# hip-compiler defaults to LLVM IR (-o model.bc); the EP loads that bitcode
# through an in-process JIT -- there is no model.dll by default. Native AOT is
# opt-in via `--mode NATIVE ... -o model.dll`.
THEROCK_DIST=<therock> hip-compiler model.mlir -o model.bc
```

The plugin is compiled into `hip-compiler` / `hip-mlir-opt` / the EP DLL; there
is no runtime plugin path to set. Its passes and dialect are registered in the
shared MLIR state, and its bitcode is linked into the model module, regardless
of whether that module is JIT-loaded (the default) or emitted as a native DLL.

### 7. Test the plugin

A plugin is testable with the same tools the in-tree sample uses, and the
compile-side checks need no GPU:

- A **unit test** that runs the static registrar and asserts the plugin's
  contributions were recorded. Use
  [test/plugin/test_static_plugins.cpp](../../test/plugin/test_static_plugins.cpp)
  as the template.
- A **LIT test** that runs the pass at a pipeline point, built with the plugin
  selected. Because the plugin is statically linked, the pass is registered by
  name -- no per-test environment needed:

```
// RUN: hip-mlir-opt \
// RUN:     --pass-pipeline='builtin.module(func.func(my-vendor-lowering))' \
// RUN:     %s | FileCheck %s
```

These run identically on Windows and Linux (static linking, no symbol export).
Because a selected plugin is active for every compile in that build, run
plugin LIT tests in a plugin-enabled build scoped to the plugin's test
directory rather than mixed with tests that assume no plugin.

## Component layout

In-tree, the mechanism is:

| File | Role |
|---|---|
| [include/hip/Compiler/PluginAPI.h](../../include/hip/Compiler/PluginAPI.h) | `hipEpRegisterPlugin_<id>` entry contract + `HIP_EP_DEFINE_PLUGIN` |
| [include/hip/Compiler/PluginRegistry.h](../../include/hip/Compiler/PluginRegistry.h) | Registry class + `PipelineSlot` enum + accessors + `dispatchPluginRegistrationsOnce` |
| [cmake/HipEpPlugins.cmake](../../cmake/HipEpPlugins.cmake) | `HIPDNN_EP_COMPILER_PLUGINS`, `hipdnn_ep_compiler_plugin_register`, generated registrar |
| [lib/Compiler/StaticPlugins.cpp](../../lib/Compiler/StaticPlugins.cpp) | Static registrar: calls each selected plugin's entry once |
| [lib/Compiler/PluginRegistry.cpp](../../lib/Compiler/PluginRegistry.cpp) | Host-side function-pointer table + contribution storage |
| [lib/Dialect/Transforms/Pipelines.cpp](../../lib/Dialect/Transforms/Pipelines.cpp) | Slot hook (`addPluginPassesForSlot`) |
| [include/hip/InitAllPasses.h](../../include/hip/InitAllPasses.h) | Dialect hook (`loadAllDialects` applies plugin dialect registrations) |
| [lib/Target/LLVM/LLVMBackend.cpp](../../lib/Target/LLVM/LLVMBackend.cpp) | Links plugin bitcode into the model module |
| [lib/Compiler/CompilerDriver.cpp](../../lib/Compiler/CompilerDriver.cpp) | Appends plugin link paths / libraries |
| [test/plugin/sample_plugin/](../../test/plugin/sample_plugin/) | Worked example exercised in CI |
| [test/plugin/test_static_plugins.cpp](../../test/plugin/test_static_plugins.cpp) | Static-registrar round-trip unit test |

The sample plugin is built behind a CMake option and exercised by the unit
test and a LIT test, so the ABI stays exercised in CI even before any
downstream plugin exists.

## Custom ops: a plugin-owned dialect end-to-end

There are two ways a plugin can handle a custom op, by how much of the pipeline
the op must survive:

- **Lower it directly in a pass** (the [walkthrough above](#what-a-downstream-team-does)).
  A pass matches the model op (e.g. `onnx.Custom("myvendor.FusedOp")`) and
  rewrites it straight to a `wrap_*` runtime call. This is the simplest path and
  the right one when the op's entire lowering is a single local rewrite and it
  never needs to exist as a typed value across passes.
- **A first-class dialect op** (this section). The op is a real `vendor.add`-style
  op that flows through bufferization and the standard HIP→LLVM lowering like an
  in-tree op. Use this when the op must survive multiple passes, participate in
  bufferization, or carry its own types and shape inference.

The first-class path lets a plugin contribute its **own dialect op** that lives
across the whole pipeline -- introduced from a model op, bufferized like an
in-tree op, and lowered to a vendor kernel -- entirely from out-of-tree code, via
three idiomatic MLIR seams that ride the same [linkage](#linkage-model) as a
plugin pass:

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
3. **HIP -> LLVM lowering** -- `convert-hip-to-llvm` walks the bufferized
   module and, for every dialect that *registers* `ConvertToLLVMPatternInterface`,
   lets it add its lowering patterns and mark its ops illegal. This is the body
   of upstream's `populateConversionTargetFromOperation`, inlined so the interface
   lookup can be guarded by `Dialect::hasPromisedInterface`: several in-tree
   upstream dialects present after bufferization (`memref`/`func`/`arith`/`cf`)
   *promise* `ConvertToLLVMPatternInterface` but never register it (we lower those
   ops via the explicit `populate*ToLLVM` calls instead), and an unguarded
   `dyn_cast` onto such a dialect `report_fatal_error`s in an assertions-enabled
   MLIR build. The guard skips promised-but-unregistered dialects and touches only
   genuine implementors (a plugin vendor dialect, whose promise is resolved when
   its extension is added). The vendor op lowers to a call into the vendor runtime
   wrapper (contributed via `addRuntimeBitcode`), which launches the vendor kernel
   (contributed via `addLibrary`).

A vendor pass (registered + slotted as above, typically at
`AfterSimplifyOnnx`) introduces the op from a model op -- e.g. rewriting
`onnx.Add` into `vendor.add`. Because the in-tree `convert-onnx-to-hip` only
matches `onnx.*` ops, the vendor op passes through untouched to bufferization.

This is exercised end-to-end (compile + GPU run + numeric check) by the
`hip-ep-plugin` example's `vendor.add` op (`out = a + b`): `onnx.Add` ->
`vendor.add` -> bufferized -> `wrap_hip_ep_vendor_add` -> vendor HIP kernel.

## Pipeline composition

Beyond inserting a pass at a fixed slot, a downstream may want to see the full
pass set and compose -- or fully replace -- the pass order. Three mechanisms
support this today; a fourth is planned.

- **Expose all passes by name.** Every production pipeline pass is
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
- **Textual pipeline override.** The driver consults the
  `HIPDNN_EP_PIPELINE` env var (`CompilerDriver::runMLIRPasses`). When set, it
  strips an optional `builtin.module(...)` wrapper and runs `parsePassPipeline`
  on the inner text (composing from in-tree passes by name); when unset, the
  in-tree default runs unchanged. A parse failure is reported up front; the
  override resolves from the host's own registry, so this path does **not**
  require shared MLIR. **Interaction with `requestPipelineSlot`:** an override
  replaces the built-in pipeline wholesale, and slot injection runs only inside
  the built-in builders, so plugin slot requests do **not** run under an
  override (the driver warns once if any are pending). A custom pipeline that
  wants a plugin pass must name it directly in the `HIPDNN_EP_PIPELINE` string
  (e.g. `func.func(my-vendor-pass), onnx-to-hip-pipeline, hip-to-llvm-pipeline`).
- **Guardrails.** When an override replaces the order, the load-bearing ordering
  invariants (host-scalar materialization before pool allocation; output-allocator
  after buffer deallocation; affine lowering after strided-metadata expansion; the
  terminal `convert-hip-to-llvm` + `generate-interface` stages) become the
  overrider's responsibility. A post-pipeline check hard-fails with a clear
  message when the produced module lacks an `inference_compute` entry point after
  an override; enforcing the remaining ordering invariants automatically is not
  yet done.
- **Planned -- a registered pipeline builder.** For passes that need
  runtime-bound state a textual string cannot carry (e.g. the
  in-memory-filesystem variant of `convert-onnx-to-hip`), a plugin would
  register a pipeline *builder* that receives a context (the filesystem and
  pipeline options) and composes passes -- including in-tree stage builders --
  in C++. This is the full-fidelity path; it would share the
  [linkage model](#linkage-model).

**Stability:** pass and pipeline names are version-pinned compiler internals,
not a frozen cross-release contract -- a plugin that composes by name pins to
a release. (Renaming a pass is a documented breaking change for such plugins.)

The custom-op path and pipeline composition share the
[linkage model](#linkage-model): the dialect, type, and pass
registries are process-global MLIR state, so any plugin-registered op, pass,
or pipeline needs the host and plugin to share one MLIR instance.
