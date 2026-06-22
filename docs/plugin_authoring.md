<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Authoring a hip-compiler Plugin

**Audience:** down-stream teams writing a plugin shared library that adds
ONNX ops, custom kernels, or MLIR passes on top of `onnx-hipdnn-ep`.

**Status:** the plugin ABI is provisional and not yet frozen. Treat
`HIP_EP_PLUGIN_API_VERSION` as a moving target and rebuild plugins when it
changes.

This guide is the practical companion to
[`docs/design/plugin-interface.md`](design/plugin-interface.md): it covers
*how* to author a plugin, while the design doc covers the architecture and
the rationale.

---

## 1. Mental model

A plugin is a regular shared library (`.dll` on Windows, `.so` on Linux)
that:

1. Exports one well-known C symbol — `hipEpGetPluginInfo` — that returns
   metadata + a callback function pointer.
2. Inside that callback, calls methods on a `HipEpPluginRegistry &` to
   contribute extensions: MLIR passes, a custom dialect and op (with its
   bufferization and HIP→LLVM-lowering interface models), runtime bitcode,
   library paths, and library names.
3. Is loaded by `hip-compiler` (or any host that links `LibHipCompiler`)
   when the user sets `HIP_EP_PLUGINS=path/to/plugin.dll`.

The host never calls back into the plugin after `RegisterCallbacks`
completes. The plugin's pass code, bitcode buffers, and library paths
must outlive every model compile in the process — in practice that
means static storage in the plugin DLL.

---

## 2. Quickstart: the in-tree sample as your worked example

`test/plugin/sample_plugin/` is a complete, building plugin that
exercises the pass, runtime-bitcode, and library callbacks. It is your
reference implementation for those. (The custom dialect + op callback,
`addDialectRegistration`, is shown by the separate end-to-end example
`rocm-ep-plugin`; see [section 7](#7-contributing-a-custom-op-a-vendor-dialect).)
Source files:

- [`sample_plugin.cpp`](../test/plugin/sample_plugin/sample_plugin.cpp) —
  the plugin entry point. Defines a sample MLIR pass, exports
  `hipEpGetPluginInfo`, and calls every method on the registry.
- [`sample_plugin_runtime.cpp`](../test/plugin/sample_plugin/sample_plugin_runtime.cpp) —
  a one-function source compiled to LLVM bitcode at build time and
  embedded as a `static const unsigned char[]` in the DLL.
- [`sample_lib.cpp`](../test/plugin/sample_plugin/sample_lib.cpp) —
  a one-function source compiled to a static library
  (`hip_ep_sample_lib.lib` on Windows / `libhip_ep_sample_lib.a` on
  Linux) that the plugin contributes via `addLibraryPath` +
  `addLibrary`.
- [`CMakeLists.txt`](../test/plugin/sample_plugin/CMakeLists.txt) —
  the build rules. Mirror this in your own plugin repo.

The unit test
[`test/plugin/test_plugin_loader.cpp`](../test/plugin/test_plugin_loader.cpp)
demonstrates how to drive the loader from C++ and asserts every public
contract.

To run the sample end-to-end:

```bash
# Build hip-compiler + sample plugin
cmake --build build/onnx-hipdnn-ep --config Release

# Set HIP_EP_PLUGINS and run any model compile.
# On PowerShell:
$env:HIP_EP_PLUGINS = "build/onnx-hipdnn-ep/bin/hip_ep_sample_plugin.dll"
hip-compiler.exe model.mlir -o model.dll
```

---

## 3. The minimal plugin

The smallest valid plugin is one that loads cleanly and does nothing.
Useful as a starting point for incremental development:

```cpp
#include "hip/Compiler/PluginAPI.h"
#include "hip/Compiler/PluginRegistry.h"

namespace {
void registerCallbacks(::hip::compiler::HipEpPluginRegistry & /*R*/) {
  // intentionally empty
}
} // namespace

extern "C" ::hip::compiler::HipEpPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
hipEpGetPluginInfo() {
  return {
      HIP_EP_PLUGIN_API_VERSION,
      "MyVendorPlugin",
      "0.1.0",
      &registerCallbacks,
  };
}
```

CMake — consume the upstream install tree via its package config:

```cmake
# Point at <upstream-install-prefix>/lib/cmake/HipDnnEp:
#   cmake -S . -B build -DHipDnnEp_DIR=<prefix>/lib/cmake/HipDnnEp ...
find_package(HipDnnEp CONFIG REQUIRED)
# The SAME MLIR build the host (hip-compiler) was compiled with. The package
# config records the build-time location in HipDnnEp_MLIR_DIR_HINT.
find_package(MLIR CONFIG REQUIRED)

add_library(my_vendor_plugin SHARED my_plugin.cpp)

# Link the plugin ABI HEADERS and MLIR HEADERS ONLY. Do NOT link the MLIR
# libraries: the plugin's MLIR symbols (mlir::PassRegistration, the pass
# registry, op TypeIDs) must stay UNDEFINED so they bind to the host's single
# copy at dlopen time. The host (hip-compiler / hip-mlir-opt) is built with
# HIPDNN_ENABLE_PLUGINS, which exports its statically-linked MLIR symbols
# (LLVM's export_executable_symbols_for_plugins, the same call mlir-opt makes).
# Linking MLIR into the plugin would give it a PRIVATE registry the host never
# reads, so its pass would be invisible (see section 4).
target_link_libraries(my_vendor_plugin PRIVATE HipDnnEp::plugin_headers)
target_include_directories(my_vendor_plugin PRIVATE ${MLIR_INCLUDE_DIRS})
target_compile_features(my_vendor_plugin PRIVATE cxx_std_17)
set_target_properties(my_vendor_plugin PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  PREFIX ""
  OUTPUT_NAME "my_vendor_plugin"
)
```

Requirements that are easy to miss:

- **Link MLIR HEADERS only, not the MLIR libraries.** The plugin's MLIR
  symbols stay undefined and resolve against the symbol-exporting host at
  `dlopen` time. This requires the host to have been built with
  `HIPDNN_ENABLE_PLUGINS` (default ON), which exports its MLIR symbols. A
  plugin that links the MLIR libraries (`MLIR` aggregate or the per-component
  archives) gets its own registry copy and its contributed pass is invisible
  to the host (see section 4).
- **Use the SAME MLIR build as the host.** The registry/op types are ABI
  objects: the plugin's MLIR headers must match the host's MLIR version and
  `LLVM_ENABLE_RTTI` setting.
- **`WINDOWS_EXPORT_ALL_SYMBOLS ON`** is required so `hipEpGetPluginInfo` is
  exported under its unmangled C name without a `.def` file.
- **The plugin links `HipDnnEp::plugin_headers`, not `LibHipCompiler`.**
  `LibHipCompiler` is a static library inside the host; there is nothing for a
  plugin to import against. The plugin reaches the host purely through the
  inline thunks in `PluginRegistry.h`.

---

## 4. Contributing an MLIR pass

Pattern (see `SamplePrintFunctionsPass` in the in-tree sample):

```cpp
struct MyVendorPass
    : public mlir::PassWrapper<MyVendorPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyVendorPass)

  llvm::StringRef getArgument() const final {
    return "my-vendor-pass"; // command-line / pipeline name
  }

  llvm::StringRef getDescription() const final {
    return "Vendor-specific transform";
  }

  void runOnOperation() override {
    // your transform here
  }
};

void registerCallbacks(::hip::compiler::HipEpPluginRegistry &R) {
  R.registerPass<MyVendorPass>();
  // The slot string is resolved with mlir::parsePassPipeline into the slot's
  // MODULE-level pass manager, so it follows --pass-pipeline syntax: a
  // func.func pass must carry its anchor nesting. MyVendorPass is an
  // OperationPass<func::FuncOp>, hence func.func(...); a ModuleOp / op-agnostic
  // pass would use the bare "my-vendor-pass".
  R.requestPipelineSlot(
      ::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
      "func.func(my-vendor-pass)");
}
```

`PipelineSlot` is an append-only enum declared in
`include/hip/Compiler/PluginRegistry.h`. The current slot list is the
seven public seams in `lib/Dialect/Transforms/Pipelines.cpp`.
`AfterConvertOnnxToHip` is the most common slot for vendor lowerings
of `onnx.Custom` ops; pick the slot that matches when your transform
needs to run.

> **Slot-string nesting:** the slot pass manager is anchored on
> `builtin.module`, so a non-module pass must be requested with its anchor
> (`func.func(<arg>)`, `func.func(<region-pass>)`, ...), exactly as you would
> write it for `--pass-pipeline`. A bare `<arg>` for a `func.func` pass fails
> to add and the pipeline prints a `[plugin-loader] WARNING` at compile time.

> **Prerequisite — host symbol export:** a contributed pass is only visible to
> the host when the host's MLIR pass registry is the one the plugin's
> `mlir::PassRegistration<>()` writes to. The host links MLIR statically, so it
> must be built with `HIPDNN_ENABLE_PLUGINS` (default ON), which exports its
> MLIR symbols (LLVM's `export_executable_symbols_for_plugins`); the plugin in
> turn links MLIR *headers only* (section 3) so its MLIR references bind to the
> host at load. If the host was built with the export off, or the plugin links
> the MLIR libraries, the registry is duplicated: `requestPipelineSlot` records
> the request but the pipeline warns and skips the pass. Runtime bitcode and
> external library contributions are unaffected either way.
>
> This binding is **ELF-only**: a Windows DLL must resolve all symbols at link
> time, so it links the MLIR libraries and gets its own registry. Plugin-pass
> registration therefore works on Linux but **not** on Windows with the
> static-MLIR build (it would need a shared-`libMLIR` build). The loader,
> bitcode, and library contributions work on every platform. See the design
> doc's "Linkage requirement" for the full story.

---

## 5. Contributing runtime LLVM bitcode

Bitcode contributions are the right path when you need to
*override* an in-tree symbol — the host links plugin bitcode with
`Linker::Flags::OverrideFromSrc`, so a vendor symbol cleanly
shadows the in-tree definition of the same name.

Pattern:

```cpp
extern "C" const unsigned char kVendorBitcode[];
extern "C" const std::size_t kVendorBitcodeSize;

void registerCallbacks(::hip::compiler::HipEpPluginRegistry &R) {
  if (kVendorBitcodeSize != 0) {
    R.addRuntimeBitcode(kVendorBitcode, kVendorBitcodeSize);
  }
}
```

The two extern symbols are produced by your build system. The in-tree
sample uses the same pipeline that produces `runtime_bc_data` — clang
`-emit-llvm` to a `.bc` file, then a Python script (`cmake/xxd.py`)
that converts the bytes to a C array. See
`test/plugin/sample_plugin/CMakeLists.txt` for a working end-to-end
example.

### Symbol naming and override semantics — important caveat

`Linker::Flags::OverrideFromSrc` is **unconditional and all-or-nothing**:
there is **no per-symbol opt-in** today. Every name collision between your
plugin's bitcode and the in-tree runtime resolves in favour of your plugin.
That is exactly what you want for the symbols you are deliberately overriding
(vendor-prefixed entry points), but it also silently shadows any *accidental*
name collision.

Practical implications:

- **Prefix your symbols.** Use `vendor_wrap_alloc` rather than
  `wrap_alloc`, `amd_internal_kernel_x` rather than `kernel_x`. This
  keeps deliberate overrides explicit and accidents impossible.
- **Audit your bitcode's symbol table** before shipping. `nm`,
  `llvm-nm`, or `dumpbin /symbols` will list everything your
  bitcode defines; cross-check against the in-tree runtime's
  `runtime_bc_data` symbols.
- If your design genuinely needs to override a specific named
  in-tree symbol, document it in your plugin's README so reviewers
  can audit it explicitly.

A future PR may switch this to per-symbol opt-in
(`addRuntimeBitcodeWithOverrides(buf, {"wrap_alloc"})`) — the
current behaviour is a starting point, not a final decision.

### Other constraints

- **Buffer lifetime: don't worry about it.** The host copies the bytes
  during `addRuntimeBitcode`, so a stack buffer or transient allocation is
  fine. The sample plugin uses a `static const unsigned char[]` only because
  that is the simplest way to embed a build-time-generated C array.
- **Empty buffers are a no-op.** `addRuntimeBitcode(nullptr, 0)`
  emits a `[plugin-loader] WARNING: ...` line and returns. You do
  not need to gate the call on `if (kVendorBitcodeSize != 0)` —
  the `if` check in the snippet above is illustrative, not
  required.
- The bitcode must be parseable by `llvm::parseBitcodeFile` against
  the LLVM version `hip-compiler` was built with. In practice this
  means compiling with the same clang that `hip-compiler` was built
  against.
- Linking happens *after* the in-tree runtime, with `OverrideFromSrc`. When
  two plugins contribute the same symbol, the later-registered one wins, by
  link order.

---

## 6. Contributing external libraries

Library contributions are the right path for vendor static libraries
(e.g., a custom kernel pack ABI-compatible with hip-compiler's
generated code) that the model module references at link time.

Pattern:

```cpp
void registerCallbacks(::hip::compiler::HipEpPluginRegistry &R) {
  R.addLibraryPath("/abs/path/to/dir/containing/libfoo");
  R.addLibrary("foo"); // matches libfoo.a or foo.lib
}
```

`addLibraryPath` takes a search directory. `addLibrary` takes either a bare
name (as above, resolved against the search paths) or an absolute path to a
library file (in which case the search paths are irrelevant).

Constraints:

- Plugin libraries are appended *after* the in-tree libraries on the
  lld-link command line. This means an in-tree library with the same
  symbol name wins; vendors who need to override an in-tree symbol
  should use the bitcode mechanism (section 5) instead.
- Library lookup happens entirely at the time hip-compiler links the
  model `.dll`. There is no install-time check that the library
  exists; if `addLibrary` references a name that lld-link cannot
  resolve, every model compile fails until the library is provided.
  (Plugins that bundle a static library should embed a
  filesystem-existence assertion at plugin load to fail fast.)
- The recorded path is taken verbatim. Use absolute paths for
  reliability — `hip-compiler`'s working directory at link time is
  not under the plugin's control.

---

## 7. Contributing a custom op (a vendor dialect)

The most complete extension is a vendor's **own dialect op** that lives across
the whole pipeline: introduced from a model op, bufferized like an in-tree op,
and lowered to a vendor kernel. The end-to-end worked example is the separate
`rocm-ep-plugin` repository's `vendor.add` op (`out = a + b`); this section is
the map to it.

A custom op uses `addDialectRegistration` together with the pass (section 4),
bitcode (section 5), and library (section 6) callbacks. The same shared-MLIR
requirement as `registerPass` applies (the dialect, op TypeIDs, and attached
interface models are process-global MLIR state) — link MLIR headers only, and
build against the host's MLIR (section 3).

### Register the dialect

`addDialectRegistration` hands the host a callback it runs against the
`mlir::DialectRegistry` the pipeline's `MLIRContext` is built from. Make it a
non-capturing function (so it converts to a plain function pointer). It does
exactly what an upstream `mlirGetDialectPluginInfo` callback does: insert the
dialect, and attach the op's interface models via `DialectExtension`s.

```cpp
static void registerVendorDialect(mlir::DialectRegistry &registry) {
  registry.insert<VendorDialect>();
  // Bufferization model for the op (tensor semantics -> memref).
  registry.addExtension(+[](mlir::MLIRContext *ctx, VendorDialect *) {
    VendorAddOp::attachInterface<VendorAddBufferizeModel>(*ctx);
  });
  // HIP->LLVM lowering for the op.
  registry.addExtension(+[](mlir::MLIRContext *, VendorDialect *d) {
    d->addInterfaces<VendorConvertToLLVMInterface>();
  });
}

void registerCallbacks(::hip::compiler::HipEpPluginRegistry &R) {
  R.addDialectRegistration(&registerVendorDialect);
  R.registerPass<ConvertOnnxAddToVendorPass>();   // introduces the op
  R.requestPipelineSlot(::hip::compiler::PipelineSlot::AfterSimplifyOnnx,
                        "func.func(my-onnx-to-vendor)");
  // ... addRuntimeBitcode + addLibrary for the op's runtime wrapper + kernel.
}
```

### The three seams the op flows through

1. **Introduce the op.** A vendor pass (section 4) rewrites a model op into the
   vendor op. Schedule it *before* `convert-onnx-to-hip` (the `AfterSimplifyOnnx`
   slot) when you are claiming an op the in-tree pipeline would otherwise lower
   itself; `convert-onnx-to-hip` only matches `onnx.*` ops, so it passes the
   vendor op through untouched.
2. **Bufferize.** `one-shot-bufferize` finds the op's attached
   `BufferizableOpInterface` model and rewrites tensor semantics to memref. If
   the bufferized op has memref operands, it must also implement
   `MemoryEffectOpInterface` (declaring which operands it reads and writes), or
   the ownership-based buffer-deallocation pass rejects it with "unknown memory
   side effects".
3. **Lower to LLVM.** `convert-hip-to-llvm` collects every dialect that
   implements `ConvertToLLVMPatternInterface` and lets it add its lowering
   patterns and mark its ops illegal. The vendor op lowers to a call into the
   vendor runtime wrapper contributed via `addRuntimeBitcode` (section 5), which
   launches the kernel contributed via `addLibrary` (section 6).

### Defining the dialect and op

`rocm-ep-plugin` defines its single op by hand (a plain C++ class registered
with `addOperations`) to avoid a TableGen build step. That is reasonable for one
trivial op; a real dialect with several ops should use ODS (a `.td` file with
`mlir_tablegen`), as the in-tree HIP dialect and the upstream `standalone`
example do — ODS generates the builders, verifiers, accessors, and interface
glue that are tedious and error-prone to write by hand.

See the upstream design doc's "Custom ops: a plugin-owned dialect end-to-end"
section for the same flow from the architecture side.

## 8. Composing a custom pipeline

Beyond inserting passes at fixed slots (section 4), the compiler lets you
compose -- or fully replace -- the pass order. Two mechanisms ship today; one
is planned.

**Available now:**

- **Reference any in-tree pass by name.** Every production pipeline pass is
  registered with a stable `getArgument()` name in the same registry the
  override resolves against (`include/hip/InitAllPasses.h::registerAllPasses`),
  so it is usable in a textual pipeline string. The composable pipeline names
  (`onnx-to-hip-pipeline`, `hip-to-llvm-pipeline`, `hipdnn-pipeline`) are
  registered too. See the published menu:
  [`docs/pipeline_pass_menu.md`](pipeline_pass_menu.md).
- **Supply a pipeline as text.** Set the `HIPDNN_EP_PIPELINE` env var to a
  pipeline string (a bare inner list, or a `builtin.module(...)` wrapper) that
  the driver parses with `parsePassPipeline`. This composes from in-tree passes
  and does **not** require shared MLIR. A few load-bearing passes are not
  individually name-registerable (`generate-interface`, `compile-hipdnn-graphs`,
  and some MLIR utility passes), so a hand-listed pipeline that needs the C-ABI
  entry point should compose the registered *pipeline names* (which include
  them) rather than enumerate every pass. The menu documents which is which.

> **`HIPDNN_EP_PIPELINE` bypasses `requestPipelineSlot`.** An override replaces
> the built-in pipeline wholesale, and slot injection (section 4) runs only
> inside the built-in builders. So a pass you scheduled with
> `requestPipelineSlot` does **not** run while `HIPDNN_EP_PIPELINE` is set (the
> driver warns once if a slot request is pending). If you compose your own
> pipeline, name your pass in the string directly, e.g.
> `func.func(my-vendor-pass), onnx-to-hip-pipeline, hip-to-llvm-pipeline`.

**Planned (not yet in the ABI):**

- **Register a pipeline builder.** For passes that need runtime-bound state a
  string cannot carry (e.g. the in-memory-filesystem variant of
  `convert-onnx-to-hip`), a plugin would register a C++ builder that receives
  the filesystem + options and composes passes, reusing in-tree stage builders.
  Like `registerPass`, this needs shared MLIR.

If you fully replace the order you own the load-bearing ordering invariants
and the required terminal stages (`convert-hip-to-llvm`, `generate-interface`);
a post-pipeline check fails loudly if the `inference_compute` entry point is
missing after the override. Pass and pipeline names are version-pinned, not a
frozen contract -- pin your plugin to a release.

See [`docs/design/plugin-interface.md`](design/plugin-interface.md), "Pipeline
composition", for the design, and
[`docs/pipeline_pass_menu.md`](pipeline_pass_menu.md) for the name/anchor/slot
reference.

## 9. Distribution checklist

Before shipping a plugin DLL to consumers:

- [ ] Plugin builds against the same LLVM/MLIR version as the
      `hip-compiler` install you are shipping for. C++ ABI compatibility is
      not guaranteed across versions, so rebuild when the host's LLVM moves.
- [ ] `hipEpGetPluginInfo` returns a non-empty `pluginName` and
      `pluginVersion`. The host loader rejects empty values with an
      `llvm::Error`.
- [ ] Plugin is loadable via `HIP_EP_PLUGINS=...` (semicolon-
      separated on Windows; same separator on Linux for parity) and
      its `RegisterCallbacks` runs end-to-end without crashing.
- [ ] Every bitcode buffer you pass to `addRuntimeBitcode` parses
      with `llvm::parseBitcodeFile` against the target host's LLVM
      version. The first four bytes must be the bitcode magic
      `42 43 C0 DE`.
- [ ] Every library name you pass to `addLibrary` resolves at link
      time on the host's filesystem layout (and not just on your
      build machine).
- [ ] CI on the consumer side runs at least one model end-to-end
      with `HIP_EP_PLUGINS` pointing at the plugin and verifies the
      expected output.

---

## 10. Where to look next

- **Design and rationale:**
  [`docs/design/plugin-interface.md`](design/plugin-interface.md).
- **The C ABI struct + version macros:**
  [`include/hip/Compiler/PluginAPI.h`](../include/hip/Compiler/PluginAPI.h).
- **The registry methods you call from `RegisterCallbacks`:**
  [`include/hip/Compiler/PluginRegistry.h`](../include/hip/Compiler/PluginRegistry.h).
- **The host loader, in case you need to inspect what your plugin is
  exposing:** [`include/hip/Compiler/PluginLoader.h`](../include/hip/Compiler/PluginLoader.h)
  and `lib/Compiler/PluginLoader.cpp`.
- **A complete in-tree example** exercising the pass, bitcode, and library
  callbacks: `test/plugin/sample_plugin/`.
- **An end-to-end custom-op example** (the `addDialectRegistration` path:
  dialect + op + bufferization + lowering + kernel, with a numeric check):
  the separate `rocm-ep-plugin` repository.

When in doubt, mirror the sample and prune. The sample is built and
tested in CI, so any divergence from its shape that breaks loading
will surface quickly.
