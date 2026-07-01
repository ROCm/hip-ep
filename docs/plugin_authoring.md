<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Authoring a hip-compiler Plugin

**Audience:** down-stream teams writing a plugin static library that adds
ONNX ops, custom kernels, or MLIR passes on top of `onnx-hipdnn-ep`.

**Status:** the plugin surface is provisional and not yet frozen. Treat
`HIP_EP_PLUGIN_API_VERSION` as a moving target and rebuild plugins when it
changes (they are built with the host, so a mismatch is a build error).

**Linkage:** plugins are linked **statically** into the host at configure time.
Because plugin and host are one binary, they share MLIR's process-global state,
so passes, custom dialects/ops, and pipeline composition all work — identically
on **Windows and Linux**, with no symbol export and no dynamic loader. Select a
plugin with `-DHIPDNN_EP_COMPILER_PLUGINS=<id>`. See the design doc's "Linkage
model" for the rationale.

This guide is the practical companion to
[`docs/design/plugin-interface.md`](design/plugin-interface.md): it covers
*how* to author a plugin, while the design doc covers the architecture and
the rationale.

---

## 1. Mental model

A plugin is a **static library** (linked into the host at configure time — see
[docs/design/plugin-interface.md](design/plugin-interface.md), "Linkage model")
that:

1. Defines one `extern "C"` entry point — `hipEpRegisterPlugin_<id>` (via the
   `HIP_EP_DEFINE_PLUGIN(<id>)` macro) — taking a `HipEpPluginRegistry &`.
2. Inside that entry, calls methods on the registry to contribute extensions:
   MLIR passes, a custom dialect and op (with its bufferization and
   HIP→LLVM-lowering interface models), runtime bitcode, library paths, and
   library names.
3. Is selected into the host with `-DHIPDNN_EP_COMPILER_PLUGINS=<id>`; a
   CMake-generated registrar calls its entry once per process. Any host that
   links `LibHipCompiler` (`hip-compiler`, `hip-mlir-opt`, the EP DLL) picks it
   up.

Because the plugin and host are one binary, they share MLIR's process-global
state (one pass registry, one set of op TypeIDs) with no symbol export and no
dlopen — identically on Windows and Linux. The host never calls back into the
plugin after registration completes; the plugin's pass code, bitcode buffers,
and library paths live in static storage for the process lifetime.

---

## 2. Quickstart: the in-tree sample as your worked example

`test/plugin/sample_plugin/` is a complete, building plugin that
exercises the pass, runtime-bitcode, and library callbacks. It is your
reference implementation for those. (The custom dialect + op callback,
`addDialectRegistration`, is shown by the separate end-to-end example
`hip-ep-plugin`; see [section 7](#7-contributing-a-custom-op-a-vendor-dialect).)
Source files:

- [`sample_plugin.cpp`](../test/plugin/sample_plugin/sample_plugin.cpp) —
  the plugin entry point. Defines a sample MLIR pass, defines
  `hipEpRegisterPlugin_sample` via `HIP_EP_DEFINE_PLUGIN(sample)`, and calls
  every method on the registry.
- [`sample_plugin_runtime.cpp`](../test/plugin/sample_plugin/sample_plugin_runtime.cpp) —
  a one-function source compiled to LLVM bitcode at build time and
  embedded as a `static const unsigned char[]` in the plugin lib.
- [`sample_lib.cpp`](../test/plugin/sample_plugin/sample_lib.cpp) —
  a one-function source compiled to a static library
  (`hip_ep_sample_lib.lib` on Windows / `libhip_ep_sample_lib.a` on
  Linux) that the plugin contributes via `addLibraryPath` +
  `addLibrary`.
- [`CMakeLists.txt`](../test/plugin/sample_plugin/CMakeLists.txt) —
  the build rules. Mirror this in your own plugin repo.

The unit test
[`test/plugin/test_static_plugins.cpp`](../test/plugin/test_static_plugins.cpp)
runs the static registrar and asserts every public contract.

To build and exercise the sample end-to-end, select it into the build:

```bash
# Configure with the sample plugin selected, then build.
cmake -S onnx-hipdnn-ep -B build/onnx-hipdnn-ep -DHIPDNN_EP_COMPILER_PLUGINS=sample ...
cmake --build build/onnx-hipdnn-ep --config Release

# The plugin is now statically linked into hip-compiler / hip-mlir-opt. Run the
# plugin LIT tests (scoped to the Plugin dir) and the unit test:
lit -sv build/onnx-hipdnn-ep/test/lit/Plugin
build/onnx-hipdnn-ep/bin/test_static_plugins
```

Note: a selected plugin is active for every compile in that build, so run the
plugin LIT tests in a plugin-enabled build scoped to the Plugin directory (the
default build selects no plugins, and those tests are UNSUPPORTED there).

---

## 3. The minimal plugin

The smallest valid plugin is one that loads cleanly and does nothing.
Useful as a starting point for incremental development:

```cpp
#include "hip/Compiler/PluginAPI.h"
#include "hip/Compiler/PluginRegistry.h"

// Defines extern "C" void hipEpRegisterPlugin_myvendor(HipEpPluginRegistry &R).
// `myvendor` must match hipdnn_ep_compiler_plugin_register(PLUGIN_ID myvendor ...)
// and -DHIPDNN_EP_COMPILER_PLUGINS=myvendor.
HIP_EP_DEFINE_PLUGIN(myvendor) {
  // intentionally empty
  (void)R;
}
```

CMake — this `CMakeLists.txt` is `add_subdirectory()`d into the host build (via
`-DHIPDNN_EP_COMPILER_PLUGIN_PATHS=<this dir>`), so the host's MLIR toolchain,
plugin-surface include path, and `hipdnn_ep_compiler_plugin_register()` helper
are already in scope. Guard on the helper so a stray standalone configure fails
with a clear message:

```cmake
if(NOT COMMAND hipdnn_ep_compiler_plugin_register)
  message(FATAL_ERROR
    "co-build this plugin into onnx-hipdnn-ep via "
    "-DHIPDNN_EP_COMPILER_PLUGIN_PATHS=<this dir>; it is not standalone.")
endif()

add_library(my_vendor_plugin STATIC my_plugin.cpp)

# The plugin surface headers (hip/Compiler/PluginAPI.h, PluginRegistry.h) are on
# the host's include path, inherited here; add MLIR headers. Declare the MLIR
# libraries the plugin references (PUBLIC) so they follow the plugin archive in
# static link order -- otherwise the plugin's mlir:: references are seen after
# the MLIR archives already passed on the host's link line and fail to resolve.
# This does NOT give the plugin a private MLIR instance: plugin and host are one
# binary. Link no hip-compiler symbol.
target_include_directories(my_vendor_plugin PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(my_vendor_plugin PUBLIC MLIRPass MLIRIR MLIRSupport)
target_compile_features(my_vendor_plugin PRIVATE cxx_std_17)
set_target_properties(my_vendor_plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Make it available; select with -DHIPDNN_EP_COMPILER_PLUGINS=myvendor.
hipdnn_ep_compiler_plugin_register(PLUGIN_ID myvendor TARGET my_vendor_plugin)
```

Requirements that are easy to miss:

- **Build a STATIC library and select it** with `-DHIPDNN_EP_COMPILER_PLUGINS=myvendor`.
  It is linked into the host as one binary, so the registration lands in the
  host's single MLIR registry — no symbol export, no dlopen, works on Windows
  and Linux.
- **Declare the MLIR libraries you reference** (PUBLIC) so static link order is
  satisfied; the host links MLIR too, so no duplicate MLIR instance results.
- **Use the SAME MLIR build as the host.** Because the plugin co-builds with the
  host, this is automatic: the registry/op types are ABI-compatible by
  construction.
- **The plugin links no `LibHipCompiler` symbol.** It reaches the host purely
  through the inline thunks in `PluginRegistry.h` (on the host include path);
  its `hipEpRegisterPlugin_<id>` is called by the host's generated registrar.

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
> to add and the pipeline prints a `[plugin] WARNING` at compile time.

> **Why the pass resolves:** a contributed pass is visible because the plugin is
> statically linked into the host — one binary, one MLIR pass registry — so the
> plugin's `mlir::PassRegistration<>()` writes the same registry the pipeline
> reads. No symbol export, no dlopen, and this works identically on Windows and
> Linux. (If a pass name still does not resolve, `requestPipelineSlot` records
> the request but the pipeline prints a `[plugin] WARNING` and skips it —
> check the pass's `getArgument()` string and the `func.func(...)` nesting.)

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
  emits a `[plugin] WARNING: ...` line and returns. You do
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
`hip-ep-plugin` repository's `vendor.add` op (`out = a + b`); this section is
the map to it.

A custom op uses `addDialectRegistration` together with the pass (section 4),
bitcode (section 5), and library (section 6) callbacks. Like `registerPass`, it
relies on the plugin sharing the host's one MLIR instance (the dialect, op
TypeIDs, and attached interface models are process-global MLIR state) — which
static linking guarantees (section 3), built against the host's MLIR.

### Register the dialect

`addDialectRegistration` hands the host a callback it runs against the
`mlir::DialectRegistry` the pipeline's `MLIRContext` is built from. Make it a
non-capturing function (so it converts to a plain function pointer). It inserts
the dialect and attaches the op's interface models via `DialectExtension`s.

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

`hip-ep-plugin` defines its single op by hand (a plain C++ class registered
with `addOperations`) to avoid a TableGen build step. That is reasonable for one
trivial op; a real dialect with several ops should use ODS (a `.td` file with
`mlir_tablegen`), as the in-tree HIP dialect does — ODS generates the builders,
verifiers, accessors, and interface glue that are tedious and error-prone to
write by hand.

See the design doc's "Custom ops: a plugin-owned dialect end-to-end" section for
the same flow from the architecture side.

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

Before shipping a plugin to consumers:

- [ ] Plugin builds as part of the same `onnx-hipdnn-ep` build (same LLVM/MLIR
      version, same flags). It is selected with
      `-DHIPDNN_EP_COMPILER_PLUGINS=<id>` and registered via
      `hipdnn_ep_compiler_plugin_register`.
- [ ] The entry point is `HIP_EP_DEFINE_PLUGIN(<id>)` and `<id>` matches the
      registration + selection.
- [ ] The static registrar dispatches your entry end-to-end without crashing
      (assert with a unit test modeled on `test_static_plugins.cpp`).
- [ ] Every bitcode buffer you pass to `addRuntimeBitcode` parses
      with `llvm::parseBitcodeFile` against the target host's LLVM
      version. The first four bytes must be the bitcode magic
      `42 43 C0 DE`.
- [ ] Every library name you pass to `addLibrary` resolves at link
      time on the host's filesystem layout (and not just on your
      build machine).
- [ ] CI runs at least one model end-to-end in a build with your plugin
      selected and verifies the expected output.

---

## 10. Where to look next

- **Design and rationale:**
  [`docs/design/plugin-interface.md`](design/plugin-interface.md).
- **The entry-point contract + `HIP_EP_DEFINE_PLUGIN`:**
  [`include/hip/Compiler/PluginAPI.h`](../include/hip/Compiler/PluginAPI.h).
- **The registry methods you call from your entry point:**
  [`include/hip/Compiler/PluginRegistry.h`](../include/hip/Compiler/PluginRegistry.h).
- **The static-plugin CMake machinery** (`HIPDNN_EP_COMPILER_PLUGINS`,
  `hipdnn_ep_compiler_plugin_register`): [`cmake/HipEpPlugins.cmake`](../cmake/HipEpPlugins.cmake)
  and the generated registrar `lib/Compiler/StaticPlugins.cpp`.
- **A complete in-tree example** exercising the pass, bitcode, and library
  callbacks: `test/plugin/sample_plugin/`.
- **An end-to-end custom-op example** (the `addDialectRegistration` path:
  dialect + op + bufferization + lowering + kernel, with a numeric check):
  the separate `hip-ep-plugin` repository.

When in doubt, mirror the sample and prune. The sample is built and
tested in CI, so any divergence from its shape that breaks loading
will surface quickly.
