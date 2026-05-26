<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Authoring a hip-compiler Plugin

**Audience:** vendor / downstream backend engineers writing a plugin DLL
that ships proprietary ONNX ops, custom kernels, or MLIR passes on top
of `onnx-hipdnn-ep`.

**Status:** the ABI is in proposal status; PRs 1–4 of the rollout have
landed. Do not ship a plugin against the in-tree headers until the
design status flips to "Stable" in
[`docs/design/plugin-extension-api.md`](design/plugin-extension-api.md).

This guide is the practical companion to the full design doc. It covers
*how* to author a plugin, not *why* the API is shaped the way it is.

---

## 1. Mental model

A plugin is a regular shared library (`.dll` on Windows, `.so` on Linux)
that:

1. Exports one well-known C symbol — `hipEpGetPluginInfo` — that returns
   metadata + a callback function pointer.
2. Inside that callback, calls methods on a `HipEpPluginRegistry &` to
   contribute extensions: MLIR passes, runtime bitcode, library paths,
   and library names.
3. Is loaded by `hip-compiler` (or any host that links `LibHipCompiler`)
   when the user sets `HIP_EP_PLUGINS=path/to/plugin.dll`.

The host never calls back into the plugin after `RegisterCallbacks`
completes. The plugin's pass code, bitcode buffers, and library paths
must outlive every model compile in the process — in practice that
means static storage in the plugin DLL.

The shape is field-for-field aligned with the LLVM/MLIR upstream plugin
pattern (see Appendix C of the design doc); if you have written an
LLVM pass plugin or an MLIR dialect plugin, the surface here will look
familiar.

---

## 2. Quickstart: the in-tree sample as your worked example

`test/plugin/sample_plugin/` is a complete, building plugin that
exercises every callback. It is your reference implementation. Source
files:

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

extern "C" ::hip::compiler::HipEpPluginLibraryInfo
hipEpGetPluginInfo() {
  return {
      HIP_EP_PLUGIN_API_VERSION,
      "MyVendorPlugin",
      "0.1.0",
      &registerCallbacks,
  };
}
```

CMake (omitting the path for brevity — adapt to your repo layout):

```cmake
add_library(my_vendor_plugin SHARED my_plugin.cpp)
target_link_libraries(my_vendor_plugin PRIVATE
  MLIRPass MLIRIR MLIRSupport MLIRFuncDialect)
target_compile_features(my_vendor_plugin PRIVATE cxx_std_17)
set_target_properties(my_vendor_plugin PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  PREFIX ""
  OUTPUT_NAME "my_vendor_plugin"
)
```

Two requirements that are easy to miss:

- **`WINDOWS_EXPORT_ALL_SYMBOLS ON`** is required so
  `hipEpGetPluginInfo` is exported under its unmangled C name without
  a `.def` file. The same workaround upstream LLVM uses for its
  examples (`llvm/examples/Bye/Bye.cpp`).
- **The plugin links MLIR libraries directly**, not `LibHipCompiler`.
  `LibHipCompiler` ships as a static library inside `hip-compiler.exe`
  / `hip-compiler.dll`; there is nothing for a plugin DLL to import
  against. The plugin reaches `LibHipCompiler` purely through inline
  thunks in `PluginRegistry.h` (see section 6 of the design doc for
  the rationale).

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
  R.requestPipelineSlot(
      ::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
      "my-vendor-pass");
}
```

`PipelineSlot` is an append-only enum declared in
`include/hip/Compiler/PluginRegistry.h`. The current slot list is the
seven public seams in `lib/Dialect/Transforms/Pipelines.cpp`.
`AfterConvertOnnxToHip` is the most common slot for vendor lowerings
of `onnx.Custom` ops; pick the slot that matches when your transform
needs to run.

> **Known limitation (open question 6 in the design doc):** because
> `hip-compiler` and the plugin both link MLIR statically, MLIR's pass
> registry is duplicated per DLL. A pass registered through
> `mlir::PassRegistration<>` from inside the plugin DLL is therefore
> *not* visible to the host's pass registry, so
> `requestPipelineSlot` records the request correctly but the
> pipeline emits a warning at compile time and skips the pass. The
> resolution requires switching to a shared MLIR (the same problem
> upstream solves with `libLLVM.so`); this is tracked separately and
> does not affect runtime bitcode or external library contribution.

---

## 5. Contributing runtime LLVM bitcode

Bitcode contributions are the right path when you need to
*override* an in-tree symbol — the host links plugin bitcode with
`Linker::Flags::OverrideFromSrc`, so a vendor `wrap_*` symbol cleanly
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

Constraints:

- The buffer must remain valid for the lifetime of `hip-compiler`.
  The registry stores the pointer + size by reference; it does not
  copy. Plugin DLLs achieve this naturally by putting the bytes in
  the read-only data segment via a `static const unsigned char[]`.
- The bitcode must be parseable by `llvm::parseBitcodeFile` against
  the LLVM version `hip-compiler` was built with. In practice this
  means compiling with the same clang that `hip-compiler` was built
  against.
- Linking happens *after* the in-tree runtime, with
  `OverrideFromSrc`. Ordering between two plugins contributing the
  same symbol follows insertion order — second-registered wins, which
  matches lld's command-line semantics.

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

Both `addLibraryPath` and `addLibrary` accept either a bare name (as
above, resolved by lld-link / the clang driver against the path list)
or an absolute file path (in which case the path list is irrelevant).

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

## 7. Distribution checklist

Before shipping a plugin DLL to consumers:

- [ ] Plugin builds against the same LLVM major version as the
      `hip-compiler` install you're shipping for. (LLVM does not
      promise C++ ABI compatibility across major versions; see
      Open Question 2 in the design doc.)
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

## 8. Where to look next

- **Why the API is shaped this way:**
  [`docs/design/plugin-extension-api.md`](design/plugin-extension-api.md).
- **The C ABI struct + version macros:**
  [`include/hip/Compiler/PluginAPI.h`](../include/hip/Compiler/PluginAPI.h).
- **The registry methods you call from `RegisterCallbacks`:**
  [`include/hip/Compiler/PluginRegistry.h`](../include/hip/Compiler/PluginRegistry.h).
- **The host loader, in case you need to inspect what your plugin is
  exposing:** [`include/hip/Compiler/PluginLoader.h`](../include/hip/Compiler/PluginLoader.h)
  and `lib/Compiler/PluginLoader.cpp`.
- **A complete in-tree example exercising every callback:**
  `test/plugin/sample_plugin/`.

When in doubt, mirror the sample and prune. The sample is built and
tested in CI, so any divergence from its shape that breaks loading
will surface quickly.
