/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_PLUGIN_REGISTRY_H
#define HIP_COMPILER_PLUGIN_REGISTRY_H

#include "llvm/ADT/StringRef.h"
#include <cstddef>

// Public registry passed to a plugin's RegisterCallbacks.
//
// In PR 1 the registry methods are stubs that simply record (or in
// the case of registerPass, ignore) the call so the loader and the
// public ABI surface can be exercised in isolation. Subsequent PRs
// fill in the bodies:
//
//   PR 2: registerPass<PassT>(), requestPipelineSlot(...)
//         (wired into lib/Dialect/Transforms/Pipelines.cpp)
//   PR 3: addRuntimeBitcode(...)
//         (wired into lib/Target/LLVM/LLVMBackend.cpp)
//   PR 4: addLibraryPath(...), addLibrary(...)
//         (wired into lib/Compiler/CompilerDriver.cpp)
//
// This shape — registry-passed-by-reference rather than five separate
// callbacks in the C struct — matches LLVM (`PassBuilder &`), MLIR
// pass plugins (no-arg + global registry), and MLIR dialect plugins
// (`DialectRegistry *`). New capabilities are added by extending this
// class, not by changing the C struct in PluginAPI.h. Old plugin DLLs
// continue to load when the registry grows new methods, as long as
// the new methods are non-virtual additions.
//
// ABI caveat: this header exposes templated and class-typed methods,
// so plugins are not pure-C consumers. They must be built against
// the same MLIR/LLVM the public hip-compiler is built against, just
// like upstream MLIR plugins. This is documented in the design
// (Open Question 2) and is the same constraint upstream lives with.
//
// See docs/design/plugin-extension-api.md.

namespace hip::compiler {

/// Well-defined slots in the public pipeline at which plugin passes
/// can be inserted.
///
/// **Append-only** across versions. Removing or renaming an enumerator
/// is an ABI break and bumps `HIP_EP_PLUGIN_API_VERSION`. Adding a
/// new enumerator at the **end** is forward-compatible: older plugins
/// simply do not reference the new slot.
///
/// The current slots match the public seams in
/// `lib/Dialect/Transforms/Pipelines.cpp`.
enum class PipelineSlot {
  /// After `SimplifyOnnxPass` in `buildOnnxToHipPipeline`.
  AfterSimplifyOnnx,
  /// After `OnnxLoopOutlinePass` in `buildOnnxToHipPipeline`.
  AfterOnnxLoopOutline,
  /// After `ConvertOnnxToHipPass` in `buildOnnxToHipPipeline`. The
  /// most common slot for vendor lowerings of `onnx.Custom` ops.
  AfterConvertOnnxToHip,
  /// In `buildOnnxToHipPipelineTail`, before bufferization.
  BeforeBufferization,
  /// In `buildOnnxToHipPipelineTail`, after pool allocation.
  AfterPoolAllocs,
  /// In `buildOnnxToHipPipelineTail`, before the final HIP→LLVM
  /// conversion.
  BeforeConvertHipToLLVM,
  /// In `buildOnnxToHipPipelineTail`, after `GenerateInterfacePass`.
  AfterGenerateInterface,
};

/// Public registry passed to plugins' RegisterCallbacks.
///
/// In PR 1 every method is a stub. The methods are non-virtual on
/// purpose: a plugin compiled against PR 1 headers and loaded into a
/// PR 4 host calls into the PR 4 method bodies (the addresses are
/// resolved at the host's load time, not the plugin's build time).
class HipEpPluginRegistry {
public:
  HipEpPluginRegistry() = default;
  ~HipEpPluginRegistry() = default;

  HipEpPluginRegistry(const HipEpPluginRegistry &) = delete;
  HipEpPluginRegistry &operator=(const HipEpPluginRegistry &) = delete;

  // ---------- MLIR passes (upstream-shaped) ---------------------------
  /// Equivalent of `mlir::PassRegistration<PassT>`. The plugin's pass
  /// is added to MLIR's global pass registry; the public pipeline
  /// instantiates it by name at the requested PipelineSlot.
  ///
  /// PR 1: stub (no-op).
  /// PR 2: delegates to `mlir::PassRegistration<PassT>`.
  template <typename PassT> void registerPass() {
    // Intentionally empty in PR 1. PR 2 fills in via
    //   mlir::PassRegistration<PassT>();
  }

  /// Request that a registered pass run at a named pipeline slot.
  /// `passName` is the pass's MLIR command-line name (the same one
  /// used for `--hip-mlir-opt --pass=...`).
  ///
  /// PR 1: stub.
  /// PR 2: records the (slot, passName) pair into a per-process
  /// registry consulted by `Pipelines.cpp`.
  void requestPipelineSlot(PipelineSlot slot, llvm::StringRef passName);

  // ---------- Extensions beyond upstream ------------------------------
  /// Contribute LLVM bitcode that will be linked into model.dll via
  /// `llvm::Linker` AFTER the in-tree `runtime_bc_data` is linked.
  /// The buffer must remain valid for the lifetime of hip-compiler
  /// (typically static storage in the plugin DLL).
  ///
  /// PR 1: stub.
  /// PR 3: wired into `LLVMBackend.cpp::linkRuntimeModule` with
  /// `Linker::Flags::OverrideFromSrc` so vendor `wrap_*` symbols
  /// shadow in-tree ones.
  void addRuntimeBitcode(const void *data, std::size_t sizeBytes);

  /// Contribute one library search path, appended to the lld-link
  /// `/LIBPATH:` list in `discoverLibraries`.
  ///
  /// PR 1: stub.
  /// PR 4: wired.
  void addLibraryPath(llvm::StringRef path);

  /// Contribute one library name (e.g., `vendor_kernels`) or a full
  /// path to a `.lib` file, appended to the lld-link command line.
  ///
  /// PR 1: stub.
  /// PR 4: wired.
  void addLibrary(llvm::StringRef nameOrFullPath);
};

} // namespace hip::compiler

#endif // HIP_COMPILER_PLUGIN_REGISTRY_H
