/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_PLUGIN_REGISTRY_H
#define HIP_COMPILER_PLUGIN_REGISTRY_H

#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>

// Public registry passed to a plugin's RegisterCallbacks.
//
// As of PR 2, registerPass and requestPipelineSlot are live: a
// plugin's callback hands its passes to MLIR's global pass registry
// and records (slot, passName) pairs that lib/Dialect/Transforms/
// Pipelines.cpp consults at each pipeline slot. The remaining
// methods are still stubs, filled in by subsequent PRs:
//
//   PR 3: addRuntimeBitcode(...)
//         (wired into lib/Target/LLVM/LLVMBackend.cpp)
//   PR 4: addLibraryPath(...), addLibrary(...)
//         (wired into lib/Compiler/CompilerDriver.cpp)
//
// This shape -- registry-passed-by-reference rather than five
// separate callbacks in the C struct -- matches LLVM
// (`PassBuilder &`), MLIR pass plugins (no-arg + global registry),
// and MLIR dialect plugins (`DialectRegistry *`). New capabilities
// are added by extending this class, not by changing the C struct in
// PluginAPI.h. Old plugin DLLs continue to load when the registry
// grows new methods, as long as the new methods follow the same
// inline-thunk-dispatching-through-vtable pattern below.
//
// Why a vtable, not direct method calls:
//
//   hip-compiler ships as a static library (`LibHipCompiler.lib`)
//   that's linked into a host process (the EP DLL, hip-mlir-opt,
//   etc.). It does *not* ship as a shared library. Plugin DLLs
//   therefore cannot resolve normal C++ method symbols across the
//   DLL boundary -- there is no `libhipcompiler.dll` to import from.
//
//   To bridge the boundary without forcing every host to ship an
//   extra DLL, the registry stores function pointers populated by
//   the host (in `lib/Compiler/PluginRegistry.cpp`). The methods
//   here are inline thunks that dispatch through those function
//   pointers. The plugin DLL therefore depends on:
//     - PluginRegistry.h (header-only, fully inline)
//     - MLIR (for `mlir::PassRegistration<T>` -- linked separately)
//   and no symbols from hip-compiler's source.
//
//   This is the same C-vtable trick COM, the V8 embedder API, and
//   ICU's plugin interface use. From the plugin author's
//   perspective the API still feels like a C++ class.
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
/// The plugin sees only inline thunks that dispatch through a
/// function-pointer table the host populates. See the file-level
/// comment for the rationale.
class HipEpPluginRegistry {
public:
  /// Function-pointer table the host fills in when constructing a
  /// registry. Plugins do not construct registries directly; they
  /// receive one by reference from the host's RegisterCallbacks
  /// dispatch and call methods on it.
  ///
  /// Stable layout:
  ///   - Pure-C signatures (no llvm::StringRef, no PipelineSlot --
  ///     `int` instead) so a plugin built against PR 5 headers can
  ///     load into a PR 2 host without any C++ ABI accidents.
  ///   - Each function takes the `self` opaque pointer first, the
  ///     same convention COM and the V8 embedder API use.
  ///   - **Append-only** across `HIP_EP_PLUGIN_API_VERSION` bumps.
  struct VTable {
    void (*requestPipelineSlot)(void *self, int slot, const char *name,
                                std::size_t nameLen);
    void (*addRuntimeBitcode)(void *self, const void *data,
                              std::size_t sizeBytes);
    void (*addLibraryPath)(void *self, const char *path, std::size_t pathLen);
    void (*addLibrary)(void *self, const char *name, std::size_t nameLen);
  };

  /// Constructed only by the host. The vtable must outlive the
  /// registry; in practice it is a process-static (see
  /// PluginRegistry.cpp).
  HipEpPluginRegistry(const VTable *vtable, void *self)
      : vtable_(vtable), self_(self) {}

  ~HipEpPluginRegistry() = default;
  HipEpPluginRegistry(const HipEpPluginRegistry &) = delete;
  HipEpPluginRegistry &operator=(const HipEpPluginRegistry &) = delete;

  // ---------- MLIR passes (upstream-shaped) ---------------------------
  /// Equivalent of `mlir::PassRegistration<PassT>`. The plugin's pass
  /// is added to MLIR's global pass registry; the public pipeline
  /// instantiates it by name at the requested `PipelineSlot`.
  ///
  /// Defined inline in this header (templates must be visible at the
  /// instantiation site -- the plugin DLL). The plugin DLL therefore
  /// links against MLIR for `mlir::PassRegistration<T>`'s definition.
  /// No hip-compiler symbol is needed.
  template <typename PassT> void registerPass() {
    mlir::PassRegistration<PassT>();
  }

  /// Request that a registered pass run at a named pipeline slot.
  /// `passName` is the pass's MLIR command-line name (the same one
  /// used for `--hip-mlir-opt --pass=...`).
  ///
  /// Records the (slot, passName) pair into the per-process registry
  /// consulted by `lib/Dialect/Transforms/Pipelines.cpp`.
  void requestPipelineSlot(PipelineSlot slot, llvm::StringRef passName) {
    vtable_->requestPipelineSlot(self_, static_cast<int>(slot), passName.data(),
                                 passName.size());
  }

  // ---------- Extensions beyond upstream ------------------------------
  /// Contribute LLVM bitcode that will be linked into model.dll via
  /// `llvm::Linker` AFTER the in-tree `runtime_bc_data` is linked.
  /// The buffer must remain valid for the lifetime of hip-compiler
  /// (typically static storage in the plugin DLL).
  ///
  /// PR 3: wired into `LLVMBackend.cpp::linkRuntimeModule` with
  /// `Linker::Flags::OverrideFromSrc` so vendor `wrap_*` symbols
  /// shadow in-tree ones.
  void addRuntimeBitcode(const void *data, std::size_t sizeBytes) {
    vtable_->addRuntimeBitcode(self_, data, sizeBytes);
  }

  /// Contribute one library search path, appended to the lld-link
  /// `/LIBPATH:` list in `discoverLibraries`.
  ///
  /// PR 4: wired.
  void addLibraryPath(llvm::StringRef path) {
    vtable_->addLibraryPath(self_, path.data(), path.size());
  }

  /// Contribute one library name (e.g., `vendor_kernels`) or a full
  /// path to a `.lib` file, appended to the lld-link command line.
  ///
  /// PR 4: wired.
  void addLibrary(llvm::StringRef nameOrFullPath) {
    vtable_->addLibrary(self_, nameOrFullPath.data(), nameOrFullPath.size());
  }

private:
  const VTable *vtable_;
  void *self_;
};

/// Process-wide plugin registry. Constructed lazily on first call;
/// every plugin's RegisterCallbacks is dispatched against the same
/// instance, so the recorded state is one process-wide view.
///
/// Used by:
///   - `dispatchPluginRegistrationsOnce` (PluginLoader.cpp) -- as
///     the registry passed to each plugin's `RegisterCallbacks`.
///   - The unit test in `test/plugin/test_plugin_loader.cpp`.
HipEpPluginRegistry &getProcessPluginRegistry();

/// Read the (slot, passName) pairs recorded by every loaded plugin's
/// `requestPipelineSlot` call, filtered by `slot`. The returned vector
/// references storage owned by the per-process plugin registry; each
/// `StringRef` is stable for the lifetime of the process.
///
/// Used by `lib/Dialect/Transforms/Pipelines.cpp` at each pipeline
/// slot to look up the requested plugin passes by name in MLIR's
/// global pass registry and add them to the active `PassManager`.
llvm::SmallVector<llvm::StringRef> pluginPassesForSlot(PipelineSlot slot);

} // namespace hip::compiler

#endif // HIP_COMPILER_PLUGIN_REGISTRY_H
