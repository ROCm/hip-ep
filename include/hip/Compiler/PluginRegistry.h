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
#include <string>

namespace mlir {
class DialectRegistry;
} // namespace mlir

// Public registry passed to a plugin's registration entry
// (`hipEpRegisterPlugin_<id>`). It exposes the contributions a plugin can make:
//
//   * registerPass<T>()        -- add a pass to MLIR's pass registry so the
//                                 pipeline can instantiate it by name at a
//                                 requested slot. Requires the host and plugin
//                                 to share one MLIR instance (see the method
//                                 comment).
//   * addDialectRegistration() -- contribute a vendor dialect (+ op
//                                 verifiers, bufferization models, and
//                                 HipToLLVM lowering via DialectExtension /
//                                 ConvertToLLVMPatternInterface) into the
//                                 DialectRegistry the pipeline's MLIRContext
//                                 is built from. Same shared-MLIR requirement
//                                 as registerPass.
//   * requestPipelineSlot(...) -- run a registered pass at a named slot; read
//                                 by lib/Dialect/Transforms/Pipelines.cpp.
//   * addRuntimeBitcode(...)   -- contribute LLVM bitcode linked into the
//                                 model module by
//                                 lib/Target/LLVM/LLVMBackend.cpp (see the
//                                 method comment for override semantics).
//   * addLibraryPath(...) /    -- contribute search paths and libraries to the
//     addLibrary(...)             native link, read by
//                                 lib/Compiler/CompilerDriver.cpp.
//
// Capabilities are methods on this class (each an inline thunk over the
// function-pointer table below) rather than bare pointers a plugin calls
// directly, so a new capability is added by extending this class without
// touching PluginAPI.h (the entry-point contract). The `static_assert` on the
// vtable size below fires on any layout change, as the reminder to bump
// HIP_EP_PLUGIN_API_VERSION in lockstep.
//
// Plugins are linked STATICALLY into the host (see cmake/HipEpPlugins.cmake and
// docs/design/plugin-interface.md, "Linkage model"): the plugin's static lib
// and the host are one binary, so the plugin and host share MLIR's process
// state by construction (one pass registry, one set of op TypeIDs) with no
// symbol export.
//
// Dispatch still goes through a function-pointer table (see PluginRegistry.cpp)
// rather than calling storage directly: harmless under static linking, and it
// keeps a plugin TU's dependency surface to this (fully inline) header plus
// MLIR for `mlir::PassRegistration<T>`.
//
// See docs/design/plugin-interface.md.

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

/// Public registry passed to a plugin's registration entry.
///
/// The plugin sees only inline thunks that dispatch through a
/// function-pointer table the host populates. See the file-level
/// comment for the rationale.
class HipEpPluginRegistry {
public:
  /// Function-pointer table the host fills in; the inline thunk methods below
  /// dispatch through it. Plugins receive a registry by reference from the
  /// static registrar (`dispatchPluginRegistrationsOnce`) and never construct
  /// one.
  ///
  /// Signatures are plain-C (`int` slot, `char*`/len rather than PipelineSlot /
  /// StringRef) and append-only, each taking the opaque `self` first. Any
  /// layout change should bump `HIP_EP_PLUGIN_API_VERSION` (a source marker,
  /// not a runtime gate under static linking); the static_assert below is the
  /// tripwire.
  struct VTable {
    void (*requestPipelineSlot)(void *self, int slot, const char *name,
                                std::size_t nameLen);
    void (*addRuntimeBitcode)(void *self, const void *data,
                              std::size_t sizeBytes);
    void (*addLibraryPath)(void *self, const char *path, std::size_t pathLen);
    void (*addLibrary)(void *self, const char *name, std::size_t nameLen);
    // V2: contribute a dialect registration callback. The callback is invoked
    // by the host against the DialectRegistry used to build the pipeline's
    // MLIRContext (see loadAllDialects). `mlir::DialectRegistry &` in the
    // signature is intentional: dialect contribution inherently requires the
    // plugin and host to share one MLIR build (same as registerPass), so the
    // MLIR ABI match this argument implies is already a precondition of using
    // it -- a bitcode/library-only plugin never touches this entry.
    void (*addDialectRegistration)(void *self,
                                   void (*registerFn)(mlir::DialectRegistry &));
  };

  /// Tripwire: the current vtable layout has exactly five function pointers.
  /// Any change to `VTable` makes this assertion fire; the compiler
  /// error is your reminder to bump `HIP_EP_PLUGIN_API_VERSION` in
  /// PluginAPI.h before the new layout ships.
  static_assert(sizeof(VTable) == 5 * sizeof(void (*)()),
                "VTable layout changed -- bump HIP_EP_PLUGIN_API_VERSION "
                "and update this assertion to match the new entry count.");

  /// Constructed only by the host. The vtable must outlive the
  /// registry; in practice it is a process-static (see
  /// PluginRegistry.cpp).
  HipEpPluginRegistry(const VTable *vtable, void *self)
      : vtable_(vtable), self_(self) {}

  ~HipEpPluginRegistry() = default;
  HipEpPluginRegistry(const HipEpPluginRegistry &) = delete;
  HipEpPluginRegistry &operator=(const HipEpPluginRegistry &) = delete;

  // ---------- MLIR passes ---------------------------------------------
  /// Register `PassT` in MLIR's global pass registry so the pipeline can
  /// instantiate it by name (via `parsePassPipeline`) at a requested
  /// `PipelineSlot`. Static linking puts the pass in the same registry
  /// `parsePassPipeline` reads (no split-registry caveat). Inline so the
  /// template is instantiated in the plugin against MLIR's own headers, needing
  /// no hip-compiler symbol.
  template <typename PassT> void registerPass() {
    mlir::PassRegistration<PassT>();
  }

  // ---------- Dialects (out-of-tree ops + interfaces) -----------------
  /// Contribute a dialect-registration callback. The host invokes `registerFn`
  /// against the `mlir::DialectRegistry` it uses to build the pipeline's
  /// MLIRContext (`hip::compiler::loadAllDialects`), so the callback is the
  /// place to insert the vendor dialect and attach its interface models:
  ///
  /// ```
  /// R.addDialectRegistration(+[](mlir::DialectRegistry &registry) {
  ///   registry.insert<VendorDialect>();
  ///   // bufferization model (DialectExtension -> attachInterface):
  ///   registry.addExtension(+[](mlir::MLIRContext *ctx, VendorDialect *) {
  ///     VendorScaleOp::attachInterface<VendorScaleBufferizeModel>(*ctx);
  ///   });
  ///   // HIP->LLVM lowering (ConvertToLLVMPatternInterface on the dialect):
  ///   registry.addExtension(+[](mlir::MLIRContext *, VendorDialect *d) {
  ///     d->addInterfaces<VendorConvertToLLVMInterface>();
  ///   });
  /// });
  /// ```
  ///
  /// Same one-MLIR-instance basis as `registerPass`: the dialect's op TypeIDs
  /// and attached interface models live in MLIR's per-context registries, and
  /// static linking puts the plugin and host in one binary sharing them.
  /// `registerFn` must be a non-capturing function (a `+[]` lambda or a free
  /// function) so it converts to a plain function pointer.
  void addDialectRegistration(void (*registerFn)(mlir::DialectRegistry &)) {
    vtable_->addDialectRegistration(self_, registerFn);
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

  // ---------- Runtime bitcode and link libraries ----------------------
  /// Contribute LLVM bitcode linked into the model module after the in-tree
  /// runtime bitcode. The host copies the bytes during this call, so a stack or
  /// transient buffer is fine. `sizeBytes == 0` is a no-op (with a stderr
  /// warning), so a plugin that produces bitcode conditionally can still call
  /// this unconditionally.
  ///
  /// CAVEAT -- symbol override: the link uses `Linker::Flags::OverrideFromSrc`,
  /// which is all-or-nothing -- every name collision with an in-tree symbol
  /// resolves to the plugin's definition, with no per-symbol opt-in, so a
  /// same-named plugin symbol silently shadows the in-tree one. Vendors should
  /// prefix their symbols (e.g. `amd_internal_wrap_alloc`, not `wrap_alloc`).
  /// See docs/plugin_authoring.md, "Symbol naming and override semantics."
  void addRuntimeBitcode(const void *data, std::size_t sizeBytes) {
    vtable_->addRuntimeBitcode(self_, data, sizeBytes);
  }

  /// Contribute one library search path, appended to the lld-link
  /// `/LIBPATH:` list in `discoverLibraries`.
  void addLibraryPath(llvm::StringRef path) {
    vtable_->addLibraryPath(self_, path.data(), path.size());
  }

  /// Contribute one library name (e.g., `vendor_kernels`) or a full
  /// path to a `.lib` file, appended to the lld-link command line.
  void addLibrary(llvm::StringRef nameOrFullPath) {
    vtable_->addLibrary(self_, nameOrFullPath.data(), nameOrFullPath.size());
  }

private:
  const VTable *vtable_;
  void *self_;
};

/// Process-wide plugin registry. Constructed lazily on first call;
/// every plugin's registration entry is dispatched against the same
/// instance, so the recorded state is one process-wide view.
///
/// Used by:
///   - `dispatchPluginRegistrationsOnce` (StaticPlugins.cpp) -- as
///     the registry passed to each statically-linked plugin's registration.
///   - The unit test in `test/plugin/test_static_plugins.cpp`.
HipEpPluginRegistry &getProcessPluginRegistry();

/// Invoke every statically-linked plugin's registration entry
/// (`hipEpRegisterPlugin_<id>`) against the per-process registry, exactly once
/// (`std::call_once`). Defined in `lib/Compiler/StaticPlugins.cpp`, into which
/// CMake generates the selected-plugin list (see cmake/HipEpPlugins.cmake).
///
/// No-op when no plugins were selected. Idempotent, so it is safe to call from
/// any entry point that might run before pipelines are built (the
/// `hip-compiler` driver, `hip-mlir-opt` main). A throwing plugin is contained
/// (logged + skipped), not fatal. After it returns, every plugin contribution
/// is queryable via the accessors below.
void dispatchPluginRegistrationsOnce();

/// Read the (slot, passName) pairs recorded by every registered plugin's
/// `requestPipelineSlot` call, filtered by `slot`. The returned vector
/// references storage owned by the per-process plugin registry; each
/// `StringRef` is stable for the lifetime of the process.
///
/// Used by `lib/Dialect/Transforms/Pipelines.cpp` at each pipeline
/// slot to look up the requested plugin passes by name in MLIR's
/// global pass registry and add them to the active `PassManager`.
llvm::SmallVector<llvm::StringRef> pluginPassesForSlot(PipelineSlot slot);

/// Read the dialect-registration callbacks recorded by every registered
/// plugin's `addDialectRegistration` call, in the order they were registered.
///
/// Used by `hip::compiler::loadAllDialects` (`include/hip/InitAllPasses.h`):
/// after the in-tree dialects are registered, each callback is invoked on the
/// same `mlir::DialectRegistry` so plugin dialects + their bufferization /
/// HIP->LLVM-lowering interface models are present in the pipeline's context.
llvm::SmallVector<void (*)(mlir::DialectRegistry &)>
pluginDialectRegistrations();

/// One bitcode buffer contributed by a plugin's `addRuntimeBitcode` call. The
/// bytes are host-owned (copied during the call) and stable for the process
/// lifetime.
struct PluginBitcodeBuffer {
  const void *data;
  std::size_t sizeBytes;
};

/// Bitcode buffers from every plugin's `addRuntimeBitcode` call, in
/// registration order. Each `data` pointer is host-owned and stable for the
/// process lifetime.
///
/// Used by `LLVMBackend::linkRuntimeModule` to link plugin bitcode into the
/// model module after the in-tree runtime with `Linker::Flags::OverrideFromSrc`
/// (see `addRuntimeBitcode` for that flag's override caveat).
llvm::SmallVector<PluginBitcodeBuffer> pluginBitcodeBuffers();

/// Library search paths from every plugin's `addLibraryPath` call, in
/// registration order. Returns owning `std::string` copies so callers cannot
/// alias internal storage (negligible -- typically 0-2 entries).
///
/// Used by `CompilerDriver::discoverLibraries` to extend the `library_paths`
/// handed to lld-link (`/LIBPATH:<path>` on Windows, `-L<path>` on Linux),
/// appended *after* the in-tree paths so in-tree libraries resolve first.
llvm::SmallVector<std::string> pluginLibraryPaths();

/// Library names (or full paths) from every plugin's `addLibrary` call, in
/// registration order. Owning copies; see `pluginLibraryPaths`.
///
/// Used by `CompilerDriver::discoverLibraries` to extend the `libraries` handed
/// to lld-link, appended *after* the in-tree libraries. Link search order means
/// a later plugin lib only contributes new symbols -- it cannot shadow an
/// in-tree one; to override an in-tree symbol use `addRuntimeBitcode` instead.
llvm::SmallVector<std::string> pluginLibraries();

} // namespace hip::compiler

#endif // HIP_COMPILER_PLUGIN_REGISTRY_H
