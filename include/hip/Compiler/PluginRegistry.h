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
// Capabilities live on this class rather than as separate function pointers in
// the C struct, so a new capability is added by extending this class without
// changing PluginAPI.h -- the C struct stays at a single callback, and an
// older plugin keeps loading as long as new methods follow the inline-thunk
// pattern below.
//
// Plugins are linked STATICALLY into the host (see cmake/HipEpPlugins.cmake and
// docs/design/plugin-interface.md, "Linkage model"): the plugin's static lib
// and the host are one binary, so the plugin and host share MLIR's process
// state by construction (one pass registry, one set of op TypeIDs) with no
// symbol export.
//
// The registry still dispatches through a function-pointer table (see
// PluginRegistry.cpp) rather than calling storage directly. That indirection is
// retained -- it is harmless under static linking and keeps the plugin's
// dependency surface to this header (fully inline) plus MLIR (for the
// definition of `mlir::PassRegistration<T>`), so a plugin TU compiles against
// only public headers.
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
  /// Function-pointer table the host fills in when constructing a
  /// registry. Plugins do not construct registries directly; they
  /// receive one by reference from the host's static registrar
  /// (`dispatchPluginRegistrationsOnce`) and call methods on it.
  ///
  /// Layout:
  ///   - Pure-C signatures (no llvm::StringRef, no PipelineSlot -- `int`
  ///     instead), which keeps the plugin's dependency on this header a plain
  ///     function-pointer shape. (Under static linking plugin and host are one
  ///     build, so this is no longer needed to survive cross-version loading --
  ///     it is kept because it is harmless and minimal.)
  ///   - Each function takes the `self` opaque pointer first.
  ///   - Append-only.
  ///
  /// IMPORTANT for maintainers: any change to this struct (a new function
  /// pointer or a changed signature) should bump `HIP_EP_PLUGIN_API_VERSION` in
  /// PluginAPI.h (a source/versioning marker; not a runtime gate under static
  /// linking). The static_assert below is the tripwire -- growing the layout
  /// without updating the size sentinel fails the build.
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
  /// Registers `PassT` so the pipeline can instantiate it by name (via
  /// `parsePassPipeline`) at a requested `PipelineSlot`.
  ///
  /// The registration writes MLIR's process-global pass registry. Because the
  /// plugin is linked statically into the host (one binary, one MLIR
  /// instance), the pass lands in the SAME registry `parsePassPipeline` reads,
  /// so it resolves by name -- no shared-library / split-registry caveat.
  ///
  /// Defined inline because the template must be instantiated in the plugin;
  /// the plugin gets `mlir::PassRegistration<T>` from MLIR headers and needs no
  /// hip-compiler symbol.
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
  /// Contribute LLVM bitcode linked into the model module, after the in-tree
  /// runtime bitcode.
  ///
  /// Buffer ownership: the host copies the bytes during this call, so the
  /// plugin's pointer need not outlive the registration entry -- a stack or
  /// transient buffer is fine. The copy is a one-time, startup cost.
  ///
  /// `sizeBytes == 0` is a no-op (with a one-line stderr warning) so a plugin
  /// that produces bitcode conditionally does not break the link with an
  /// opaque bitcode-parse error.
  ///
  /// Symbol-override semantics (CAVEAT): the link uses
  /// `Linker::Flags::OverrideFromSrc`, which is unconditional and
  /// all-or-nothing -- every name collision with an in-tree symbol resolves
  /// to the plugin's definition, with no per-symbol opt-in. A plugin symbol
  /// that happens to share a name with an in-tree symbol therefore silently
  /// shadows it. Vendors should prefix their symbols (e.g.
  /// `amd_internal_wrap_alloc` rather than `wrap_alloc`). See
  /// docs/plugin_authoring.md, "Symbol naming and override semantics."
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
/// per process (`std::call_once`). Defined in `lib/Compiler/StaticPlugins.cpp`,
/// which CMake generates the plugin list into (see cmake/HipEpPlugins.cmake).
///
/// No-op when no plugins were selected (`HIPDNN_EP_COMPILER_PLUGINS` empty).
/// Idempotent, so it is safe to call from every entry point that might run
/// before pipelines are built (the `hip-compiler` driver, `hip-mlir-opt`
/// main, etc.). A throwing plugin is contained (logged + skipped), matching
/// the earlier dynamic loader's degrade-and-continue posture.
///
/// After this returns, every selected plugin's `requestPipelineSlot` /
/// `addRuntimeBitcode` / `addLibrary` / `addDialectRegistration` /
/// `registerPass<T>()` calls have run and are queryable via the accessors
/// below. Because the plugin is statically linked, `registerPass<T>()` lands in
/// the host's single MLIR pass registry (no shared-library caveat).
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

/// One bitcode buffer contributed by a plugin's `addRuntimeBitcode`
/// call. The bytes are owned by the per-process plugin registry
/// (the host copies the plugin's data during the
/// `addRuntimeBitcode` call) and remain valid for the lifetime of
/// the process. `data` is stable across the lifetime of the
/// returned buffer view.
struct PluginBitcodeBuffer {
  const void *data;
  std::size_t sizeBytes;
};

/// Read the bitcode buffers recorded by every registered plugin's
/// `addRuntimeBitcode` call, in the order they were registered.
/// Each `data` pointer is host-owned and stable for the lifetime of
/// the process.
///
/// Used by `lib/Target/LLVM/LLVMBackend.cpp::linkRuntimeModule` to
/// link plugin-contributed bitcode into the model module after the
/// in-tree runtime, with `Linker::Flags::OverrideFromSrc` so vendor
/// definitions shadow in-tree ones. See `addRuntimeBitcode` for the
/// caveat on what "shadow" means in that flag's actual LLVM
/// semantics.
llvm::SmallVector<PluginBitcodeBuffer> pluginBitcodeBuffers();

/// Read the library search paths recorded by every registered plugin's
/// `addLibraryPath` call, in the order they were registered.
///
/// Returns owning `std::string` copies (rather than `StringRef`s
/// into the per-process registry) so callers cannot accidentally
/// alias internal storage. The cost is one short-string copy per
/// entry, which is negligible -- there are typically 0-2 entries
/// per process.
///
/// Used by `lib/Compiler/CompilerDriver.cpp::discoverLibraries` to
/// extend the `library_paths` argument vector handed to lld-link
/// (rendered as one `/LIBPATH:<path>` per entry on Windows; `-L<path>`
/// on Linux). Paths are appended *after* the in-tree paths so that
/// in-tree libraries continue to resolve from their canonical
/// location.
llvm::SmallVector<std::string> pluginLibraryPaths();

/// Read the library names (or full library paths) recorded by every
/// registered plugin's `addLibrary` call, in the order they were
/// registered. Returns owning `std::string` copies; see
/// `pluginLibraryPaths` for the rationale.
///
/// Used by `lib/Compiler/CompilerDriver.cpp::discoverLibraries` to
/// extend the `libraries` argument vector handed to lld-link.
/// Entries are appended *after* the in-tree libraries; lld-link
/// command-line search order means a plugin lib later in the list
/// only contributes new symbols, it does not shadow in-tree ones.
/// Vendors who need to override an in-tree symbol should use the
/// bitcode mechanism (`addRuntimeBitcode`) instead -- see the
/// caveat on its symbol override semantics.
llvm::SmallVector<std::string> pluginLibraries();

} // namespace hip::compiler

#endif // HIP_COMPILER_PLUGIN_REGISTRY_H
