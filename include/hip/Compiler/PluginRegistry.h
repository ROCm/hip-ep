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

// Public registry passed to a plugin's RegisterCallbacks.
//
// As of PR 5, all five plugin contributions are live and wired:
//
//   * registerPass<T>()        -- adds T to MLIR's pass registry
//                                 (CAVEAT: cross-DLL behaviour is
//                                 not yet correct, see Open Question
//                                 6 in the design doc; tracked as
//                                 a follow-up to this rollout).
//   * requestPipelineSlot(...) -- consulted by
//                                 lib/Dialect/Transforms/Pipelines.cpp
//                                 at each PipelineSlot enum value.
//   * addRuntimeBitcode(...)   -- linked into the model module by
//                                 lib/Target/LLVM/LLVMBackend.cpp
//                                 with `OverrideFromSrc` semantics
//                                 (see CAVEAT on `addRuntimeBitcode`
//                                 below).
//   * addLibraryPath(...)      -- appended to the lld-link
//   * addLibrary(...)             argument vector by
//                                 lib/Compiler/CompilerDriver.cpp.
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
  ///
  /// IMPORTANT for maintainers: any change to this struct (adding a
  /// new function pointer, changing a signature) MUST bump
  /// `HIP_EP_PLUGIN_API_VERSION` in PluginAPI.h. The static_assert
  /// below is a tripwire: if you grow the layout without updating
  /// the size sentinel, compilation fails and you remember to bump
  /// the version. This mirrors how LLVM upstream protects
  /// `PassPluginLibraryInfo` -- they bump their version on every
  /// layout change.
  struct VTable {
    void (*requestPipelineSlot)(void *self, int slot, const char *name,
                                std::size_t nameLen);
    void (*addRuntimeBitcode)(void *self, const void *data,
                              std::size_t sizeBytes);
    void (*addLibraryPath)(void *self, const char *path, std::size_t pathLen);
    void (*addLibrary)(void *self, const char *name, std::size_t nameLen);
  };

  /// Tripwire: the V1 vtable layout has exactly four function pointers.
  /// Any change to `VTable` makes this assertion fire; the compiler
  /// error is your reminder to bump `HIP_EP_PLUGIN_API_VERSION` in
  /// PluginAPI.h before the new layout ships.
  static_assert(sizeof(VTable) == 4 * sizeof(void (*)()),
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

  // ---------- MLIR passes (upstream-shaped) ---------------------------
  /// Wraps `mlir::PassRegistration<PassT>()`. The intent is that the
  /// plugin's pass becomes resolvable by name from `parsePassPipeline`
  /// at the requested `PipelineSlot`.
  ///
  /// **Known limitation (Open Question 6, tracked as follow-up to
  /// PR 5).** When `hip-compiler` is statically linked into the host
  /// (its only shipping mode today) AND the plugin DLL also links
  /// MLIR statically, the call below writes to the **plugin DLL's**
  /// copy of `mlir::passRegistry`, not the host's. The host's
  /// `parsePassPipeline` then fails to find the pass and emits a
  /// `[plugin-loader] WARNING: pass '...' not registered in MLIR's
  /// pass registry` line. The PR-2 LIT test for the sample plugin's
  /// pass is XFAIL'd for exactly this reason.
  ///
  /// Workarounds today:
  ///   - Plugin pass is registered but never executed (everything
  ///     else, including bitcode and library contribution, still
  ///     works). Useful for "show me the slot wiring" demos.
  ///
  /// Planned fix: route registerPass through the vtable so the host
  /// TU calls `mlir::registerPass(allocator)`, mirroring LLVM's
  /// `PassBuilder &`-based plugin pattern. See Open Question 6 in
  /// docs/design/plugin-extension-api.md.
  ///
  /// Defined inline because templates must be visible at the
  /// instantiation site (the plugin DLL). The plugin DLL therefore
  /// links against MLIR for `mlir::PassRegistration<T>`'s definition;
  /// no hip-compiler symbol is needed.
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
  ///
  /// Buffer ownership: the host **copies** the bytes during this call.
  /// The plugin's pointer/lifetime do not need to outlive
  /// `RegisterCallbacks` -- a stack buffer or transient allocation
  /// is fine. (Earlier versions of this design borrowed the pointer;
  /// PR 6 switched to a copy after we measured the cost: vendor
  /// runtime bitcode is 100 kB-1 MB, and the copy happens once per
  /// process at startup, well below noise.)
  ///
  /// `sizeBytes == 0` is treated as a no-op (with a one-line stderr
  /// warning) so a plugin that conditionally produces bitcode does
  /// not break the host's link with a cryptic
  /// `llvm::parseBitcodeFile` "file too small to contain bitcode
  /// header" error. See the impl in PluginRegistry.cpp.
  ///
  /// **Symbol override semantics (CAVEAT).** PR 3 wires this with
  /// `Linker::Flags::OverrideFromSrc`. Per the LLVM source
  /// (`llvm/lib/Linker/LinkModules.cpp::shouldLinkFromSource`), that
  /// flag is **unconditional and all-or-nothing**: every name
  /// collision between plugin bitcode and in-tree bitcode resolves
  /// to the plugin's definition, with no per-symbol opt-in. There
  /// is no facility today for "override these symbols and only
  /// these." Implication: any function or global in the plugin's
  /// bitcode that happens to share a name with an in-tree symbol
  /// silently shadows the in-tree definition. Vendors should
  /// prefix their symbols (`amd_internal_wrap_alloc` rather than
  /// `wrap_alloc`) until we either (a) switch to per-symbol opt-in
  /// override or (b) explicitly bless `OverrideFromSrc` as the
  /// design intent. See docs/plugin_authoring.md, "Symbol naming
  /// and override semantics."
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

/// Read the bitcode buffers recorded by every loaded plugin's
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

/// Read the library search paths recorded by every loaded plugin's
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
/// loaded plugin's `addLibrary` call, in the order they were
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
