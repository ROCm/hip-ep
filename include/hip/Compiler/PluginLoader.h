/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_PLUGIN_LOADER_H
#define HIP_COMPILER_PLUGIN_LOADER_H

#include "hip/Compiler/PluginAPI.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

// Loader for hip-compiler plugins, modeled after `llvm::PassPlugin` and
// `mlir::PassPlugin`.
//
// Production callers should use `dispatchPluginRegistrationsOnce()`,
// which loads every DLL listed in `HIP_EP_PLUGINS` and runs each
// plugin's `RegisterCallbacks` exactly once per process. After it
// returns:
//   - all plugin passes are in MLIR's global pass registry
//   - all (slot, passName) requests are queryable via
//     `hip::compiler::pluginPassesForSlot(slot)` (see PluginRegistry.h)
//
// Tests and tools that need finer-grained control can use the lower
// level `loadPluginsOnce()` and dispatch `registerCallbacks` manually.

namespace hip::compiler {

class HipEpPluginRegistry;

/// One loaded plugin DLL. Mirrors `llvm::PassPlugin` and
/// `mlir::PassPlugin`.
class HipEpPluginLoader {
public:
  /// Load the plugin DLL at `filename`, look up the
  /// `hipEpGetPluginInfo` symbol, call it, and validate the returned
  /// `HipEpPluginLibraryInfo`.
  ///
  /// On success returns the loaded plugin (which holds a permanent
  /// reference to the underlying `llvm::sys::DynamicLibrary`).
  ///
  /// Failure modes (all return `llvm::Error`):
  ///  - DLL cannot be opened (path wrong, missing dependency, etc.).
  ///  - `hipEpGetPluginInfo` symbol not exported by the DLL.
  ///  - `APIVersion` returned by the plugin does not match
  ///    `HIP_EP_PLUGIN_API_VERSION` of this hip-compiler build.
  static llvm::Expected<HipEpPluginLoader> Load(const std::string &filename);

  /// Path the plugin was loaded from. Useful for diagnostics.
  llvm::StringRef getFilename() const { return filename_; }

  /// Plugin name reported by the plugin via `hipEpGetPluginInfo`.
  /// Stable for the lifetime of this `HipEpPluginLoader`.
  llvm::StringRef getPluginName() const { return info_.PluginName; }

  /// Plugin version reported by the plugin via `hipEpGetPluginInfo`.
  llvm::StringRef getPluginVersion() const { return info_.PluginVersion; }

  /// The API version reported by the plugin (matches
  /// `HIP_EP_PLUGIN_API_VERSION` after a successful `Load`).
  uint32_t getAPIVersion() const { return info_.APIVersion; }

  /// Invoke the plugin's `RegisterCallbacks` against `R`.
  ///
  /// No-op if the plugin set `RegisterCallbacks` to nullptr (which
  /// is allowed for placeholder/test plugins — see PluginAPI.h).
  void registerCallbacks(HipEpPluginRegistry &R) const {
    if (info_.RegisterCallbacks)
      info_.RegisterCallbacks(R);
  }

private:
  HipEpPluginLoader(std::string filename, llvm::sys::DynamicLibrary library,
                    HipEpPluginLibraryInfo info)
      : filename_(std::move(filename)), library_(library), info_(info) {}

  std::string filename_;
  // `llvm::sys::DynamicLibrary` is a value-type handle; the underlying
  // OS handle is owned by an internal LLVM cache (loaded permanently
  // via `getPermanentLibrary`). Copying / moving the loader is safe.
  llvm::sys::DynamicLibrary library_;
  HipEpPluginLibraryInfo info_;
};

/// Read `HIP_EP_PLUGINS` (semicolon-separated paths, on every
/// platform — colon clashes with Windows drive letters) and load
/// every plugin listed. Returns owning copies of each loaded plugin.
///
/// Lifetime / caching:
///   - The list is loaded **once per process**; subsequent calls
///     return the same vector. The env var is read on the first call
///     only — later changes to `HIP_EP_PLUGINS` in the same process
///     are ignored. Set the env var before the first compilation.
///   - Duplicate paths in `HIP_EP_PLUGINS` are deduplicated, so
///     `foo.dll;foo.dll` produces exactly one loaded plugin (and
///     thus exactly one `RegisterCallbacks` invocation).
///
/// Failure handling:
///   - A plugin that fails to load (bad path, missing symbol, API
///     mismatch) is **always** logged to stderr as a single-line
///     `[plugin-loader] WARNING: ...` and skipped. The rest of the
///     plugin list continues to load. We deliberately do not gate
///     this warning on `HIPDNN_EP_DEBUG`, because the most common
///     failure mode is a typo in `HIP_EP_PLUGINS` and a silent skip
///     led to "is my plugin loaded?" debugging sessions.
///   - Empty / unset env var is the normal "no plugins" case and
///     returns an empty vector.
const std::vector<HipEpPluginLoader> &loadPluginsOnce();

/// Load all plugins (via `loadPluginsOnce`) and invoke each plugin's
/// `RegisterCallbacks` against the per-process plugin registry.
/// Idempotent: subsequent calls are no-ops, so it is safe to call
/// from every entry point that might run before pipelines are built
/// (`hip-compiler` driver, `hip-mlir-opt` main, etc.).
///
/// A throwing plugin is contained: the loader catches `std::exception`
/// and `...` from `RegisterCallbacks` and logs a `[plugin-loader]
/// WARNING: ...` line, then continues with the next plugin. This
/// matters because a plugin built against a different MSVC CRT or
/// libstdc++ version can hit cross-DLL exception-propagation
/// undefined behaviour; we bound the blast radius rather than crash
/// the host.
///
/// After this returns:
///   - every plugin's `requestPipelineSlot(slot, name)` request is
///     queryable via `hip::compiler::pluginPassesForSlot(slot)`
///   - every plugin's `addRuntimeBitcode` buffer is queryable via
///     `hip::compiler::pluginBitcodeBuffers()`
///   - every plugin's `addLibraryPath` / `addLibrary` entry is
///     queryable via the corresponding accessor
///   - every plugin's `registerPass<T>()` call has run (but see the
///     CAVEAT on `HipEpPluginRegistry::registerPass` -- the pass is
///     registered in the **plugin DLL's** copy of MLIR's static
///     `passRegistry`, not the host's, so it is not yet resolvable
///     by name from `parsePassPipeline`).
///
/// Cost when no plugins are configured: one `getenv` lookup, no DLL
/// loads.
void dispatchPluginRegistrationsOnce();

} // namespace hip::compiler

#endif // HIP_COMPILER_PLUGIN_LOADER_H
