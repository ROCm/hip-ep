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

/// Read `HIP_EP_PLUGINS` (semicolon-separated paths) and load every
/// plugin listed. The list is loaded **once per process**; subsequent
/// calls return the same vector. Plugins that fail to load are logged
/// (when `HIPDNN_EP_DEBUG` is set) and skipped — a single bad path
/// does not abort the rest of compilation.
///
/// Empty / unset env var is a normal "no plugins" case and returns
/// an empty vector.
const std::vector<HipEpPluginLoader> &loadPluginsOnce();

/// Load all plugins (via `loadPluginsOnce`) and invoke each plugin's
/// `RegisterCallbacks` against the per-process plugin registry.
/// Idempotent: subsequent calls are no-ops, so it is safe to call
/// from every entry point that might run before pipelines are built
/// (`hip-compiler` driver, `hip-mlir-opt` main, etc.).
///
/// After this returns:
///   - every plugin pass is in `mlir::PassRegistry` (global)
///   - every (slot, passName) request is queryable via
///     `hip::compiler::pluginPassesForSlot(slot)`
///   - every (future) bitcode buffer / library entry recorded by
///     plugins is queryable via the corresponding accessor
///
/// Cost when no plugins are configured: one `getenv` lookup, no DLL
/// loads.
void dispatchPluginRegistrationsOnce();

} // namespace hip::compiler

#endif // HIP_COMPILER_PLUGIN_LOADER_H
