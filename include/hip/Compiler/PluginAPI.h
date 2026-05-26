/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_PLUGIN_API_H
#define HIP_COMPILER_PLUGIN_API_H

#include "llvm/Support/Compiler.h"
#include <cstdint>

// Plugin extension ABI for hip-compiler.
//
// This is the public C ABI surface that vendor plugins compile against.
// Field-for-field aligned with the upstream LLVM/MLIR plugin pattern
// (see `llvm/include/llvm/Plugins/PassPlugin.h` and
// `mlir/include/mlir/Tools/Plugins/PassPlugin.h`).
//
// See docs/design/plugin-extension-api.md for design notes,
// rationale, and the alignment matrix against upstream.
//
// STATUS (2026-05): proposal under vendor-team review. The struct
// layout is not yet frozen. Until PRs 1-5 land, plugins should treat
// HIP_EP_PLUGIN_API_VERSION as a moving target.

namespace hip::compiler {
class HipEpPluginRegistry;
} // namespace hip::compiler

namespace hip::compiler {

/// API version understood by this plugin.
///
/// The version is incremented for ANY ABI-breaking change to the
/// HipEpPluginLibraryInfo struct (callbacks added, removed, or
/// reordered). This matches the upstream LLVM/MLIR convention; we
/// intentionally do not split into major/minor. Drivers reject
/// mismatched versions.
#define HIP_EP_PLUGIN_API_VERSION 1

extern "C" {

/// Plugin info struct returned by hipEpGetPluginInfo().
///
/// Returned **by value** so the plugin owns no allocations the host
/// would have to free. The string fields point at static storage in
/// the plugin DLL and remain valid for the lifetime of the loaded
/// plugin (i.e., until the host calls dlclose / FreeLibrary, which
/// hip-compiler does not do today — plugins are loaded permanently
/// for the lifetime of the process).
struct HipEpPluginLibraryInfo {
  /// API version understood by this plugin. Should equal
  /// HIP_EP_PLUGIN_API_VERSION at the time the plugin was built.
  uint32_t APIVersion;

  /// Human-readable plugin name, used in load-time logging.
  /// E.g., "AMDInternalAcceleratorPlugin".
  const char *PluginName;

  /// Vendor's own version string, e.g., "1.2.3" or a git SHA.
  /// Used for diagnostic logging only; the host does not parse it.
  const char *PluginVersion;

  /// The single registration callback. The plugin uses the supplied
  /// registry to register passes, request pipeline-slot insertions,
  /// contribute runtime bitcode, and contribute external libraries.
  /// Called exactly once at plugin load.
  ///
  /// Set this to nullptr if the plugin has no registrations to make
  /// (e.g., a placeholder plugin used only for ABI smoke testing);
  /// the loader will still consider the load successful.
  void (*RegisterCallbacks)(HipEpPluginRegistry &);
};

} // extern "C"

} // namespace hip::compiler

/// Public entry point for a hip-compiler plugin.
///
/// The host (hip-compiler.dll) looks up this symbol by name in the
/// plugin DLL and calls it to obtain the plugin info struct.
///
/// `LLVM_ATTRIBUTE_WEAK` is intentional and matches upstream:
/// - On non-Windows targets it lets the same source be either
///   statically linked into a tool (where the tool resolves the
///   symbol at link time) or dynamically loaded.
/// - On Windows `LLVM_ATTRIBUTE_WEAK` is a no-op (per upstream
///   `llvm/Support/Compiler.h`); plugin authors should annotate the
///   *definition* with `__declspec(dllexport)` to ensure the symbol
///   is exported. CMake-side, setting
///   `WINDOWS_EXPORT_ALL_SYMBOLS` on the plugin target is the
///   simplest way to achieve this.
///
/// Example plugin implementation:
///
/// ```
/// extern "C" ::hip::compiler::HipEpPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
/// hipEpGetPluginInfo() {
///   return {
///       HIP_EP_PLUGIN_API_VERSION,
///       "MyVendorPlugin",
///       "0.1.0",
///       [](::hip::compiler::HipEpPluginRegistry &R) {
///         // PR 2: R.registerPass<MyPass>() + R.requestPipelineSlot(...)
///         // PR 3: R.addRuntimeBitcode(my_bc, my_bc_size)
///         // PR 4: R.addLibraryPath(...) + R.addLibrary(...)
///       }};
/// }
/// ```
extern "C" ::hip::compiler::HipEpPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
hipEpGetPluginInfo();

#endif // HIP_COMPILER_PLUGIN_API_H
