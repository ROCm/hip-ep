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
// This is the public C ABI surface that vendor plugins compile against. A
// plugin is a shared library that exports `hipEpGetPluginInfo()`; the
// compiler loads it, validates the API version, and invokes its single
// registration callback. See docs/design/plugin-interface.md for the
// design rationale.
//
// The struct layout is not yet frozen, so treat `HIP_EP_PLUGIN_API_VERSION`
// as provisional until the design is ratified.

namespace hip::compiler {
class HipEpPluginRegistry;
} // namespace hip::compiler

namespace hip::compiler {

/// API version understood by this plugin.
///
/// Incremented for ANY ABI-breaking change to the HipEpPluginLibraryInfo
/// struct (a callback added, removed, or reordered) OR to the
/// HipEpPluginRegistry::VTable layout (a registry capability added or
/// reordered). There is no major/minor split: the loader rejects any version
/// it does not equal.
///
/// History:
///   1 -- initial: registerPass, requestPipelineSlot, addRuntimeBitcode,
///        addLibraryPath, addLibrary.
///   2 -- added addDialectRegistration (vtable entry 5) for out-of-tree
///        dialect + op + bufferization/lowering-interface contribution.
#define HIP_EP_PLUGIN_API_VERSION 2

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
/// The host looks up this symbol by name in the plugin library and calls it
/// to obtain the plugin info struct.
///
/// `LLVM_ATTRIBUTE_WEAK` lets the same source be either statically linked
/// into a tool (symbol resolved at link time) or dynamically loaded. It is a
/// no-op on Windows, where the plugin must export the *definition*
/// explicitly: annotate it with `__declspec(dllexport)`, or set
/// `WINDOWS_EXPORT_ALL_SYMBOLS` on the plugin's CMake target.
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
///         R.registerPass<MyPass>();
///         R.requestPipelineSlot(
///             ::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
///             "my-pass");
///         R.addRuntimeBitcode(my_bc, my_bc_size);
///         R.addLibraryPath("/path/to/libs");
///         R.addLibrary("vendor_kernels");
///       }};
/// }
/// ```
extern "C" ::hip::compiler::HipEpPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
hipEpGetPluginInfo();

#endif // HIP_COMPILER_PLUGIN_API_H
