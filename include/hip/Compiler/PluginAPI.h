/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_COMPILER_PLUGIN_API_H
#define HIP_COMPILER_PLUGIN_API_H

// Plugin extension entry point for hip-compiler.
//
// Plugins are linked STATICALLY into the host at configure time (see
// cmake/HipEpPlugins.cmake and docs/design/plugin-interface.md). A plugin is a
// static library that defines ONE registration entry point:
//
//   extern "C" void hipEpRegisterPlugin_<id>(hip::compiler::HipEpPluginRegistry
//   &R);
//
// where `<id>` matches the plugin's id in `HIPDNN_EP_COMPILER_PLUGINS`. The
// host's CMake-generated registrar (StaticPlugins.cpp) declares and CALLS this
// function once per process; the body uses the supplied registry to register
// passes, dialects/ops, pipeline-slot requests, runtime bitcode, and link
// libraries.
//
// Why static rather than a runtime-loaded shared library: an MLIR-contributing
// plugin must share the host's process-global MLIR state (one pass registry,
// one set of op TypeIDs). Sharing that across a dynamic-load boundary would
// require the host to export its entire `mlir::` symbol surface, which exceeds
// the export-table capacity some object formats impose. Static linking
// sidesteps the question entirely: the plugin and host become one binary that
// shares MLIR by construction, with no symbol export and identical behavior on
// every platform.
//
// There is no runtime version handshake: because the plugin and host are built
// and linked together, an ABI mismatch is a build error, not a load-time
// surprise. HIP_EP_PLUGIN_API_VERSION remains only as a documentation / source
// marker for the registry surface.

namespace hip::compiler {
class HipEpPluginRegistry;
} // namespace hip::compiler

/// Source-level marker for the plugin registry surface (see PluginRegistry.h).
/// Bumped when the HipEpPluginRegistry method set changes. Not consulted at
/// runtime -- static linking guarantees plugin and host agree by construction.
///
/// History:
///   1 -- registerPass, requestPipelineSlot, addRuntimeBitcode,
///        addLibraryPath, addLibrary.
///   2 -- added addDialectRegistration (dialect + op + bufferization/lowering
///        interface contribution).
///   3 -- static linkage model: per-id `hipEpRegisterPlugin_<id>` entry point
///        replaces the dynamic `hipEpGetPluginInfo` + `HipEpPluginLibraryInfo`.
#define HIP_EP_PLUGIN_API_VERSION 3

/// Convenience macro for a plugin's registration entry point. Expands to the
/// correctly-named `extern "C"` function signature; follow it with a body.
///
/// ```
/// HIP_EP_DEFINE_PLUGIN(myvendor) {
///   R.registerPass<MyPass>();
///   R.requestPipelineSlot(
///       ::hip::compiler::PipelineSlot::AfterConvertOnnxToHip, "my-pass");
///   R.addRuntimeBitcode(my_bc, my_bc_size);
///   R.addLibraryPath("/path/to/libs");
///   R.addLibrary("vendor_kernels");
/// }
/// ```
///
/// The `<id>` (here `myvendor`) must match the id passed to
/// `hipdnn_ep_compiler_plugin_register(PLUGIN_ID myvendor ...)` and listed in
/// `HIPDNN_EP_COMPILER_PLUGINS`.
#define HIP_EP_DEFINE_PLUGIN(id)                                               \
  extern "C" void hipEpRegisterPlugin_##id(                                    \
      ::hip::compiler::HipEpPluginRegistry &R)

#endif // HIP_COMPILER_PLUGIN_API_H
