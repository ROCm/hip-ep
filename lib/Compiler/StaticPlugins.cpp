/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/PluginRegistry.h"

#include "llvm/Support/raw_ostream.h"

#include <exception>
#include <mutex>

// Static compiler-plugin dispatch (see cmake/HipEpPlugins.cmake).
//
// Statically-linked plugins are selected at configure time via
// `HIPDNN_EP_COMPILER_PLUGINS`; CMake generates
// `hip/Compiler/StaticLinkedPlugins.inc` with one `HANDLE_PLUGIN_ID(<id>)` line
// per selected plugin. We include it twice: once to declare each plugin's
// per-id registration entry, once to CALL it. When no plugins are selected the
// include is empty and `dispatchPluginRegistrationsOnce()` is a no-op.
//
// Correctness: the registration MUST be the explicit call below, never a static
// initializer in the plugin lib. The linker's dead-code elimination
// (`--gc-sections`, `/OPT:REF`) discards an object whose symbols are otherwise
// unreferenced; the explicit call is the reference that keeps the plugin's
// registration object alive.

namespace hip::compiler {

// Declare each selected plugin's per-id entry point. Each plugin's static lib
// defines `extern "C" void hipEpRegisterPlugin_<id>(HipEpPluginRegistry &)`.
#define HANDLE_PLUGIN_ID(id)                                                   \
  extern "C" void hipEpRegisterPlugin_##id(                                    \
      ::hip::compiler::HipEpPluginRegistry &);
#include "hip/Compiler/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID

void dispatchPluginRegistrationsOnce() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    HipEpPluginRegistry &registry = getProcessPluginRegistry();
    (void)registry; // silences "unused" when no plugins are selected.

    // A plugin that throws is contained: log and continue with the next one,
    // rather than aborting the host. (Same degrade-and-continue posture the
    // dynamic loader used.)
#define HANDLE_PLUGIN_ID(id)                                                   \
  try {                                                                        \
    hipEpRegisterPlugin_##id(registry);                                        \
  } catch (const std::exception &e) {                                          \
    llvm::errs() << "[plugin] WARNING: hipEpRegisterPlugin_" #id               \
                 << " threw std::exception: " << e.what()                      \
                 << "; skipping this plugin.\n";                               \
  } catch (...) {                                                              \
    llvm::errs() << "[plugin] WARNING: hipEpRegisterPlugin_" #id               \
                 << " threw a non-std exception; skipping this plugin.\n";     \
  }
#include "hip/Compiler/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID
  });
}

} // namespace hip::compiler
