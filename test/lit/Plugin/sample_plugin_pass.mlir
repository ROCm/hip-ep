// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// REQUIRES: hip_static_plugins
//
// Validates the pass-instance half of the plugin surface end-to-end: the sample
// plugin registers an MLIR pass with mlir::PassRegistration<>() from its
// hipEpRegisterPlugin_sample entry, and the host resolves + runs it by name.
//
// The sample plugin is linked STATICALLY into hip-mlir-opt (via the static
// registrar, cmake/HipEpPlugins.cmake), so its registration lands in the host's
// single MLIR pass registry -- no symbol export, no dlopen. hip-mlir-opt's main
// runs dispatchPluginRegistrationsOnce() before parsing the command line, so
// --hip-ep-sample-print-functions is a recognised pass name.
//
// hip_static_plugins is set when the build selected the sample plugin
// (-DHIPDNN_EP_COMPILER_PLUGINS=sample). The default build selects no plugins,
// so this test is UNSUPPORTED there rather than failing. Static linking (no
// symbol export, no dlopen) means this works identically on Windows and Linux.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-ep-sample-print-functions %s 2>&1 | FileCheck %s

// CHECK: [hip-ep-sample] visited f
func.func @f(%arg0 : i32) -> i32 {
  return %arg0 : i32
}

// CHECK: [hip-ep-sample] visited g
func.func @g() {
  return
}
