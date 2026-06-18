// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// REQUIRES: hip_plugins_enabled
//
// Validates the pass-instance half of the public plugin ABI end-to-end: a
// plugin loaded via HIP_EP_PLUGINS registers an MLIR pass with
// mlir::PassRegistration<>(), and the host resolves + runs it by name.
//
// This works because the tools are built with HIPDNN_ENABLE_PLUGINS, which
// makes hip-mlir-opt export its statically-linked MLIR symbols (LLVM's
// export_executable_symbols_for_plugins, the same call mlir-opt makes). The
// plugin's MLIR registry references then bind to the host's single copy at
// dlopen time, so the plugin's registration lands in the registry the host
// reads. (Built WITHOUT that export, host and plugin would have separate
// static MLIR registries and the lookup would miss -- hence the
// hip_plugins_enabled gate.) The slot-recording / bitcode / library halves of
// the ABI work regardless and are covered by test/plugin/test_plugin_loader.
//===----------------------------------------------------------------------===//

// RUN: env HIP_EP_PLUGINS=%hip-ep-sample-plugin \
// RUN:   hip-mlir-opt --hip-ep-sample-print-functions %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: [hip-ep-sample] visited f
func.func @f(%arg0 : i32) -> i32 {
  return %arg0 : i32
}

// CHECK: [hip-ep-sample] visited g
func.func @g() {
  return
}
