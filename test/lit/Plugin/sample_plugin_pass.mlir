// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// REQUIRES: hip_static_plugins
//
// Pass half of the plugin surface: the sample plugin registers a pass from
// hipEpRegisterPlugin_sample and hip-mlir-opt resolves + runs it by name
// (dispatchPluginRegistrationsOnce runs before CLI parsing). Statically linked,
// so it shares the host's pass registry. UNSUPPORTED unless the build selected
// the sample (hip_static_plugins / -DHIPDNN_EP_COMPILER_PLUGINS=sample).
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
