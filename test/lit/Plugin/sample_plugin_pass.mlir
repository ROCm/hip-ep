// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// XFAIL: *
//
// This test is intentionally expected to fail today.
//
// The slot-recording half of the public plugin ABI works end-to-end
// (validated by the unit test in test/plugin/test_plugin_loader.cpp: the
// plugin's requestPipelineSlot call lands in host-side storage and
// Pipelines.cpp queries it). What does NOT work in the default build is the
// pass-instance half: the host and the plugin each link MLIR statically, so
// the plugin's mlir::PassRegistration<SamplePrintFunctionsPass>() writes into
// the plugin's own copy of MLIR's global pass registry, not the host's, and
// the host's pass lookup never finds it.
//
// Resolution requires the host and the plugin to share one MLIR instance (a
// shared MLIR library) -- a build-system change. This test will start passing
// once that build mode lands; removing the XFAIL line is then the only change
// needed. Until then it is kept on disk (XFAIL'd) so the shape of a plugin
// LIT test is reviewable.
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
