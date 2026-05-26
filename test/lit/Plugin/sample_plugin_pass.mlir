// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// XFAIL: *
//
// This test is intentionally expected to fail today.
//
// The slot-recording half of the public plugin ABI works end-to-end
// (validated by the unit test in test/plugin/test_plugin_loader.cpp:
// the plugin's requestPipelineSlot call lands in host-side storage
// and Pipelines.cpp queries it correctly). What does NOT work in the
// current prebuilt is the *MLIR-pass-instance* half: the plugin DLL
// statically links MLIRPass / MLIRIR / MLIRFuncDialect from
// prebuilt-local/, which gives the plugin its own copy of MLIR's
// global pass registry, dialect registry, and TypeID state. The
// plugin's mlir::PassRegistration<SamplePrintFunctionsPass>() then
// writes into the plugin's copy, not the host's.
//
// Resolution requires shipping MLIR as a shared library that both
// the host process and the plugin link against -- the same problem
// upstream LLVM solves with libLLVM.so. That is a build-system
// change beyond PR 2's scope; this LIT test will start passing once
// the project's MLIR-build mode lands.
//
// Until then we keep this test on disk (XFAIL'd) so:
//   1. The shape of a vendor LIT test is reviewable.
//   2. Once shared MLIR lands, removing the XFAIL line is the only
//      change needed to flip this from XFAIL -> PASS.
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
