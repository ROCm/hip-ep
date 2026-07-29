// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// REQUIRES: hip_static_plugins
//
// Dialect half of the plugin surface, and the regression guard for the
// hip-mlir-opt fix (pluginDialectRegistrations() applied to the tool's
// registry). The sample plugin contributes the `hip_ep_sample` dialect + its
// ConvertToLLVMPatternInterface via addDialectRegistration. For hip-mlir-opt to
// parse a plugin op AND lower it under --convert-hip-to-llvm, the tool must run
// the plugin dialect registrations into its own registry -- exactly what the
// fix adds (companion to dispatchPluginRegistrationsOnce). Without it the tool
// either cannot parse hip_ep_sample.marker (dialect never inserted) or leaves
// it unlowered (interface never attached, hasPromisedInterface guard rejects).
//
// The sample lowering just erases the marker, so a successful convert-hip-to-llvm
// leaves a body with only the terminator -- and no hip_ep_sample op survives.
// UNSUPPORTED unless the sample was selected (hip_static_plugins /
// -DHIPDNN_EP_COMPILER_PLUGINS=sample).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --convert-hip-to-llvm 2>&1 | FileCheck %s

// The plugin op is registered (parses) and lowered away (erased) -- proving the
// dialect + its ConvertToLLVMPatternInterface reached the tool's registry.
// convert-hip-to-llvm rewrites func.func -> llvm.func (see the abs/add tests),
// and the sample lowering erases the marker, so no hip_ep_sample op survives.
// CHECK-LABEL: llvm.func @marker_lowers
// CHECK-NOT: hip_ep_sample.marker
func.func @marker_lowers() {
  "hip_ep_sample.marker"() : () -> ()
  return
}
