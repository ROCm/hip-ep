// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Out-of-tree plugin PoC — positive path.
//
// Loads the hip-fusion plugin (.so) into hip-mlir-opt at runtime. The plugin:
//   * registers the real op  hip.fused_mul_add  into the `hip` dialect, and
//   * registers the pass     --hip-fuse-mul-add
// with NO rebuild of hip-mlir-opt (beyond the one-time registerPluginOps hook).
//
// This checks the full chain: the pass fuses hip.add(hip.mul(x,b),a) into the
// plugin op, and because the plugin ALSO attaches the bufferization external
// model, --one-shot-bufferize legalizes it cleanly into memref form.
//===----------------------------------------------------------------------===//

// REQUIRES: hip_plugins
//
// NOTE on invocation form: we drive the plugin pass via --pass-pipeline rather
// than the generated --hip-fuse-mul-add shorthand flag. When a dialect plugin
// and a pass plugin are BOTH loaded, MlirOptMain's individual-pass cl flags
// (e.g. --hip-fuse-mul-add) regress to printing --help; the textual
// --pass-pipeline parser is unaffected and is the robust form for plugin passes.
// RUN: hip-mlir-opt \
// RUN:   --load-dialect-plugin=%hip_fusion_plugin \
// RUN:   --load-pass-plugin=%hip_fusion_plugin \
// RUN:   --pass-pipeline="builtin.module(func.func(hip-fuse-mul-add),one-shot-bufferize{bufferize-function-boundaries})" \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_mul_add_lhs_mul
// CHECK-NOT:     hip.mul
// CHECK-NOT:     hip.add
// CHECK:         hip.fused_mul_add
// CHECK-SAME:      memref
// CHECK:         return
func.func @fuse_mul_add_lhs_mul(
    %ctx: !hip.context,
    %x: tensor<4xf32>,
    %b: tensor<4xf32>,
    %a: tensor<4xf32>) -> tensor<4xf32> {
  %init_mul = tensor.empty() : tensor<4xf32>
  %m = hip.mul(%ctx)
      ins(%x, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%init_mul : tensor<4xf32>)
      -> tensor<4xf32>
  %init_add = tensor.empty() : tensor<4xf32>
  %y = hip.add(%ctx)
      ins(%m, %a : tensor<4xf32>, tensor<4xf32>)
      outs(%init_add : tensor<4xf32>)
      -> tensor<4xf32>
  return %y : tensor<4xf32>
}

// ----------------------------------------------------------------------------
// Mirror operand ordering: add(a, mul(x, b)) fuses the same way.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @fuse_mul_add_rhs_mul
// CHECK-NOT:     hip.mul
// CHECK-NOT:     hip.add
// CHECK:         hip.fused_mul_add
// CHECK:         return
func.func @fuse_mul_add_rhs_mul(
    %ctx: !hip.context,
    %x: tensor<4xf32>,
    %b: tensor<4xf32>,
    %a: tensor<4xf32>) -> tensor<4xf32> {
  %init_mul = tensor.empty() : tensor<4xf32>
  %m = hip.mul(%ctx)
      ins(%x, %b : tensor<4xf32>, tensor<4xf32>)
      outs(%init_mul : tensor<4xf32>)
      -> tensor<4xf32>
  %init_add = tensor.empty() : tensor<4xf32>
  %y = hip.add(%ctx)
      ins(%a, %m : tensor<4xf32>, tensor<4xf32>)
      outs(%init_add : tensor<4xf32>)
      -> tensor<4xf32>
  return %y : tensor<4xf32>
}
