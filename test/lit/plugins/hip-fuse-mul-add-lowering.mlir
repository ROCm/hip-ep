// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Touchpoint E — HipToLLVM lowering test for hip.fused_mul_add.
//
// Verifies the full chain:
//   fusion pass  ->  one-shot-bufferize  ->  hip-to-llvm-with-fusion-plugin
//
// The last stage is a plugin-registered pipeline that runs the standard
// convert-hip-to-llvm pass (in-tree ops) followed by the plugin's
// FusedMulAddToLLVMPass (hip.fused_mul_add -> llvm.call @wrap_fused_mul_add).
//
// Without this second stage, hip.fused_mul_add would remain as an illegal op
// and applyPartialConversion would fail — demonstrating why touchpoint E
// requires a pipeline extension, not just a pattern registration.
//===----------------------------------------------------------------------===//

// REQUIRES: hip_plugins
//
// NOTE: use hip-to-llvm-with-fusion-plugin (plugin-registered pipeline),
// NOT the bare convert-hip-to-llvm pass — the in-tree pass has no hook for
// plugin lowering patterns and would leave hip.fused_mul_add unresolved.
// RUN: hip-mlir-opt \
// RUN:   --load-dialect-plugin=%hip_fusion_plugin \
// RUN:   --load-pass-plugin=%hip_fusion_plugin \
// RUN:   --pass-pipeline="builtin.module(func.func(hip-fuse-mul-add),one-shot-bufferize{bufferize-function-boundaries},hip-to-llvm-with-fusion-plugin)" \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: llvm.func @test_lowering
// CHECK-NOT:   hip.fused_mul_add
// CHECK:       llvm.call @wrap_fused_mul_add
func.func @test_lowering(
    %ctx: !hip.context,
    %x: tensor<8xf32>,
    %b: tensor<8xf32>,
    %a: tensor<8xf32>) -> tensor<8xf32> {
  %init_mul = tensor.empty() : tensor<8xf32>
  %m = hip.mul(%ctx)
      ins(%x, %b : tensor<8xf32>, tensor<8xf32>)
      outs(%init_mul : tensor<8xf32>)
      -> tensor<8xf32>
  %init_add = tensor.empty() : tensor<8xf32>
  %y = hip.add(%ctx)
      ins(%m, %a : tensor<8xf32>, tensor<8xf32>)
      outs(%init_add : tensor<8xf32>)
      -> tensor<8xf32>
  return %y : tensor<8xf32>
}
