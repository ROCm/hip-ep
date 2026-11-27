// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// f32 attention without matrix acceleration merges only when its score
// matrix is strictly larger than attention-f32-min-score-bytes. The scores
// here are 1x2x64x64 f32, 32768 bytes. With f32 in gemm-gemm-accel-types the
// cutoff does not apply.

// RUN: hip-mlir-opt --hip-rocmlir-prepare-quant --hip-fuse-rocmlir-elementwise \
// RUN:   --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16 attention-f32-min-score-bytes=32768" %s \
// RUN:   | FileCheck %s --check-prefix=APART
// RUN: hip-mlir-opt --hip-rocmlir-prepare-quant --hip-fuse-rocmlir-elementwise \
// RUN:   --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16 attention-f32-min-score-bytes=32767" %s \
// RUN:   | FileCheck %s --check-prefix=MERGED
// RUN: hip-mlir-opt --hip-rocmlir-prepare-quant --hip-fuse-rocmlir-elementwise \
// RUN:   --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16" %s \
// RUN:   | FileCheck %s --check-prefix=APART
// RUN: hip-mlir-opt --hip-rocmlir-prepare-quant --hip-fuse-rocmlir-elementwise \
// RUN:   --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16,f32" %s \
// RUN:   | FileCheck %s --check-prefix=MERGED

// APART-LABEL: func.func @main_graph
// APART: hip.rocmlir
// APART: hip.miopen.softmax
// APART: hip.rocmlir

// MERGED: func.func @mlir_matmul_miopen.softmax_matmul_
// MERGED-LABEL: func.func @main_graph
// MERGED-NOT: hip.miopen.softmax
// MERGED: hip.rocmlir
// MERGED-NOT: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf32>,
                      %k: tensor<1x2x32x64xf32>, %v: tensor<1x2x64x32xf32>)
    -> tensor<1x2x64x32xf32> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf32>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf32>, tensor<1x2x32x64xf32>)
      outs(%e0 : tensor<1x2x64x64xf32>) : tensor<1x2x64x64xf32>
  %e2 = tensor.empty() : tensor<1x2x64x64xf32>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf32>)
      outs(%e2 : tensor<1x2x64x64xf32>) -> tensor<1x2x64x64xf32>
  %e3 = tensor.empty() : tensor<1x2x64x32xf32>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf32>, tensor<1x2x64x32xf32>)
      outs(%e3 : tensor<1x2x64x32xf32>) : tensor<1x2x64x32xf32>
  return %y : tensor<1x2x64x32xf32>
}
