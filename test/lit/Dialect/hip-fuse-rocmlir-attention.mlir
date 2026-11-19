// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Attention: a gemm kernel, the last-axis softmax of its only result and the
// gemm kernel that reads the softmax as its A operand merge into one kernel,
// which rocMLIR lowers to rock.attention. Both gemms must read a float type in
// gemm-gemm-accel-types (or f32 with a score matrix larger than
// attention-f32-min-score-bytes, here a 2 MiB cache), and both head dims must
// be at most 256.

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16,bf16 attention-f32-min-score-bytes=2097152" %s \
// RUN:   | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16,bf16 attention-f32-min-score-bytes=2097152 attention=false" %s \
// RUN:   | FileCheck %s --check-prefix=OFF
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16,bf16" %s \
// RUN:   | FileCheck %s --check-prefix=NOF32

// Without attention-f32-min-score-bytes only the f16 attention below merges.
// NOF32: func.func @mlir_matmul_mul_miopen.softmax_matmul_
// NOF32-NOT: func.func @{{.*}}softmax_matmul

// Scaled f16 attention with BHSD inputs: one kernel runs matmul -> mul ->
// softmax -> matmul and reads q, k and v.

// CHECK-LABEL: func.func @mlir_matmul_mul_miopen.softmax_matmul_
// CHECK-SAME: (%[[Q:.*]]: tensor<1x2x64x32xf16>, %[[K:.*]]: tensor<1x2x32x64xf16>, %[[V:.*]]: tensor<1x2x64x32xf16>) -> tensor<1x2x64x32xf16>
// CHECK: %[[S:.*]] = hip.matmul({{.*}}) ins(%[[Q]], %[[K]]
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[S]]
// CHECK: %[[P:.*]] = hip.miopen.softmax({{.*}}) ins(%[[M]]
// CHECK: %[[Y:.*]] = hip.matmul({{.*}}) ins(%[[P]], %[[V]]
// CHECK: return %[[Y]]

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir
// CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>, tensor<1x2x64x32xf16>)
// CHECK-NOT: hip.rocmlir
// CHECK-NOT: hip.miopen.softmax

// OFF-LABEL: func.func @main_graph
// OFF: hip.rocmlir
// OFF: hip.miopen.softmax
// OFF: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %v: tensor<1x2x64x32xf16>)
    -> tensor<1x2x64x32xf16> {
  %scale = hip.constant {value = dense<1.767580e-01> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e1 = tensor.empty() : tensor<1x2x64x64xf16>
  %m = hip.mul(%ctx) ins(%s, %scale : tensor<1x2x64x64xf16>, tensor<f16>)
      outs(%e1 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%m : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e3 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y : tensor<1x2x64x32xf16>
}

// -----

// f32 has no matrix acceleration listed and its 32 KiB score matrix fits in
// the 2 MiB cache, so the gemms and the softmax stay apart.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
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

// -----

// An f32 score matrix of 256 MiB is larger than the 2 MiB cache and would go
// to memory between unfused kernels, so f32 fuses too. Q is transposed by the
// matmul, so the head dim is its second-to-last dimension.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir
// CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<1x16x64x2048xf32>, tensor<1x16x64x2048xf32>, tensor<1x16x2048x64xf32>)
// CHECK-NOT: hip.rocmlir
// CHECK-NOT: hip.miopen.softmax
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x16x64x2048xf32>,
                      %k: tensor<1x16x64x2048xf32>, %v: tensor<1x16x2048x64xf32>)
    -> tensor<1x16x2048x64xf32> {
  %e0 = tensor.empty() : tensor<1x16x2048x2048xf32>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x16x64x2048xf32>, tensor<1x16x64x2048xf32>)
      outs(%e0 : tensor<1x16x2048x2048xf32>) {transA = 1 : i64} : tensor<1x16x2048x2048xf32>
  %e2 = tensor.empty() : tensor<1x16x2048x2048xf32>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x16x2048x2048xf32>)
      outs(%e2 : tensor<1x16x2048x2048xf32>) -> tensor<1x16x2048x2048xf32>
  %e3 = tensor.empty() : tensor<1x16x2048x64xf32>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x16x2048x2048xf32>, tensor<1x16x2048x64xf32>)
      outs(%e3 : tensor<1x16x2048x64xf32>) : tensor<1x16x2048x64xf32>
  return %y : tensor<1x16x2048x64xf32>
}

// -----

// A head dim of 512 exceeds what the attention kernel keeps on chip.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x512xf16>,
                      %k: tensor<1x2x512x64xf16>, %v: tensor<1x2x64x32xf16>)
    -> tensor<1x2x64x32xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x512xf16>, tensor<1x2x512x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e3 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y : tensor<1x2x64x32xf16>
}

// -----

// The probabilities are also a graph output, so the softmax must stay a
// separate kernel that writes them.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %v: tensor<1x2x64x32xf16>)
    -> (tensor<1x2x64x32xf16>, tensor<1x2x64x64xf16>) {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e3 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y, %p : tensor<1x2x64x32xf16>, tensor<1x2x64x64xf16>
}
