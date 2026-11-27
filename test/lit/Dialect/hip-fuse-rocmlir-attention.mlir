// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Attention: a gemm kernel, the last-axis softmax of its only result and the
// gemm kernel that reads the softmax as its A operand merge into one kernel,
// which rocMLIR lowers to rock.attention. Both gemms must read a float type in
// gemm-gemm-accel-types (or f32 with a score matrix larger than
// attention-f32-min-score-bytes, here a 2 MiB cache), both head dims must be
// at most 256, and only batch reshapes and one paired cast may sit between the
// softmax and the second gemm. hip-fuse-rocmlir-attention-f32-cutoff.mlir
// checks the cutoff itself.

// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16,bf16 attention-f32-min-score-bytes=2097152" %s \
// RUN:   | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16,bf16 attention-f32-min-score-bytes=2097152 attention=false" %s \
// RUN:   | FileCheck %s --check-prefix=OFF
// RUN: hip-mlir-opt --split-input-file --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16,bf16" %s \
// RUN:   | FileCheck %s --check-prefix=NOF32

// Merged cases end their checks at the return of main_graph, because the
// next case's merged kernel, printed before its main_graph, holds a softmax.

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
// CHECK: return

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
// CHECK: return

// Without attention-f32-min-score-bytes the same f32 attention stays apart.
// NOF32-LABEL: func.func @main_graph({{.*}}tensor<1x16x64x2048xf32>
// NOF32: hip.rocmlir
// NOF32: hip.miopen.softmax
// NOF32: hip.rocmlir
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

// -----

// The second matmul transposes the probabilities, so it would reduce over the
// rows the softmax did not normalize.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %v: tensor<1x2x64x32xf16>)
    -> tensor<1x2x64x32xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e3 : tensor<1x2x64x32xf16>) {transA = 1 : i64} : tensor<1x2x64x32xf16>
  return %y : tensor<1x2x64x32xf16>
}

// -----

// A Q.K head dim of 256 is at the limit and merges.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir
// CHECK-NOT: hip.rocmlir
// CHECK-NOT: hip.miopen.softmax
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x256xf16>,
                      %k: tensor<1x2x256x64xf16>, %v: tensor<1x2x64x64xf16>)
    -> tensor<1x2x64x64xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x256xf16>, tensor<1x2x256x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x64xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x64xf16>)
      outs(%e3 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  return %y : tensor<1x2x64x64xf16>
}

// -----

// A Q.K head dim of 257 is over the limit.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x257xf16>,
                      %k: tensor<1x2x257x64xf16>, %v: tensor<1x2x64x64xf16>)
    -> tensor<1x2x64x64xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x257xf16>, tensor<1x2x257x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x64xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x64xf16>)
      outs(%e3 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  return %y : tensor<1x2x64x64xf16>
}

// -----

// A V head dim of 256 is at the limit and merges.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir
// CHECK-NOT: hip.rocmlir
// CHECK-NOT: hip.miopen.softmax
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x64xf16>,
                      %k: tensor<1x2x64x64xf16>, %v: tensor<1x2x64x256xf16>)
    -> tensor<1x2x64x256xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x64xf16>, tensor<1x2x64x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x256xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x256xf16>)
      outs(%e3 : tensor<1x2x64x256xf16>) : tensor<1x2x64x256xf16>
  return %y : tensor<1x2x64x256xf16>
}

// -----

// A V head dim of 257 is over the limit.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x64xf16>,
                      %k: tensor<1x2x64x64xf16>, %v: tensor<1x2x64x257xf16>)
    -> tensor<1x2x64x257xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x64xf16>, tensor<1x2x64x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x257xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x257xf16>)
      outs(%e3 : tensor<1x2x64x257xf16>) : tensor<1x2x64x257xf16>
  return %y : tensor<1x2x64x257xf16>
}

// -----

// A scale and an additive padding mask before the softmax: the mask becomes
// an input of the merged kernel, after Q and K and before V.

// CHECK-LABEL: func.func @mlir_matmul_mul_add_miopen.softmax_matmul_
// CHECK-SAME: (%[[Q:.*]]: tensor<1x2x64x32xf16>, %[[K:.*]]: tensor<1x2x32x64xf16>, %[[MASK:.*]]: tensor<1x1x1x64xf16>, %[[V:.*]]: tensor<1x2x64x32xf16>)
// CHECK: %[[S:.*]] = hip.matmul({{.*}}) ins(%[[Q]], %[[K]]
// CHECK: %[[M:.*]] = hip.mul({{.*}}) ins(%[[S]]
// CHECK: %[[B:.*]] = hip.add({{.*}}) ins(%[[M]], %[[MASK]]
// CHECK: %[[P:.*]] = hip.miopen.softmax({{.*}}) ins(%[[B]]
// CHECK: hip.matmul({{.*}}) ins(%[[P]], %[[V]]

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir
// CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>, tensor<1x1x1x64xf16>, tensor<1x2x64x32xf16>)
// CHECK-NOT: hip.rocmlir
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %mask: tensor<1x1x1x64xf16>,
                      %v: tensor<1x2x64x32xf16>) -> tensor<1x2x64x32xf16> {
  %scale = hip.constant {value = dense<1.767580e-01> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e1 = tensor.empty() : tensor<1x2x64x64xf16>
  %m = hip.mul(%ctx) ins(%s, %scale : tensor<1x2x64x64xf16>, tensor<f16>)
      outs(%e1 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %eb = tensor.empty() : tensor<1x2x64x64xf16>
  %b = hip.add(%ctx) ins(%m, %mask : tensor<1x2x64x64xf16>, tensor<1x1x1x64xf16>)
      outs(%eb : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%b : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e3 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y : tensor<1x2x64x32xf16>
}

// -----

// The softmax runs in f32 between f16 gemms: the cast to f32 rides in the
// first kernel's epilogue and the cast back in the second kernel's prologue,
// the pair rocMLIR's attention matcher takes.

// CHECK-LABEL: func.func @mlir_matmul_cast_miopen.softmax_cast_matmul_
// CHECK: %[[S:.*]] = hip.matmul
// CHECK: %[[S32:.*]] = hip.cast({{.*}}) ins(%[[S]] : tensor<1x2x64x64xf16>)
// CHECK: %[[P:.*]] = hip.miopen.softmax({{.*}}) ins(%[[S32]] : tensor<1x2x64x64xf32>)
// CHECK: %[[P16:.*]] = hip.cast({{.*}}) ins(%[[P]] : tensor<1x2x64x64xf32>)
// CHECK: hip.matmul({{.*}}) ins(%[[P16]],

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir
// CHECK-NOT: hip.rocmlir
// CHECK-NOT: hip.miopen.softmax
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %v: tensor<1x2x64x32xf16>)
    -> tensor<1x2x64x32xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %ec = tensor.empty() : tensor<1x2x64x64xf32>
  %s32 = hip.cast(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%ec : tensor<1x2x64x64xf32>) {to = 1 : i64} : tensor<1x2x64x64xf32>
  %e2 = tensor.empty() : tensor<1x2x64x64xf32>
  %p = hip.miopen.softmax(%ctx) ins(%s32 : tensor<1x2x64x64xf32>)
      outs(%e2 : tensor<1x2x64x64xf32>) -> tensor<1x2x64x64xf32>
  %ed = tensor.empty() : tensor<1x2x64x64xf16>
  %p16 = hip.cast(%ctx) ins(%p : tensor<1x2x64x64xf32>)
      outs(%ed : tensor<1x2x64x64xf16>) {to = 10 : i64} : tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%p16, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e3 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y : tensor<1x2x64x32xf16>
}

// -----

// Two casts between the softmax and the second gemm are more than rocMLIR's
// attention matcher skips.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %v: tensor<1x2x64x32xf16>)
    -> tensor<1x2x64x32xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %ec = tensor.empty() : tensor<1x2x64x64xf32>
  %p32 = hip.cast(%ctx) ins(%p : tensor<1x2x64x64xf16>)
      outs(%ec : tensor<1x2x64x64xf32>) {to = 1 : i64} : tensor<1x2x64x64xf32>
  %ed = tensor.empty() : tensor<1x2x64x64xf16>
  %p16 = hip.cast(%ctx) ins(%p32 : tensor<1x2x64x64xf32>)
      outs(%ed : tensor<1x2x64x64xf16>) {to = 10 : i64} : tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%p16, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e3 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y : tensor<1x2x64x32xf16>
}

// -----

// An f16 softmax cast to f32 for an f32 second gemm: V is f32 while Q and K
// are f16, and no cast before the softmax pairs with the one after it.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x16x1024x64xf16>,
                      %k: tensor<1x16x64x1024xf16>, %v: tensor<1x16x1024x64xf32>)
    -> tensor<1x16x1024x64xf32> {
  %e0 = tensor.empty() : tensor<1x16x1024x1024xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x16x1024x64xf16>, tensor<1x16x64x1024xf16>)
      outs(%e0 : tensor<1x16x1024x1024xf16>) : tensor<1x16x1024x1024xf16>
  %e2 = tensor.empty() : tensor<1x16x1024x1024xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x16x1024x1024xf16>)
      outs(%e2 : tensor<1x16x1024x1024xf16>) -> tensor<1x16x1024x1024xf16>
  %ec = tensor.empty() : tensor<1x16x1024x1024xf32>
  %p32 = hip.cast(%ctx) ins(%p : tensor<1x16x1024x1024xf16>)
      outs(%ec : tensor<1x16x1024x1024xf32>) {to = 1 : i64} : tensor<1x16x1024x1024xf32>
  %e3 = tensor.empty() : tensor<1x16x1024x64xf32>
  %y = hip.matmul(%ctx) ins(%p32, %v : tensor<1x16x1024x1024xf32>, tensor<1x16x1024x64xf32>)
      outs(%e3 : tensor<1x16x1024x64xf32>) : tensor<1x16x1024x64xf32>
  return %y : tensor<1x16x1024x64xf32>
}

// -----

// A pointwise op between the softmax and the second gemm is not a link
// rocMLIR's attention matcher skips.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %v: tensor<1x2x64x32xf16>)
    -> tensor<1x2x64x32xf16> {
  %keep = hip.constant {value = dense<9.000000e-01> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e1 = tensor.empty() : tensor<1x2x64x64xf16>
  %d = hip.mul(%ctx) ins(%p, %keep : tensor<1x2x64x64xf16>, tensor<f16>)
      outs(%e1 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%d, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e3 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y : tensor<1x2x64x32xf16>
}

// -----

// The scores before the softmax are also a graph output.

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
  return %y, %s : tensor<1x2x64x32xf16>, tensor<1x2x64x64xf16>
}

// -----

// V is computed from the scores, so a merged kernel would read its own
// intermediate as an input.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>) -> tensor<1x2x64x64xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %ev = tensor.empty() : tensor<1x2x64x64xf16>
  %v = hip.tanh(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%ev : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e3 = tensor.empty() : tensor<1x2x64x64xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x64xf16>)
      outs(%e3 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  return %y : tensor<1x2x64x64xf16>
}

// -----

// hip.gemm anchors with K read transposed (transB) merge like matmuls.

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir
// CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<64x32xf16>, tensor<64x32xf16>, tensor<64x32xf16>)
// CHECK-NOT: hip.rocmlir
// CHECK-NOT: hip.miopen.softmax
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %q: tensor<64x32xf16>,
                      %k: tensor<64x32xf16>, %v: tensor<64x32xf16>)
    -> tensor<64x32xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %s = hip.gemm(%ctx) ins(%q, %k : tensor<64x32xf16>, tensor<64x32xf16>)
      outs(%e0 : tensor<64x64xf16>)
      {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : i64, transB = 1 : i64}
      : tensor<64x64xf16>
  %e2 = tensor.empty() : tensor<64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<64x64xf16>)
      outs(%e2 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %e3 = tensor.empty() : tensor<64x32xf16>
  %y = hip.gemm(%ctx) ins(%p, %v : tensor<64x64xf16>, tensor<64x32xf16>)
      outs(%e3 : tensor<64x32xf16>)
      {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : i64, transB = 0 : i64}
      : tensor<64x32xf16>
  return %y : tensor<64x32xf16>
}

// -----

// A reshape that folds the heads into the batch keeps the rows and columns of
// the probabilities, so it may sit between the softmax and the second gemm.

// CHECK-LABEL: func.func @mlir_matmul_miopen.softmax_{{[a-z_]*}}matmul_
// CHECK-SAME: (%{{.*}}: tensor<1x2x64x32xf16>, %{{.*}}: tensor<1x2x32x64xf16>, %{{.*}}: tensor<2x64x32xf16>) -> tensor<2x64x32xf16>
// CHECK: %[[P:.*]] = hip.miopen.softmax
// CHECK: %[[R:.*]] = tensor.collapse_shape %[[P]]
// CHECK-SAME: into tensor<2x64x64xf16>
// CHECK: hip.matmul({{.*}}) ins(%[[R]],

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir
// CHECK-NOT: hip.rocmlir
// CHECK-NOT: hip.miopen.softmax
// CHECK: return
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %v: tensor<2x64x32xf16>)
    -> tensor<2x64x32xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %r = tensor.collapse_shape %p [[0, 1], [2], [3]]
      : tensor<1x2x64x64xf16> into tensor<2x64x64xf16>
  %e3 = tensor.empty() : tensor<2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%r, %v : tensor<2x64x64xf16>, tensor<2x64x32xf16>)
      outs(%e3 : tensor<2x64x32xf16>) : tensor<2x64x32xf16>
  return %y : tensor<2x64x32xf16>
}

// -----

// A reshape that folds the heads into the rows gives the second gemm another
// m and batch count than the first, which rocMLIR's attention does not take.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %v: tensor<1x64x32xf16>)
    -> tensor<1x128x32xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %r = tensor.collapse_shape %p [[0], [1, 2], [3]]
      : tensor<1x2x64x64xf16> into tensor<1x128x64xf16>
  %e3 = tensor.empty() : tensor<1x128x32xf16>
  %y = hip.matmul(%ctx) ins(%r, %v : tensor<1x128x64xf16>, tensor<1x64x32xf16>)
      outs(%e3 : tensor<1x128x32xf16>) : tensor<1x128x32xf16>
  return %y : tensor<1x128x32xf16>
}

// -----

// A reshape that changes the last dim makes the second gemm reduce over
// something other than the keys the softmax normalized.

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK: hip.miopen.softmax
// CHECK: hip.rocmlir
func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x32xf16>,
                      %k: tensor<1x2x32x64xf16>, %v: tensor<1x4096x32xf16>)
    -> tensor<1x2x32xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %r = tensor.collapse_shape %p [[0], [1], [2, 3]]
      : tensor<1x2x64x64xf16> into tensor<1x2x4096xf16>
  %e3 = tensor.empty() : tensor<1x2x32xf16>
  %y = hip.matmul(%ctx) ins(%r, %v : tensor<1x2x4096xf16>, tensor<1x4096x32xf16>)
      outs(%e3 : tensor<1x2x32xf16>) : tensor<1x2x32xf16>
  return %y : tensor<1x2x32xf16>
}
