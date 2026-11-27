// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Attention merges dispatches whose kernels other dispatches may also call.
// A merge erases a kernel only once nothing calls it any more, and erases a
// kernel that both gemms of the triple call once.

// RUN: hip-mlir-opt --split-input-file \
// RUN:   --hip-fuse-rocmlir-anchor="gemm-gemm-accel-types=f16,bf16" %s \
// RUN:   | FileCheck %s

// Two attention blocks call the same score and P·V kernels. The first merge
// must leave both kernels to the second block, which merges too; then
// neither is called and both go.

// CHECK: func.func @mlir_matmul_mul_miopen.softmax_matmul_
// CHECK: func.func @mlir_matmul_mul_miopen.softmax_matmul_
// CHECK-NOT: func.func @score
// CHECK-NOT: func.func @pv
// CHECK: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir({{.*}}) @mlir_matmul_mul_miopen.softmax_matmul_
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir({{.*}}) @mlir_matmul_mul_miopen.softmax_matmul_
// CHECK-NOT: hip.miopen.softmax
// CHECK-NOT: hip.rocmlir
// CHECK: return

func.func @score(%q: tensor<1x2x64x32xf16>, %k: tensor<1x2x32x64xf16>)
    -> tensor<1x2x64x64xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %scale = hip.constant {value = dense<1.767580e-01> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e1 = tensor.empty() : tensor<1x2x64x64xf16>
  %m = hip.mul(%ctx) ins(%s, %scale : tensor<1x2x64x64xf16>, tensor<f16>)
      outs(%e1 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  return %m : tensor<1x2x64x64xf16>
}

func.func @pv(%p: tensor<1x2x64x64xf16>, %v: tensor<1x2x64x32xf16>)
    -> tensor<1x2x64x32xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e0 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e0 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y : tensor<1x2x64x32xf16>
}

func.func @main_graph(%ctx: !hip.context,
                      %q0: tensor<1x2x64x32xf16>, %k0: tensor<1x2x32x64xf16>,
                      %v0: tensor<1x2x64x32xf16>, %q1: tensor<1x2x64x32xf16>,
                      %k1: tensor<1x2x32x64xf16>, %v1: tensor<1x2x64x32xf16>)
    -> (tensor<1x2x64x32xf16>, tensor<1x2x64x32xf16>) {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s0 = hip.rocmlir(%ctx) @score ins(%q0, %k0 : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e1 = tensor.empty() : tensor<1x2x64x64xf16>
  %p0 = hip.miopen.softmax(%ctx) ins(%s0 : tensor<1x2x64x64xf16>)
      outs(%e1 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x32xf16>
  %y0 = hip.rocmlir(%ctx) @pv ins(%p0, %v0 : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e2 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  %e3 = tensor.empty() : tensor<1x2x64x64xf16>
  %s1 = hip.rocmlir(%ctx) @score ins(%q1, %k1 : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e3 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e4 = tensor.empty() : tensor<1x2x64x64xf16>
  %p1 = hip.miopen.softmax(%ctx) ins(%s1 : tensor<1x2x64x64xf16>)
      outs(%e4 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e5 = tensor.empty() : tensor<1x2x64x32xf16>
  %y1 = hip.rocmlir(%ctx) @pv ins(%p1, %v1 : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e5 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y0, %y1 : tensor<1x2x64x32xf16>, tensor<1x2x64x32xf16>
}

// -----

// The score kernel also runs on its own for a second output, so it stays;
// the P·V kernel has no other caller and goes.

// CHECK: func.func @mlir_matmul_mul_miopen.softmax_matmul_
// CHECK-NOT: func.func @pv
// CHECK: func.func @score(
// CHECK-NOT: func.func @pv
// CHECK: func.func @main_graph
// CHECK: hip.rocmlir({{.*}}) @mlir_matmul_mul_miopen.softmax_matmul_
// CHECK: hip.rocmlir({{.*}}) @score ins(

func.func @score(%q: tensor<1x2x64x32xf16>, %k: tensor<1x2x32x64xf16>)
    -> tensor<1x2x64x64xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %scale = hip.constant {value = dense<1.767580e-01> : tensor<f16>} : tensor<f16>
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.matmul(%ctx) ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e1 = tensor.empty() : tensor<1x2x64x64xf16>
  %m = hip.mul(%ctx) ins(%s, %scale : tensor<1x2x64x64xf16>, tensor<f16>)
      outs(%e1 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  return %m : tensor<1x2x64x64xf16>
}

func.func @pv(%p: tensor<1x2x64x64xf16>, %v: tensor<1x2x64x32xf16>)
    -> tensor<1x2x64x32xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e0 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.matmul(%ctx) ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e0 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  return %y : tensor<1x2x64x32xf16>
}

func.func @main_graph(%ctx: !hip.context,
                      %q: tensor<1x2x64x32xf16>, %k: tensor<1x2x32x64xf16>,
                      %v: tensor<1x2x64x32xf16>, %q2: tensor<1x2x64x32xf16>)
    -> (tensor<1x2x64x32xf16>, tensor<1x2x64x64xf16>) {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.rocmlir(%ctx) @score ins(%q, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e1 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e1 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x32xf16>
  %y = hip.rocmlir(%ctx) @pv ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x32xf16>)
      outs(%e2 : tensor<1x2x64x32xf16>) : tensor<1x2x64x32xf16>
  %e3 = tensor.empty() : tensor<1x2x64x64xf16>
  %s2 = hip.rocmlir(%ctx) @score ins(%q2, %k : tensor<1x2x64x32xf16>, tensor<1x2x32x64xf16>)
      outs(%e3 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  return %y, %s2 : tensor<1x2x64x32xf16>, tensor<1x2x64x64xf16>
}

// -----

// With a head dim equal to the sequence length both gemms have one
// signature and call the same kernel, which the merge erases once.

// CHECK: func.func @mlir_matmul_miopen.softmax_matmul_
// CHECK-NOT: func.func @mm
// CHECK: func.func @main_graph
// CHECK-NOT: hip.miopen.softmax
// CHECK: hip.rocmlir({{.*}}) @mlir_matmul_miopen.softmax_matmul_
// CHECK-NOT: @mm

func.func @mm(%a: tensor<1x2x64x64xf16>, %b: tensor<1x2x64x64xf16>)
    -> tensor<1x2x64x64xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %y = hip.matmul(%ctx) ins(%a, %b : tensor<1x2x64x64xf16>, tensor<1x2x64x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  return %y : tensor<1x2x64x64xf16>
}

func.func @main_graph(%ctx: !hip.context, %q: tensor<1x2x64x64xf16>,
                      %k: tensor<1x2x64x64xf16>, %v: tensor<1x2x64x64xf16>)
    -> tensor<1x2x64x64xf16> {
  %e0 = tensor.empty() : tensor<1x2x64x64xf16>
  %s = hip.rocmlir(%ctx) @mm ins(%q, %k : tensor<1x2x64x64xf16>, tensor<1x2x64x64xf16>)
      outs(%e0 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  %e1 = tensor.empty() : tensor<1x2x64x64xf16>
  %p = hip.miopen.softmax(%ctx) ins(%s : tensor<1x2x64x64xf16>)
      outs(%e1 : tensor<1x2x64x64xf16>) -> tensor<1x2x64x64xf16>
  %e2 = tensor.empty() : tensor<1x2x64x64xf16>
  %y = hip.rocmlir(%ctx) @mm ins(%p, %v : tensor<1x2x64x64xf16>, tensor<1x2x64x64xf16>)
      outs(%e2 : tensor<1x2x64x64xf16>) : tensor<1x2x64x64xf16>
  return %y : tensor<1x2x64x64xf16>
}
