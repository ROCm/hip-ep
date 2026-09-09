// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip ops that decompose to more than one TOSA op inside a
// rock.kernel function, so rocMLIR can absorb them into a fused kernel, and
// verify the forms the conversion rejects.
//
// hip.leaky_relu is not a BinaryConverter/UnaryConverter mapping: TOSA has no
// leaky-relu op. The lowering is tosa.maximum(x, tosa.mul(x, splat(alpha)))
// with nan_mode = IGNORE.
//
// hip.miopen.softmax is last-dim softmax. TOSA has no softmax op; the
// lowering is reduce_max/sub/exp/reduce_sum/reciprocal/mul, which TosaToRock
// attention matching expects.
//
// hip.sqrt has no TOSA sqrt; the lowering is tosa.reciprocal(tosa.rsqrt(x)).
// rocMLIR folds that pair back to math.sqrt.
//
// FILE LAYOUT:
// Everything that converts lives in the first --split-input-file chunk, so
// it is one module and therefore also covers several ops converting in a
// single pass run. Each rejected form then gets its own chunk: full
// conversion turns a rejection into a pass failure that aborts the run for
// the whole module, so sharing a chunk would let one rejection mask the cases
// after it. A failing chunk contributes no output, while the chunks that
// convert still print for FileCheck.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @leaky_relu
// CHECK: %[[ALPHA:.*]] = "tosa.const"{{.*}}tensor<2x8xf16>
// CHECK: %[[SHIFT:.*]] = "tosa.const"{{.*}}dense<0> : tensor<1xi8>
// CHECK: %[[SCALED:.*]] = tosa.mul %arg1, %[[ALPHA]], %[[SHIFT]]
// CHECK: tosa.maximum %arg1, %[[SCALED]] {{.*}}nan_mode = IGNORE
// CHECK-NOT: hip.leaky_relu
func.func @leaky_relu(%ctx: !hip.context, %x: tensor<2x8xf16>,
                      %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.leaky_relu(%ctx) ins(%x : tensor<2x8xf16>)
                            outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @leaky_relu_alpha
// CHECK: tosa.mul
// CHECK: tosa.maximum {{.*}}nan_mode = IGNORE
// CHECK-NOT: hip.leaky_relu
func.func @leaky_relu_alpha(%ctx: !hip.context, %x: tensor<4xf32>,
                           %init: tensor<4xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %r = hip.leaky_relu(%ctx) ins(%x : tensor<4xf32>)
                            outs(%init : tensor<4xf32>)
                            {alpha = 0.2 : f64} : tensor<4xf32>
  return %r : tensor<4xf32>
}

// The shape hip-fuse-rocmlir actually produces: the !hip.context is a
// ub.poison materialized inside the outlined kernel rather than a block
// argument.
// CHECK-LABEL: func.func @leaky_relu_outlined_kernel
// CHECK: tosa.maximum {{.*}}nan_mode = IGNORE
// CHECK-NOT: hip.leaky_relu
func.func @leaky_relu_outlined_kernel(%x: tensor<2x8xf16>,
                                     %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.leaky_relu(%ctx) ins(%x : tensor<2x8xf16>)
                            outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// The pass early-returns unless the function is an outlined kernel.
// CHECK-LABEL: func.func @leaky_relu_not_a_kernel
// CHECK: hip.leaky_relu
// CHECK-NOT: tosa.maximum
func.func @leaky_relu_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xf16>,
                                   %init: tensor<2x8xf16>) -> tensor<2x8xf16> {
  %r = hip.leaky_relu(%ctx) ins(%x : tensor<2x8xf16>)
                            outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @softmax
// CHECK: tosa.reduce_max %arg1 {axis = 1 : i32}
// CHECK: tosa.sub
// CHECK: tosa.exp
// CHECK: tosa.reduce_sum {{.*}}{axis = 1 : i32}
// CHECK: tosa.reciprocal
// CHECK: tosa.mul
// CHECK-NOT: hip.miopen.softmax
func.func @softmax(%ctx: !hip.context, %x: tensor<2x8xf16>,
                   %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.miopen.softmax(%ctx) ins(%x : tensor<2x8xf16>)
                                 outs(%init : tensor<2x8xf16>)
                                 -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// 3D [B,S,D] softmax is over the last dim (axis = 2).
// CHECK-LABEL: func.func @softmax_3d
// CHECK: tosa.reduce_max %arg1 {axis = 2 : i32}
// CHECK: tosa.reduce_sum {{.*}}{axis = 2 : i32}
// CHECK-NOT: hip.miopen.softmax
func.func @softmax_3d(%ctx: !hip.context, %x: tensor<2x4x8xf16>,
                      %init: tensor<2x4x8xf16>) -> tensor<2x4x8xf16>
    attributes {rock.kernel} {
  %r = hip.miopen.softmax(%ctx) ins(%x : tensor<2x4x8xf16>)
                                 outs(%init : tensor<2x4x8xf16>)
                                 -> tensor<2x4x8xf16>
  return %r : tensor<2x4x8xf16>
}

// CHECK-LABEL: func.func @softmax_outlined_kernel
// CHECK: tosa.reduce_max %arg0 {axis = 1 : i32}
// CHECK-NOT: hip.miopen.softmax
func.func @softmax_outlined_kernel(%x: tensor<2x8xf16>,
                                   %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.miopen.softmax(%ctx) ins(%x : tensor<2x8xf16>)
                                 outs(%init : tensor<2x8xf16>)
                                 -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @softmax_not_a_kernel
// CHECK: hip.miopen.softmax
// CHECK-NOT: tosa.reduce_max
func.func @softmax_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xf16>,
                                %init: tensor<2x8xf16>) -> tensor<2x8xf16> {
  %r = hip.miopen.softmax(%ctx) ins(%x : tensor<2x8xf16>)
                                 outs(%init : tensor<2x8xf16>)
                                 -> tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @sqrt
// CHECK: %[[RSQRT:.*]] = tosa.rsqrt %arg1
// CHECK: tosa.reciprocal %[[RSQRT]]
// CHECK-NOT: hip.sqrt
func.func @sqrt(%ctx: !hip.context, %x: tensor<2x8xf16>,
                %init: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %r = hip.sqrt(%ctx) ins(%x : tensor<2x8xf16>)
                      outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @sqrt_outlined_kernel
// CHECK: tosa.rsqrt
// CHECK: tosa.reciprocal
// CHECK-NOT: hip.sqrt
func.func @sqrt_outlined_kernel(%x: tensor<2x8xf16>, %init: tensor<2x8xf16>)
    -> tensor<2x8xf16> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.sqrt(%ctx) ins(%x : tensor<2x8xf16>)
                      outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// CHECK-LABEL: func.func @sqrt_not_a_kernel
// CHECK: hip.sqrt
// CHECK-NOT: tosa.rsqrt
func.func @sqrt_not_a_kernel(%ctx: !hip.context, %x: tensor<2x8xf16>,
                               %init: tensor<2x8xf16>) -> tensor<2x8xf16> {
  %r = hip.sqrt(%ctx) ins(%x : tensor<2x8xf16>)
                      outs(%init : tensor<2x8xf16>) : tensor<2x8xf16>
  return %r : tensor<2x8xf16>
}

// -----

// Dynamic shapes give the pattern no static shape to reason about.
func.func @dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf16>,
                         %init: tensor<?x8xf16>) -> tensor<?x8xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.leaky_relu'}}
  %r = hip.leaky_relu(%ctx) ins(%x : tensor<?x8xf16>)
                            outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}

// -----

// Integer operands are not a valid ONNX LeakyRelu and would have no TOSA
// float-profile lowering for the mul/maximum pair.
func.func @integer_operand(%ctx: !hip.context, %x: tensor<4xi32>,
                           %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.leaky_relu'}}
  %r = hip.leaky_relu(%ctx) ins(%x : tensor<4xi32>)
                            outs(%init : tensor<4xi32>) : tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

func.func @softmax_dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf16>,
                                  %init: tensor<?x8xf16>) -> tensor<?x8xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.miopen.softmax'}}
  %r = hip.miopen.softmax(%ctx) ins(%x : tensor<?x8xf16>)
                                 outs(%init : tensor<?x8xf16>)
                                 -> tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}

// -----

func.func @softmax_integer_operand(%ctx: !hip.context, %x: tensor<4xi32>,
                                  %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.miopen.softmax'}}
  %r = hip.miopen.softmax(%ctx) ins(%x : tensor<4xi32>)
                                 outs(%init : tensor<4xi32>)
                                 -> tensor<4xi32>
  return %r : tensor<4xi32>
}

// -----

func.func @softmax_rank0(%ctx: !hip.context, %x: tensor<f16>,
                          %init: tensor<f16>) -> tensor<f16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.miopen.softmax'}}
  %r = hip.miopen.softmax(%ctx) ins(%x : tensor<f16>)
                                 outs(%init : tensor<f16>) -> tensor<f16>
  return %r : tensor<f16>
}

// -----

func.func @sqrt_dynamic_shape(%ctx: !hip.context, %x: tensor<?x8xf16>,
                               %init: tensor<?x8xf16>) -> tensor<?x8xf16>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.sqrt'}}
  %r = hip.sqrt(%ctx) ins(%x : tensor<?x8xf16>)
                      outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
  return %r : tensor<?x8xf16>
}

// -----

func.func @sqrt_integer_operand(%ctx: !hip.context, %x: tensor<4xi32>,
                                 %init: tensor<4xi32>) -> tensor<4xi32>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.sqrt'}}
  %r = hip.sqrt(%ctx) ins(%x : tensor<4xi32>)
                      outs(%init : tensor<4xi32>) : tensor<4xi32>
  return %r : tensor<4xi32>
}
