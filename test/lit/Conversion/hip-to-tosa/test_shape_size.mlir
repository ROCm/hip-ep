// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the OnnxToHip residues of Shape / Size / Identity / ConstantOfShape
// convert to TOSA inside a rock.kernel function.
//
// TOSA has no shape/size/constant_of_shape. Identity is 1-1 tosa.identity.
// After convert-onnx-to-hip:
//   Identity          SSA forward (or a same-type tensor.cast)
//   Shape (static)    arith.constant dims + tensor.from_elements
//   Size (static)     arith.constant / hip.size folded to tosa.const
//   ConstantOfShape   arith.constant splat or tensor.splat
// Runtime shape queries (dynamic hip.size, tensor.dim) stay unconverted.
//
// FILE LAYOUT:
// Converting cases live in the first --split-input-file chunk. Each rejected
// form gets its own chunk so a legalization failure cannot mask later cases.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// onnx.Identity is forwarded at OnnxToHip (no hip.identity op). TOSA's 1-1
// mapping is tosa.identity; after forwarding the kernel just returns the input.
// CHECK-LABEL: func.func @identity
// CHECK: return %arg0
func.func @identity(%x: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  return %x : tensor<2x8xf16>
}

// The outlined-kernel form: context is ub.poison, value is the forwarded input.
// CHECK-LABEL: func.func @identity_outlined_kernel
// CHECK: return %arg0
func.func @identity_outlined_kernel(%x: tensor<2x8xf16>) -> tensor<2x8xf16>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  return %x : tensor<2x8xf16>
}

// Static onnx.Shape is tensor.from_elements of i64 constants.
// CHECK-LABEL: func.func @shape_static
// CHECK: tosa.const
// CHECK-NOT: tensor.from_elements
func.func @shape_static(%x: tensor<2x3x4xf16>) -> tensor<3xi64>
    attributes {rock.kernel} {
  %c0 = arith.constant 2 : i64
  %c1 = arith.constant 3 : i64
  %c2 = arith.constant 4 : i64
  %s = tensor.from_elements %c0, %c1, %c2 : tensor<3xi64>
  return %s : tensor<3xi64>
}

// Static Size / ConstantOfShape fold to a dense arith.constant.
// CHECK-LABEL: func.func @size_folded_const
// CHECK: tosa.const
// CHECK-NOT: arith.constant dense<6>
func.func @size_folded_const() -> tensor<i64> attributes {rock.kernel} {
  %s = arith.constant dense<6> : tensor<i64>
  return %s : tensor<i64>
}

// hip.size on a static input is prod(shape).
// CHECK-LABEL: func.func @size_static
// CHECK: tosa.const
// CHECK-NOT: hip.size
func.func @size_static(%ctx: !hip.context, %x: tensor<2x3xf32>,
                       %init: tensor<i64>) -> tensor<i64>
    attributes {rock.kernel} {
  %r = hip.size(%ctx) ins(%x : tensor<2x3xf32>)
                      outs(%init : tensor<i64>) : tensor<i64>
  return %r : tensor<i64>
}

// CHECK-LABEL: func.func @size_outlined_kernel
// CHECK: tosa.const
// CHECK-NOT: hip.size
func.func @size_outlined_kernel(%x: tensor<2x4x8xf16>, %init: tensor<i64>)
    -> tensor<i64> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %r = hip.size(%ctx) ins(%x : tensor<2x4x8xf16>)
                      outs(%init : tensor<i64>) : tensor<i64>
  return %r : tensor<i64>
}

// CHECK-LABEL: func.func @constant_of_shape_splat_const
// CHECK: tosa.const
// CHECK-NOT: arith.constant dense<0.000000e+00> : tensor<2x3xf32>
func.func @constant_of_shape_splat_const() -> tensor<2x3xf32>
    attributes {rock.kernel} {
  %r = arith.constant dense<0.000000e+00> : tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @constant_of_shape_splat_op
// CHECK: tosa.const
// CHECK-NOT: tensor.splat
func.func @constant_of_shape_splat_op() -> tensor<2x2xf32>
    attributes {rock.kernel} {
  %v = arith.constant 1.500000e+00 : f32
  %r = tensor.splat %v : tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @constant_of_shape_i64
// CHECK: tosa.const
func.func @constant_of_shape_i64() -> tensor<3xi64> attributes {rock.kernel} {
  %r = arith.constant dense<7> : tensor<3xi64>
  return %r : tensor<3xi64>
}

// TOSA number tensors cannot be empty. Empty ONNX residues (Reduce axes
// dense<[]>) stay arith.constant.
// CHECK-LABEL: func.func @empty_tensor_const
// CHECK: arith.constant dense<> : tensor<0xi64>
// CHECK-NOT: tosa.const
func.func @empty_tensor_const() -> tensor<0xi64> attributes {rock.kernel} {
  %r = arith.constant dense<[]> : tensor<0xi64>
  return %r : tensor<0xi64>
}

// -----

func.func @size_dynamic(%ctx: !hip.context, %x: tensor<?x?xf32>,
                        %init: tensor<i64>) -> tensor<i64>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.size'}}
  %r = hip.size(%ctx) ins(%x : tensor<?x?xf32>)
                      outs(%init : tensor<i64>) : tensor<i64>
  return %r : tensor<i64>
}
