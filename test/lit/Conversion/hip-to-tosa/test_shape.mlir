// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the shape-manipulating ops lower to their TOSA counterparts inside a
// rock.kernel function, so rocMLIR can fold them into the neighbouring conv or
// gemm instead of paying for a separate kernel, and verify the forms the
// conversion leaves alone.
//
// Only ONNX Transpose reaches this pass as a hip op. OnnxToHip decomposes
// Reshape, Squeeze, Unsqueeze and Flatten into tensor.collapse_shape and
// tensor.expand_shape, and decomposes the constant, unit-stride half of Slice
// into tensor.extract_slice, so those four ops are covered through the builtin
// tensor ops they become. The remaining half of Slice stays as hip.slice,
// which tosa.slice cannot express and this pass does not claim.
//
// FILE LAYOUT:
// Everything that converts or passes through lives in the first
// --split-input-file chunk, so it is one module and therefore also covers
// several ops converting in a single pass run. The rejected form gets its own
// chunk: a rejection aborts the run for the whole module, so sharing a chunk
// would let it mask the cases after it.
//
// This test validates:
// - hip.transpose maps to tosa.transpose with the permutation unchanged
// - The hip context and DPS outs operand are both dropped
// - collapse_shape and expand_shape map to tosa.reshape, covering Reshape,
//   Squeeze, Unsqueeze and Flatten
// - extract_slice maps to tosa.slice, with start and size as tosa.const_shape
// - Strided, rank-reducing and dynamically shaped tensor ops are left in place
//   for rocMLIR to consume directly, rather than failing the pass
// - The pass is a no-op on functions without rock.kernel
// - A dynamically shaped hip.transpose is rejected rather than lowered to
//   invalid TOSA
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Transpose.
//===----------------------------------------------------------------------===//

// The NHWC-to-NCHW permutation an ONNX image model opens with.
// CHECK-LABEL: func.func @transpose_nhwc
// CHECK: tosa.transpose %arg1 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x224x224x3xf32>) -> tensor<1x3x224x224xf32>
// CHECK-NOT: hip.transpose
func.func @transpose_nhwc(%ctx: !hip.context, %x: tensor<1x224x224x3xf32>,
                          %init: tensor<1x3x224x224xf32>)
    -> tensor<1x3x224x224xf32> attributes {rock.kernel} {
  %r = hip.transpose(%ctx) ins(%x : tensor<1x224x224x3xf32>)
                           outs(%init : tensor<1x3x224x224xf32>)
                           {perm = [0, 3, 1, 2]} : tensor<1x3x224x224xf32>
  return %r : tensor<1x3x224x224xf32>
}

// The last-two-dimension swap that feeds an attention matmul.
// CHECK-LABEL: func.func @transpose_2d
// CHECK: tosa.transpose %arg1 {perms = array<i32: 1, 0>} : (tensor<3x5xf16>) -> tensor<5x3xf16>
func.func @transpose_2d(%ctx: !hip.context, %x: tensor<3x5xf16>,
                        %init: tensor<5x3xf16>) -> tensor<5x3xf16>
    attributes {rock.kernel} {
  %r = hip.transpose(%ctx) ins(%x : tensor<3x5xf16>)
                           outs(%init : tensor<5x3xf16>)
                           {perm = [1, 0]} : tensor<5x3xf16>
  return %r : tensor<5x3xf16>
}

//===----------------------------------------------------------------------===//
// Reshape, Flatten, Squeeze and Unsqueeze, via the tensor ops they become.
//===----------------------------------------------------------------------===//

// ResNet's Flatten between the global pool and the classifier.
// CHECK-LABEL: func.func @flatten
// CHECK: %[[SHAPE:.*]] = tosa.const_shape {values = dense<[1, 2048]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: tosa.reshape %arg0, %[[SHAPE]] : (tensor<1x2048x1x1xf16>, !tosa.shape<2>) -> tensor<1x2048xf16>
// CHECK-NOT: tensor.collapse_shape
func.func @flatten(%x: tensor<1x2048x1x1xf16>) -> tensor<1x2048xf16>
    attributes {rock.kernel} {
  %c = tensor.collapse_shape %x [[0], [1, 2, 3]]
      : tensor<1x2048x1x1xf16> into tensor<1x2048xf16>
  return %c : tensor<1x2048xf16>
}

// Squeeze drops the size-1 dimensions.
// CHECK-LABEL: func.func @squeeze
// CHECK: %[[SHAPE:.*]] = tosa.const_shape {values = dense<4> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK: tosa.reshape %arg0, %[[SHAPE]] : (tensor<1x4x1xf32>, !tosa.shape<1>) -> tensor<4xf32>
func.func @squeeze(%x: tensor<1x4x1xf32>) -> tensor<4xf32>
    attributes {rock.kernel} {
  %c = tensor.collapse_shape %x [[0, 1, 2]]
      : tensor<1x4x1xf32> into tensor<4xf32>
  return %c : tensor<4xf32>
}

// Unsqueeze adds one back.
// CHECK-LABEL: func.func @unsqueeze
// CHECK: %[[SHAPE:.*]] = tosa.const_shape {values = dense<[2, 3, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK: tosa.reshape %arg0, %[[SHAPE]] : (tensor<2x3xf16>, !tosa.shape<3>) -> tensor<2x3x1xf16>
// CHECK-NOT: tensor.expand_shape
func.func @unsqueeze(%x: tensor<2x3xf16>) -> tensor<2x3x1xf16>
    attributes {rock.kernel} {
  %e = tensor.expand_shape %x [[0], [1, 2]] output_shape [2, 3, 1]
      : tensor<2x3xf16> into tensor<2x3x1xf16>
  return %e : tensor<2x3x1xf16>
}

//===----------------------------------------------------------------------===//
// Slice.
//===----------------------------------------------------------------------===//

// tosa.slice carries start and size as shape operands rather than attributes.
// CHECK-LABEL: func.func @slice
// CHECK-DAG: %[[SIZE:.*]] = tosa.const_shape {values = dense<[2, 6]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG: %[[START:.*]] = tosa.const_shape {values = dense<[1, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK: tosa.slice %arg0, %[[START]], %[[SIZE]] : (tensor<4x6xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x6xf32>
// CHECK-NOT: tensor.extract_slice
func.func @slice(%x: tensor<4x6xf32>) -> tensor<2x6xf32>
    attributes {rock.kernel} {
  %s = tensor.extract_slice %x[1, 0] [2, 6] [1, 1]
      : tensor<4x6xf32> to tensor<2x6xf32>
  return %s : tensor<2x6xf32>
}

//===----------------------------------------------------------------------===//
// Forms left in place. rocMLIR consumes the tensor ops directly, so a form
// with no TOSA spelling is passed through rather than failing the pass.
//===----------------------------------------------------------------------===//

// tosa.slice has no stride operand.
// CHECK-LABEL: func.func @slice_strided
// CHECK: tensor.extract_slice
// CHECK-NOT: tosa.slice
func.func @slice_strided(%x: tensor<2x4xf32>) -> tensor<1x2xf32>
    attributes {rock.kernel} {
  %s = tensor.extract_slice %x[1, 0] [1, 2] [1, 2]
      : tensor<2x4xf32> to tensor<1x2xf32>
  return %s : tensor<1x2xf32>
}

// tosa.slice reads a window of the same rank as its input.
// CHECK-LABEL: func.func @slice_rank_reducing
// CHECK: tensor.extract_slice
// CHECK-NOT: tosa.slice
func.func @slice_rank_reducing(%x: tensor<4x6xf32>) -> tensor<6xf32>
    attributes {rock.kernel} {
  %s = tensor.extract_slice %x[1, 0] [1, 6] [1, 1]
      : tensor<4x6xf32> to tensor<6xf32>
  return %s : tensor<6xf32>
}

// tosa.reshape's shape operand is a compile-time constant.
// CHECK-LABEL: func.func @collapse_dynamic
// CHECK: tensor.collapse_shape
// CHECK-NOT: tosa.reshape
func.func @collapse_dynamic(%x: tensor<?x2x3xf16>) -> tensor<?x6xf16>
    attributes {rock.kernel} {
  %c = tensor.collapse_shape %x [[0], [1, 2]]
      : tensor<?x2x3xf16> into tensor<?x6xf16>
  return %c : tensor<?x6xf16>
}

// Without rock.kernel the pass returns before looking at the body.
// CHECK-LABEL: func.func @not_a_kernel
// CHECK: tensor.extract_slice
// CHECK-NOT: tosa.slice
func.func @not_a_kernel(%x: tensor<4x6xf32>) -> tensor<2x6xf32> {
  %s = tensor.extract_slice %x[1, 0] [2, 6] [1, 1]
      : tensor<4x6xf32> to tensor<2x6xf32>
  return %s : tensor<2x6xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Rejected form. Unlike the tensor ops above, a hip op this pass claims has to
// convert or the pass fails.
//===----------------------------------------------------------------------===//

// tosa.transpose needs a static result shape to permute.
func.func @transpose_dynamic(%ctx: !hip.context, %x: tensor<?x4xf32>,
                             %init: tensor<4x?xf32>) -> tensor<4x?xf32>
    attributes {rock.kernel} {
  // expected-error @+1 {{failed to legalize operation 'hip.transpose'}}
  %r = hip.transpose(%ctx) ins(%x : tensor<?x4xf32>)
                           outs(%init : tensor<4x?xf32>)
                           {perm = [1, 0]} : tensor<4x?xf32>
  return %r : tensor<4x?xf32>
}
