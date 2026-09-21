// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the standard-tensor residues produced by OnnxToHip for Split and
// Concat convert to their direct TOSA counterparts inside a rock.kernel.
//
// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file \
// RUN:   --verify-diagnostics %s | FileCheck %s

// OnnxToHip lowers Split to one tensor.extract_slice per result.
// CHECK-LABEL: func.func @split
// CHECK-DAG: tosa.slice %arg0
// CHECK-DAG: tosa.slice %arg0
// CHECK-NOT: tensor.extract_slice
func.func @split(%x: tensor<2x6xf32>)
    -> (tensor<2x2xf32>, tensor<2x4xf32>) attributes {rock.kernel} {
  %left = tensor.extract_slice %x[0, 0] [2, 2] [1, 1]
      : tensor<2x6xf32> to tensor<2x2xf32>
  %right = tensor.extract_slice %x[0, 2] [2, 4] [1, 1]
      : tensor<2x6xf32> to tensor<2x4xf32>
  return %left, %right : tensor<2x2xf32>, tensor<2x4xf32>
}

// OnnxToHip lowers Concat to a tensor.empty plus a contiguous insert_slice
// chain. The complete chain maps to one tosa.concat.
// CHECK-LABEL: func.func @concat
// CHECK: tosa.concat %arg0, %arg1, %arg2 {axis = 1 : i32}
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: tensor.empty
func.func @concat(%a: tensor<2x3xf16>, %b: tensor<2x4xf16>,
                  %c: tensor<2x5xf16>) -> tensor<2x12xf16>
    attributes {rock.kernel} {
  %init = tensor.empty() : tensor<2x12xf16>
  %0 = tensor.insert_slice %a into %init[0, 0] [2, 3] [1, 1]
      : tensor<2x3xf16> into tensor<2x12xf16>
  %1 = tensor.insert_slice %b into %0[0, 3] [2, 4] [1, 1]
      : tensor<2x4xf16> into tensor<2x12xf16>
  %2 = tensor.insert_slice %c into %1[0, 7] [2, 5] [1, 1]
      : tensor<2x5xf16> into tensor<2x12xf16>
  return %2 : tensor<2x12xf16>
}

// A partial insertion is not a Concat decomposition and stays unchanged.
// CHECK-LABEL: func.func @partial_insert
// CHECK: tensor.insert_slice
// CHECK-NOT: tosa.concat
func.func @partial_insert(%x: tensor<2x2xf32>) -> tensor<2x4xf32>
    attributes {rock.kernel} {
  %init = tensor.empty() : tensor<2x4xf32>
  %r = tensor.insert_slice %x into %init[0, 0] [2, 2] [1, 1]
      : tensor<2x2xf32> into tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}
