// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-infer-shapes %s | FileCheck %s

// Phase 0 of --hip-infer-shapes: a constant-index tensor.extract from a shape
// vector assembled with tensor.insert_slice resolves to the scalar it reads,
// either an element of a tensor.from_elements or of an inline integer
// hip.constant. This is the Gemma-4 KV-sharing-layer pattern
// ConstantOfShape(Concat(Shape(q)[0:1], [0, 512])), which otherwise reads all
// three dims back to the host at runtime.

// CHECK-LABEL: func.func @fold_shape_vector
//  CHECK-SAME:   (%[[T:.*]]: tensor<?x2048xf16>)
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : i64
//   CHECK-DAG:   %[[C512:.*]] = arith.constant 512 : i64
//   CHECK-DAG:   %[[D:.*]] = tensor.dim %[[T]]
//   CHECK-DAG:   %[[DI:.*]] = arith.index_cast %[[D]] : index to i64
//   CHECK-NOT:   tensor.extract
//       CHECK:   return %[[DI]], %[[C0]], %[[C512]]
func.func @fold_shape_vector(%t: tensor<?x2048xf16>) -> (i64, i64, i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d = tensor.dim %t, %c0 : tensor<?x2048xf16>
  %di = arith.index_cast %d : index to i64
  %v0 = tensor.from_elements %di : tensor<1xi64>
  %cst = hip.constant {value = dense<[0, 512]> : tensor<2xi64>} : tensor<2xi64>
  %e = tensor.empty() : tensor<3xi64>
  %v1 = tensor.insert_slice %v0 into %e[0] [1] [1] : tensor<1xi64> into tensor<3xi64>
  %v2 = tensor.insert_slice %cst into %v1[1] [2] [1] : tensor<2xi64> into tensor<3xi64>
  %x0 = tensor.extract %v2[%c0] : tensor<3xi64>
  %x1 = tensor.extract %v2[%c1] : tensor<3xi64>
  %x2 = tensor.extract %v2[%c2] : tensor<3xi64>
  return %x0, %x1, %x2 : i64, i64, i64
}

// A dynamic index cannot be resolved at compile time.
// CHECK-LABEL: func.func @keep_dynamic_index
//       CHECK:   tensor.extract
func.func @keep_dynamic_index(%i: index) -> i64 {
  %cst = hip.constant {value = dense<[0, 512]> : tensor<2xi64>} : tensor<2xi64>
  %x = tensor.extract %cst[%i] : tensor<2xi64>
  return %x : i64
}

// A carrier without an inline value (file-backed) has nothing to read.
// CHECK-LABEL: func.func @keep_file_backed_constant
//       CHECK:   tensor.extract
func.func @keep_file_backed_constant() -> i64 {
  %c1 = arith.constant 1 : index
  %cst = hip.constant {location = "w.bin", offset = 0 : i64, size = 16 : i64} : tensor<2xi64>
  %x = tensor.extract %cst[%c1] : tensor<2xi64>
  return %x : i64
}
