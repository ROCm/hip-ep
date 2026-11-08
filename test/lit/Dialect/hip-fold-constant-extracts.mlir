// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --pass-pipeline='builtin.module(func.func(hip-fold-constant-extracts))' %s | FileCheck %s

// A constant-index tensor.extract from a vector assembled with
// tensor.insert_slice resolves to the scalar it reads: an element of a
// tensor.from_elements, of an inline integer hip.constant, or of another
// integer constant. This is the Gemma-4 KV-sharing-layer pattern
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

// An index outside the last slice walks back through every later insert to
// the one that wrote it; an element no insert wrote (tensor.empty) is kept.
// CHECK-LABEL: func.func @walk_back_through_dests
//  CHECK-SAME:   (%[[X:.*]]: i64)
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : i64
//   CHECK-DAG:   %[[C7:.*]] = arith.constant 7 : i64
//       CHECK:   %[[E:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<5xi64>
//       CHECK:   return %[[X]], %[[C5]], %[[C7]], %[[E]]
func.func @walk_back_through_dests(%x: i64) -> (i64, i64, i64, i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %a = tensor.from_elements %x : tensor<1xi64>
  %b = hip.constant {value = dense<5> : tensor<1xi64>} : tensor<1xi64>
  %c = hip.constant {value = dense<[6, 7]> : tensor<2xi64>} : tensor<2xi64>
  %e = tensor.empty() : tensor<5xi64>
  %v1 = tensor.insert_slice %a into %e[0] [1] [1] : tensor<1xi64> into tensor<5xi64>
  %v2 = tensor.insert_slice %b into %v1[1] [1] [1] : tensor<1xi64> into tensor<5xi64>
  %v3 = tensor.insert_slice %c into %v2[2] [2] [1] : tensor<2xi64> into tensor<5xi64>
  %x0 = tensor.extract %v3[%c0] : tensor<5xi64>
  %x1 = tensor.extract %v3[%c1] : tensor<5xi64>
  %x3 = tensor.extract %v3[%c3] : tensor<5xi64>
  %x4 = tensor.extract %v3[%c4] : tensor<5xi64>
  return %x0, %x1, %x3, %x4 : i64, i64, i64, i64
}

// Multi-dimensional indices are rebased into the slice and linearized
// row-major into the constant.
// CHECK-LABEL: func.func @multi_dim
//   CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : i64
//       CHECK:   %[[E:.*]] = tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<3x3xi64>
//       CHECK:   return %[[C5]], %[[E]]
func.func @multi_dim() -> (i64, i64) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = hip.constant {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>} : tensor<2x3xi64>
  %e = tensor.empty() : tensor<3x3xi64>
  %v = tensor.insert_slice %cst into %e[1, 0] [2, 3] [1, 1] : tensor<2x3xi64> into tensor<3x3xi64>
  %in = tensor.extract %v[%c2, %c1] : tensor<3x3xi64>
  %out = tensor.extract %v[%c0, %c1] : tensor<3x3xi64>
  return %in, %out : i64, i64
}

// Other integer constants resolve too, including index elements.
// CHECK-LABEL: func.func @arith_constant_index
//   CHECK-NOT:   tensor.extract
//       CHECK:   %[[C9:.*]] = arith.constant 9 : index
//       CHECK:   return %[[C9]]
func.func @arith_constant_index(%s: index) -> index {
  %c2 = arith.constant 2 : index
  %a = tensor.from_elements %s : tensor<1xindex>
  %cst = arith.constant dense<[8, 9]> : tensor<2xindex>
  %e = tensor.empty() : tensor<3xindex>
  %v1 = tensor.insert_slice %a into %e[0] [1] [1] : tensor<1xindex> into tensor<3xindex>
  %v2 = tensor.insert_slice %cst into %v1[1] [2] [1] : tensor<2xindex> into tensor<3xindex>
  %x = tensor.extract %v2[%c2] : tensor<3xindex>
  return %x : index
}

// Nothing but the resolved extracts changes: foldable ops and code that was
// already dead are left to the canonicalizer.
// CHECK-LABEL: func.func @no_unrelated_changes
//       CHECK:   arith.addi
//       CHECK:   tensor.extract_slice
func.func @no_unrelated_changes(%t: tensor<4xi64>) -> i64 {
  %a = arith.constant 1 : i64
  %b = arith.constant 2 : i64
  %s = arith.addi %a, %b : i64
  %dead = tensor.extract_slice %t[0] [2] [1] : tensor<4xi64> to tensor<2xi64>
  return %s : i64
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

// A slice with a dynamic offset cannot be placed at compile time.
// CHECK-LABEL: func.func @keep_dynamic_offset
//       CHECK:   tensor.extract
func.func @keep_dynamic_offset(%o: index) -> i64 {
  %c0 = arith.constant 0 : index
  %cst = hip.constant {value = dense<[0, 512]> : tensor<2xi64>} : tensor<2xi64>
  %e = tensor.empty() : tensor<3xi64>
  %v = tensor.insert_slice %cst into %e[%o] [2] [1] : tensor<2xi64> into tensor<3xi64>
  %x = tensor.extract %v[%c0] : tensor<3xi64>
  return %x : i64
}

// Strided slices are not resolved.
// CHECK-LABEL: func.func @keep_strided_slice
//       CHECK:   tensor.extract
func.func @keep_strided_slice() -> i64 {
  %c2 = arith.constant 2 : index
  %cst = hip.constant {value = dense<[0, 512]> : tensor<2xi64>} : tensor<2xi64>
  %e = tensor.empty() : tensor<4xi64>
  %v = tensor.insert_slice %cst into %e[0] [2] [2] : tensor<2xi64> into tensor<4xi64>
  %x = tensor.extract %v[%c2] : tensor<4xi64>
  return %x : i64
}

// Rank-reducing slices are not resolved.
// CHECK-LABEL: func.func @keep_rank_reducing_slice
//       CHECK:   tensor.extract
func.func @keep_rank_reducing_slice() -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = hip.constant {value = dense<[0, 512]> : tensor<2xi64>} : tensor<2xi64>
  %e = tensor.empty() : tensor<1x3xi64>
  %v = tensor.insert_slice %cst into %e[0, 1] [1, 2] [1, 1] : tensor<2xi64> into tensor<1x3xi64>
  %x = tensor.extract %v[%c0, %c1] : tensor<1x3xi64>
  return %x : i64
}

// Only integer and index elements are resolved.
// CHECK-LABEL: func.func @keep_float
//       CHECK:   tensor.extract
func.func @keep_float() -> f32 {
  %c1 = arith.constant 1 : index
  %cst = hip.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
  %x = tensor.extract %cst[%c1] : tensor<2xf32>
  return %x : f32
}
