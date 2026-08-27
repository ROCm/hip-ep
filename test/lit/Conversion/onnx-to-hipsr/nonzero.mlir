// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Converts onnx.NonZero to hipsr.nonzero, the hipsr.copy_d2h that brings the
// count to the host, and the hipsr.compute that narrows the worst-case search
// destination. Rejected forms live in nonzero-invalid.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// The search is DPS, so its placeholder stays empty for
// hipsr-populate-shape-region. The narrowing is a compute, so this fills the
// barrier region from the host copy of the count.
// CHECK-LABEL: func.func @nonzero_mask(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[MASK:.+]]: tensor<?x?xi8, #hipsr.mem<device>>) -> tensor<2x?xi64, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INITS:.+]]:2 = hipsr.placeholder(%[[CTX]]) ins(%[[MASK]] : tensor<?x?xi8, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[SEARCH:.+]]:2 = hipsr.nonzero(%[[CTX]]) ins(%[[MASK]] : tensor<?x?xi8, #hipsr.mem<device>>) outs(%[[INITS]]#0, %[[INITS]]#1 : tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>) : tensor<2x?xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// A copy keeps the source shape, so the host destination forwards the shape of
// the search's count destination.
// CHECK-NEXT:    %[[HOST_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INITS]]#1 : tensor<1xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:    ^bb0(%[[COUNT_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT:      hipsr.shape_yield %[[COUNT_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// The copy's own result goes unused: the shape graph names the destination.
// CHECK-NEXT:    %{{.+}} = hipsr.copy_d2h(%[[CTX]]) ins(%[[SEARCH]]#1 : tensor<1xi64, #hipsr.mem<device>>) outs(%[[HOST_INIT]] : tensor<1xi64, #hipsr.mem<host>>) : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[HOST_INIT]] : tensor<1xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<2x?xi64, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[COUNT:.+]]: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[ZERO:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[FOUND:.+]] = tensor.extract %[[COUNT]]{{\[}}%[[ZERO]]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[COLUMNS:.+]] = arith.index_cast %[[FOUND]] : i64 to index
// CHECK-NEXT:      %[[ROWS:.+]] = arith.constant 2 : index
// CHECK-NEXT:      %[[SHAPE:.+]] = shape.from_extents %[[ROWS]], %[[COLUMNS]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[NARROWED:.+]] = hipsr.compute(%[[CTX]]) ins(%[[SEARCH]]#0 : tensor<2x?xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[POSITIONS:.+]]: tensor<2x?xi64, #hipsr.mem<device>>, %[[DEST:.+]]: tensor<2x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[ONE:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM:.+]] = tensor.dim %[[DEST]], %[[ONE]] : tensor<2x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[SLICE:.+]] = tensor.extract_slice %[[POSITIONS]][0, 0] [2, %[[DIM]]] [1, 1] : tensor<2x?xi64, #hipsr.mem<device>> to tensor<2x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[SLICE]] : tensor<2x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<2x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[NARROWED]] : tensor<2x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @nonzero_mask(%ctx: !hipsr.context,
                        %mask: tensor<?x?xi8>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%mask) : (tensor<?x?xi8>) -> tensor<2x?xi64>
  "onnx.Return"(%0) : (tensor<2x?xi64>) -> ()
}

// -----

// A static input pins the worst case at its element count, so the search
// destination is static even though the published result stays dynamic. A
// rank-1 input names a position with a single row.
// CHECK-LABEL: func.func @static_capacity_rank_one(
// CHECK-SAME:    %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:    %[[MASK:.+]]: tensor<12xi1, #hipsr.mem<device>>) -> tensor<1x?xi64, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INITS:.+]]:2 = hipsr.placeholder(%[[CTX]]) ins(%[[MASK]] : tensor<12xi1, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1x12xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[SEARCH:.+]]:2 = hipsr.nonzero(%[[CTX]]) ins(%[[MASK]] : tensor<12xi1, #hipsr.mem<device>>) outs(%[[INITS]]#0, %[[INITS]]#1 : tensor<1x12xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>) : tensor<1x12xi64, #hipsr.mem<device>>, tensor<1xi64, #hipsr.mem<device>>
// CHECK-NEXT:    %[[HOST_INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[INITS]]#1 : tensor<1xi64, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<1xi64, #hipsr.mem<host>> shape_region {
// CHECK-NEXT:    ^bb0(%[[COUNT_SHAPE:.+]]: !shape.shape):
// CHECK-NEXT:      hipsr.shape_yield %[[COUNT_SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.+}} = hipsr.copy_d2h(%[[CTX]]) ins(%[[SEARCH]]#1 : tensor<1xi64, #hipsr.mem<device>>) outs(%[[HOST_INIT]] : tensor<1xi64, #hipsr.mem<host>>) : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:    %[[INIT:.+]] = hipsr.placeholder(%[[CTX]]) ins(%[[HOST_INIT]] : tensor<1xi64, #hipsr.mem<host>>) {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<1x?xi64, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[COUNT:.+]]: tensor<1xi64, #hipsr.mem<host>>):
// CHECK-NEXT:      %[[ZERO:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[FOUND:.+]] = tensor.extract %[[COUNT]]{{\[}}%[[ZERO]]] : tensor<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[COLUMNS:.+]] = arith.index_cast %[[FOUND]] : i64 to index
// CHECK-NEXT:      %[[ROWS:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[SHAPE:.+]] = shape.from_extents %[[ROWS]], %[[COLUMNS]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[NARROWED:.+]] = hipsr.compute(%[[CTX]]) ins(%[[SEARCH]]#0 : tensor<1x12xi64, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<1x?xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.+}}: !hipsr.context, %[[POSITIONS:.+]]: tensor<1x12xi64, #hipsr.mem<device>>, %[[DEST:.+]]: tensor<1x?xi64, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[ONE:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[DIM:.+]] = tensor.dim %[[DEST]], %[[ONE]] : tensor<1x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      %[[SLICE:.+]] = tensor.extract_slice %[[POSITIONS]][0, 0] [1, %[[DIM]]] [1, 1] : tensor<1x12xi64, #hipsr.mem<device>> to tensor<1x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[SLICE]] : tensor<1x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<1x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:    return %[[NARROWED]] : tensor<1x?xi64, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @static_capacity_rank_one(%ctx: !hipsr.context,
                                    %mask: tensor<12xi1>) -> tensor<1x?xi64> {
  %0 = "onnx.NonZero"(%mask) : (tensor<12xi1>) -> tensor<1x?xi64>
  "onnx.Return"(%0) : (tensor<1x?xi64>) -> ()
}
