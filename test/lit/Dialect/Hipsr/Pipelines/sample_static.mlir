// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// A sample for testing --hipsr-pipeline, with every extent static.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME:      %[[CTX:.+]]: !hipsr.context,
// CHECK-SAME:      %[[A:.+]]: tensor<2x3xf16, #hipsr.mem<device>> {onnx.name = "a"},
// CHECK-SAME:      %[[B:.+]]: tensor<2x4xf32, #hipsr.mem<device>> {onnx.name = "b"})
// CHECK-SAME:      -> (tensor<2x2xf32, #hipsr.mem<device>> {onnx.name = "y"})
// CHECK-SAME:      attributes {onnx.graph.name = "main_graph"} {

// The barrier the expand needs splits the graph in two. Everything up to it
// lands in domain 0, which yields both halves of the shape and data graphs for
// domain 1 to pick up. Inside a domain the constants come first, then the shape
// computations, then the allocations they size, then the data ops, and last a
// hipsr.preserve_shape tying each shape to the value filling the buffer it
// sized.
//
// The shape computations are flat here, and carry extent tensors rather than
// !shape.shape. -hipsr-convert-shape-to-extent inlined the scf.execute_region
// and shape.assuming that -hipsr-materialize-init-tensors wraps them in, and
// -remove-shape-constraints erased the shape.cstr_eq, shape.cstr_broadcastable,
// and shape.assuming_all that produced the witness.
// CHECK-NEXT:    %[[D0:.+]]:5 = hipsr.pool_domain(%[[CTX]], %[[A]], %[[B]] : !hipsr.context, tensor<2x3xf16, #hipsr.mem<device>>, tensor<2x4xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%[[D0_CTX:.+]]: !hipsr.context, %[[D0_A:.+]]: tensor<2x3xf16, #hipsr.mem<device>>, %[[D0_B:.+]]: tensor<2x4xf32, #hipsr.mem<device>>):

// The extent vector's length follows the rank of the operand, not its extents,
// so the operand's shape goes unread and the whole computation is the constant
// 2. Constants lead the block ahead of even the weights, because the greedy
// driver in -remove-shape-constraints hoists them there.
// CHECK-NEXT:      %[[EXTENTS_SHAPE:.+]] = shape.const_shape [2] : tensor<1xindex>
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index

// Both weights lead the operations. Only the first one has a shape a
// computation reads; the second is here because a constant takes no operands,
// so the pass brings every one of them over ahead of the computations.
// CHECK-NEXT:      %[[W1:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<3x1xf16>} : tensor<3x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[W2:.+]] = hipsr.constant {value = dense<{{.*}}> : tensor<4x2xf32>} : tensor<4x2xf32, #hipsr.mem<device>>

// The matmul shape: broadcast the batch dimensions, then append M from the lhs
// and N from the rhs. shape.shape_of recovers a static rank from its operand,
// which is what keeps the two shape.get_extent reads statically indexable.
//
// The split_at results stay tensor<?xindex> on purpose. --convert-shape-to-std
// lowers split_at to a tensor.extract_slice whose size is computed, so it
// produces tensor<?xindex> whatever result type it is asked for; claiming a
// static rank here would leave the extract_slice unable to replace it. That
// dynamic rank is then carried by the broadcast and the concat below.
// CHECK-NEXT:      %[[A_SHAPE:.+]] = shape.shape_of %[[D0_A]] : tensor<2x3xf16, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:      %[[W1_SHAPE:.+]] = shape.shape_of %[[W1]] : tensor<3x1xf16, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:      %[[A_BATCH:.+]], %[[A_TAIL:.+]] = "shape.split_at"(%[[A_SHAPE]], %[[C0]]) : (tensor<2xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK-NEXT:      %[[W1_BATCH:.+]], %[[W1_TAIL:.+]] = "shape.split_at"(%[[W1_SHAPE]], %[[C0]]) : (tensor<2xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK-NEXT:      %[[BATCH:.+]] = shape.broadcast %[[A_BATCH]], %[[W1_BATCH]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
// CHECK-NEXT:      %[[M:.+]] = shape.get_extent %[[A_SHAPE]], %[[C0]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[N:.+]] = shape.get_extent %[[W1_SHAPE]], %[[C1]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[MATRIX:.+]] = tensor.from_elements %[[M]], %[[N]] : tensor<2xindex>
// CHECK-NEXT:      %[[MM1_SHAPE:.+]] = tensor.concat dim(0) %[[BATCH]], %[[MATRIX]] : (tensor<?xindex>, tensor<2xindex>) -> tensor<?xindex>

// The cast keeps its operand's shape, so it has no computation of its own left:
// inlining the forwarding scf.execute_region leaves the matmul's shape reaching
// the cast's hipsr.preserve_shape directly.

// Every extent is in the types, so no allocation reads a shape back.
// CHECK-NEXT:      %[[MM1_INIT:.+]] = tensor.empty() : tensor<2x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CAST_INIT:.+]] = tensor.empty() : tensor<2x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXTENTS_INIT:.+]] = tensor.empty() : tensor<2xi64, #hipsr.mem<host>>

// CHECK-NEXT:      %[[MM1:.+]] = hipsr.matmul(%[[D0_CTX]]) ins(%[[D0_A]], %[[W1]] : tensor<2x3xf16, #hipsr.mem<device>>, tensor<3x1xf16, #hipsr.mem<device>>) outs(%[[MM1_INIT]] : tensor<2x1xf16, #hipsr.mem<device>>) : tensor<2x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[CAST:.+]] = hipsr.cast(%[[D0_CTX]]) ins(%[[MM1]] : tensor<2x1xf16, #hipsr.mem<device>>) outs(%[[CAST_INIT]] : tensor<2x1xf32, #hipsr.mem<device>>) : tensor<2x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXTENTS:.+]] = hipsr.compute(%[[D0_CTX]]) ins(%[[D0_B]] : tensor<2x4xf32, #hipsr.mem<device>>) outs(%[[EXTENTS_INIT]] : tensor<2xi64, #hipsr.mem<host>>) {
// CHECK-NEXT:      ^bb0(%{{.+}}: !hipsr.context, %{{.+}}: tensor<2x4xf32, #hipsr.mem<device>>, %{{.+}}: tensor<2xi64, #hipsr.mem<host>>):
// CHECK-NEXT:        %[[EXTENT_VECTOR:.+]] = arith.constant dense<[2, 4]> : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:        hipsr.compute_yield %[[EXTENT_VECTOR]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      } : tensor<2xi64, #hipsr.mem<host>>

// CHECK-NEXT:      hipsr.preserve_shape %[[MM1_SHAPE]], %[[MM1]] : tensor<?xindex>, tensor<2x1xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[MM1_SHAPE]], %[[CAST]] : tensor<?xindex>, tensor<2x1xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[EXTENTS_SHAPE]], %[[EXTENTS]] : tensor<1xindex>, tensor<2xi64, #hipsr.mem<host>>

// The second weight has no inputs, which keeps it in domain 0, and only the
// yield uses it there.
// CHECK-NEXT:      hipsr.pool_domain_yield %[[CAST_INIT]], %[[CAST]], %[[EXTENTS_INIT]], %[[EXTENTS]], %[[W2]] : tensor<2x1xf32, #hipsr.mem<device>>, tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<2xi64, #hipsr.mem<host>>, tensor<4x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    } -> tensor<2x1xf32, #hipsr.mem<device>>, tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<2xi64, #hipsr.mem<host>>, tensor<4x2xf32, #hipsr.mem<device>> {domain_id = 0 : i64}

// Domain 1 takes the shape-graph results first and the data results after, so
// the barrier keeps reading the buffers the shape graph names while the
// operations keep following results.
// CHECK-NEXT:    %[[D1:.+]] = hipsr.pool_domain(%[[CTX]], %[[D0]]#0, %[[D0]]#2, %[[D0]]#1, %[[D0]]#3, %[[D0]]#4 : !hipsr.context, tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>, tensor<4x2xf32, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%[[D1_CTX:.+]]: !hipsr.context, %[[D1_CAST_INIT:.+]]: tensor<2x1xf32, #hipsr.mem<device>>, %[[D1_EXTENTS_INIT:.+]]: tensor<2xi64, #hipsr.mem<host>>, %[[D1_CAST:.+]]: tensor<2x1xf32, #hipsr.mem<device>>, %[[D1_EXTENTS:.+]]: tensor<2xi64, #hipsr.mem<host>>, %[[D1_W2:.+]]: tensor<4x2xf32, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[D1_C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[D1_C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[EXTRACT_I1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[EXTRACT_I0:.+]] = arith.constant 0 : index

// The expand's destination is the reason for the cut: its extents come out of
// the host vector domain 0 wrote, so the barrier's shape computation reads that
// buffer instead of a shape. Both operands of the broadcast have a static rank,
// so its result does too, which is why the expand's preserve_shape below takes
// a tensor<2xindex> where the matmul's takes a tensor<?xindex>.
// CHECK-NEXT:      %[[IN_SHAPE:.+]] = shape.shape_of %[[D1_CAST_INIT]] : tensor<2x1xf32, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:      %[[E0:.+]] = tensor.extract %[[D1_EXTENTS_INIT]]{{\[}}%[[EXTRACT_I0]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[X0:.+]] = arith.index_cast %[[E0]] : i64 to index
// CHECK-NEXT:      %[[E1:.+]] = tensor.extract %[[D1_EXTENTS_INIT]]{{\[}}%[[EXTRACT_I1]]] : tensor<2xi64, #hipsr.mem<host>>
// CHECK-NEXT:      %[[X1:.+]] = arith.index_cast %[[E1]] : i64 to index
// CHECK-NEXT:      %[[REQ:.+]] = tensor.from_elements %[[X0]], %[[X1]] : tensor<2xindex>
// CHECK-NEXT:      %[[EXPAND_SHAPE:.+]] = shape.broadcast %[[IN_SHAPE]], %[[REQ]] : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>

// CHECK-NEXT:      %[[W2_SHAPE:.+]] = shape.shape_of %[[D1_W2]] : tensor<4x2xf32, #hipsr.mem<device>> -> tensor<2xindex>
// CHECK-NEXT:      %[[E_BATCH:.+]], %[[E_TAIL:.+]] = "shape.split_at"(%[[EXPAND_SHAPE]], %[[D1_C0]]) : (tensor<2xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK-NEXT:      %[[W2_BATCH:.+]], %[[W2_TAIL:.+]] = "shape.split_at"(%[[W2_SHAPE]], %[[D1_C0]]) : (tensor<2xindex>, index) -> (tensor<?xindex>, tensor<?xindex>)
// CHECK-NEXT:      %[[BATCH2:.+]] = shape.broadcast %[[E_BATCH]], %[[W2_BATCH]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
// CHECK-NEXT:      %[[M2:.+]] = shape.get_extent %[[EXPAND_SHAPE]], %[[D1_C0]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[N2:.+]] = shape.get_extent %[[W2_SHAPE]], %[[D1_C1]] : tensor<2xindex>, index -> index
// CHECK-NEXT:      %[[MATRIX2:.+]] = tensor.from_elements %[[M2]], %[[N2]] : tensor<2xindex>
// CHECK-NEXT:      %[[MM2_SHAPE:.+]] = tensor.concat dim(0) %[[BATCH2]], %[[MATRIX2]] : (tensor<?xindex>, tensor<2xindex>) -> tensor<?xindex>

// CHECK-NEXT:      %[[EXPAND_INIT:.+]] = tensor.empty() : tensor<2x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MM2_INIT:.+]] = tensor.empty() : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[EXPAND:.+]] = hipsr.expand(%[[D1_CTX]]) ins(%[[D1_CAST]], %[[D1_EXTENTS]] : tensor<2x1xf32, #hipsr.mem<device>>, tensor<2xi64, #hipsr.mem<host>>) outs(%[[EXPAND_INIT]] : tensor<2x4xf32, #hipsr.mem<device>>) : tensor<2x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      %[[MM2:.+]] = hipsr.matmul(%[[D1_CTX]]) ins(%[[EXPAND]], %[[D1_W2]] : tensor<2x4xf32, #hipsr.mem<device>>, tensor<4x2xf32, #hipsr.mem<device>>) outs(%[[MM2_INIT]] : tensor<2x2xf32, #hipsr.mem<device>>) : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[EXPAND_SHAPE]], %[[EXPAND]] : tensor<2xindex>, tensor<2x4xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.preserve_shape %[[MM2_SHAPE]], %[[MM2]] : tensor<?xindex>, tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.pool_domain_yield %[[MM2]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:    } -> tensor<2x2xf32, #hipsr.mem<device>> {domain_id = 1 : i64}

// CHECK-NEXT:    return %[[D1]] : tensor<2x2xf32, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @main_graph(%a: tensor<2x3xf16> {onnx.name = "a"},
                      %b: tensor<2x4xf32> {onnx.name = "b"})
    -> (tensor<2x2xf32> {onnx.name = "y"})
    attributes {onnx.graph.name = "main_graph"} {
  %w1 = "onnx.Constant"() {value = dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf16>}
      : () -> tensor<3x1xf16>
  %mm1 = "onnx.MatMul"(%a, %w1) : (tensor<2x3xf16>, tensor<3x1xf16>)
      -> tensor<2x1xf16>
  %cast = "onnx.Cast"(%mm1) {to = f32} : (tensor<2x1xf16>) -> tensor<2x1xf32>
  %shape = "onnx.Shape"(%b) : (tensor<2x4xf32>) -> tensor<2xi64>
  %expand = "onnx.Expand"(%cast, %shape)
      : (tensor<2x1xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  %w2 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0],
                                          [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>}
      : () -> tensor<4x2xf32>
  %y = "onnx.MatMul"(%expand, %w2) : (tensor<2x4xf32>, tensor<4x2xf32>)
      -> tensor<2x2xf32>
  "onnx.Return"(%y) : (tensor<2x2xf32>) -> ()
}
