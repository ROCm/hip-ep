// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// hipsr.preserve_shape through one-shot-bufferize.
//
// The op is metadata only: it has no results and no init operand, so it is not
// DPS and carries a hand-written BufferizableOpInterface model. What the model
// has to get right is that the `$data` operand follows the tensor into whatever
// buffer that tensor was folded into, while the `$shape` operand and the op's
// position are left alone. The contiguous CHECK-NEXT runs below pin that
// position; the NOCOPY run states separately that reporting neither a read nor
// a write keeps the op from forcing a copy on the buffer it names.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s
// RUN: hip-mlir-opt --split-input-file --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s --check-prefix=NOCOPY

// NOCOPY-NOT: memref.copy

// The case the op exists for: the shape describes a tensor.empty that a DPS op
// then writes in place. After bufferization both name the one memref.alloc, so
// the dynamic extent stays attached to the buffer it belongs to.
//
// The run stops at hip.matmul: what follows is a memref.cast that bufferization
// leaves dead behind the returned buffer, an artifact of the return-type
// conversion that has nothing to do with this op.
// CHECK-LABEL: func.func @dps_init(
// CHECK-SAME: %[[CTX:.+]]: !hip.context, %[[D0:.+]]: index,
// CHECK-SAME: -> memref<?x2048xf32> {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : memref<?x2048xf32>
// CHECK-NEXT: hip.matmul(%[[CTX]]) ins({{.+}}) outs(%[[ALLOC]] : memref<?x2048xf32>)
func.func @dps_init(%ctx: !hip.context, %d0: index, %lhs: tensor<?x64xf32>,
                    %rhs: tensor<64x2048xf32>) -> tensor<?x2048xf32> {
  %c2048 = arith.constant 2048 : index
  %shape = "shape.from_extents"(%d0, %c2048) : (index, index) -> !shape.shape
  %init = tensor.empty(%d0) : tensor<?x2048xf32>
  hipsr.preserve_shape %shape, %init : tensor<?x2048xf32>
  %out = hip.matmul(%ctx) ins(%lhs, %rhs : tensor<?x64xf32>, tensor<64x2048xf32>)
                          outs(%init : tensor<?x2048xf32>) : tensor<?x2048xf32>
  return %out : tensor<?x2048xf32>
}

// -----

// Naming the DPS result instead of its init reaches the same buffer, because
// the result was bufferized in place onto the init. The op also stays where it
// was, after the producer.
// CHECK-LABEL: func.func @dps_result(
// CHECK-SAME: %[[CTX:.+]]: !hip.context, %[[D0:.+]]: index,
// CHECK-SAME: -> memref<?x2048xf32> {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32>
// CHECK-NEXT: hip.matmul(%[[CTX]]) ins({{.+}}) outs(%[[ALLOC]] : memref<?x2048xf32>)
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : memref<?x2048xf32>
func.func @dps_result(%ctx: !hip.context, %d0: index, %lhs: tensor<?x64xf32>,
                      %rhs: tensor<64x2048xf32>) -> tensor<?x2048xf32> {
  %c2048 = arith.constant 2048 : index
  %shape = "shape.from_extents"(%d0, %c2048) : (index, index) -> !shape.shape
  %init = tensor.empty(%d0) : tensor<?x2048xf32>
  %out = hip.matmul(%ctx) ins(%lhs, %rhs : tensor<?x64xf32>, tensor<64x2048xf32>)
                          outs(%init : tensor<?x2048xf32>) : tensor<?x2048xf32>
  hipsr.preserve_shape %shape, %out : tensor<?x2048xf32>
  return %out : tensor<?x2048xf32>
}

// -----

// A tensor block argument has no producer to bufferize: the model resolves it
// to the memref the function boundary handed over, layout and all. Nothing is
// allocated for it, which the contiguous run to the closing brace establishes.
// CHECK-LABEL: func.func @func_arg(
// CHECK-SAME: %[[D0:.+]]: index,
// CHECK-SAME: %[[DATA:.+]]: memref<?x2048xf32, strided<[?, ?], offset: ?>>) {
// CHECK-NEXT: %[[C2048:.+]] = arith.constant 2048 : index
// CHECK-NEXT: %[[SHAPE:.+]] = shape.from_extents %[[D0]], %[[C2048]] : index, index
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[DATA]] : memref<?x2048xf32, strided<[?, ?], offset: ?>>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @func_arg(%d0: index, %data: tensor<?x2048xf32>) {
  %c2048 = arith.constant 2048 : index
  %shape = "shape.from_extents"(%d0, %c2048) : (index, index) -> !shape.shape
  hipsr.preserve_shape %shape, %data : tensor<?x2048xf32>
  return
}

// -----

// An opaque shape is not a tensor either, so bufferization carries it over
// untouched just like the from_extents form. This is the shape the intended
// producer builds, out of an scf.execute_region.
// CHECK-LABEL: func.func @opaque_shape(
// CHECK-SAME: %[[D0:.+]]: index, %[[SHAPE:.+]]: !shape.shape) {
// CHECK-NEXT: %[[ALLOC:.+]] = memref.alloc(%[[D0]]) {{.*}}: memref<?x2048xf32>
// CHECK-NEXT: hipsr.preserve_shape %[[SHAPE]], %[[ALLOC]] : memref<?x2048xf32>
// CHECK-NEXT: return
// CHECK-NEXT: }
func.func @opaque_shape(%d0: index, %shape: !shape.shape) {
  %init = tensor.empty(%d0) : tensor<?x2048xf32>
  hipsr.preserve_shape %shape, %init : tensor<?x2048xf32>
  return
}
