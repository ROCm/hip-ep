// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// The reshape conversion sends every regroup through the 1-D form instead of
// looking for the single grouping that would express it, because
// ComposeExpandOfCollapseOp already finds that grouping. These check it does,
// and that it leaves the pair alone when no single grouping exists.
//
// reshape.mlir checks what the conversion emits before canonicalization.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --split-input-file -allow-unregistered-dialect -convert-onnx-to-hipsr --canonicalize | FileCheck %s

// The stated 32 and the input's 8x4 cancel, so the inferred extent is the
// batch read straight off the input's shape. The two body ops nest into
// [[0], [1, 2]], so the pair composes away. This is the flatten-all-but-batch
// shape ONNX graphs use most, and the hardest case to compose because one
// extent stays dynamic. Nothing is left but the single collapse.
// CHECK-LABEL: func.func @dynamic_extent_composes(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<?x8x4xf16, #hipsr.mem<device>>) -> tensor<?x32xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8x4xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?x32xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%[[IN_SHAPE:.*]]: !shape.shape):
// CHECK-NEXT:      %[[COLS:.*]] = arith.constant 32 : index
// CHECK-NEXT:      %[[AXIS:.*]] = shape.const_size 0
// CHECK-NEXT:      %[[SIZE:.*]] = shape.get_extent %[[IN_SHAPE]], %[[AXIS]] : !shape.shape, !shape.size -> !shape.size
// CHECK-NEXT:      %[[ROWS:.*]] = shape.size_to_index %[[SIZE]] : !shape.size
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.from_extents %[[ROWS]], %[[COLS]] : index, index
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<?x8x4xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<?x32xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<?x8x4xf16, #hipsr.mem<device>>, %{{.*}}: tensor<?x32xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[OUT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0], [1, 2]] : tensor<?x8x4xf16, #hipsr.mem<device>> into tensor<?x32xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<?x32xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<?x32xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<?x32xf16, #hipsr.mem<device>>
func.func @dynamic_extent_composes(%ctx: !hipsr.context,
                                   %input: tensor<?x8x4xf16>) -> tensor<?x?xf16> {
  %shape = "onnx.Constant"() {value = dense<[-1, 32]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %0 = "onnx.Reshape"(%input, %shape) {allowzero = 0 : si64}
      : (tensor<?x8x4xf16>, tensor<2xi64>) -> tensor<?x?xf16>
  "onnx.Return"(%0) : (tensor<?x?xf16>) -> ()
}

// -----

// 4x3 into 2x6 cuts the row-major order at 3, 6 and 9 on one side and at 6 on
// the other, so no single grouping merges and splits at once. The pair is the
// answer here and canonicalization leaves it alone, folding only the static
// extents in the shape region.
// CHECK-LABEL: func.func @interleaved_stays_paired(
// CHECK-SAME:    %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:    %[[INPUT:.*]]: tensor<4x3xf16, #hipsr.mem<device>>,
// CHECK-SAME:    %{{.*}}: tensor<2xi64, #hipsr.mem<device>>) -> tensor<2x6xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[INIT:.*]] = hipsr.placeholder(%[[CTX]]) ins(%[[INPUT]] : tensor<4x3xf16, #hipsr.mem<device>>) {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2x6xf16, #hipsr.mem<device>> shape_region {
// CHECK-NEXT:    ^bb0(%{{.*}}: !shape.shape):
// CHECK-NEXT:      %[[SHAPE:.*]] = shape.const_shape [2, 6] : !shape.shape
// CHECK-NEXT:      hipsr.shape_yield %[[SHAPE]] : !shape.shape
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[RESULT:.*]] = hipsr.compute(%[[CTX]]) ins(%[[INPUT]] : tensor<4x3xf16, #hipsr.mem<device>>) outs(%[[INIT]] : tensor<2x6xf16, #hipsr.mem<device>>) {
// CHECK-NEXT:    ^bb0(%{{.*}}: !hipsr.context, %[[IN:.*]]: tensor<4x3xf16, #hipsr.mem<device>>, %{{.*}}: tensor<2x6xf16, #hipsr.mem<device>>):
// CHECK-NEXT:      %[[FLAT:.*]] = tensor.collapse_shape %[[IN]] {{\[}}[0, 1]] : tensor<4x3xf16, #hipsr.mem<device>> into tensor<12xf16, #hipsr.mem<device>>
// CHECK-NEXT:      %[[OUT:.*]] = tensor.expand_shape %[[FLAT]] {{\[}}[0, 1]] output_shape [2, 6] : tensor<12xf16, #hipsr.mem<device>> into tensor<2x6xf16, #hipsr.mem<device>>
// CHECK-NEXT:      hipsr.compute_yield %[[OUT]] : tensor<2x6xf16, #hipsr.mem<device>>
// CHECK-NEXT:    } : tensor<2x6xf16, #hipsr.mem<device>>{{$}}
// CHECK-NEXT:    return %[[RESULT]] : tensor<2x6xf16, #hipsr.mem<device>>
func.func @interleaved_stays_paired(%ctx: !hipsr.context,
                                    %input: tensor<4x3xf16>,
                                    %shape: tensor<2xi64>) -> tensor<2x6xf16> {
  %0 = "onnx.Reshape"(%input, %shape)
      : (tensor<4x3xf16>, tensor<2xi64>) -> tensor<2x6xf16>
  "onnx.Return"(%0) : (tensor<2x6xf16>) -> ()
}
