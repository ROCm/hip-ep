// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s
//
// E2E coverage for BuildShapeFunctionPass (--hip-build-shape-fn) inside the
// full --hipdnn-pipeline. Dynamic leading-dim reshape:
//   [?,128,32] -> [?,32]   (collapse dims [0,1]; out_dim0 = in_dim0 * 128).
//
// This pins three things the standalone --hip-build-shape-fn LIT
// (test/lit/Dialect/hip-build-shape-fn.mlir) cannot:
//   1. The pass runs EARLY enough (immediately after --convert-onnx-to-hip)
//      that it sees `tensor.collapse_shape` and folds the product. Run on the
//      raw onnx.Reshape it would emit kDynamic instead -- the reshape's shape
//      operand is opaque to the reify-driven dim folds.
//   2. @infer_shapes carries the unused !hip.context arg0 (lowered to
//      !llvm.ptr) so it SURVIVES the ctx-requiring passes hip-lower-allocs /
//      hip-resolve-extern-constants instead of aborting their func-arg-0
//      assertion.
//   3. --convert-hip-to-llvm lowers it and GenerateInterface wraps it as the
//      exported C-ABI `inference_infer_shapes` that the EP calls before
//      ctx.GetOutput() to size output buffers.

// No ONNX or unrealized casts may survive the full lowering.
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: builtin.unrealized_conversion_cast

// The lowered shape program: arg0 is the (unused) !hip.context -> !llvm.ptr;
// the input dims follow as i64. out_dim0 = batch * 128, out_dim1 = 32, packed
// into the returned struct.
// CHECK-LABEL: llvm.func @infer_shapes
// CHECK-SAME:  (%{{[^:]+}}: !llvm.ptr, %[[B:[^:]+]]: i64, %{{[^:]+}}: i64, %{{[^:]+}}: i64) -> !llvm.struct<(i64, i64)>
// CHECK-DAG:   %[[C128:.+]] = llvm.mlir.constant(128 : index) : i64
// CHECK:       llvm.mul %[[B]], %[[C128]]

// The exported C-ABI wrapper: passes null for the unused ctx, calls the
// lowered shape program, scatters the results into the output_shapes rows.
// CHECK-LABEL: llvm.func @inference_infer_shapes
// CHECK-SAME:  llvm.emit_c_interface
// CHECK:       llvm.mlir.zero : !llvm.ptr
// CHECK:       llvm.call @infer_shapes

module {
  func.func @main_graph(%arg0: tensor<?x128x32xf16> {onnx.name = "input_0"}) -> (tensor<?x32xf16> {onnx.name = "output_0"}) {
    %shape = "onnx.Constant"() {value = dense<[-1, 32]> : tensor<2xi64>} : () -> tensor<2xi64>
    %result = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64, onnx_node_name = "Reshape_0"} : (tensor<?x128x32xf16>, tensor<2xi64>) -> tensor<?x32xf16>
    "onnx.Return"(%result) : (tensor<?x32xf16>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
