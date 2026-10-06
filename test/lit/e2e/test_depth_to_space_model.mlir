// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// DepthToSpace through the full pipeline. The channel permutation is a
// hip.transpose, which lowers to wrap_transpose. The surrounding reshapes
// are metadata.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.DepthToSpace
// CHECK: llvm.func @wrap_transpose
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x8x2x2xf32> {onnx.name = "input"})
      -> (tensor<1x2x4x4xf32> {onnx.name = "output"}) {
    %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 2 : si64, mode = "DCR",
                                    onnx_node_name = "depth_to_space"}
        : (tensor<1x8x2x2xf32>) -> tensor<1x2x4x4xf32>
    "onnx.Return"(%0) : (tensor<1x2x4x4xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
