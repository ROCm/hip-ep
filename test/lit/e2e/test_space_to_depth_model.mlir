// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// SpaceToDepth through the full pipeline. The spatial-to-channel permutation
// is a hip.transpose, which lowers to wrap_transpose. The surrounding
// reshapes are metadata.

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 1
// CHECK-SAME: hipdnn.output_count = 1
// CHECK-NOT: onnx.SpaceToDepth
// CHECK: llvm.func @wrap_transpose
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
module {
  func.func @main_graph(%arg0: tensor<1x3x4x4xf32> {onnx.name = "input"})
      -> (tensor<1x12x2x2xf32> {onnx.name = "output"}) {
    %0 = "onnx.SpaceToDepth"(%arg0) {blocksize = 2 : si64,
                                    onnx_node_name = "space_to_depth"}
        : (tensor<1x3x4x4xf32>) -> tensor<1x12x2x2xf32>
    "onnx.Return"(%0) : (tensor<1x12x2x2xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
