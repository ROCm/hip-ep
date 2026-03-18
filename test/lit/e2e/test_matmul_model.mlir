// RUN: env HIPDNN_EP_DEBUG=1 hip-mlir-opt %s --morphizen-pipeline 2>&1 | FileCheck %s

// Verifies that --morphizen-pipeline falls back to DiskFileSystem when a model has
// onnx.Constant ops and no FileSystem is provided. Models with constants
// onnx.Constant ops and no FileSystem is provided (troubleshooting mode).

// CHECK: llvm.mlir.global private constant @__constant_
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK-NOT: onnx.Constant
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/input_layernorm/output_0"}) -> (tensor<1x128x1024xf16> {onnx.name = "/model/layers.0/attn/k_proj/MatMul/output_0"}) {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<4096x1024xf16>} : () -> tensor<4096x1024xf16>
    %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "/model/layers.0/attn/k_proj/MatMul"} : (tensor<1x128x4096xf16>, tensor<4096x1024xf16>) -> tensor<1x128x1024xf16>
    "onnx.Return"(%1) : (tensor<1x128x1024xf16>) -> ()
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
