// RUN: hip-opt %s --morphizen-pipeline 2>&1 | FileCheck %s

// Verifies that --morphizen-pipeline falls back to DiskFileSystem when a model has
// onnx.Constant ops and no FileSystem is provided. Models with constants
// onnx.Constant ops and no FileSystem is provided (troubleshooting mode).

// CHECK: Wrote constant
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/post_attention_layernorm/output_0"}) -> (tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/mlp/down_proj/MatMul/output_0"}) {
    %0 = onnx.Constant dense<1.000000e+00> : tensor<4096x14336xf16>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<4096x14336xf16>
    %2 = onnx.Constant dense<1.000000e+00> : tensor<14336x4096xf16>
    %3 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "/model/layers.0/mlp/gate_proj/MatMul"} : (tensor<1x128x4096xf16>, tensor<4096x14336xf16>) -> tensor<1x128x14336xf16>
    %4 = "onnx.MatMul"(%arg0, %1) {onnx_node_name = "/model/layers.0/mlp/up_proj/MatMul"} : (tensor<1x128x4096xf16>, tensor<4096x14336xf16>) -> tensor<1x128x14336xf16>
    %5 = "onnx.Sigmoid"(%3) {onnx_node_name = "/model/layers.0/mlp/act_fn/Sigmoid"} : (tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    %6 = "onnx.Mul"(%3, %5) {onnx_node_name = "/model/layers.0/mlp/act_fn/Mul"} : (tensor<1x128x14336xf16>, tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    %7 = "onnx.Mul"(%6, %4) {onnx_node_name = "/model/layers.0/mlp/Mul"} : (tensor<1x128x14336xf16>, tensor<1x128x14336xf16>) -> tensor<1x128x14336xf16>
    %8 = "onnx.MatMul"(%7, %2) {onnx_node_name = "/model/layers.0/mlp/down_proj/MatMul"} : (tensor<1x128x14336xf16>, tensor<14336x4096xf16>) -> tensor<1x128x4096xf16>
    onnx.Return %8 : tensor<1x128x4096xf16>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
