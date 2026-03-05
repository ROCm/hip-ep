// RUN: hip-mlir-opt %s --morphizen-pipeline 2>&1 | FileCheck %s

// Verifies that --morphizen-pipeline falls back to DiskFileSystem when a model has
// onnx.Constant ops and no FileSystem is provided. Models with constants
// onnx.Constant ops and no FileSystem is provided (troubleshooting mode).

// CHECK: Wrote constant
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/attn/q_proj/MatMul/output_0"}, %arg1: tensor<1x128xi64> {onnx.name = "position_ids"}) -> (tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/attn/q_rotary/RotaryEmbedding/output_0"}) {
    %0 = onnx.Constant dense<1.000000e+00> : tensor<131072x64xf16>
    %1 = onnx.Constant dense<1.000000e+00> : tensor<131072x64xf16>
    %2 = "onnx.Custom"(%arg0, %arg1, %0, %1) <{function_name = "RotaryEmbedding"}> {domain_name = "com.microsoft", interleaved = 0 : si64, num_heads = 0 : si64, onnx_node_name = "/model/layers.0/attn/q_rotary/RotaryEmbedding", rotary_embedding_dim = 0 : si64} : (tensor<1x128x4096xf16>, tensor<1x128xi64>, tensor<131072x64xf16>, tensor<131072x64xf16>) -> tensor<1x128x4096xf16>
    onnx.Return %2 : tensor<1x128x4096xf16>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
