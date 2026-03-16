// RUN: env HIPDNN_EP_DEBUG=1 hip-mlir-opt %s --morphizen-pipeline 2>&1 | FileCheck %s

// Verifies that --morphizen-pipeline falls back to DiskFileSystem when a model has
// onnx.Constant ops and no FileSystem is provided. Models with constants
// onnx.Constant ops and no FileSystem is provided (troubleshooting mode).

// CHECK: Wrote constant
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16> {onnx.name = "/model/embed_tokens/Gather/output_0"}) -> (tensor<1x128x4096xf16> {onnx.name = "/model/layers.0/input_layernorm/output_0"}) {
    %0 = onnx.Constant dense<1.000000e+00> : tensor<4096xf16>
    %1 = "onnx.Custom"(%arg0, %0) <{function_name = "SimplifiedLayerNormalization"}> {axis = -1 : si64, domain_name = "", epsilon = 9.99999974E-6 : f32, onnx_node_name = "/model/layers.0/input_layernorm/LayerNorm", stash_type = 1 : si64} : (tensor<1x128x4096xf16>, tensor<4096xf16>) -> tensor<1x128x4096xf16>
    onnx.Return %1 : tensor<1x128x4096xf16>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}
