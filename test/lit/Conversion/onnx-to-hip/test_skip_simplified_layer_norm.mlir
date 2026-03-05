// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x128x4096xf16>,
                         %skip: tensor<1x128x4096xf16>,
                         %gamma: tensor<4096xf16>)
      -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>) {
    %output, %skip_output = "onnx.Custom"(%input, %skip, %gamma) {
      function_name = "SkipSimplifiedLayerNormalization",
      domain_name = "com.microsoft",
      epsilon = 9.99999974E-6 : f32
    } : (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>,
         tensor<4096xf16>)
        -> (tensor<1x128x4096xf16>, tensor<1x128x4096xf16>)
    onnx.Return %output, %skip_output : tensor<1x128x4096xf16>, tensor<1x128x4096xf16>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: !hip.context
// CHECK-SAME: tensor<1x128x4096xf16>
// CHECK: tensor.empty() : tensor<1x128x4096xf16>
// CHECK: tensor.empty() : tensor<1x128x4096xf16>
// CHECK: hip.skip_simplified_layer_norm
// CHECK-SAME: epsilon = 9.99999974E-6
// CHECK-NOT: hip.alloc
