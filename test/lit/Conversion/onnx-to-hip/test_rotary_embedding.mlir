// RUN: hip-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%input: tensor<1x128x4096xf16>,
                         %position_ids: tensor<1x128xi64>,
                         %cos_cache: tensor<131072x64xf16>,
                         %sin_cache: tensor<131072x64xf16>)
      -> tensor<1x128x4096xf16> {
    %output = "onnx.Custom"(%input, %position_ids, %cos_cache, %sin_cache) {
      function_name = "RotaryEmbedding",
      domain_name = "com.microsoft",
      interleaved = 0 : si64,
      num_heads = 0 : si64,
      rotary_embedding_dim = 0 : si64
    } : (tensor<1x128x4096xf16>, tensor<1x128xi64>,
         tensor<131072x64xf16>, tensor<131072x64xf16>)
        -> tensor<1x128x4096xf16>
    return %output : tensor<1x128x4096xf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: !hip.context
// CHECK-SAME: tensor<1x128x4096xf16>
// CHECK: tensor.empty()
// CHECK: hip.rotary_embedding
// CHECK-SAME: interleaved = 0
// CHECK-SAME: num_heads = 0
// CHECK-SAME: rotary_embedding_dim = 0
// CHECK-NOT: hip.alloc
