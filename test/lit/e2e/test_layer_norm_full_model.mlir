// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// CHECK: llvm.func @wrap_layer_norm
// CHECK-NOT: onnx.LayerNormalization

module {
  func.func @main_graph(%x: tensor<2x4x16xf32>, %gamma: tensor<16xf32>,
                        %beta: tensor<16xf32>) -> (tensor<2x4x16xf32>) {
    %0 = "onnx.LayerNormalization"(%x, %gamma, %beta)
        {axis = -1 : si64, epsilon = 1.0e-05 : f32, stash_type = 1 : si64}
        : (tensor<2x4x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
    "onnx.Return"(%0) : (tensor<2x4x16xf32>) -> ()
  }

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
