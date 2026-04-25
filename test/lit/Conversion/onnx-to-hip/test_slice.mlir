// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Dummy entry point required by generateModuleMetadata.
  func.func @main_graph(%arg0: tensor<128xf32>) -> tensor<128xf32> {
    return %arg0 : tensor<128xf32>
  }

  // ---- Slice with default step=1 along axis 0 ----
  // input: 8x16xf32, take rows [2, 6) -> output 4x16xf32, starts=[2, 0]
  func.func @test_slice_basic(%arg0: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %starts = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends   = "onnx.Constant"() {value = dense<6> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes   = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes)
        : (tensor<8x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }
  // CHECK-LABEL: func.func @test_slice_basic
  // CHECK: hip.slice(%{{.*}})
  // CHECK-SAME: starts = [2, 0]
  // CHECK-SAME: steps = [1, 1]

  // ---- Slice with step=2 along axis 1 ----
  // input: 8x16xf32, take cols [0, 8) step 2 -> output 8x4xf32, starts=[0, 0]
  func.func @test_slice_step(%arg0: tensor<8x16xf32>) -> tensor<8x4xf32> {
    %starts = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %ends   = "onnx.Constant"() {value = dense<8> : tensor<1xi64>} : () -> tensor<1xi64>
    %axes   = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %steps  = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
    %0 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps)
        : (tensor<8x16xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>)
            -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
  // CHECK-LABEL: func.func @test_slice_step
  // CHECK: hip.slice(%{{.*}})
  // CHECK-SAME: starts = [0, 0]
  // CHECK-SAME: steps = [1, 2]

  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
