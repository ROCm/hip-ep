// RUN: hip-mlir-opt %s --torch-hipdnn-pipeline | FileCheck %s

// Test Torch RMSNorm E2E full pipeline
// Input:  tensor<1x128x4096xf16>
// Scale:  tensor<4096xf16>
// Output: tensor<1x128x4096xf16>
//
// Verifies:
// 1. convert-torch-to-hip: torch.aten.rms_norm → hip.rms_norm
// 2. convert-hip-to-llvm: hip.rms_norm → wrap_miopenT5LayerNormForward
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// CHECK: llvm.func @wrap_miopenT5LayerNormForward
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK-NOT: torch.aten.rms_norm
module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16>, %arg1: tensor<4096xf16>) -> tensor<1x128x4096xf16> {
    %float1.000000e-06 = "torch.constant.float"() {value = 1.000000e-06 : f64} : () -> !torch.float
    %int4096 = "torch.constant.int"() {value = 4096 : i64} : () -> !torch.int
    %0 = "torch.prim.ListConstruct"(%int4096) : (!torch.int) -> !torch.list<int>
    %1 = "torch.aten.rms_norm"(%arg0, %0, %arg1, %float1.000000e-06) : (tensor<1x128x4096xf16>, !torch.list<int>, tensor<4096xf16>, !torch.float) -> tensor<1x128x4096xf16>
    return %1 : tensor<1x128x4096xf16>
  }
}
