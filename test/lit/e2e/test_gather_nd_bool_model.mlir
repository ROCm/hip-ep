// RUN: hip-mlir-opt %s --hipdnn-pipeline | FileCheck %s

// Test bool GatherND E2E full pipeline.
// A bool (i1) tensor is stored as one byte per element and reaches the
// runtime as HIPDNN_EP_DATATYPE_UINT8 (7). bf16 LLMs gather rows of a bool
// attention mask this way (here a 512 x 512 mask with batch_dims = 1).
//
// Verifies the complete hipdnn-pipeline:
// 1. convert-onnx-to-hip: onnx.GatherND -> hip.gather_nd
// 2. convert-hip-to-llvm: hip.gather_nd -> wrap_gather_nd with uint8
// 3. generate-interface: Create inference_init/compute/cleanup/metadata

// CHECK: module attributes {
// CHECK-SAME: hipdnn.input_count = 2
// CHECK-SAME: hipdnn.output_count = 1
// The last wrap_gather_nd argument is data_type, UINT8 (7).
// CHECK: llvm.func @wrap_gather_nd
// CHECK: llvm.call @hipdnn_ep_alloc_output
// CHECK: [[DT:%[0-9]+]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_gather_nd({{.*}}, [[DT]])
// CHECK: llvm.func @inference_init
// CHECK: llvm.func @inference_compute
// CHECK: llvm.func @inference_cleanup
// CHECK: llvm.func @inference_get_metadata_json
// CHECK-NOT: onnx.GatherND
module {
  func.func @main_graph(%arg0: tensor<1x512x512xi1> {onnx.name = "mask"},
                        %arg1: tensor<1x128x1xi64> {onnx.name = "indices"})
      -> (tensor<1x128x512xi1> {onnx.name = "output"}) {
    %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 1 : si64, onnx_node_name = "gather_nd_mask"} : (tensor<1x512x512xi1>, tensor<1x128x1xi64>) -> tensor<1x128x512xi1>
    "onnx.Return"(%0) : (tensor<1x128x512xi1>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
