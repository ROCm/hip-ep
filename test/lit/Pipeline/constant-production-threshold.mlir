// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: mkdir -p %t/out
// RUN: hip-mlir-opt --onnx-to-hip-pipeline='externalize-min-num-elements=1 externalize-output-dir=%t/out' --mlir-print-ir-after=hip-externalize-constants %t/gbq.mlir 2>&1 | FileCheck %s --check-prefix=GBQ
// RUN: hip-mlir-opt --onnx-to-hip-pipeline='externalize-min-num-elements=1 externalize-output-dir=%t/out skip-constant-data=true' --mlir-print-ir-after=hip-externalize-constants %t/external.mlir 2>&1 | FileCheck %s --check-prefix=EXTERNAL

// Production threshold 1 keeps packed GBQ storage semantically unsigned in IR
// while preserving baseline integer strings in the JSON artifact.
// GBQ: IR Dump After ExternalizeConstantsPass
// GBQ-DAG: memref.global "private" @hip_ext_constant_data_0 : memref<512x64xui8>
// GBQ-DAG: hip.gather_block_quantized
// GBQ-SAME: tensor<512x64xui8>
// GBQ-SAME: unsigned_quant_storage

// skip-constant-data streaming does not open the missing file. The FileRef
// metadata is retained for runtime and the pass/pipeline complete.
// EXTERNAL: IR Dump After ExternalizeConstantsPass
// EXTERNAL: hipdnn.constant_file_offsets = array<i64: 123>
// EXTERNAL-SAME: hipdnn.constant_file_paths = ["/definitely/missing/production_weights.bin"]
// EXTERNAL-SAME: hipdnn.constant_source_kinds = array<i32: 2>
// EXTERNAL: memref.global "private" @hip_ext_constant_external_weight_0 : memref<2xi8>
// EXTERNAL-SAME: hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 2 : i64}

//--- gbq.mlir
module {
  func.func @main_graph(%indices: tensor<4xi32> {onnx.name = "indices"})
      -> tensor<4x64xf32> {
    %data = "onnx.Constant"() {
      onnx_node_name = "data",
      value = dense<1> : tensor<512x64xui8>
    } : () -> tensor<512x64xui8>
    %scales = "onnx.Constant"() {
      onnx_node_name = "scales",
      value = dense<1.000000e+00> : tensor<512x2xf32>
    } : () -> tensor<512x2xf32>
    %out = "onnx.Custom"(%data, %indices, %scales) {
      function_name = "GatherBlockQuantized",
      domain_name = "com.microsoft",
      bits = 8 : si64,
      block_size = 32 : si64,
      gather_axis = 0 : si64,
      quantize_axis = 1 : si64,
      onnx_node_name = "GatherBlockQuantized"
    } : (tensor<512x64xui8>, tensor<4xi32>, tensor<512x2xf32>)
        -> tensor<4x64xf32>
    "onnx.Return"(%out) : (tensor<4x64xf32>) -> ()
  }
}

//--- external.mlir
module {
  func.func @main_graph() -> tensor<2xi8> {
    %weight = "onnx.Constant"() {
      onnx_node_name = "external_weight",
      location = "/definitely/missing/production_weights.bin",
      offset = 123 : i64,
      size = 2 : i64
    } : () -> tensor<2xi8>
    "onnx.Return"(%weight) : (tensor<2xi8>) -> ()
  }
}
