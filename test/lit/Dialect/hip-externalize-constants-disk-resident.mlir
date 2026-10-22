// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 skip-constant-data=true' %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: hipdnn.constant_offsets = array<i64: 0, 0, 2819260416, 6845792256, 0, 10872324096, 11878957056>
// CHECK-SAME: hipdnn.constant_source_kinds = array<i32: 4, 2, 2, 2, 4, 2, 2>
// CHECK-DAG: memref.global "private" @hip_ext_constant_ple_0 {{.*}}hip.external_data = {disk_resident, index = 0 : i64
// CHECK-DAG: memref.global "private" @hip_ext_constant_embed_1 {{.*}}hip.external_data = {index = 1 : i64
// CHECK-DAG: memref.global "private" @hip_ext_constant_shared_2 {{.*}}hip.external_data = {index = 2 : i64
// CHECK-DAG: memref.global "private" @hip_ext_constant_inner_3 {{.*}}hip.external_data = {index = 3 : i64
// CHECK-DAG: memref.global "private" @hip_ext_constant_ple_q4_4 {{.*}}hip.external_data = {disk_resident, index = 4 : i64
// CHECK-DAG: memref.global "private" @hip_ext_constant_ple_q8_5 {{.*}}hip.external_data = {index = 5 : i64
// CHECK-DAG: memref.global "private" @hip_ext_constant_ple_q4_qa0_6 {{.*}}hip.external_data = {index = 6 : i64

// ple_q4 and ple_q8 store the same bytes; 4-bit packs two elements per byte, so
// only ple_q4 reaches the logical-element threshold. ple_q4_qa0 is packed along
// the gather axis, so one stored row holds two table rows and it stays on GPU.
func.func @tables(%ctx: !hip.context, %idx: tensor<2xi64>,
                  %rows_ple: tensor<2x7680xf16>, %rows_emb: tensor<2x5376xf16>,
                  %cols: tensor<262144x2xf16>,
                  %scales_q4: tensor<262144x240xf16>,
                  %scales_q8: tensor<262144x120xf16>,
                  %rows_q8: tensor<2x3840xf16>,
                  %scales_qa0: tensor<8192x7680xf16>)
    -> (tensor<2x7680xf16>, tensor<2x5376xf16>, tensor<2x7680xf16>,
        tensor<262144x7680xf16>, tensor<262144x2xf16>, tensor<2x7680xf16>,
        tensor<2x3840xf16>, tensor<2x7680xf16>) {
  %ple = hip.constant {
    location = "/models/weights.data", offset = 0 : i64, size = 4026531840 : i64,
    symbol_name_hint = "ple"
  } : tensor<262144x7680xf16>
  %embed = hip.constant {
    location = "/models/weights.data", offset = 4026531840 : i64,
    size = 2819260416 : i64, symbol_name_hint = "embed"
  } : tensor<262208x5376xf16>
  %shared = hip.constant {
    location = "/models/weights.data", offset = 6845792256 : i64,
    size = 4026531840 : i64, symbol_name_hint = "shared"
  } : tensor<262144x7680xf16>
  %inner = hip.constant {
    location = "/models/weights.data", offset = 10872324096 : i64,
    size = 4026531840 : i64, symbol_name_hint = "inner"
  } : tensor<262144x7680xf16>
  %0 = hip.gather(%ctx) ins(%ple, %idx : tensor<262144x7680xf16>, tensor<2xi64>)
      outs(%rows_ple : tensor<2x7680xf16>) {axis = 0 : i64} : tensor<2x7680xf16>
  %1 = hip.gather(%ctx) ins(%embed, %idx : tensor<262208x5376xf16>, tensor<2xi64>)
      outs(%rows_emb : tensor<2x5376xf16>) {axis = 0 : i64} : tensor<2x5376xf16>
  %2 = hip.gather(%ctx) ins(%shared, %idx : tensor<262144x7680xf16>, tensor<2xi64>)
      outs(%rows_ple : tensor<2x7680xf16>) {axis = 0 : i64} : tensor<2x7680xf16>
  %3 = hip.gather(%ctx) ins(%inner, %idx : tensor<262144x7680xf16>, tensor<2xi64>)
      outs(%cols : tensor<262144x2xf16>) {axis = 1 : i64} : tensor<262144x2xf16>
  %ple_q4 = hip.constant {
    location = "/models/weights.data", offset = 14898855936 : i64,
    size = 1006632960 : i64, symbol_name_hint = "ple_q4"
  } : tensor<262144x3840xui8>
  %ple_q8 = hip.constant {
    location = "/models/weights.data", offset = 15905488896 : i64,
    size = 1006632960 : i64, symbol_name_hint = "ple_q8"
  } : tensor<262144x3840xui8>
  %4 = hip.gather_block_quantized(%ctx)
      ins(%ple_q4, %idx, %scales_q4 : tensor<262144x3840xui8>, tensor<2xi64>,
          tensor<262144x240xf16>)
      outs(%rows_ple : tensor<2x7680xf16>)
      {bits = 4 : i64, block_size = 32 : i64, gather_axis = 0 : i64,
       quantize_axis = 1 : i64} : tensor<2x7680xf16>
  %5 = hip.gather_block_quantized(%ctx)
      ins(%ple_q8, %idx, %scales_q8 : tensor<262144x3840xui8>, tensor<2xi64>,
          tensor<262144x120xf16>)
      outs(%rows_q8 : tensor<2x3840xf16>)
      {bits = 8 : i64, block_size = 32 : i64, gather_axis = 0 : i64,
       quantize_axis = 1 : i64} : tensor<2x3840xf16>
  %ple_q4_qa0 = hip.constant {
    location = "/models/weights.data", offset = 16912121856 : i64,
    size = 1006632960 : i64, symbol_name_hint = "ple_q4_qa0"
  } : tensor<131072x7680xui8>
  %6 = hip.gather_block_quantized(%ctx)
      ins(%ple_q4_qa0, %idx, %scales_qa0 : tensor<131072x7680xui8>,
          tensor<2xi64>, tensor<8192x7680xf16>)
      outs(%rows_ple : tensor<2x7680xf16>)
      {bits = 4 : i64, block_size = 32 : i64, gather_axis = 0 : i64,
       quantize_axis = 0 : i64} : tensor<2x7680xf16>
  return %0, %1, %2, %shared, %3, %4, %5, %6
      : tensor<2x7680xf16>, tensor<2x5376xf16>, tensor<2x7680xf16>,
        tensor<262144x7680xf16>, tensor<262144x2xf16>, tensor<2x7680xf16>,
        tensor<2x3840xf16>, tensor<2x7680xf16>
}
