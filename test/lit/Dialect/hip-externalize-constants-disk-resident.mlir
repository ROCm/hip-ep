// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 skip-constant-data=true disk-resident-gather=true' %s | FileCheck %s --check-prefix=DISK
// RUN: hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 skip-constant-data=true' %s | FileCheck %s --check-prefix=OFF --implicit-check-not=disk_resident

// DISK: module attributes
// DISK-SAME: hipdnn.constant_offsets = array<i64: 0, 0, 2819260416, 6845792256>
// DISK-SAME: hipdnn.constant_source_kinds = array<i32: 4, 2, 2, 2>
// DISK-DAG: memref.global "private" @hip_ext_constant_ple_0 {{.*}}hip.external_data = {disk_resident, index = 0 : i64
// DISK-DAG: memref.global "private" @hip_ext_constant_embed_1 {{.*}}hip.external_data = {index = 1 : i64
// DISK-DAG: memref.global "private" @hip_ext_constant_shared_2 {{.*}}hip.external_data = {index = 2 : i64
// DISK-DAG: memref.global "private" @hip_ext_constant_inner_3 {{.*}}hip.external_data = {index = 3 : i64

// OFF: module attributes
// OFF-SAME: hipdnn.constant_source_kinds = array<i32: 2, 2, 2, 2>
// OFF: memref.global "private" @hip_ext_constant_ple_0

func.func @tables(%ctx: !hip.context, %idx: tensor<2xi64>,
                  %rows_ple: tensor<2x7680xf16>, %rows_emb: tensor<2x5376xf16>,
                  %cols: tensor<262144x2xf16>)
    -> (tensor<2x7680xf16>, tensor<2x5376xf16>, tensor<2x7680xf16>,
        tensor<262144x7680xf16>, tensor<262144x2xf16>) {
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
  return %0, %1, %2, %shared, %3
      : tensor<2x7680xf16>, tensor<2x5376xf16>, tensor<2x7680xf16>,
        tensor<262144x7680xf16>, tensor<262144x2xf16>
}
