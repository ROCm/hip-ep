// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 skip-constant-data=true disk-resident-gather-min-bytes=64' %t/tables.mlir | FileCheck %s --check-prefix=DISK
// RUN: hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 skip-constant-data=true' %t/tables.mlir | FileCheck %s --check-prefix=OFF --implicit-check-not=disk_resident
// RUN: printf '\000\000\000\000\000\000\000\000' > %t/full.bin
// RUN: mkdir -p %t/out
// RUN: cd %t && hip-mlir-opt --hip-externalize-constants='externalize-min-num-elements=1 externalize-output-dir=%t/out disk-resident-gather-min-bytes=8' full.mlir | FileCheck %s --check-prefix=FULL --implicit-check-not=disk_resident

// DISK: module attributes
// DISK-SAME: hipdnn.constant_offsets = array<i64: 0, 0, 64>
// DISK-SAME: hipdnn.constant_source_kinds = array<i32: 4, 2, 2>
// DISK-DAG: memref.global "private" @hip_ext_constant_ple_0 {{.*}}hip.external_data = {disk_resident, index = 0 : i64
// DISK-DAG: memref.global "private" @hip_ext_constant_shared_1 {{.*}}hip.external_data = {index = 1 : i64
// DISK-DAG: memref.global "private" @hip_ext_constant_inner_2 {{.*}}hip.external_data = {index = 2 : i64

// OFF: module attributes
// OFF-SAME: hipdnn.constant_offsets = array<i64: 0, 64, 128>
// OFF-SAME: hipdnn.constant_source_kinds = array<i32: 2, 2, 2>
// OFF: memref.global "private" @hip_ext_constant_ple_0

// FULL: memref.global "private" @hip_ext_constant_table_0

//--- tables.mlir
func.func @tables(%ctx: !hip.context, %idx: tensor<2xi64>,
                  %init_rows: tensor<2x8xf16>, %init_cols: tensor<4x2xf16>)
    -> (tensor<2x8xf16>, tensor<2x8xf16>, tensor<4x8xf16>, tensor<4x2xf16>) {
  %ple = hip.constant {
    location = "/models/embed.onnx_data", offset = 0 : i64, size = 64 : i64,
    symbol_name_hint = "ple"
  } : tensor<4x8xf16>
  %shared = hip.constant {
    location = "/models/embed.onnx_data", offset = 64 : i64, size = 64 : i64,
    symbol_name_hint = "shared"
  } : tensor<4x8xf16>
  %inner = hip.constant {
    location = "/models/embed.onnx_data", offset = 128 : i64, size = 64 : i64,
    symbol_name_hint = "inner"
  } : tensor<4x8xf16>
  %0 = hip.gather(%ctx) ins(%ple, %idx : tensor<4x8xf16>, tensor<2xi64>)
      outs(%init_rows : tensor<2x8xf16>) {axis = 0 : i64} : tensor<2x8xf16>
  %1 = hip.gather(%ctx) ins(%shared, %idx : tensor<4x8xf16>, tensor<2xi64>)
      outs(%init_rows : tensor<2x8xf16>) {axis = 0 : i64} : tensor<2x8xf16>
  %2 = hip.gather(%ctx) ins(%inner, %idx : tensor<4x8xf16>, tensor<2xi64>)
      outs(%init_cols : tensor<4x2xf16>) {axis = 1 : i64} : tensor<4x2xf16>
  return %0, %1, %shared, %2
      : tensor<2x8xf16>, tensor<2x8xf16>, tensor<4x8xf16>, tensor<4x2xf16>
}

//--- full.mlir
func.func @full(%ctx: !hip.context, %idx: tensor<2xi64>,
                %init: tensor<2x2xf16>) -> tensor<2x2xf16> {
  %table = hip.constant {
    location = "full.bin", offset = 0 : i64, size = 8 : i64,
    symbol_name_hint = "table"
  } : tensor<2x2xf16>
  %0 = hip.gather(%ctx) ins(%table, %idx : tensor<2x2xf16>, tensor<2xi64>)
      outs(%init : tensor<2x2xf16>) {axis = 0 : i64} : tensor<2x2xf16>
  return %0 : tensor<2x2xf16>
}
