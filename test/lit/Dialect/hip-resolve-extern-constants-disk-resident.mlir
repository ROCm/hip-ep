// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt --hip-resolve-extern-constants %t/gather.mlir | FileCheck %s
// RUN: not hip-mlir-opt --hip-resolve-extern-constants %t/other-use.mlir 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-LABEL: func.func @gather
// CHECK:       %[[T:.*]] = hip.get_constant(%{{.*}}, %{{.*}}) {hip.disk_resident} : memref<4x8xf16>
// CHECK:       hip.gather(%{{.*}}) ins(%[[T]], %{{.*}} : memref<4x8xf16>, memref<2xi64>)
// CHECK:       hip.get_constant(%{{.*}}, %{{.*}}) : memref<4x8xf16>
// CHECK:       %[[Q:.*]] = hip.get_constant(%{{.*}}, %{{.*}}) {hip.disk_resident} : memref<4x4xui8>
// CHECK:       hip.gather_block_quantized(%{{.*}}) ins(%[[Q]], %{{.*}}, %{{.*}} : memref<4x4xui8>

// ERR: error: disk-resident constant has no GPU copy; only hip.gather / hip.gather_block_quantized data may use it

//--- gather.mlir
module {
  memref.global "private" @ple : memref<4x8xf16> {
    alignment = 64 : i64,
    hip.external_data = {disk_resident, index = 0 : i64, offset = 0 : i64, size = 64 : i64}
  }
  memref.global "private" @dense : memref<4x8xf16> {
    alignment = 64 : i64,
    hip.external_data = {index = 1 : i64, offset = 0 : i64, size = 64 : i64}
  }
  memref.global "private" @ple_q4 : memref<4x4xui8> {
    alignment = 64 : i64,
    hip.external_data = {disk_resident, index = 2 : i64, offset = 0 : i64, size = 16 : i64}
  }
  func.func @gather(%ctx: !hip.context, %idx: memref<2xi64>,
                    %out: memref<2x8xf16>, %scales: memref<4x1xf16>)
      -> memref<4x8xf16> {
    %t = memref.get_global @ple : memref<4x8xf16>
    hip.gather(%ctx) ins(%t, %idx : memref<4x8xf16>, memref<2xi64>)
        outs(%out : memref<2x8xf16>) {axis = 0 : i64}
    %d = memref.get_global @dense : memref<4x8xf16>
    %q = memref.get_global @ple_q4 : memref<4x4xui8>
    hip.gather_block_quantized(%ctx)
        ins(%q, %idx, %scales : memref<4x4xui8>, memref<2xi64>, memref<4x1xf16>)
        outs(%out : memref<2x8xf16>)
        {bits = 4 : i64, block_size = 8 : i64, gather_axis = 0 : i64,
         quantize_axis = 1 : i64}
    return %d : memref<4x8xf16>
  }
}

//--- other-use.mlir
module {
  memref.global "private" @ple : memref<4x8xf16> {
    alignment = 64 : i64,
    hip.external_data = {disk_resident, index = 0 : i64, offset = 0 : i64, size = 64 : i64}
  }
  func.func @other_use(%ctx: !hip.context) -> memref<4x8xf16> {
    %t = memref.get_global @ple : memref<4x8xf16>
    return %t : memref<4x8xf16>
  }
}
