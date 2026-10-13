// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt --hip-resolve-extern-constants %t/gather.mlir | FileCheck %s
// RUN: not hip-mlir-opt --hip-resolve-extern-constants %t/other-use.mlir 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-LABEL: func.func @gather
// CHECK:       %[[T:.*]] = hip.get_constant(%{{.*}}, %{{.*}}) {hip.disk_resident} : memref<4x8xf16>
// CHECK:       hip.gather(%{{.*}}) ins(%[[T]], %{{.*}} : memref<4x8xf16>, memref<2xi64>)
// CHECK:       hip.get_constant(%{{.*}}, %{{.*}}) : memref<4x8xf16>

// ERR: error: disk-resident constant has no GPU copy; only hip.gather data may use it

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
  func.func @gather(%ctx: !hip.context, %idx: memref<2xi64>,
                    %out: memref<2x8xf16>) -> memref<4x8xf16> {
    %t = memref.get_global @ple : memref<4x8xf16>
    hip.gather(%ctx) ins(%t, %idx : memref<4x8xf16>, memref<2xi64>)
        outs(%out : memref<2x8xf16>) {axis = 0 : i64}
    %d = memref.get_global @dense : memref<4x8xf16>
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
