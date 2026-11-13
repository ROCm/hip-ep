// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-llvm %s | FileCheck %s
// RUN: hip-mlir-opt --convert-hip-to-llvm='report-host-scalars=true' %s -o %t 2>&1 | FileCheck %s --check-prefix=REPORT

// hip.geglu accesses only its device buffers in stream order, so a
// host-written length stays forwarded across it.

// REPORT: remark: host scalars forwarded: 1

// CHECK-LABEL: llvm.func @geglu_keeps_forwarding(
// CHECK: llvm.store %arg1,
// CHECK: llvm.call @wrap_geglu
// CHECK-NOT: llvm.call @hipdnn_ep_readback
// CHECK: llvm.return
func.func @geglu_keeps_forwarding(%ctx: !hip.context, %n: i32,
                                  %gate: memref<8x16xf16, 1>,
                                  %up: memref<8x16xf16, 1>,
                                  %output: memref<8x16xf16, 1>) -> i32 {
  %c0 = arith.constant 0 : index
  %size = arith.constant 64 : index
  %scratch = hip.get_host_scratch(%ctx, %size) : memref<?xi8>
  %length = memref.view %scratch[%c0][] : memref<?xi8> to memref<1xi32>
  memref.store %n, %length[%c0] : memref<1xi32>
  hip.geglu(%ctx) ins(%gate, %up : memref<8x16xf16, 1>, memref<8x16xf16, 1>)
                  outs(%output : memref<8x16xf16, 1>) {approximate = "tanh"}
  %r = hip.readback_scalar(%ctx, %length : memref<1xi32>) -> i32
  return %r : i32
}
