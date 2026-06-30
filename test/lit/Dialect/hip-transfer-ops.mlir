// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Parse/verify round-trip for the explicit transfer op family:
//   hip.transfer (tensor-phase, value-preserving)
//   hip.memcpy_h2d_async / hip.memcpy_d2h_async (memref-phase async copy)
//   hip.stream_sync (host-side barrier)
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s 2>&1 | FileCheck %s

// --- hip.transfer: tensor -> tensor, target space in the `to` clause. ---
// CHECK-LABEL: func.func @transfer_to_host
// CHECK: %[[H:.*]] = hip.transfer(%{{.*}}, %{{.*}} : tensor<8xi64>) to <host> -> tensor<8xi64>
func.func @transfer_to_host(%ctx: !hip.context, %pads: tensor<8xi64>) -> tensor<8xi64> {
  %h = hip.transfer(%ctx, %pads : tensor<8xi64>) to #hip.mem<host> -> tensor<8xi64>
  return %h : tensor<8xi64>
}

// --- hip.memcpy_d2h_async: device src -> host dst (the primary direction). ---
// CHECK-LABEL: func.func @memcpy_d2h
// CHECK: hip.memcpy_d2h_async(%{{.*}}, %{{.*}}, %{{.*}} : memref<8xi64, #hip.mem<host>>, memref<8xi64, #hip.mem<device>>)
// CHECK: hip.stream_sync(%{{.*}})
func.func @memcpy_d2h(%ctx: !hip.context,
                      %dst: memref<8xi64, #hip.mem<host>>,
                      %src: memref<8xi64, #hip.mem<device>>) {
  hip.memcpy_d2h_async(%ctx, %dst, %src
      : memref<8xi64, #hip.mem<host>>, memref<8xi64, #hip.mem<device>>)
  hip.stream_sync(%ctx)
  return
}

// --- hip.memcpy_h2d_async: host src -> device dst (wired for symmetry). ---
// CHECK-LABEL: func.func @memcpy_h2d
// CHECK: hip.memcpy_h2d_async(%{{.*}}, %{{.*}}, %{{.*}} : memref<4xi32, #hip.mem<device>>, memref<4xi32, #hip.mem<host>>)
func.func @memcpy_h2d(%ctx: !hip.context,
                      %dst: memref<4xi32, #hip.mem<device>>,
                      %src: memref<4xi32, #hip.mem<host>>) {
  hip.memcpy_h2d_async(%ctx, %dst, %src
      : memref<4xi32, #hip.mem<device>>, memref<4xi32, #hip.mem<host>>)
  return
}

// --- A pinned src is also accepted on the host side of H2D/D2H. ---
// CHECK-LABEL: func.func @memcpy_d2h_pinned
// CHECK: hip.memcpy_d2h_async(%{{.*}}, %{{.*}}, %{{.*}} : memref<2xi64, #hip.mem<pinned>>, memref<2xi64, #hip.mem<device>>)
func.func @memcpy_d2h_pinned(%ctx: !hip.context,
                             %dst: memref<2xi64, #hip.mem<pinned>>,
                             %src: memref<2xi64, #hip.mem<device>>) {
  hip.memcpy_d2h_async(%ctx, %dst, %src
      : memref<2xi64, #hip.mem<pinned>>, memref<2xi64, #hip.mem<device>>)
  return
}
