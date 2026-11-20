// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-materialize-host-scalars \
// RUN:   --expand-strided-metadata --lower-affine --convert-hip-to-llvm | FileCheck %s

// The readback slot of a padded element is aligned to a power of two, not to
// its 6-byte payload.
// CHECK-LABEL: llvm.func @padded_load_after_copy(
// CHECK: llvm.alloca %{{.*}} x i48 {alignment = 8 : i64}
// CHECK: llvm.call @hipdnn_ep_readback_scalar
// CHECK: llvm.load %{{.*}} : !llvm.ptr -> i48
func.func @padded_load_after_copy(%ctx: !hip.context, %src: memref<1xi48>)
    -> i48 {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<1xi48>
  memref.copy %src, %a : memref<1xi48> to memref<1xi48>
  %v = memref.load %a[%c0] : memref<1xi48>
  return %v : i48
}

// hip.stream_wait waits for the session stream.
// CHECK-LABEL: llvm.func @stream_wait(
// CHECK: %[[STREAM:.*]] = llvm.call @hipdnn_ep_state_get_stream(
// CHECK: llvm.call @wrap_hipStreamSynchronize(%[[STREAM]])
func.func @stream_wait(%ctx: !hip.context) {
  hip.stream_wait(%ctx)
  return
}
