// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Test: memref.copy lowering in --convert-hip-to-llvm.
//
// Covers three regimes:
//
//   1. Static dense row-major  -> single wrap_hipMemcpyAsync (total bytes)
//   2. Static strided suffix   -> wrap_hipMemcpy2DAsync (width / pitch / height)
//   3. Dynamic strides / shape -> runtime-computed wrap_hipMemcpy2DAsync,
//      or wrap_hipMemcpyAsync for rank-1
//
// Each function takes an !hip.context first arg so that the converted
// llvm.func has the state pointer in argument 0 (which
// MemRefCopyOpLowering reads as the runtime state pointer).  Memref
// arguments flatten into multiple LLVM args (the descriptor struct), so
// the FileCheck patterns avoid pinning the full signature.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  //===--------------------------------------------------------------------===//
  // 1. Static dense row-major: src and dst both fully-static identity.
  //    Expect a single wrap_hipMemcpyAsync of the total byte count
  //    (4 * 8 * 2 = 64 bytes).
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: llvm.func @copy_static_dense
  func.func @copy_static_dense(%ctx: !hip.context,
                                %src: memref<4x8xf16>,
                                %dst: memref<4x8xf16>) {
    // CHECK:   %[[N:.+]] = llvm.mlir.constant(64 : i64)
    // CHECK:   llvm.call @wrap_hipMemcpyAsync({{.*}}, %[[N]])
    memref.copy %src, %dst : memref<4x8xf16> to memref<4x8xf16>
    // CHECK:   llvm.return
    return
  }

  //===--------------------------------------------------------------------===//
  // 2. Static strided suffix (no holes inside the last dim): exercises the
  //    fully-static pitched path. dst is identity, src is strided<[8, 1]>
  //    over a wider parent — pitch differs.
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: llvm.func @copy_static_strided_2d
  func.func @copy_static_strided_2d(%ctx: !hip.context,
                                     %src: memref<2x4xf16,
                                                   strided<[8, 1], offset: 0>>,
                                     %dst: memref<2x4xf16>) {
    // CHECK:   llvm.call @wrap_hipMemcpy2DAsync(
    memref.copy %src, %dst
      : memref<2x4xf16, strided<[8, 1], offset: 0>> to memref<2x4xf16>
    // CHECK:   llvm.return
    return
  }

  //===--------------------------------------------------------------------===//
  // 3a. Dynamic-shape identity rank-1: rank-1 falls to a single 1D D2D copy.
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: llvm.func @copy_dynamic_rank1
  func.func @copy_dynamic_rank1(%ctx: !hip.context,
                                 %src: memref<?xf16>,
                                 %dst: memref<?xf16>) {
    // CHECK:   %[[ELEM:.+]] = llvm.mlir.constant(2 : i64)
    // CHECK:   %[[N:.+]] = llvm.mul %{{.+}}, %[[ELEM]]
    // CHECK:   llvm.call @wrap_hipMemcpyAsync({{.*}}, %[[N]])
    memref.copy %src, %dst : memref<?xf16> to memref<?xf16>
    // CHECK:   llvm.return
    return
  }

  //===--------------------------------------------------------------------===//
  // 3b. Dynamic-shape identity rank-2: width = inner-dim * elem_bytes,
  //     height = outer dim, src pitch from descriptor.
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: llvm.func @copy_dynamic_identity_2d
  func.func @copy_dynamic_identity_2d(%ctx: !hip.context,
                                       %src: memref<?x?xf16>,
                                       %dst: memref<?x?xf16>) {
    // CHECK:   llvm.call @wrap_hipMemcpy2DAsync(
    memref.copy %src, %dst : memref<?x?xf16> to memref<?x?xf16>
    // CHECK:   llvm.return
    return
  }

  //===--------------------------------------------------------------------===//
  // 3c. Canonical promote-strided pattern: rank-4 source with dynamic outer
  //     strides and static inner strides, identity dst alloc. This is the
  //     shape produced for `onnx.Split(axis=-1)` / `onnx.Slice` in
  //     dynamic-shape models like Qwen3.5-text.
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: llvm.func @copy_dynamic_strided_split_4d
  func.func @copy_dynamic_strided_split_4d(
      %ctx: !hip.context,
      %src: memref<?x?x16x256xf16,
                    strided<[?, ?, 256, 1], offset: ?>>,
      %dst: memref<?x?x16x256xf16>) {
    // CHECK:   llvm.call @wrap_hipMemcpy2DAsync(
    memref.copy %src, %dst
      : memref<?x?x16x256xf16, strided<[?, ?, 256, 1], offset: ?>>
        to memref<?x?x16x256xf16>
    // CHECK:   llvm.return
    return
  }

  //===--------------------------------------------------------------------===//
  // 4. Negative: source's last dim is NOT stride 1 (e.g. transposed view).
  //    We bail out — the upstream MemRefToLLVM fallback then takes over.
  //    Verify we do NOT emit a wrap_hipMemcpy* call for this pattern; an
  //    upstream `memrefCopy` libcall is acceptable (it would surface as a
  //    link error in production, which is the intended signal to add a
  //    new fusion / lowering).
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: llvm.func @copy_dynamic_nonunit_inner_stride
  func.func @copy_dynamic_nonunit_inner_stride(
      %ctx: !hip.context,
      %src: memref<?x?xf16, strided<[1, ?], offset: 0>>,
      %dst: memref<?x?xf16, strided<[1, ?], offset: 0>>) {
    // CHECK-NOT: llvm.call @wrap_hipMemcpyAsync(
    // CHECK-NOT: llvm.call @wrap_hipMemcpy2DAsync(
    memref.copy %src, %dst
      : memref<?x?xf16, strided<[1, ?], offset: 0>>
        to memref<?x?xf16, strided<[1, ?], offset: 0>>
    // CHECK:   llvm.return
    return
  }
}
