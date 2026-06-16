// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify `memref.copy` is lowered to one of our runtime helpers
// (`wrap_hipMemcpyAsync` for fully-dense d2d, `wrap_hipMemcpy2DAsync` for
// pitched 2D copies with wide rows, `wrap_strided_copy` for degenerate
// pitched 2D copies with thin rows over a large height) and never falls
// through to MLIR's default `MemRef -> LLVM` lowering, which would emit an
// external `memrefCopy` C-runner-utils call that we do not link.
//
// The pitched 2D path covers the canonical `tensor.insert_slice` ->
// `memref.subview + memref.copy` shape produced by ONNX Concat / Split
// bufferization on a non-zero axis: the destination is a subview into a
// parent buffer (outer stride keeps parent's pitch, inner suffix dense),
// the source is a dense input. Either side may be a strided subview as
// long as both share a common contiguous suffix.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm --split-input-file %s | FileCheck %s

// -----

// Test 1: fully dense d2d copy of a rank-2 memref. Both src and dst are
// identity-layout dense memrefs -> single `wrap_hipMemcpyAsync` call.
//
// CHECK-LABEL: llvm.func @copy_dense_2d
// CHECK:         llvm.call @wrap_hipMemcpyAsync({{.*}}) :
// CHECK-NOT:     llvm.call @memrefCopy
// CHECK-NOT:     llvm.call @wrap_hipMemcpy2DAsync
func.func @copy_dense_2d(%ctx: !hip.context,
                         %src: memref<4x8xf16>,
                         %dst: memref<4x8xf16>) {
  memref.copy %src, %dst : memref<4x8xf16> to memref<4x8xf16>
  return
}

// -----

// Test 2: pitched 2D copy from a strided subview source into a dense
// destination (the canonical Split / promote-strided-operands pattern).
// `src` has strides [262144, 2048, 1] (the slice's stride[1] = 2048 is
// the parent's pitch, not the slice's dense pitch 1024); `dst` is dense
// row-major. Expect `wrap_hipMemcpy2DAsync`, NOT a plain memcpy or the
// memrefCopy fallback.
//
// CHECK-LABEL: llvm.func @copy_strided_src_dense_dst
// CHECK:         llvm.call @wrap_hipMemcpy2DAsync({{.*}}) :
// CHECK-NOT:     llvm.call @memrefCopy
func.func @copy_strided_src_dense_dst(
    %ctx: !hip.context,
    %src: memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>,
    %dst: memref<1x128x1024xf16>) {
  memref.copy %src, %dst
      : memref<1x128x1024xf16, strided<[262144, 2048, 1], offset: 1024>>
        to memref<1x128x1024xf16>
  return
}

// -----

// Test 3: pitched 2D copy from a dense source into a strided subview
// destination -- the canonical onnx.Concat axis>0 pattern. Bufferization
// of `tensor.insert_slice` along axis=1 produces a `memref.subview` with
// strides inherited from the parent (here parent shape [1,256,64,64]
// concatenated of two halves; the slice covers half along axis=1, so
// dst.strides[0] = 1048576 keeps the parent pitch). Source is the full
// dense input. Pre-fix this fell through to MLIR's default lowering,
// emitting an undefined `memrefCopy` symbol at link time.
//
// CHECK-LABEL: llvm.func @copy_dense_src_strided_dst_concat
// CHECK:         llvm.call @wrap_hipMemcpy2DAsync({{.*}}) :
// CHECK-NOT:     llvm.call @memrefCopy
func.func @copy_dense_src_strided_dst_concat(
    %ctx: !hip.context,
    %src: memref<1x128x64x64xf16>,
    %dst: memref<1x128x64x64xf16, strided<[1048576, 4096, 64, 1], offset: 0>>) {
  memref.copy %src, %dst
      : memref<1x128x64x64xf16>
        to memref<1x128x64x64xf16, strided<[1048576, 4096, 64, 1], offset: 0>>
  return
}

// -----

// Test 4: pitched 2D copy from a dense source into a strided subview
// destination at a NON-ZERO offset (second slice of a 2-input Concat).
// `extractMemRefDataPtr` must factor the descriptor offset (524288) into
// the destination pointer so the copy lands at the second half of the
// parent buffer.
//
// CHECK-LABEL: llvm.func @copy_dense_src_strided_dst_concat_offset
// CHECK:         llvm.call @wrap_hipMemcpy2DAsync({{.*}}) :
// CHECK-NOT:     llvm.call @memrefCopy
func.func @copy_dense_src_strided_dst_concat_offset(
    %ctx: !hip.context,
    %src: memref<1x128x64x64xf16>,
    %dst: memref<1x128x64x64xf16, strided<[1048576, 4096, 64, 1], offset: 524288>>) {
  memref.copy %src, %dst
      : memref<1x128x64x64xf16>
        to memref<1x128x64x64xf16, strided<[1048576, 4096, 64, 1], offset: 524288>>
  return
}

// -----

// Test 5: rank-1 dense copy is still handled by the plain memcpy path
// (no pitched 2D fallback needed).
//
// CHECK-LABEL: llvm.func @copy_dense_1d
// CHECK:         llvm.call @wrap_hipMemcpyAsync({{.*}}) :
// CHECK-NOT:     llvm.call @memrefCopy
// CHECK-NOT:     llvm.call @wrap_hipMemcpy2DAsync
func.func @copy_dense_1d(%ctx: !hip.context,
                         %src: memref<256xf32>,
                         %dst: memref<256xf32>) {
  memref.copy %src, %dst : memref<256xf32> to memref<256xf32>
  return
}

// -----

// Test 6: DEGENERATE pitched 2D -- a dense source scattered into a strided
// destination with a 1-element contiguous row over a very large height. This
// is the sinusoidal position-embedding interleave: Concat(unsqueeze(sin),
// unsqueeze(cos), axis=last) bufferizes to a copy of `memref<...x64x1>` into
// one half (stride 2) of the `...x64x2` parent. splitDim lands on the last
// dim -> widthElems=1 (widthBytes=2), height=40000, dstPitch=2 elems. Because
// the row is thin (<= 256B) and the height is large (>= 256), this must lower
// to the parallel `wrap_strided_copy` kernel, NOT the per-row
// `wrap_hipMemcpy2DAsync` (which serializes into 40000 micro-transfers).
//
// CHECK-LABEL: llvm.func @copy_degenerate_thin_interleave
// CHECK:         llvm.call @wrap_strided_copy({{.*}}) :
// CHECK-NOT:     llvm.call @wrap_hipMemcpy2DAsync
// CHECK-NOT:     llvm.call @memrefCopy
func.func @copy_degenerate_thin_interleave(
    %ctx: !hip.context,
    %src: memref<1x25x25x64x1xf16>,
    %dst: memref<1x25x25x64x1xf16, strided<[80000, 3200, 128, 2, 1], offset: 0>>) {
  memref.copy %src, %dst
      : memref<1x25x25x64x1xf16>
        to memref<1x25x25x64x1xf16, strided<[80000, 3200, 128, 2, 1], offset: 0>>
  return
}
