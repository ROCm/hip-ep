// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// When a graph output is returned through memref.collapse_shape (the internal
// compute rank is higher than the ONNX / func.return rank), the EP output
// allocator callback (hipdnn_ep_alloc_output) MUST receive the RETURNED rank
// and dims -- not the root alloc rank -- or ORT rejects the pre-bound output
// OrtValue with a shape mismatch ("has shape {252,2560} but the computed
// output shape ... is {1,252,2560}").
//
// The rank-reducing collapse is captured by --hip-use-output-allocator as the
// `hipdnn.abi_shape` / `hipdnn.abi_groups` attrs WHILE collapse_shape is still
// intact, then consumed by --convert-hip-to-llvm. The critical property is that
// this works even after --expand-strided-metadata has decomposed collapse_shape
// into reinterpret_cast + extract_strided_metadata (the real pipeline order):
// the attrs live on the hip.alloc_output op, which that decomposition does not
// touch. (A previous attempt re-derived the shape by walking the view chain in
// the lowering; it silently no-op'd here because the chain no longer exists by
// then -- this test locks in the ordering that regression missed.)

// Producer: the ABI attrs are stamped from the collapse reassociation.
// RUN: hip-mlir-opt %s --hip-use-output-allocator | FileCheck %s --check-prefix=STAMP

// Consumer, REAL pipeline order (decompose collapse_shape, then lower): the
// callback uses the returned (rank-2) shape, proving the attrs survive
// expand-strided-metadata.
// RUN: hip-mlir-opt %s --hip-use-output-allocator --expand-strided-metadata --convert-hip-to-llvm | FileCheck %s --check-prefix=DECOMP

// Vision-encoder-class case: internal rank-3 (both leading dims dynamic),
// ONNX/OGA output rank-2. abi_groups=[2,1] -> external dim0 folds internal
// dims 0,1 (runtime product); external dim1 = internal dim2 (static 2560).
// STAMP-LABEL: func.func @vision_internal3_return2
// STAMP:       hip.alloc_output
// STAMP-SAME:    hipdnn.abi_groups = array<i64: 2, 1>
// STAMP-SAME:    hipdnn.abi_shape = array<i64: -9223372036854775808, 2560>
//
// DECOMP-LABEL: llvm.func @vision_internal3_return2
// The callback shape array is rank 2, not the internal rank 3 ...
// DECOMP:       llvm.alloca %{{.*}} x !llvm.array<2 x i64>
// ... and the rank argument passed to the runtime is 2.
// DECOMP:       %[[RANK:.*]] = llvm.mlir.constant(2 : i64) : i64
// DECOMP:       llvm.call @hipdnn_ep_alloc_output(%{{.*}}, %{{.*}}, %{{.*}}, %[[RANK]], %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
func.func @vision_internal3_return2(%ctx: !hip.context, %d0: index, %d1: index)
    -> memref<?x2560xf16> {
  %out = memref.alloc(%d0, %d1) : memref<?x?x2560xf16>
  %ret = memref.collapse_shape %out [[0, 1], [2]]
      : memref<?x?x2560xf16> into memref<?x2560xf16>
  return %ret : memref<?x2560xf16>
}

// Conv-style flatten: internal rank 4, returned rank 2, all static.
// abi_groups=[1,3] -> external dim0 = internal dim0 (1); external dim1 folds
// internal dims 1,2,3 (64*56*56 = 200704).
// STAMP-LABEL: func.func @collapse_static
// STAMP:       hip.alloc_output
// STAMP-SAME:    hipdnn.abi_groups = array<i64: 1, 3>
// STAMP-SAME:    hipdnn.abi_shape = array<i64: 1, 200704>
//
// DECOMP-LABEL: llvm.func @collapse_static
// The callback shape array is rank 2, not the internal rank 4, and the rank
// argument to the runtime is 2 (proving the collapse rank-reduction is honored).
// DECOMP:       llvm.alloca %{{.*}} x !llvm.array<2 x i64>
// DECOMP:       %[[RANK2:.*]] = llvm.mlir.constant(2 : i64) : i64
// DECOMP:       llvm.call @hipdnn_ep_alloc_output(%{{.*}}, %{{.*}}, %{{.*}}, %[[RANK2]], %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
func.func @collapse_static(%ctx: !hip.context) -> memref<1x200704xf32> {
  %out = memref.alloc() : memref<1x64x56x56xf32>
  %ret = memref.collapse_shape %out [[0], [1, 2, 3]]
      : memref<1x64x56x56xf32> into memref<1x200704xf32>
  return %ret : memref<1x200704xf32>
}

// DETR logits-style expand: internal rank 2, ONNX return rank 3.
// abi_groups=[2,1] -> ext dim0 static 1, ext dim1 = internal dim0, ext dim2 = 92.
// STAMP-LABEL: func.func @expand_logits
// STAMP:       hip.alloc_output
// STAMP-SAME:    hipdnn.abi_groups = array<i64: 2, 1>
// STAMP-SAME:    hipdnn.abi_shape = array<i64: 1, -9223372036854775808, 92>
//
// DECOMP-LABEL: llvm.func @expand_logits
// DECOMP:       llvm.alloca %{{.*}} x !llvm.array<3 x i64>
// DECOMP:       %[[RANK3:.*]] = llvm.mlir.constant(3 : i64) : i64
// DECOMP:       llvm.call @hipdnn_ep_alloc_output(%{{.*}}, %{{.*}}, %{{.*}}, %[[RANK3]], %{{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
func.func @expand_logits(%ctx: !hip.context, %n: index) -> memref<1x?x92xf16> {
  %out = memref.alloc(%n) : memref<?x92xf16>
  %ret = memref.expand_shape %out [[0, 1], [2]] output_shape [1, %n, 92]
      : memref<?x92xf16> into memref<1x?x92xf16>
  return %ret : memref<1x?x92xf16>
}
