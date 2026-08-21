// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the two ONNX Slice lowering paths:
//
//   1. SliceDecompose (preferred) — all slice params are compile-time
//      constants with positive unit stride, so onnx.Slice is rewritten to
//      a zero-cost tensor.extract_slice.
//   2. SliceToHip (fallback) — non-constant indices or negative steps fall
//      through to a native hip.slice op whose runtime is a stub today.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: classic prefix slice (axis 0, [1:3], stride 1) — decomposes to
  // tensor.extract_slice.
  func.func @test_slice_decompose_simple(%input: tensor<4x6xf32>) -> tensor<2x6xf32> {
    // CHECK-LABEL: func.func @test_slice_decompose_simple
    %starts = arith.constant dense<[1]> : tensor<1xi64>
    %ends   = arith.constant dense<[3]> : tensor<1xi64>
    %axes   = arith.constant dense<[0]> : tensor<1xi64>
    %steps  = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<4x6xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<2x6xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK-NOT: hip.slice
    // CHECK: tensor.extract_slice {{.*}}[1, 0] [2, 6] [1, 1]

    return %r : tensor<2x6xf32>
  }

  // Test 2: per-axis slice with stride > 1 — still decomposes (extract_slice
  // supports strides).
  func.func @test_slice_decompose_stride(%input: tensor<2x4xf32>) -> tensor<1x2xf32> {
    // CHECK-LABEL: func.func @test_slice_decompose_stride
    %starts = arith.constant dense<[1, 0]> : tensor<2xi64>
    %ends   = arith.constant dense<[2, 3]> : tensor<2xi64>
    %axes   = arith.constant dense<[0, 1]> : tensor<2xi64>
    %steps  = arith.constant dense<[1, 2]> : tensor<2xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>,
           tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK: tensor.extract_slice {{.*}}[1, 0] [1, 2] [1, 2]

    return %r : tensor<1x2xf32>
  }

  // Test 3: omitted axes / steps — default to all axes, unit stride.
  func.func @test_slice_decompose_default_axes(%input: tensor<4x6xf32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @test_slice_decompose_default_axes
    %starts = arith.constant dense<[0, 0]> : tensor<2xi64>
    %ends   = arith.constant dense<[2, 3]> : tensor<2xi64>
    %r = "onnx.Slice"(%input, %starts, %ends)
        : (tensor<4x6xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x3xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK: tensor.extract_slice {{.*}}[0, 0] [2, 3] [1, 1]

    return %r : tensor<2x3xf32>
  }

  // Test 4: negative step forces the native fallback (hip.slice).  The
  // runtime is a stub today, but the conversion + bufferization pipeline
  // must still produce valid IR.
  func.func @test_slice_native_negative_step(%input: tensor<6xf32>) -> tensor<3xf32> {
    // CHECK-LABEL: func.func @test_slice_native_negative_step
    %starts = arith.constant dense<[5]> : tensor<1xi64>
    %ends   = arith.constant dense<[-1]> : tensor<1xi64>
    %axes   = arith.constant dense<[0]> : tensor<1xi64>
    %steps  = arith.constant dense<[-2]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<6xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK: tensor.empty() : tensor<3xf32>
    // CHECK: hip.slice({{.*}}) ins({{.*}}, {{.*}}, {{.*}} : tensor<6xf32>, tensor<1xi64>, tensor<1xi64>)

    return %r : tensor<3xf32>
  }

  // Test 5: non-constant starts (block argument) also falls back to the
  // native op — the decompose pattern needs to read the values.
  func.func @test_slice_native_dynamic_starts(
      %input: tensor<8xf32>, %starts: tensor<1xi64>) -> tensor<4xf32> {
    // CHECK-LABEL: func.func @test_slice_native_dynamic_starts
    %ends   = arith.constant dense<[7]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends)
        : (tensor<8xf32>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK: hip.slice({{.*}}) ins(
    return %r : tensor<4xf32>
  }

  // Test 6: SliceDecompose on a tensor with a dynamic non-sliced axis.
  // axis 0 is sliced (input dim is static = 4), axis 1 is left alone
  // (input dim is ? -> the corresponding extract_slice size is a
  // tensor.dim Value, not a constant).
  func.func @test_slice_decompose_dyn_untouched(%input: tensor<4x?xf32>) -> tensor<2x?xf32> {
    // CHECK-LABEL: func.func @test_slice_decompose_dyn_untouched
    %starts = arith.constant dense<[1]> : tensor<1xi64>
    %ends   = arith.constant dense<[3]> : tensor<1xi64>
    %axes   = arith.constant dense<[0]> : tensor<1xi64>
    %steps  = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<4x?xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<2x?xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK-NOT: hip.slice
    // CHECK-DAG: %[[A1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[A1]] : tensor<4x?xf32>
    // The first dim's slice is [start=1, size=2, step=1]; the second
    // dim is untouched and uses the runtime dim value.
    // CHECK: tensor.extract_slice %{{.*}}[1, 0] [2, %[[DIM]]] [1, 1]
    return %r : tensor<2x?xf32>
  }

  // Test 7: SliceDecompose bails when a sliced axis has a dynamic
  // input dim (ONNX clamping rules need the static dim size); falls
  // through to hip.slice. The output extent is still the slice length --
  // clamp(end) - clamp(start) -- not the data dim, which is only an upper
  // bound and would be handed to every consumer as the shape.
  func.func @test_slice_native_dyn_axis(%input: tensor<?xf32>) -> tensor<?xf32> {
    // CHECK-LABEL: func.func @test_slice_native_dyn_axis
    %starts = arith.constant dense<[1]> : tensor<1xi64>
    %ends   = arith.constant dense<[3]> : tensor<1xi64>
    %axes   = arith.constant dense<[0]> : tensor<1xi64>
    %steps  = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<?xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>

    // CHECK-NOT: tensor.extract_slice
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[DIM:.*]] = tensor.dim %{{.*}} : tensor<?xf32>
    // CHECK: %[[LO:.*]] = arith.minsi %{{.*}}, %[[DIM]] : index
    // CHECK: %[[HI:.*]] = arith.minsi %{{.*}}, %[[DIM]] : index
    // CHECK: %[[LEN:.*]] = arith.subi %[[HI]], %[[LO]] : index
    // CHECK: %[[EXT:.*]] = arith.maxsi %[[LEN]], %{{.*}} : index
    // CHECK: tensor.empty(%[[EXT]]) : tensor<?xf32>
    // CHECK: hip.slice({{.*}}) ins({{.*}}, {{.*}}, {{.*}} : tensor<?xf32>, tensor<1xi64>, tensor<1xi64>)
    return %r : tensor<?xf32>
  }

  // Test 8: the decode-mask idiom from Gemma-4 26B-A4B. `starts` is
  // Shape(attn)[1] - Shape(ids)[1] and `ends` is Shape(attn)[1], so the slice
  // keeps the current query positions only -- one row during decode. Both
  // bounds are host arithmetic over onnx.Shape, so the extent is computable
  // with no device readback, and the resulting empty must NOT be sized by the
  // data dim: that is what inflated the causal mask to [1, S, S] and cost 60%
  // of a decode step.
  func.func @test_slice_native_shape_sub_extent(
      %data: tensor<?x?xi64>, %ids: tensor<?x?xi64>, %attn: tensor<?x?xi64>)
      -> tensor<?x?xi64> {
    // CHECK-LABEL: func.func @test_slice_native_shape_sub_extent
    %axes = arith.constant dense<[1]> : tensor<1xi64>
    %ids_len  = "onnx.Shape"(%ids)  {start = 1 : si64, end = 2 : si64}
        : (tensor<?x?xi64>) -> tensor<1xi64>
    %attn_len = "onnx.Shape"(%attn) {start = 1 : si64, end = 2 : si64}
        : (tensor<?x?xi64>) -> tensor<1xi64>
    %starts = "onnx.Sub"(%attn_len, %ids_len)
        : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %r = "onnx.Slice"(%data, %starts, %attn_len, %axes)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<?x?xi64>

    // SliceToHip fires before the operands are rewritten, so `starts` resolves
    // through the ONNX spelling (onnx.Sub of two onnx.Shape) and the walker
    // emits the tensor.dim / arith.subi below itself. That is also the order
    // Gemma-4 takes in the real pipeline, so this is the production path; test
    // 10 covers the post-conversion spelling, which the same walk accepts
    // because the greedy driver does not guarantee either order.
    // CHECK: %[[D0:.*]] = tensor.dim %arg1, %{{.*}} : tensor<?x?xi64>
    // CHECK: %[[D1:.*]] = tensor.dim %arg1, %{{.*}} : tensor<?x?xi64>
    // CHECK: arith.subi %{{.*}}, %{{.*}} : index
    // CHECK: %[[LO:.*]] = arith.minsi %{{.*}}, %[[D1]] : index
    // CHECK: %[[HI:.*]] = arith.minsi %{{.*}}, %[[D1]] : index
    // CHECK: %[[LEN:.*]] = arith.subi %[[HI]], %[[LO]] : index
    // CHECK: %[[EXT:.*]] = arith.maxsi %[[LEN]], %{{.*}} : index
    // CHECK: tensor.empty(%[[D0]], %[[EXT]]) : tensor<?x?xi64>
    // CHECK: hip.slice
    return %r : tensor<?x?xi64>
  }

  // Test 9: an opaque `starts` (a graph input, not shape arithmetic) is not
  // host-resolvable, so the extent falls back to the data dim upper bound
  // rather than emitting an extent that cannot be justified.
  func.func @test_slice_native_opaque_starts(
      %data: tensor<?xi64>, %starts: tensor<1xi64>, %ends: tensor<1xi64>)
      -> tensor<?xi64> {
    // CHECK-LABEL: func.func @test_slice_native_opaque_starts
    %axes = arith.constant dense<[0]> : tensor<1xi64>
    %r = "onnx.Slice"(%data, %starts, %ends, %axes)
        : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<?xi64>

    // CHECK-NOT: arith.minsi
    // CHECK: %[[DIM:.*]] = tensor.dim %arg1, %{{.*}} : tensor<?xi64>
    // CHECK: tensor.empty(%[[DIM]]) : tensor<?xi64>
    // CHECK: hip.slice
    return %r : tensor<?xi64>
  }

  // Test 10: the same idiom already rewritten into the post-conversion
  // spelling, which is the other order the greedy driver may produce and which
  // test 8 therefore does not reach. Feeding it pre-lowered pins that branch of
  // the walk: `from_elements(index_cast(tensor.dim))` for the shapes and
  // hip.sub for the arithmetic, with the index_cast unwrapped rather than cast
  // a second time. Without a test here, an ordering change elsewhere would
  // silently drop back to the data-dim upper bound and nothing would fail.
  func.func @test_slice_native_lowered_shape_sub_extent(
      %ctx: !hip.context, %data: tensor<?x?xi64>, %ids: tensor<?x?xi64>,
      %attn: tensor<?x?xi64>) -> tensor<?x?xi64> {
    // CHECK-LABEL: func.func @test_slice_native_lowered_shape_sub_extent
    %c1 = arith.constant 1 : index
    %axes = arith.constant dense<[1]> : tensor<1xi64>

    %ids_dim = tensor.dim %ids, %c1 : tensor<?x?xi64>
    %ids_i64 = arith.index_cast %ids_dim : index to i64
    %ids_len = tensor.from_elements %ids_i64 : tensor<1xi64>

    %attn_dim = tensor.dim %attn, %c1 : tensor<?x?xi64>
    %attn_i64 = arith.index_cast %attn_dim : index to i64
    %attn_len = tensor.from_elements %attn_i64 : tensor<1xi64>

    %sub_init = tensor.empty() : tensor<1xi64>
    %starts = hip.sub(%ctx) ins(%attn_len, %ids_len : tensor<1xi64>,
        tensor<1xi64>) outs(%sub_init : tensor<1xi64>) : tensor<1xi64>

    %r = "onnx.Slice"(%data, %starts, %attn_len, %axes)
        : (tensor<?x?xi64>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>) -> tensor<?x?xi64>

    // The bounds resolve to the two dims the from_elements were packed from,
    // with the index_cast unwrapped, so `starts` is an index-domain subi of
    // them; the extent is then the clamped end minus the clamped start, not the
    // data dim.
    // CHECK: %[[D0:.*]] = tensor.dim %arg1, %{{.*}} : tensor<?x?xi64>
    // CHECK: %[[D1:.*]] = tensor.dim %arg1, %{{.*}} : tensor<?x?xi64>
    // CHECK: arith.subi %{{.*}}, %{{.*}} : index
    // CHECK: %[[LO:.*]] = arith.minsi %{{.*}}, %[[D1]] : index
    // CHECK: %[[HI:.*]] = arith.minsi %{{.*}}, %[[D1]] : index
    // CHECK: %[[LEN:.*]] = arith.subi %[[HI]], %[[LO]] : index
    // CHECK: %[[EXT:.*]] = arith.maxsi %[[LEN]], %{{.*}} : index
    // CHECK: tensor.empty(%[[D0]], %[[EXT]]) : tensor<?x?xi64>
    // CHECK: hip.slice
    return %r : tensor<?x?xi64>
  }
}
