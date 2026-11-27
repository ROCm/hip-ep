// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

// A view reads a buffer through different extents or strides. How many strides
// it has and whether they match its shape is the encoding's business; what is
// left for the operation is that reading bytes differently does not reinterpret
// what they hold.

func.func @view_element_type(%arg0: tensor<1x1x2x3xf32, #ggml.layout<
    ne = [3, 2, 1, 1], nb = [4, 12, 24, 24], offset = 0>>) {
  // expected-error @below {{requires input and result tensor element types to match}}
  %0 = "ggml.view"(%arg0) : (tensor<1x1x2x3xf32, #ggml.layout<
      ne = [3, 2, 1, 1], nb = [4, 12, 24, 24], offset = 0>>)
    -> tensor<1x1x2x3xf16, #ggml.layout<
      ne = [3, 2, 1, 1], nb = [2, 6, 12, 12], offset = 0>>
  return
}

// -----

func.func @permute_axes(%arg0: tensor<1x2x3x4xf32>) {
  // expected-error @below {{requires `axes` to be a permutation of [0, 4); duplicate axis 2}}
  %0 = "ggml.permute"(%arg0) <{axes = [0, 2, 2, 3]}>
    : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  return
}

// -----

func.func @permute_result_shape(%arg0: tensor<1x2x3x4xf32>) {
  // expected-error @below {{requires result shape to match the requested permutation}}
  %0 = "ggml.permute"(%arg0) <{axes = [0, 2, 1, 3]}>
    : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  return
}

// -----

func.func @permute_valid(%arg0: tensor<1x2x3x4xf32>)
    -> tensor<1x3x2x4xf32> {
  %0 = "ggml.permute"(%arg0) <{axes = [0, 2, 1, 3]}>
    : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
  return %0 : tensor<1x3x2x4xf32>
}

// -----

func.func @set_rows_result(
    %updates: tensor<1x1x2x4xf32>, %indices: tensor<1x1x1x2xi64>,
    %destination: tensor<1x1x8x4xf16>) {
  // expected-error @below {{requires the result shape and element type to equal the destination}}
  %0 = "ggml.set_rows"(%updates, %indices, %destination)
    : (tensor<1x1x2x4xf32>, tensor<1x1x1x2xi64>, tensor<1x1x8x4xf16>)
      -> tensor<1x1x8x4xf32>
  return
}

// -----

func.func @glu_split_without_up(%gate: tensor<1x1x2x8xf32>) {
  // expected-error @below {{requires `split` to match the presence of the `up` operand}}
  %0 = "ggml.glu"(%gate) <{
    alpha = 0.000000e+00 : f32, glu_kind = #ggml.glu_kind<swiglu>,
    limit = 0.000000e+00 : f32, split = true, swapped = false
  }> : (tensor<1x1x2x8xf32>) -> tensor<1x1x2x8xf32>
  return
}

// -----

func.func @rope_dimensions(
    %input: tensor<1x2x4x8xf32>, %positions: tensor<1x1x1x2xi32>) {
  // expected-error @below {{requires positive, even `dimensions`}}
  %0 = "ggml.rope"(%input, %positions) <{
    attention_factor = 1.000000e+00 : f32,
    beta_fast = 3.200000e+01 : f32,
    beta_slow = 1.000000e+00 : f32,
    dimensions = 7 : i64,
    extension_factor = 0.000000e+00 : f32,
    frequency_base = 1.000000e+04 : f32,
    frequency_scale = 1.000000e+00 : f32,
    legacy_context = 0 : i64,
    mode = #ggml.rope_mode<normal>,
    n_offs = 0 : i64,
    n_past = 0 : i64,
    original_context = 4096 : i64,
    sections = [0, 0, 0, 0]
  }> : (tensor<1x2x4x8xf32>, tensor<1x1x1x2xi32>)
    -> tensor<1x2x4x8xf32>
  return
}

// -----

func.func @attention_bias_without_mask(
    %query: tensor<1x2x4x8xf32>, %key: tensor<1x2x4x8xf32>,
    %value: tensor<1x2x4x8xf32>) {
  // expected-error @below {{positive `max_bias` requires a mask operand}}
  %0 = "ggml.flash_attention_ext"(%query, %key, %value) <{
    logit_softcap = 0.000000e+00 : f32,
    max_bias = 1.000000e+00 : f32,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>,
    precision = #ggml.compute_precision<f32>,
    scale = 1.000000e+00 : f32
  }> : (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>,
        tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf32>
  return
}

// -----

func.func @get_rows_indices(
    %data: tensor<1x1x8x4xf32>, %indices: tensor<1x1x1x2xf32>) {
  // expected-error @below {{requires i32 indices}}
  %0 = "ggml.get_rows"(%data, %indices)
    : (tensor<1x1x8x4xf32>, tensor<1x1x1x2xf32>)
      -> tensor<1x1x2x4xf32>
  return
}

// -----

func.func @rms_norm_type(%input: tensor<1x1x2x4xf16>) {
  // expected-error @below {{requires f32 input and result tensors}}
  %0 = "ggml.rms_norm"(%input) <{epsilon = 1.000000e-5 : f32}>
    : (tensor<1x1x2x4xf16>) -> tensor<1x1x2x4xf16>
  return
}

// -----

func.func @mul_mat_hint(
    %weight: tensor<1x1x3x4xf32>, %input: tensor<1x1x2x4xf32>) {
  // expected-error @below {{requires `hint` to be 0 or 1}}
  %0 = "ggml.mul_mat"(%weight, %input) <{
    hint = 99 : i64, precision = #ggml.compute_precision<default>
  }> : (tensor<1x1x3x4xf32>, tensor<1x1x2x4xf32>)
    -> tensor<1x1x2x3xf32>
  return
}

// -----

func.func @binary_result_type(
    %lhs: tensor<1x1x2x4xf32>, %rhs: tensor<1x1x1x4xf32>) {
  // expected-error @below {{requires the result shape and element type to match the lhs}}
  %0 = "ggml.add"(%lhs, %rhs)
    : (tensor<1x1x2x4xf32>, tensor<1x1x1x4xf32>)
      -> tensor<1x1x2x4xf64>
  return
}

// -----

// The layout says the count even where the shape says only `?`, so a reshape
// that changes it is refused whether or not the extents are known.

func.func @reshape_cardinality(%input: tensor<1x1x?x4xf32, #ggml.layout<
    ne = [4, 1*token_count, 1, 1],
    nb = [4, 16, 16*token_count, 16*token_count], offset = 0>>) {
  // expected-error @below {{requires equal input and result element counts, but holds #ggml.affine<4*token_count> and #ggml.affine<12*token_count>}}
  %0 = "ggml.reshape"(%input) : (tensor<1x1x?x4xf32, #ggml.layout<
      ne = [4, 1*token_count, 1, 1],
      nb = [4, 16, 16*token_count, 16*token_count], offset = 0>>)
    -> tensor<1x1x?x12xf32, #ggml.layout<
      ne = [12, 1*token_count, 1, 1],
      nb = [4, 48, 48*token_count, 48*token_count], offset = 0>>
  return
}

// -----

// GGML lays a reshape out contiguously, so strides it did not choose are not a
// reshape's -- and unlike the extents, nothing else would notice.

func.func @reshape_dense(%input: tensor<1x1x2x4xf32, #ggml.layout<
    ne = [4, 2, 1, 1], nb = [4, 16, 32, 32], offset = 0>>) {
  // expected-error @below {{requires a dense result}}
  %0 = "ggml.reshape"(%input) : (tensor<1x1x2x4xf32, #ggml.layout<
      ne = [4, 2, 1, 1], nb = [4, 16, 32, 32], offset = 0>>)
    -> tensor<1x1x4x2xf32, #ggml.layout<
      ne = [2, 4, 1, 1], nb = [4, 16, 64, 64], offset = 0>>
  return
}

// -----

// A count that does not fit in an int64 is a graph no target can run, and it is
// the extents that overflow -- the strides below are beside the point.

func.func @reshape_cardinality_overflow(
    %input: tensor<1x1x4294967296x4294967296xf32, #ggml.layout<
      ne = [4294967296, 4294967296, 1, 1], nb = [4, 4, 4, 4], offset = 0>>) {
  // expected-error @below {{has an element count that extent product overflows int64}}
  %0 = "ggml.reshape"(%input)
    : (tensor<1x1x4294967296x4294967296xf32, #ggml.layout<
        ne = [4294967296, 4294967296, 1, 1], nb = [4, 4, 4, 4], offset = 0>>)
      -> tensor<1x1x4294967296x4294967296xf32, #ggml.layout<
        ne = [4294967296, 4294967296, 1, 1], nb = [4, 4, 4, 4], offset = 0>>
  return
}
