// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @gather_valid(%ctx: !hip.context,
                        %data: memref<2x3x4xf32, 1>,
                        %indices: memref<5xi64, 1>,
                        %out: memref<2x5x4xf32, 1>) {
  hip.gather(%ctx)
    ins(%data, %indices : memref<2x3x4xf32, 1>, memref<5xi64, 1>)
    outs(%out : memref<2x5x4xf32, 1>)
    {axis = 1 : i64}
  return
}

// -----

func.func @gather_wrong_output(%ctx: !hip.context,
                               %data: memref<2x3x4xf32, 1>,
                               %indices: memref<5xi64, 1>,
                               %out: memref<2x6x4xf32, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 5}}
  hip.gather(%ctx)
    ins(%data, %indices : memref<2x3x4xf32, 1>, memref<5xi64, 1>)
    outs(%out : memref<2x6x4xf32, 1>)
    {axis = -2 : i64}
  return
}

// -----

func.func @one_hot_memref_valid(
    %ctx: !hip.context, %indices: memref<2x4xi64, 1>,
    %depth: memref<i64, 1>, %values: memref<2xf32, 1>,
    %output: memref<2x?x4xf32, 1>) {
  hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        memref<2x4xi64, 1>, memref<i64, 1>, memref<2xf32, 1>)
    outs(%output : memref<2x?x4xf32, 1>) {axis = 1 : i64}
  return
}

// -----

func.func @one_hot_indices_shape_contradiction(
    %ctx: !hip.context, %indices: tensor<2x4xi64>, %depth: tensor<i64>,
    %values: tensor<2xf32>, %output: tensor<3x?x4xf32>) {
  // expected-error @+1 {{dim 0 of result mismatch: expected 2}}
  %0 = hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        tensor<2x4xi64>, tensor<i64>, tensor<2xf32>)
    outs(%output : tensor<3x?x4xf32>) {axis = 1 : i64}
    : tensor<3x?x4xf32>
  return
}

// -----

func.func @one_hot_depth_shape_contradiction(
    %ctx: !hip.context, %indices: tensor<2x4xi64>,
    %values: tensor<2xf32>, %output: tensor<2x8x4xf32>) {
  %depth = arith.constant dense<7> : tensor<i64>
  // expected-error @+1 {{dim 1 of result mismatch: expected 7}}
  %0 = hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        tensor<2x4xi64>, tensor<i64>, tensor<2xf32>)
    outs(%output : tensor<2x8x4xf32>) {axis = 1 : i64}
    : tensor<2x8x4xf32>
  return
}

// -----

func.func @one_hot_hip_constant_depth_contradiction(
    %ctx: !hip.context, %indices: tensor<2x4xi64>,
    %values: tensor<2xf32>, %output: tensor<2x8x4xf32>) {
  %depth = hip.constant {value = dense<7> : tensor<i64>} : tensor<i64>
  // expected-error @+1 {{dim 1 of result mismatch: expected 7}}
  %0 = hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        tensor<2x4xi64>, tensor<i64>, tensor<2xf32>)
    outs(%output : tensor<2x8x4xf32>) {axis = 1 : i64}
    : tensor<2x8x4xf32>
  return
}

// -----

// External hip.constant carriers have no structural payload value. A static
// destination therefore remains legal when only the carrier location is known.
func.func @one_hot_location_only_depth_is_unknown(
    %ctx: !hip.context, %indices: tensor<2x4xi64>,
    %values: tensor<2xf32>, %output: tensor<2x8x4xf32>) {
  %depth = hip.constant {
    location = "/unused/depth.bin", offset = 0 : i64, size = 8 : i64
  } : tensor<i64>
  %0 = hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        tensor<2x4xi64>, tensor<i64>, tensor<2xf32>)
    outs(%output : tensor<2x8x4xf32>) {axis = 1 : i64}
    : tensor<2x8x4xf32>
  return
}

// -----

memref.global "private" constant @one_hot_depth7 :
    memref<i64> = dense<7>

func.func @one_hot_global_depth_valid(
    %ctx: !hip.context, %indices: memref<2x4xi64>,
    %values: memref<2xf32>, %output: memref<2x7x4xf32>) {
  %depth = memref.get_global @one_hot_depth7 : memref<i64>
  hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        memref<2x4xi64>, memref<i64>, memref<2xf32>)
    outs(%output : memref<2x7x4xf32>) {axis = 1 : i64}
  return
}

// -----

memref.global "private" constant @one_hot_depth7 :
    memref<i64> = dense<7>

func.func @one_hot_global_depth_contradiction(
    %ctx: !hip.context, %indices: memref<2x4xi64>,
    %values: memref<2xf32>, %output: memref<2x8x4xf32>) {
  %depth = memref.get_global @one_hot_depth7 : memref<i64>
  // expected-error @+1 {{dim 1 of result mismatch: expected 7}}
  hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        memref<2x4xi64>, memref<i64>, memref<2xf32>)
    outs(%output : memref<2x8x4xf32>) {axis = 1 : i64}
  return
}

// -----

func.func @one_hot_malformed_non_empty_init(
    %ctx: !hip.context, %indices: tensor<2x4xi64>, %depth: tensor<i64>,
    %values: tensor<2xf32>, %output: tensor<2x4xf32>) {
  // expected-error @+1 {{output rank must equal indices rank plus one}}
  %0 = hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        tensor<2x4xi64>, tensor<i64>, tensor<2xf32>)
    outs(%output : tensor<2x4xf32>) {axis = 1 : i64}
    : tensor<2x4xf32>
  return
}

// -----

func.func @one_hot_invalid_axis(
    %ctx: !hip.context, %indices: tensor<2x4xi64>, %depth: tensor<i64>,
    %values: tensor<2xf32>, %output: tensor<2x?x4xf32>) {
  // expected-error @+1 {{axis must be in the range [-output.rank, output.rank)}}
  %0 = hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        tensor<2x4xi64>, tensor<i64>, tensor<2xf32>)
    outs(%output : tensor<2x?x4xf32>) {axis = 3 : i64}
    : tensor<2x?x4xf32>
  return
}

// -----

func.func @one_hot_malformed_values(
    %ctx: !hip.context, %indices: tensor<2x4xi64>, %depth: tensor<i64>,
    %values: tensor<3xf32>, %output: tensor<2x?x4xf32>) {
  // expected-error @+1 {{values must be a statically two-element tensor}}
  %0 = hip.one_hot(%ctx)
    ins(%indices, %depth, %values :
        tensor<2x4xi64>, tensor<i64>, tensor<3xf32>)
    outs(%output : tensor<2x?x4xf32>) {axis = 1 : i64}
    : tensor<2x?x4xf32>
  return
}

// -----

func.func @gather_nd_valid(%ctx: !hip.context,
                           %data: memref<2x3x4xf32, 1>,
                           %indices: memref<2x5x1xi64, 1>,
                           %out: memref<2x5x4xf32, 1>) {
  hip.gather_nd(%ctx)
    ins(%data, %indices : memref<2x3x4xf32, 1>,
         memref<2x5x1xi64, 1>)
    outs(%out : memref<2x5x4xf32, 1>)
    {batch_dims = 1 : i64}
  return
}

// -----

func.func @gather_nd_dynamic_tuple_fallback(
    %ctx: !hip.context,
    %data: memref<?x?x?xf32, 1>,
    %indices: memref<?x?xi64, 1>,
    %out: memref<?x?xf32, 1>) {
  hip.gather_nd(%ctx)
    ins(%data, %indices : memref<?x?x?xf32, 1>, memref<?x?xi64, 1>)
    outs(%out : memref<?x?xf32, 1>)
    {batch_dims = 0 : i64}
  return
}

// -----

func.func @gather_nd_tuple_width_addition_overflow(
    %ctx: !hip.context,
    %data: memref<1x1xf32, 1>,
    %indices: memref<1x9223372036854775807xi64, 1>,
    %out: memref<?xf32, 1>) {
  // expected-error @+1 {{indices tuple width and batch_dims are incompatible with data rank}}
  hip.gather_nd(%ctx)
    ins(%data, %indices : memref<1x1xf32, 1>,
         memref<1x9223372036854775807xi64, 1>)
    outs(%out : memref<?xf32, 1>)
    {batch_dims = 1 : i64}
  return
}

// -----

func.func @gather_nd_bad_batch_prefix(
    %ctx: !hip.context,
    %data: memref<2x3x4xf32, 1>,
    %indices: memref<7x5x1xi64, 1>,
    %out: memref<7x5x4xf32, 1>) {
  // expected-error @+1 {{data and indices batch dimensions must match}}
  hip.gather_nd(%ctx)
    ins(%data, %indices : memref<2x3x4xf32, 1>,
         memref<7x5x1xi64, 1>)
    outs(%out : memref<7x5x4xf32, 1>)
    {batch_dims = 1 : i64}
  return
}

// -----

func.func @transpose_valid(%ctx: !hip.context,
                           %input: memref<2x3x4xf16, 1>,
                           %out: memref<4x2x3xf16, 1>) {
  hip.transpose(%ctx)
    ins(%input : memref<2x3x4xf16, 1>)
    outs(%out : memref<4x2x3xf16, 1>)
    {perm = [2, 0, 1]}
  return
}

// -----

func.func @transpose_wrong_output(%ctx: !hip.context,
                                  %input: memref<2x3x4xf16, 1>,
                                  %out: memref<4x3x2xf16, 1>) {
  // expected-error @+1 {{dim 1 of result mismatch: expected 2}}
  hip.transpose(%ctx)
    ins(%input : memref<2x3x4xf16, 1>)
    outs(%out : memref<4x3x2xf16, 1>)
    {perm = [2, 0, 1]}
  return
}

// -----

func.func @matmul_nbits_valid(
    %ctx: !hip.context,
    %a: memref<2x3x64xf16, 1>,
    %b: memref<32x2x16xui8, 1>,
    %scales: memref<32x2xf16, 1>,
    %out: memref<2x3x32xf16, 1>) {
  hip.matmul_nbits(%ctx)
    ins(%a, %b, %scales : memref<2x3x64xf16, 1>,
        memref<32x2x16xui8, 1>, memref<32x2xf16, 1>)
    outs(%out : memref<2x3x32xf16, 1>)
    {K = 64 : i64, N = 32 : i64, bits = 4 : i64,
     block_size = 32 : i64, accuracy_level = 0 : i64,
     zp_elem_size = 2 : i64}
  return
}

// -----

func.func @matmul_nbits_wrong_output(
    %ctx: !hip.context,
    %a: memref<2x3x64xf16, 1>,
    %b: memref<32x2x16xui8, 1>,
    %scales: memref<32x2xf16, 1>,
    %out: memref<2x3x31xf16, 1>) {
  // expected-error @+1 {{dim 2 of result mismatch: expected 32}}
  hip.matmul_nbits(%ctx)
    ins(%a, %b, %scales : memref<2x3x64xf16, 1>,
        memref<32x2x16xui8, 1>, memref<32x2xf16, 1>)
    outs(%out : memref<2x3x31xf16, 1>)
    {K = 64 : i64, N = 32 : i64, bits = 4 : i64,
     block_size = 32 : i64, accuracy_level = 0 : i64,
     zp_elem_size = 2 : i64}
  return
}

// -----

func.func @size_valid(%ctx: !hip.context,
                      %input: memref<?x4xf32, 1>,
                      %out: memref<i64, 1>) {
  hip.size(%ctx)
    ins(%input : memref<?x4xf32, 1>)
    outs(%out : memref<i64, 1>)
  return
}

// -----

func.func @size_wrong_rank(%ctx: !hip.context,
                           %input: memref<?x4xf32, 1>,
                           %out: memref<1xi64, 1>) {
  // expected-error @+1 {{output must be a rank-zero i64 tensor or memref}}
  hip.size(%ctx)
    ins(%input : memref<?x4xf32, 1>)
    outs(%out : memref<1xi64, 1>)
  return
}
