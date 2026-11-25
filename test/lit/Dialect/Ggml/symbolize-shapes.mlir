// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt %t/derive.mlir --ggml-symbolize-shapes | FileCheck %s --check-prefix=DERIVE
// RUN: not hip-mlir-opt %t/contradiction.mlir --ggml-symbolize-shapes 2>&1 | FileCheck %s --check-prefix=CONTRADICTION
// RUN: not hip-mlir-opt %t/no-seed.mlir --ggml-symbolize-shapes 2>&1 | FileCheck %s --check-prefix=NO-SEED
// RUN: hip-mlir-opt %t/sliced.mlir --ggml-symbolize-shapes | FileCheck %s --check-prefix=SLICED

//--- derive.mlir
// The capture ran four tokens through a two-wide embedding of eight rows and
// one three-column projection, so every extent below is either four -- the
// request size -- or a property of the model. The module says where the
// request size enters: the first extent of the input named "tokens".

// What was derived is the layout, and a value's layout is its type. Four
// distinct ones survive; every other value shares them.

// DERIVE-DAG: #[[SEED:layout[0-9]*]] = #ggml.layout<ne = [1*token_count, 1, 1, 1], nb = [4, 4*token_count, 4*token_count, 4*token_count], offset = 0>
// DERIVE-DAG: #[[TABLE:layout[0-9]*]] = #ggml.layout<ne = [2, 8, 1, 1], nb = [4, 8, 64, 64], offset = 0>
// DERIVE-DAG: #[[GATHERED:layout[0-9]*]] = #ggml.layout<ne = [2, 1*token_count, 1, 1], nb = [4, 8, 8*token_count, 8*token_count], offset = 0>
// DERIVE-DAG: #[[PROJECTED:layout[0-9]*]] = #ggml.layout<ne = [3, 1*token_count, 1, 1], nb = [4, 12, 12*token_count, 12*token_count], offset = 0>

// The seed: the request size enters here and nowhere else. A weight's extents
// are the model's, which no request changes.
// DERIVE-LABEL: func.func @main_graph(
// DERIVE-SAME: %arg0: tensor<1x1x1x?xi32, #[[SEED]]>
// DERIVE-SAME: %arg1: tensor<1x1x8x2xf32, #[[TABLE]]>

// The result is the value returned there, which the type now says outright.
// DERIVE-SAME: -> (tensor<1x1x?x3xf32, #[[PROJECTED]]>

// The gather takes its width from the table and its height from the indices.
// DERIVE: ggml.get_rows
// DERIVE-SAME: -> tensor<1x1x?x2xf32, #[[GATHERED]]>

// The projection replaces the reduction axis with the weight's output width
// and carries the activation's height through.
// DERIVE: ggml.mul_mat
// DERIVE-SAME: -> tensor<1x1x?x3xf32, #[[PROJECTED]]>

#tokens = #ggml.layout<ne = [4, 1, 1, 1], nb = [4, 16, 16, 16], offset = 0>
#table = #ggml.layout<ne = [2, 8, 1, 1], nb = [4, 8, 64, 64], offset = 0>
#rows = #ggml.layout<ne = [2, 4, 1, 1], nb = [4, 8, 32, 32], offset = 0>
#weight = #ggml.layout<ne = [2, 3, 1, 1], nb = [4, 8, 24, 24], offset = 0>
#projected = #ggml.layout<ne = [3, 4, 1, 1], nb = [4, 12, 48, 48], offset = 0>

module attributes {ggml.symbols = [{name = "token_count", type = "index",
    minimum = 1 : i64, maximum = 512 : i64,
    inputs = [{input = "tokens", dimension = 0 : i64, scale = 1 : i64}]}]} {
func.func @main_graph(%tokens: tensor<1x1x1x4xi32, #tokens> {
      ggml.io_kind = "runtime", ggml.io_name = "runtime.0",
      ggml.source_name = "tokens", ggml.value_id = 1 : i64},
    %table: tensor<1x1x8x2xf32, #table> {
      ggml.io_kind = "runtime", ggml.io_name = "runtime.1",
      ggml.source_name = "token_embd.weight", ggml.value_id = 0 : i64},
    %weight: tensor<1x1x3x2xf32, #weight> {
      ggml.io_kind = "runtime", ggml.io_name = "runtime.2",
      ggml.source_name = "blk.0.attn_q.weight", ggml.value_id = 3 : i64})
    -> (tensor<1x1x4x3xf32, #projected> {
      ggml.io_kind = "output", ggml.io_name = "output.0", ggml.value_id = 4 : i64}) {
  %rows = "ggml.get_rows"(%table, %tokens) {
    ggml.value_id = 2 : i64
  } : (tensor<1x1x8x2xf32, #table>, tensor<1x1x1x4xi32, #tokens>) -> tensor<1x1x4x2xf32, #rows>

  %projected = "ggml.mul_mat"(%weight, %rows)
    <{precision = #ggml.compute_precision<default>, hint = 0 : i64}> {
    ggml.value_id = 4 : i64
  } : (tensor<1x1x3x2xf32, #weight>, tensor<1x1x4x2xf32, #rows>) -> tensor<1x1x4x3xf32, #projected>

  return %projected : tensor<1x1x4x3xf32, #projected>
}
}

//--- contradiction.mlir
// The gather's recorded strides are not the ones its extents imply, so the
// derivation fails on the graph that produced it rather than emitting a layout
// nothing observed.
//
// The contradiction is in the strides and not the extents, because an extent
// no longer has anywhere to hide: `#ggml.layout` requires `ne` to be the shape
// reversed, and `ggml.get_rows`' own verifier requires the result's rows to be
// the indices'. A wrong height is refused before this pass runs. Nothing checks
// a stride but this.

// CONTRADICTION: 'ggml.get_rows' op derived stride 32 at dimension 2 but the capture observed 40

#tokens = #ggml.layout<ne = [4, 1, 1, 1], nb = [4, 16, 16, 16], offset = 0>
#table = #ggml.layout<ne = [2, 8, 1, 1], nb = [4, 8, 64, 64], offset = 0>
#rows = #ggml.layout<ne = [2, 4, 1, 1], nb = [4, 8, 40, 40], offset = 0>

module attributes {ggml.symbols = [{name = "token_count", type = "index",
    minimum = 1 : i64, maximum = 512 : i64,
    inputs = [{input = "tokens", dimension = 0 : i64, scale = 1 : i64}]}]} {
func.func @main_graph(%tokens: tensor<1x1x1x4xi32, #tokens> {
      ggml.source_name = "tokens", ggml.value_id = 1 : i64},
    %table: tensor<1x1x8x2xf32, #table> {
      ggml.source_name = "token_embd.weight", ggml.value_id = 0 : i64})
    -> tensor<1x1x4x2xf32, #rows> {
  %rows = "ggml.get_rows"(%table, %tokens) {
    ggml.value_id = 2 : i64
  } : (tensor<1x1x8x2xf32, #table>, tensor<1x1x1x4xi32, #tokens>) -> tensor<1x1x4x2xf32, #rows>
  return %rows : tensor<1x1x4x2xf32, #rows>
}
}

//--- no-seed.mlir
// A symbol the module declares and no input carries is nowhere for a request
// size to enter, so there is nothing to recover and every extent would
// silently stay constant.

// NO-SEED: no runtime input declares a request-dependent extent

#table = #ggml.layout<ne = [2, 8, 1, 1], nb = [4, 8, 64, 64], offset = 0>

module attributes {ggml.symbols = [{name = "token_count", type = "index",
    minimum = 1 : i64, maximum = 512 : i64,
    inputs = [{input = "tokens", dimension = 0 : i64, scale = 1 : i64}]}]} {
func.func @main_graph(%table: tensor<1x1x8x2xf32, #table> {
      ggml.source_name = "token_embd.weight", ggml.value_id = 0 : i64})
    -> tensor<1x1x8x2xf32, #table> {
  return %table : tensor<1x1x8x2xf32, #table>
}
}

//--- sliced.mlir
// A view of part of its source. GGML records the offset, the four extents and
// the four strides its caller asked for, and none of them follow from
// anything, so the recovery is by stride: a dimension stepping by one of the
// source's strides is walking that dimension of the source, and either spans
// the whole of it -- symbol and all -- or a constant part of it.
//
// Both views below step the same way and differ only in where they start:
// one takes the first column of every row, the other the second.

// SLICED-DAG: #[[GATHERED:layout[0-9]*]] = #ggml.layout<ne = [2, 1*token_count, 1, 1], nb = [4, 8, 8*token_count, 8*token_count], offset = 0>
// SLICED-DAG: #[[FIRST:layout[0-9]*]] = #ggml.layout<ne = [1, 1*token_count, 1, 1], nb = [4, 8, 8*token_count, 8*token_count], offset = 0>
// SLICED-DAG: #[[SECOND:layout[0-9]*]] = #ggml.layout<ne = [1, 1*token_count, 1, 1], nb = [4, 8, 8*token_count, 8*token_count], offset = 4>

// SLICED-LABEL: func.func @main_graph(

// The height is the whole of the source's, so it is the source's expression.
// The width is one of two, so it is the number the caller chose.
// SLICED: ggml.get_rows
// SLICED-SAME: -> tensor<1x1x?x2xf32, #[[GATHERED]]>
// SLICED: ggml.view
// SLICED-SAME: -> tensor<1x1x?x1xf32, #[[FIRST]]>
// SLICED: ggml.view
// SLICED-SAME: -> tensor<1x1x?x1xf32, #[[SECOND]]>

#tokens = #ggml.layout<ne = [4, 1, 1, 1], nb = [4, 16, 16, 16], offset = 0>
#table = #ggml.layout<ne = [2, 8, 1, 1], nb = [4, 8, 64, 64], offset = 0>
#gathered = #ggml.layout<ne = [2, 4, 1, 1], nb = [4, 8, 32, 32], offset = 0>
#first = #ggml.layout<ne = [1, 4, 1, 1], nb = [4, 8, 32, 32], offset = 0>
#second = #ggml.layout<ne = [1, 4, 1, 1], nb = [4, 8, 32, 32], offset = 4>

module attributes {ggml.symbols = [{name = "token_count", type = "index",
    minimum = 1 : i64, maximum = 512 : i64,
    inputs = [{input = "tokens", dimension = 0 : i64, scale = 1 : i64}]}]} {
func.func @main_graph(%tokens: tensor<1x1x1x4xi32, #tokens> {
      ggml.source_name = "tokens", ggml.value_id = 1 : i64},
    %table: tensor<1x1x8x2xf32, #table> {
      ggml.source_name = "token_embd.weight", ggml.value_id = 0 : i64})
    -> (tensor<1x1x4x1xf32, #first>, tensor<1x1x4x1xf32, #second>) {
  %gathered = "ggml.get_rows"(%table, %tokens) {
    ggml.value_id = 2 : i64
  } : (tensor<1x1x8x2xf32, #table>, tensor<1x1x1x4xi32, #tokens>) -> tensor<1x1x4x2xf32, #gathered>
  %first = "ggml.view"(%gathered) {
    ggml.value_id = 3 : i64
  } : (tensor<1x1x4x2xf32, #gathered>) -> tensor<1x1x4x1xf32, #first>
  %second = "ggml.view"(%gathered) {
    ggml.value_id = 4 : i64
  } : (tensor<1x1x4x2xf32, #gathered>) -> tensor<1x1x4x1xf32, #second>
  return %first, %second : tensor<1x1x4x1xf32, #first>, tensor<1x1x4x1xf32, #second>
}
}
