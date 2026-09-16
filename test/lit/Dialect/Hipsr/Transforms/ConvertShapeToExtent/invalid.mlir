// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

//===----------------------------------------------------------------------===//
// Error paths of -hipsr-convert-shape-to-extent and of the relaxed
// hipsr.preserve_shape verifier. Positive coverage lives in
// convert-shape-to-extent.mlir.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -split-input-file -verify-diagnostics -hipsr-convert-shape-to-extent

// The prologue inlines a shape.assuming only when its witness is a passing
// shape.const_witness, which -remove-shape-constraints creates. A surviving
// witness means that pass did not run.
//
// The pass has no pattern for the region, so the region keeps its !shape.size
// result and the conversion reports it as illegal.
func.func @assuming_region_survives(%a: tensor<?x4xf16>, %b: tensor<?x4xf16>) -> index {
  %sa = shape.shape_of %a : tensor<?x4xf16> -> !shape.shape
  %sb = shape.shape_of %b : tensor<?x4xf16> -> !shape.shape
  %w = shape.cstr_eq %sa, %sb : !shape.shape, !shape.shape
  // expected-error@+1 {{failed to legalize operation 'shape.assuming'}}
  %s = shape.assuming %w -> !shape.size {
    %0 = shape.shape_of %a : tensor<?x4xf16> -> !shape.shape
    %n = shape.num_elements %0 : !shape.shape -> !shape.size
    shape.assuming_yield %n : !shape.size
  }
  %i = shape.size_to_index %s : !shape.size
  return %i : index
}

// -----

// Accepting extent tensors and memrefs widens the operand type, so only the
// element type still separates a shape from ordinary data. An i64 host buffer
// is the likely mistake.
func.func @shape_operand_is_not_an_extent_tensor(%extents: tensor<2xi64>,
                                                 %data: tensor<?x4xf16>) {
  // expected-error@+1 {{operand #0 must be shape, extent tensor, or extent memref}}
  hipsr.preserve_shape %extents, %data : tensor<2xi64>, tensor<?x4xf16>
  return
}

// -----

// An extent tensor carries its length, so the verifier can compare it with the
// data rank. An opaque !shape.shape carries no length, so the check only
// applies after conversion.
func.func @extent_count_disagrees_with_data_rank(%d0: index, %data: tensor<?x4xf16>) {
  %shape = tensor.from_elements %d0 : tensor<1xindex>
  // expected-error@+1 {{extent count 1 does not match data rank 2}}
  hipsr.preserve_shape %shape, %data : tensor<1xindex>, tensor<?x4xf16>
  return
}
