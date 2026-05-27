// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the SSA-trace branch of InferOnnxShapes records the correct
// per-output-dim mult when a Reshape merges by a non-power-of-2 factor.
//
// The trace handles Reshape patterns like `<dyn, A> -> <dyn/K, A*K>` by
// recording mult = 1/K on the output dim's origin. Power-of-2 K is exact
// in IEEE 754 (e.g. K=4 → mult=0.25); non-power-of-2 K like K=3 forces
// 1/3 = 0x3FD5555555555555 (≈ 0.333333...). The runtime's `std::llround`
// tolerates the 1-ULP drift for shape values up to ~2^52, but the test
// pins the bit-pattern so a future change to the encoding (or to the
// mult-composition logic) trips the regression.
//
// Single-function file so the module-level `hip.refined_output_dim_origins`
// attribute (which the pass attaches per FuncOp via `setAttr` on the
// enclosing ModuleOp — last writer wins when multiple funcs share a
// module) reliably reflects this test's trace and nothing else. Do NOT
// add more functions to this file.
// ============================================================================

// RUN: hip-mlir-opt --infer-onnx-shapes %s | FileCheck %s

// Reshape <num_patches, 3> -> <num_patches/3, 9>:
//   in_other_product  = 3      (trailing dim)
//   out_other_product = 9
//   mult = in_other / out_other = 3 / 9 = 1/3
//
// IEEE 754 binary64 for 1.0/3.0 is 0x3FD5555555555555 — decimal
// 4599676419421066581. The bit-cast lives in the encoded payload as
// the third int per dim (arg_idx, dim_idx, mult_bits). For arg 0
// dim 0 with mult = 1/3 the triple is `[0, 0, 4599676419421066581]`.
func.func @reshape_k3_mult(%x: tensor<?x3xf32>) -> tensor<?x9xf32> {
  %shape = "onnx.Constant"() {value = dense<[-1, 9]> : tensor<2xi64>}
      : () -> tensor<2xi64>
  %r = "onnx.Reshape"(%x, %shape)
      : (tensor<?x3xf32>, tensor<2xi64>) -> tensor<?x9xf32>
  return %r : tensor<?x9xf32>
}

// The module-level attribute is ArrayAttr<ArrayAttr<IntegerAttr<i64>>>:
// outer index = function result, inner = flat `(arg, dim, mult_bits)`
// triples (3 i64 per output dim). One result, 2 dims:
//   dim 0: traced → arg=0, dim=0, mult = 1/3 = 0x3FD5555555555555
//                                       = 4599676419421066581
//   dim 1: static → sentinel (-1, -1, 1.0_bits) where 1.0_bits =
//          0x3FF0000000000000 = 4607182418800017408
// CHECK: hip.refined_output_dim_origins = {{\[}}[0, 0, 4599676419421066581, -1, -1, 4607182418800017408]]
