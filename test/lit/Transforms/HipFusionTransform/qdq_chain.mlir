// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// Second-phase fusions of per-tensor UINT16 Q/DQ chains on f32 activations:
// the quantized SwiGLU becomes hip.qswiglu, and a Q read only by a matching
// DQ becomes hip.fake_quantize. Both are exact, so they must decline whenever
// a consumer reads an intermediate with parameters other than the ones its
// producer wrote it with.

// CHECK-LABEL: func.func @qswiglu
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[G:.*]]: tensor<4x64xf32>, %[[U:.*]]: tensor<4x64xf32>)
// CHECK-NOT:   hip.quantize_linear
// CHECK-NOT:   hip.qsigmoid
// CHECK-NOT:   hip.qmul
// CHECK:       %[[Y:.*]] = hip.qswiglu(%[[CTX]]) ins(%[[G]], %[[U]] : tensor<4x64xf32>, tensor<4x64xf32>)
// CHECK-SAME:  dq_scale = 4.000000e+00 : f32, dq_zp = 50 : i64, gate_scale = 5.000000e-01 : f32, gate_zp = 100 : i64, mul_scale = 4.000000e+00 : f32, mul_zp = 50 : i64, sigmoid_scale = 2.500000e-01 : f32, sigmoid_zp = 0 : i64, silu_scale = 1.250000e-01 : f32, silu_zp = 7 : i64, up_scale = 2.000000e+00 : f32, up_zp = 33 : i64
// CHECK-NOT:   hip.dequantize_linear
// CHECK:       return %[[Y]]
func.func @qswiglu(%ctx: !hip.context, %g: tensor<4x64xf32>,
                   %u: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %sg = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %zg = hip.constant {value = dense<100> : tensor<ui16>} : tensor<ui16>
  %su = hip.constant {value = dense<2.000000e+00> : tensor<f32>} : tensor<f32>
  %zu = hip.constant {value = dense<33> : tensor<ui16>} : tensor<ui16>
  %sd = hip.constant {value = dense<4.000000e+00> : tensor<f32>} : tensor<f32>
  %zd = hip.constant {value = dense<50> : tensor<ui16>} : tensor<ui16>

  %e0 = tensor.empty() : tensor<4x64xui16>
  %qg = hip.quantize_linear(%ctx) ins(%g, %sg : tensor<4x64xf32>, tensor<f32>)
      zero_point(%zg : tensor<ui16>) outs(%e0 : tensor<4x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x64xui16>
  %e1 = tensor.empty() : tensor<4x64xui16>
  %qs = hip.qsigmoid(%ctx) ins(%qg : tensor<4x64xui16>)
      outs(%e1 : tensor<4x64xui16>)
      {input_scale = 5.000000e-01 : f32, input_zp = 100 : i64,
       output_scale = 2.500000e-01 : f32, output_zp = 0 : i64}
      : tensor<4x64xui16>
  %e2 = tensor.empty() : tensor<4x64xui16>
  %qa = hip.qmul(%ctx) ins(%qg, %qs : tensor<4x64xui16>, tensor<4x64xui16>)
      outs(%e2 : tensor<4x64xui16>)
      {lhs_scale = 5.000000e-01 : f32, lhs_zp = 100 : i64,
       rhs_scale = 2.500000e-01 : f32, rhs_zp = 0 : i64,
       output_scale = 1.250000e-01 : f32, output_zp = 7 : i64}
      : tensor<4x64xui16>
  %e3 = tensor.empty() : tensor<4x64xui16>
  %qu = hip.quantize_linear(%ctx) ins(%u, %su : tensor<4x64xf32>, tensor<f32>)
      zero_point(%zu : tensor<ui16>) outs(%e3 : tensor<4x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x64xui16>
  %e4 = tensor.empty() : tensor<4x64xui16>
  %qo = hip.qmul(%ctx) ins(%qa, %qu : tensor<4x64xui16>, tensor<4x64xui16>)
      outs(%e4 : tensor<4x64xui16>)
      {lhs_scale = 1.250000e-01 : f32, lhs_zp = 7 : i64,
       rhs_scale = 2.000000e+00 : f32, rhs_zp = 33 : i64,
       output_scale = 4.000000e+00 : f32, output_zp = 50 : i64}
      : tensor<4x64xui16>
  %e5 = tensor.empty() : tensor<4x64xf32>
  %y = hip.dequantize_linear(%ctx) ins(%qo, %sd : tensor<4x64xui16>, tensor<f32>)
      zero_point(%zd : tensor<ui16>) outs(%e5 : tensor<4x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x64xf32>
  return %y : tensor<4x64xf32>
}

// -----

// Both multiplies commuted, and the export's primitive DQ/sigmoid/mul/Q form
// rather than the first-phase qsigmoid/qmul: the first phase forms those ops
// and the second still finds the chain. Dynamic rows trace to one activation.
// CHECK-LABEL: func.func @qswiglu_from_primitives_commuted
// CHECK-NOT:   hip.sigmoid
// CHECK-NOT:   hip.mul
// CHECK-NOT:   hip.qmul
// CHECK:       hip.qswiglu
// CHECK-SAME:  gate_scale = 5.000000e-01 : f32, gate_zp = 100 : i64
// CHECK-SAME:  up_scale = 2.000000e+00 : f32, up_zp = 33 : i64
func.func @qswiglu_from_primitives_commuted(
    %ctx: !hip.context, %x: tensor<?x8xf32>, %wg: tensor<8x64xf32>,
    %wu: tensor<8x64xf32>) -> tensor<?x64xf32> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %x, %c0 : tensor<?x8xf32>
  %sg = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %zg = hip.constant {value = dense<100> : tensor<ui16>} : tensor<ui16>
  %ss = hip.constant {value = dense<2.500000e-01> : tensor<f32>} : tensor<f32>
  %zs = hip.constant {value = dense<0> : tensor<ui16>} : tensor<ui16>
  %sa = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %za = hip.constant {value = dense<7> : tensor<ui16>} : tensor<ui16>
  %su = hip.constant {value = dense<2.000000e+00> : tensor<f32>} : tensor<f32>
  %zu = hip.constant {value = dense<33> : tensor<ui16>} : tensor<ui16>
  %so = hip.constant {value = dense<4.000000e+00> : tensor<f32>} : tensor<f32>
  %zo = hip.constant {value = dense<50> : tensor<ui16>} : tensor<ui16>

  %mg = tensor.empty(%n) : tensor<?x64xf32>
  %g = hip.matmul(%ctx) ins(%x, %wg : tensor<?x8xf32>, tensor<8x64xf32>)
       outs(%mg : tensor<?x64xf32>) : tensor<?x64xf32>
  %mu = tensor.empty(%n) : tensor<?x64xf32>
  %u = hip.matmul(%ctx) ins(%x, %wu : tensor<?x8xf32>, tensor<8x64xf32>)
       outs(%mu : tensor<?x64xf32>) : tensor<?x64xf32>

  %q0 = tensor.empty(%n) : tensor<?x64xui16>
  %qg = hip.quantize_linear(%ctx) ins(%g, %sg : tensor<?x64xf32>, tensor<f32>)
      zero_point(%zg : tensor<ui16>) outs(%q0 : tensor<?x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x64xui16>
  %f0 = tensor.empty(%n) : tensor<?x64xf32>
  %dg = hip.dequantize_linear(%ctx) ins(%qg, %sg : tensor<?x64xui16>, tensor<f32>)
      zero_point(%zg : tensor<ui16>) outs(%f0 : tensor<?x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x64xf32>
  %f1 = tensor.empty(%n) : tensor<?x64xf32>
  %s = hip.sigmoid(%ctx) ins(%dg : tensor<?x64xf32>)
       outs(%f1 : tensor<?x64xf32>) : tensor<?x64xf32>
  %q1 = tensor.empty(%n) : tensor<?x64xui16>
  %qs = hip.quantize_linear(%ctx) ins(%s, %ss : tensor<?x64xf32>, tensor<f32>)
      zero_point(%zs : tensor<ui16>) outs(%q1 : tensor<?x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x64xui16>
  %f2 = tensor.empty(%n) : tensor<?x64xf32>
  %ds = hip.dequantize_linear(%ctx) ins(%qs, %ss : tensor<?x64xui16>, tensor<f32>)
      zero_point(%zs : tensor<ui16>) outs(%f2 : tensor<?x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x64xf32>
  %f3 = tensor.empty(%n) : tensor<?x64xf32>
  %a = hip.mul(%ctx) ins(%ds, %dg : tensor<?x64xf32>, tensor<?x64xf32>)
       outs(%f3 : tensor<?x64xf32>) -> tensor<?x64xf32>
  %q2 = tensor.empty(%n) : tensor<?x64xui16>
  %qa = hip.quantize_linear(%ctx) ins(%a, %sa : tensor<?x64xf32>, tensor<f32>)
      zero_point(%za : tensor<ui16>) outs(%q2 : tensor<?x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x64xui16>
  %f4 = tensor.empty(%n) : tensor<?x64xf32>
  %da = hip.dequantize_linear(%ctx) ins(%qa, %sa : tensor<?x64xui16>, tensor<f32>)
      zero_point(%za : tensor<ui16>) outs(%f4 : tensor<?x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x64xf32>
  %q3 = tensor.empty(%n) : tensor<?x64xui16>
  %qu = hip.quantize_linear(%ctx) ins(%u, %su : tensor<?x64xf32>, tensor<f32>)
      zero_point(%zu : tensor<ui16>) outs(%q3 : tensor<?x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x64xui16>
  %f5 = tensor.empty(%n) : tensor<?x64xf32>
  %du = hip.dequantize_linear(%ctx) ins(%qu, %su : tensor<?x64xui16>, tensor<f32>)
      zero_point(%zu : tensor<ui16>) outs(%f5 : tensor<?x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x64xf32>
  %f6 = tensor.empty(%n) : tensor<?x64xf32>
  %o = hip.mul(%ctx) ins(%du, %da : tensor<?x64xf32>, tensor<?x64xf32>)
       outs(%f6 : tensor<?x64xf32>) -> tensor<?x64xf32>
  %q4 = tensor.empty(%n) : tensor<?x64xui16>
  %qo = hip.quantize_linear(%ctx) ins(%o, %so : tensor<?x64xf32>, tensor<f32>)
      zero_point(%zo : tensor<ui16>) outs(%q4 : tensor<?x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x64xui16>
  %f7 = tensor.empty(%n) : tensor<?x64xf32>
  %y = hip.dequantize_linear(%ctx) ins(%qo, %so : tensor<?x64xui16>, tensor<f32>)
      zero_point(%zo : tensor<ui16>) outs(%f7 : tensor<?x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x64xf32>
  return %y : tensor<?x64xf32>
}

// -----

// hip.qmul reads the gate code with a different scale than the quantize wrote
// it with, so fusing would change the result.
// CHECK-LABEL: func.func @qswiglu_scale_mismatch
// CHECK-NOT:   hip.qswiglu
// CHECK:       hip.qsigmoid
// CHECK:       hip.qmul
// CHECK:       hip.qmul
func.func @qswiglu_scale_mismatch(%ctx: !hip.context, %g: tensor<4x64xf32>,
                                  %u: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %sg = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %zg = hip.constant {value = dense<100> : tensor<ui16>} : tensor<ui16>
  %su = hip.constant {value = dense<2.000000e+00> : tensor<f32>} : tensor<f32>
  %zu = hip.constant {value = dense<33> : tensor<ui16>} : tensor<ui16>
  %e0 = tensor.empty() : tensor<4x64xui16>
  %qg = hip.quantize_linear(%ctx) ins(%g, %sg : tensor<4x64xf32>, tensor<f32>)
      zero_point(%zg : tensor<ui16>) outs(%e0 : tensor<4x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x64xui16>
  %e1 = tensor.empty() : tensor<4x64xui16>
  %qs = hip.qsigmoid(%ctx) ins(%qg : tensor<4x64xui16>)
      outs(%e1 : tensor<4x64xui16>)
      {input_scale = 5.000000e-01 : f32, input_zp = 100 : i64,
       output_scale = 2.500000e-01 : f32, output_zp = 0 : i64}
      : tensor<4x64xui16>
  %e2 = tensor.empty() : tensor<4x64xui16>
  %qa = hip.qmul(%ctx) ins(%qg, %qs : tensor<4x64xui16>, tensor<4x64xui16>)
      outs(%e2 : tensor<4x64xui16>)
      {lhs_scale = 7.500000e-01 : f32, lhs_zp = 100 : i64,
       rhs_scale = 2.500000e-01 : f32, rhs_zp = 0 : i64,
       output_scale = 1.250000e-01 : f32, output_zp = 7 : i64}
      : tensor<4x64xui16>
  %e3 = tensor.empty() : tensor<4x64xui16>
  %qu = hip.quantize_linear(%ctx) ins(%u, %su : tensor<4x64xf32>, tensor<f32>)
      zero_point(%zu : tensor<ui16>) outs(%e3 : tensor<4x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<4x64xui16>
  %e4 = tensor.empty() : tensor<4x64xui16>
  %qo = hip.qmul(%ctx) ins(%qa, %qu : tensor<4x64xui16>, tensor<4x64xui16>)
      outs(%e4 : tensor<4x64xui16>)
      {lhs_scale = 1.250000e-01 : f32, lhs_zp = 7 : i64,
       rhs_scale = 2.000000e+00 : f32, rhs_zp = 33 : i64,
       output_scale = 4.000000e+00 : f32, output_zp = 50 : i64}
      : tensor<4x64xui16>
  %e5 = tensor.empty() : tensor<4x64xf32>
  %y = hip.dequantize_linear(%ctx) ins(%qo, %su : tensor<4x64xui16>, tensor<f32>)
      zero_point(%zu : tensor<ui16>) outs(%e5 : tensor<4x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<4x64xf32>
  return %y : tensor<4x64xf32>
}

// -----

// CHECK-LABEL: func.func @fake_quantize
// CHECK-SAME:  (%[[CTX:.*]]: !hip.context, %[[X:.*]]: tensor<?x5120xf32>)
// CHECK-NOT:   hip.quantize_linear
// CHECK:       %[[Y:.*]] = hip.fake_quantize(%[[CTX]]) ins(%[[X]] : tensor<?x5120xf32>)
// CHECK-SAME:  {qmax = 65535 : i64, qmin = 0 : i64, scale = 1.250000e-01 : f32, zero_point = 32768 : i64}
// CHECK-NOT:   hip.dequantize_linear
// CHECK:       return %[[Y]]
func.func @fake_quantize(%ctx: !hip.context,
                         %x: tensor<?x5120xf32>) -> tensor<?x5120xf32> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %x, %c0 : tensor<?x5120xf32>
  %s = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %e0 = tensor.empty(%n) : tensor<?x5120xui16>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<?x5120xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%e0 : tensor<?x5120xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x5120xui16>
  %n1 = tensor.dim %q, %c0 : tensor<?x5120xui16>
  %e1 = tensor.empty(%n1) : tensor<?x5120xf32>
  %y = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<?x5120xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%e1 : tensor<?x5120xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x5120xf32>
  return %y : tensor<?x5120xf32>
}

// -----

// Signed 8-bit storage takes its range from the element type.
// CHECK-LABEL: func.func @fake_quantize_i8
// CHECK:       hip.fake_quantize
// CHECK-SAME:  qmax = 127 : i64, qmin = -128 : i64
func.func @fake_quantize_i8(%ctx: !hip.context,
                            %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %s = hip.constant {value = dense<2.500000e-01> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<3> : tensor<si8>} : tensor<si8>
  %e0 = tensor.empty() : tensor<8x16xsi8>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<8x16xf32>, tensor<f32>)
      zero_point(%z : tensor<si8>) outs(%e0 : tensor<8x16xsi8>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8x16xsi8>
  %e1 = tensor.empty() : tensor<8x16xf32>
  %y = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<8x16xsi8>, tensor<f32>)
      zero_point(%z : tensor<si8>) outs(%e1 : tensor<8x16xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// The DQ uses another scale, so the round trip is not an identity on the grid.
// CHECK-LABEL: func.func @fake_quantize_scale_mismatch
// CHECK-NOT:   hip.fake_quantize
// CHECK:       hip.quantize_linear
// CHECK:       hip.dequantize_linear
func.func @fake_quantize_scale_mismatch(%ctx: !hip.context,
                                        %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %s = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %s2 = hip.constant {value = dense<2.500000e-01> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %e0 = tensor.empty() : tensor<8x16xui16>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<8x16xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%e0 : tensor<8x16xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8x16xui16>
  %e1 = tensor.empty() : tensor<8x16xf32>
  %y = hip.dequantize_linear(%ctx) ins(%q, %s2 : tensor<8x16xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%e1 : tensor<8x16xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// The codes are also a function result, so the quantize has to stay.
// CHECK-LABEL: func.func @fake_quantize_code_escapes
// CHECK-NOT:   hip.fake_quantize
// CHECK:       hip.quantize_linear
// CHECK:       hip.dequantize_linear
func.func @fake_quantize_code_escapes(%ctx: !hip.context, %x: tensor<8x16xf32>)
    -> (tensor<8x16xf32>, tensor<8x16xui16>) {
  %s = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %e0 = tensor.empty() : tensor<8x16xui16>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<8x16xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%e0 : tensor<8x16xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8x16xui16>
  %e1 = tensor.empty() : tensor<8x16xf32>
  %y = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<8x16xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%e1 : tensor<8x16xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x16xf32>
  return %y, %q : tensor<8x16xf32>, tensor<8x16xui16>
}

// -----

// Two projections read the activation through their own DQ of one code. One
// hip.fake_quantize serves both.
// CHECK-LABEL: func.func @fake_quantize_shared
// CHECK-COUNT-1: hip.fake_quantize
// CHECK-NOT:   hip.fake_quantize
// CHECK-NOT:   hip.dequantize_linear
// CHECK:       return %[[Y:.*]], %[[Y]]
func.func @fake_quantize_shared(%ctx: !hip.context, %x: tensor<8x16xf32>)
    -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  %s = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %z = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %e0 = tensor.empty() : tensor<8x16xui16>
  %q = hip.quantize_linear(%ctx) ins(%x, %s : tensor<8x16xf32>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%e0 : tensor<8x16xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<8x16xui16>
  %e1 = tensor.empty() : tensor<8x16xf32>
  %y0 = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<8x16xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%e1 : tensor<8x16xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x16xf32>
  %e2 = tensor.empty() : tensor<8x16xf32>
  %y1 = hip.dequantize_linear(%ctx) ins(%q, %s : tensor<8x16xui16>, tensor<f32>)
      zero_point(%z : tensor<ui16>) outs(%e2 : tensor<8x16xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<8x16xf32>
  return %y0, %y1 : tensor<8x16xf32>, tensor<8x16xf32>
}

// -----

// Gate and up projections each read the activation through their own DQ of
// one code tensor, so their dynamic rows are proven equal through the Q/DQ.
// CHECK-LABEL: func.func @qswiglu_separate_activation_dqs
// CHECK:       hip.fake_quantize
// CHECK:       hip.qswiglu
// CHECK-NOT:   hip.qmul
func.func @qswiglu_separate_activation_dqs(
    %ctx: !hip.context, %x: tensor<?x8xf32>, %wg: tensor<8x64xf32>,
    %wu: tensor<8x64xf32>) -> tensor<?x64xf32> {
  %c0 = arith.constant 0 : index
  %n = tensor.dim %x, %c0 : tensor<?x8xf32>
  %sx = hip.constant {value = dense<1.250000e-01> : tensor<f32>} : tensor<f32>
  %zx = hip.constant {value = dense<32768> : tensor<ui16>} : tensor<ui16>
  %sg = hip.constant {value = dense<5.000000e-01> : tensor<f32>} : tensor<f32>
  %zg = hip.constant {value = dense<100> : tensor<ui16>} : tensor<ui16>
  %su = hip.constant {value = dense<2.000000e+00> : tensor<f32>} : tensor<f32>
  %zu = hip.constant {value = dense<33> : tensor<ui16>} : tensor<ui16>
  %sd = hip.constant {value = dense<4.000000e+00> : tensor<f32>} : tensor<f32>
  %zd = hip.constant {value = dense<50> : tensor<ui16>} : tensor<ui16>

  %ex = tensor.empty(%n) : tensor<?x8xui16>
  %qx = hip.quantize_linear(%ctx) ins(%x, %sx : tensor<?x8xf32>, tensor<f32>)
      zero_point(%zx : tensor<ui16>) outs(%ex : tensor<?x8xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x8xui16>
  %ea = tensor.empty(%n) : tensor<?x8xf32>
  %xa = hip.dequantize_linear(%ctx) ins(%qx, %sx : tensor<?x8xui16>, tensor<f32>)
      zero_point(%zx : tensor<ui16>) outs(%ea : tensor<?x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x8xf32>
  %eb = tensor.empty(%n) : tensor<?x8xf32>
  %xb = hip.dequantize_linear(%ctx) ins(%qx, %sx : tensor<?x8xui16>, tensor<f32>)
      zero_point(%zx : tensor<ui16>) outs(%eb : tensor<?x8xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x8xf32>
  %mg = tensor.empty(%n) : tensor<?x64xf32>
  %g = hip.matmul(%ctx) ins(%xa, %wg : tensor<?x8xf32>, tensor<8x64xf32>)
       outs(%mg : tensor<?x64xf32>) : tensor<?x64xf32>
  %mu = tensor.empty(%n) : tensor<?x64xf32>
  %u = hip.matmul(%ctx) ins(%xb, %wu : tensor<?x8xf32>, tensor<8x64xf32>)
       outs(%mu : tensor<?x64xf32>) : tensor<?x64xf32>

  %e0 = tensor.empty(%n) : tensor<?x64xui16>
  %qg = hip.quantize_linear(%ctx) ins(%g, %sg : tensor<?x64xf32>, tensor<f32>)
      zero_point(%zg : tensor<ui16>) outs(%e0 : tensor<?x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x64xui16>
  %e1 = tensor.empty(%n) : tensor<?x64xui16>
  %qs = hip.qsigmoid(%ctx) ins(%qg : tensor<?x64xui16>)
      outs(%e1 : tensor<?x64xui16>)
      {input_scale = 5.000000e-01 : f32, input_zp = 100 : i64,
       output_scale = 2.500000e-01 : f32, output_zp = 0 : i64}
      : tensor<?x64xui16>
  %e2 = tensor.empty(%n) : tensor<?x64xui16>
  %qa = hip.qmul(%ctx) ins(%qs, %qg : tensor<?x64xui16>, tensor<?x64xui16>)
      outs(%e2 : tensor<?x64xui16>)
      {lhs_scale = 2.500000e-01 : f32, lhs_zp = 0 : i64,
       rhs_scale = 5.000000e-01 : f32, rhs_zp = 100 : i64,
       output_scale = 1.250000e-01 : f32, output_zp = 7 : i64}
      : tensor<?x64xui16>
  %e3 = tensor.empty(%n) : tensor<?x64xui16>
  %qu = hip.quantize_linear(%ctx) ins(%u, %su : tensor<?x64xf32>, tensor<f32>)
      zero_point(%zu : tensor<ui16>) outs(%e3 : tensor<?x64xui16>)
      {axis = 1 : i64, block_size = 0 : i64, precision = 0 : i64,
       saturate = 1 : i64} : tensor<?x64xui16>
  %e4 = tensor.empty(%n) : tensor<?x64xui16>
  %qo = hip.qmul(%ctx) ins(%qu, %qa : tensor<?x64xui16>, tensor<?x64xui16>)
      outs(%e4 : tensor<?x64xui16>)
      {lhs_scale = 2.000000e+00 : f32, lhs_zp = 33 : i64,
       rhs_scale = 1.250000e-01 : f32, rhs_zp = 7 : i64,
       output_scale = 4.000000e+00 : f32, output_zp = 50 : i64}
      : tensor<?x64xui16>
  %e5 = tensor.empty(%n) : tensor<?x64xf32>
  %y = hip.dequantize_linear(%ctx) ins(%qo, %sd : tensor<?x64xui16>, tensor<f32>)
      zero_point(%zd : tensor<ui16>) outs(%e5 : tensor<?x64xf32>)
      {axis = 1 : i64, block_size = 0 : i64} : tensor<?x64xf32>
  return %y : tensor<?x64xf32>
}
