// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @promoted_static
// CHECK-NOT: hip.cast
// CHECK-NOT: hip.sigmoid
// CHECK-NOT: hip.mul
// CHECK: hip.swiglu
// CHECK-SAME: promote_activation = true
// CHECK-NOT: hip.cast
func.func @promoted_static(%ctx: !hip.context, %g: tensor<8x128xf16>, %u: tensor<8x128xf16>) -> tensor<8x128xf16> {
  %eg = tensor.empty() : tensor<8x128xf32>
  %gf = hip.cast(%ctx) ins(%g : tensor<8x128xf16>) outs(%eg : tensor<8x128xf32>) {to = 1 : i64} : tensor<8x128xf32>
  %eu = tensor.empty() : tensor<8x128xf32>
  %uf = hip.cast(%ctx) ins(%u : tensor<8x128xf16>) outs(%eu : tensor<8x128xf32>) {to = 1 : i64} : tensor<8x128xf32>
  %es = tensor.empty() : tensor<8x128xf32>
  %s = hip.sigmoid(%ctx) ins(%gf : tensor<8x128xf32>) outs(%es : tensor<8x128xf32>) : tensor<8x128xf32>
  %ea = tensor.empty() : tensor<8x128xf32>
  %a = hip.mul(%ctx) ins(%gf, %s : tensor<8x128xf32>, tensor<8x128xf32>) outs(%ea : tensor<8x128xf32>) -> tensor<8x128xf32>
  %ey = tensor.empty() : tensor<8x128xf32>
  %y = hip.mul(%ctx) ins(%a, %uf : tensor<8x128xf32>, tensor<8x128xf32>) outs(%ey : tensor<8x128xf32>) -> tensor<8x128xf32>
  %eo = tensor.empty() : tensor<8x128xf16>
  %o = hip.cast(%ctx) ins(%y : tensor<8x128xf32>) outs(%eo : tensor<8x128xf16>) {to = 10 : i64} : tensor<8x128xf16>
  return %o : tensor<8x128xf16>
}

// -----

// CHECK-LABEL: func.func @promoted_shared_destination_extent
// CHECK-NOT: hip.cast
// CHECK-NOT: hip.mul
// CHECK: hip.swiglu
// CHECK-SAME: promote_activation = true
// CHECK-NOT: hip.cast
func.func @promoted_shared_destination_extent(%ctx: !hip.context, %g: tensor<?x128xf16>, %u: tensor<?x128xf16>, %n: index, %m: index) -> tensor<?x128xf16> {
  %egh = tensor.empty(%n) : tensor<?x128xf16>
  %gh = hip.sigmoid(%ctx) ins(%g : tensor<?x128xf16>)
        outs(%egh : tensor<?x128xf16>) : tensor<?x128xf16>
  %euh = tensor.empty(%n) : tensor<?x128xf16>
  %uh = hip.sigmoid(%ctx) ins(%u : tensor<?x128xf16>)
        outs(%euh : tensor<?x128xf16>) : tensor<?x128xf16>
  %eg = tensor.empty(%n) : tensor<?x128xf32>
  %gf = hip.cast(%ctx) ins(%gh : tensor<?x128xf16>) outs(%eg : tensor<?x128xf32>) {to = 1 : i64} : tensor<?x128xf32>
  %eu = tensor.empty(%n) : tensor<?x128xf32>
  %uf = hip.cast(%ctx) ins(%uh : tensor<?x128xf16>) outs(%eu : tensor<?x128xf32>) {to = 1 : i64} : tensor<?x128xf32>
  %es = tensor.empty(%n) : tensor<?x128xf32>
  %s = hip.sigmoid(%ctx) ins(%gf : tensor<?x128xf32>) outs(%es : tensor<?x128xf32>) : tensor<?x128xf32>
  %ea = tensor.empty(%n) : tensor<?x128xf32>
  %a = hip.mul(%ctx) ins(%gf, %s : tensor<?x128xf32>, tensor<?x128xf32>) outs(%ea : tensor<?x128xf32>) -> tensor<?x128xf32>
  %ey = tensor.empty(%n) : tensor<?x128xf32>
  %y = hip.mul(%ctx) ins(%a, %uf : tensor<?x128xf32>, tensor<?x128xf32>) outs(%ey : tensor<?x128xf32>) -> tensor<?x128xf32>
  %eo = tensor.empty(%n) : tensor<?x128xf16>
  %o = hip.cast(%ctx) ins(%y : tensor<?x128xf32>) outs(%eo : tensor<?x128xf16>) {to = 10 : i64} : tensor<?x128xf16>
  return %o : tensor<?x128xf16>
}

// -----

// CHECK-LABEL: func.func @unproven_dynamic
// CHECK: hip.cast
// CHECK-NOT: promote_activation = true
func.func @unproven_dynamic(%ctx: !hip.context, %g: tensor<?x128xf16>, %u: tensor<?x128xf16>, %n: index, %m: index) -> tensor<?x128xf16> {
  %egh = tensor.empty(%n) : tensor<?x128xf16>
  %gh = hip.sigmoid(%ctx) ins(%g : tensor<?x128xf16>)
        outs(%egh : tensor<?x128xf16>) : tensor<?x128xf16>
  %euh = tensor.empty(%m) : tensor<?x128xf16>
  %uh = hip.sigmoid(%ctx) ins(%u : tensor<?x128xf16>)
        outs(%euh : tensor<?x128xf16>) : tensor<?x128xf16>
  %eg = tensor.empty(%n) : tensor<?x128xf32>
  %gf = hip.cast(%ctx) ins(%gh : tensor<?x128xf16>) outs(%eg : tensor<?x128xf32>) {to = 1 : i64} : tensor<?x128xf32>
  %eu = tensor.empty(%m) : tensor<?x128xf32>
  %uf = hip.cast(%ctx) ins(%uh : tensor<?x128xf16>) outs(%eu : tensor<?x128xf32>) {to = 1 : i64} : tensor<?x128xf32>
  %es = tensor.empty(%n) : tensor<?x128xf32>
  %s = hip.sigmoid(%ctx) ins(%gf : tensor<?x128xf32>) outs(%es : tensor<?x128xf32>) : tensor<?x128xf32>
  %ea = tensor.empty(%n) : tensor<?x128xf32>
  %a = hip.mul(%ctx) ins(%gf, %s : tensor<?x128xf32>, tensor<?x128xf32>) outs(%ea : tensor<?x128xf32>) -> tensor<?x128xf32>
  %ey = tensor.empty(%n) : tensor<?x128xf32>
  %y = hip.mul(%ctx) ins(%a, %uf : tensor<?x128xf32>, tensor<?x128xf32>) outs(%ey : tensor<?x128xf32>) -> tensor<?x128xf32>
  %eo = tensor.empty(%n) : tensor<?x128xf16>
  %o = hip.cast(%ctx) ins(%y : tensor<?x128xf32>) outs(%eo : tensor<?x128xf16>) {to = 10 : i64} : tensor<?x128xf16>
  return %o : tensor<?x128xf16>
}

// -----

// CHECK-LABEL: func.func @preserve_cast_value_consumer
// CHECK: hip.cast
// CHECK-NOT: promote_activation = true
func.func @preserve_cast_value_consumer(%ctx: !hip.context, %g: tensor<8x128xf16>, %u: tensor<8x128xf16>) -> (tensor<8x128xf16>, tensor<8x128xf32>) {
  %eg = tensor.empty() : tensor<8x128xf32>
  %gf = hip.cast(%ctx) ins(%g : tensor<8x128xf16>) outs(%eg : tensor<8x128xf32>) {to = 1 : i64} : tensor<8x128xf32>
  %eu = tensor.empty() : tensor<8x128xf32>
  %uf = hip.cast(%ctx) ins(%u : tensor<8x128xf16>) outs(%eu : tensor<8x128xf32>) {to = 1 : i64} : tensor<8x128xf32>
  %es = tensor.empty() : tensor<8x128xf32>
  %s = hip.sigmoid(%ctx) ins(%gf : tensor<8x128xf32>) outs(%es : tensor<8x128xf32>) : tensor<8x128xf32>
  %ea = tensor.empty() : tensor<8x128xf32>
  %a = hip.mul(%ctx) ins(%gf, %s : tensor<8x128xf32>, tensor<8x128xf32>) outs(%ea : tensor<8x128xf32>) -> tensor<8x128xf32>
  %ey = tensor.empty() : tensor<8x128xf32>
  %y = hip.mul(%ctx) ins(%a, %uf : tensor<8x128xf32>, tensor<8x128xf32>) outs(%ey : tensor<8x128xf32>) -> tensor<8x128xf32>
  %eo = tensor.empty() : tensor<8x128xf16>
  %o = hip.cast(%ctx) ins(%y : tensor<8x128xf32>) outs(%eo : tensor<8x128xf16>) {to = 10 : i64} : tensor<8x128xf16>
  return %o, %gf : tensor<8x128xf16>, tensor<8x128xf32>
}

// -----

// The shared anchor has more than 32 producers before it; only the local
// branch provenance is needed. This is the decoder gated-normalization view.
// CHECK-LABEL: func.func @promoted_decoder_views_deep_anchor
// CHECK: hip.swiglu
// CHECK-SAME: promote_activation = true
// CHECK-NOT: hip.cast
func.func @promoted_decoder_views_deep_anchor(%ctx: !hip.context, %h: tensor<?x?x6144xf16>) -> tensor<?x?x6144xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b0 = tensor.dim %h, %c0 : tensor<?x?x6144xf16>
  %n0 = tensor.dim %h, %c1 : tensor<?x?x6144xf16>
  %eh0 = tensor.empty(%b0, %n0) : tensor<?x?x6144xf16>
  %h0 = hip.sigmoid(%ctx) ins(%h : tensor<?x?x6144xf16>) outs(%eh0 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b1 = tensor.dim %h0, %c0 : tensor<?x?x6144xf16>
  %n1 = tensor.dim %h0, %c1 : tensor<?x?x6144xf16>
  %eh1 = tensor.empty(%b1, %n1) : tensor<?x?x6144xf16>
  %h1 = hip.sigmoid(%ctx) ins(%h0 : tensor<?x?x6144xf16>) outs(%eh1 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b2 = tensor.dim %h1, %c0 : tensor<?x?x6144xf16>
  %n2 = tensor.dim %h1, %c1 : tensor<?x?x6144xf16>
  %eh2 = tensor.empty(%b2, %n2) : tensor<?x?x6144xf16>
  %h2 = hip.sigmoid(%ctx) ins(%h1 : tensor<?x?x6144xf16>) outs(%eh2 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b3 = tensor.dim %h2, %c0 : tensor<?x?x6144xf16>
  %n3 = tensor.dim %h2, %c1 : tensor<?x?x6144xf16>
  %eh3 = tensor.empty(%b3, %n3) : tensor<?x?x6144xf16>
  %h3 = hip.sigmoid(%ctx) ins(%h2 : tensor<?x?x6144xf16>) outs(%eh3 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b4 = tensor.dim %h3, %c0 : tensor<?x?x6144xf16>
  %n4 = tensor.dim %h3, %c1 : tensor<?x?x6144xf16>
  %eh4 = tensor.empty(%b4, %n4) : tensor<?x?x6144xf16>
  %h4 = hip.sigmoid(%ctx) ins(%h3 : tensor<?x?x6144xf16>) outs(%eh4 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b5 = tensor.dim %h4, %c0 : tensor<?x?x6144xf16>
  %n5 = tensor.dim %h4, %c1 : tensor<?x?x6144xf16>
  %eh5 = tensor.empty(%b5, %n5) : tensor<?x?x6144xf16>
  %h5 = hip.sigmoid(%ctx) ins(%h4 : tensor<?x?x6144xf16>) outs(%eh5 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b6 = tensor.dim %h5, %c0 : tensor<?x?x6144xf16>
  %n6 = tensor.dim %h5, %c1 : tensor<?x?x6144xf16>
  %eh6 = tensor.empty(%b6, %n6) : tensor<?x?x6144xf16>
  %h6 = hip.sigmoid(%ctx) ins(%h5 : tensor<?x?x6144xf16>) outs(%eh6 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b7 = tensor.dim %h6, %c0 : tensor<?x?x6144xf16>
  %n7 = tensor.dim %h6, %c1 : tensor<?x?x6144xf16>
  %eh7 = tensor.empty(%b7, %n7) : tensor<?x?x6144xf16>
  %h7 = hip.sigmoid(%ctx) ins(%h6 : tensor<?x?x6144xf16>) outs(%eh7 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b8 = tensor.dim %h7, %c0 : tensor<?x?x6144xf16>
  %n8 = tensor.dim %h7, %c1 : tensor<?x?x6144xf16>
  %eh8 = tensor.empty(%b8, %n8) : tensor<?x?x6144xf16>
  %h8 = hip.sigmoid(%ctx) ins(%h7 : tensor<?x?x6144xf16>) outs(%eh8 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b9 = tensor.dim %h8, %c0 : tensor<?x?x6144xf16>
  %n9 = tensor.dim %h8, %c1 : tensor<?x?x6144xf16>
  %eh9 = tensor.empty(%b9, %n9) : tensor<?x?x6144xf16>
  %h9 = hip.sigmoid(%ctx) ins(%h8 : tensor<?x?x6144xf16>) outs(%eh9 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b10 = tensor.dim %h9, %c0 : tensor<?x?x6144xf16>
  %n10 = tensor.dim %h9, %c1 : tensor<?x?x6144xf16>
  %eh10 = tensor.empty(%b10, %n10) : tensor<?x?x6144xf16>
  %h10 = hip.sigmoid(%ctx) ins(%h9 : tensor<?x?x6144xf16>) outs(%eh10 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b11 = tensor.dim %h10, %c0 : tensor<?x?x6144xf16>
  %n11 = tensor.dim %h10, %c1 : tensor<?x?x6144xf16>
  %eh11 = tensor.empty(%b11, %n11) : tensor<?x?x6144xf16>
  %h11 = hip.sigmoid(%ctx) ins(%h10 : tensor<?x?x6144xf16>) outs(%eh11 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b12 = tensor.dim %h11, %c0 : tensor<?x?x6144xf16>
  %n12 = tensor.dim %h11, %c1 : tensor<?x?x6144xf16>
  %eh12 = tensor.empty(%b12, %n12) : tensor<?x?x6144xf16>
  %h12 = hip.sigmoid(%ctx) ins(%h11 : tensor<?x?x6144xf16>) outs(%eh12 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b13 = tensor.dim %h12, %c0 : tensor<?x?x6144xf16>
  %n13 = tensor.dim %h12, %c1 : tensor<?x?x6144xf16>
  %eh13 = tensor.empty(%b13, %n13) : tensor<?x?x6144xf16>
  %h13 = hip.sigmoid(%ctx) ins(%h12 : tensor<?x?x6144xf16>) outs(%eh13 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b14 = tensor.dim %h13, %c0 : tensor<?x?x6144xf16>
  %n14 = tensor.dim %h13, %c1 : tensor<?x?x6144xf16>
  %eh14 = tensor.empty(%b14, %n14) : tensor<?x?x6144xf16>
  %h14 = hip.sigmoid(%ctx) ins(%h13 : tensor<?x?x6144xf16>) outs(%eh14 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b15 = tensor.dim %h14, %c0 : tensor<?x?x6144xf16>
  %n15 = tensor.dim %h14, %c1 : tensor<?x?x6144xf16>
  %eh15 = tensor.empty(%b15, %n15) : tensor<?x?x6144xf16>
  %h15 = hip.sigmoid(%ctx) ins(%h14 : tensor<?x?x6144xf16>) outs(%eh15 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b16 = tensor.dim %h15, %c0 : tensor<?x?x6144xf16>
  %n16 = tensor.dim %h15, %c1 : tensor<?x?x6144xf16>
  %eh16 = tensor.empty(%b16, %n16) : tensor<?x?x6144xf16>
  %h16 = hip.sigmoid(%ctx) ins(%h15 : tensor<?x?x6144xf16>) outs(%eh16 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b17 = tensor.dim %h16, %c0 : tensor<?x?x6144xf16>
  %n17 = tensor.dim %h16, %c1 : tensor<?x?x6144xf16>
  %eh17 = tensor.empty(%b17, %n17) : tensor<?x?x6144xf16>
  %h17 = hip.sigmoid(%ctx) ins(%h16 : tensor<?x?x6144xf16>) outs(%eh17 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b18 = tensor.dim %h17, %c0 : tensor<?x?x6144xf16>
  %n18 = tensor.dim %h17, %c1 : tensor<?x?x6144xf16>
  %eh18 = tensor.empty(%b18, %n18) : tensor<?x?x6144xf16>
  %h18 = hip.sigmoid(%ctx) ins(%h17 : tensor<?x?x6144xf16>) outs(%eh18 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b19 = tensor.dim %h18, %c0 : tensor<?x?x6144xf16>
  %n19 = tensor.dim %h18, %c1 : tensor<?x?x6144xf16>
  %eh19 = tensor.empty(%b19, %n19) : tensor<?x?x6144xf16>
  %h19 = hip.sigmoid(%ctx) ins(%h18 : tensor<?x?x6144xf16>) outs(%eh19 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b20 = tensor.dim %h19, %c0 : tensor<?x?x6144xf16>
  %n20 = tensor.dim %h19, %c1 : tensor<?x?x6144xf16>
  %eh20 = tensor.empty(%b20, %n20) : tensor<?x?x6144xf16>
  %h20 = hip.sigmoid(%ctx) ins(%h19 : tensor<?x?x6144xf16>) outs(%eh20 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b21 = tensor.dim %h20, %c0 : tensor<?x?x6144xf16>
  %n21 = tensor.dim %h20, %c1 : tensor<?x?x6144xf16>
  %eh21 = tensor.empty(%b21, %n21) : tensor<?x?x6144xf16>
  %h21 = hip.sigmoid(%ctx) ins(%h20 : tensor<?x?x6144xf16>) outs(%eh21 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b22 = tensor.dim %h21, %c0 : tensor<?x?x6144xf16>
  %n22 = tensor.dim %h21, %c1 : tensor<?x?x6144xf16>
  %eh22 = tensor.empty(%b22, %n22) : tensor<?x?x6144xf16>
  %h22 = hip.sigmoid(%ctx) ins(%h21 : tensor<?x?x6144xf16>) outs(%eh22 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b23 = tensor.dim %h22, %c0 : tensor<?x?x6144xf16>
  %n23 = tensor.dim %h22, %c1 : tensor<?x?x6144xf16>
  %eh23 = tensor.empty(%b23, %n23) : tensor<?x?x6144xf16>
  %h23 = hip.sigmoid(%ctx) ins(%h22 : tensor<?x?x6144xf16>) outs(%eh23 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b24 = tensor.dim %h23, %c0 : tensor<?x?x6144xf16>
  %n24 = tensor.dim %h23, %c1 : tensor<?x?x6144xf16>
  %eh24 = tensor.empty(%b24, %n24) : tensor<?x?x6144xf16>
  %h24 = hip.sigmoid(%ctx) ins(%h23 : tensor<?x?x6144xf16>) outs(%eh24 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b25 = tensor.dim %h24, %c0 : tensor<?x?x6144xf16>
  %n25 = tensor.dim %h24, %c1 : tensor<?x?x6144xf16>
  %eh25 = tensor.empty(%b25, %n25) : tensor<?x?x6144xf16>
  %h25 = hip.sigmoid(%ctx) ins(%h24 : tensor<?x?x6144xf16>) outs(%eh25 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b26 = tensor.dim %h25, %c0 : tensor<?x?x6144xf16>
  %n26 = tensor.dim %h25, %c1 : tensor<?x?x6144xf16>
  %eh26 = tensor.empty(%b26, %n26) : tensor<?x?x6144xf16>
  %h26 = hip.sigmoid(%ctx) ins(%h25 : tensor<?x?x6144xf16>) outs(%eh26 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b27 = tensor.dim %h26, %c0 : tensor<?x?x6144xf16>
  %n27 = tensor.dim %h26, %c1 : tensor<?x?x6144xf16>
  %eh27 = tensor.empty(%b27, %n27) : tensor<?x?x6144xf16>
  %h27 = hip.sigmoid(%ctx) ins(%h26 : tensor<?x?x6144xf16>) outs(%eh27 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b28 = tensor.dim %h27, %c0 : tensor<?x?x6144xf16>
  %n28 = tensor.dim %h27, %c1 : tensor<?x?x6144xf16>
  %eh28 = tensor.empty(%b28, %n28) : tensor<?x?x6144xf16>
  %h28 = hip.sigmoid(%ctx) ins(%h27 : tensor<?x?x6144xf16>) outs(%eh28 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b29 = tensor.dim %h28, %c0 : tensor<?x?x6144xf16>
  %n29 = tensor.dim %h28, %c1 : tensor<?x?x6144xf16>
  %eh29 = tensor.empty(%b29, %n29) : tensor<?x?x6144xf16>
  %h29 = hip.sigmoid(%ctx) ins(%h28 : tensor<?x?x6144xf16>) outs(%eh29 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b30 = tensor.dim %h29, %c0 : tensor<?x?x6144xf16>
  %n30 = tensor.dim %h29, %c1 : tensor<?x?x6144xf16>
  %eh30 = tensor.empty(%b30, %n30) : tensor<?x?x6144xf16>
  %h30 = hip.sigmoid(%ctx) ins(%h29 : tensor<?x?x6144xf16>) outs(%eh30 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b31 = tensor.dim %h30, %c0 : tensor<?x?x6144xf16>
  %n31 = tensor.dim %h30, %c1 : tensor<?x?x6144xf16>
  %eh31 = tensor.empty(%b31, %n31) : tensor<?x?x6144xf16>
  %h31 = hip.sigmoid(%ctx) ins(%h30 : tensor<?x?x6144xf16>) outs(%eh31 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b32 = tensor.dim %h31, %c0 : tensor<?x?x6144xf16>
  %n32 = tensor.dim %h31, %c1 : tensor<?x?x6144xf16>
  %eh32 = tensor.empty(%b32, %n32) : tensor<?x?x6144xf16>
  %h32 = hip.sigmoid(%ctx) ins(%h31 : tensor<?x?x6144xf16>) outs(%eh32 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b33 = tensor.dim %h32, %c0 : tensor<?x?x6144xf16>
  %n33 = tensor.dim %h32, %c1 : tensor<?x?x6144xf16>
  %eh33 = tensor.empty(%b33, %n33) : tensor<?x?x6144xf16>
  %h33 = hip.sigmoid(%ctx) ins(%h32 : tensor<?x?x6144xf16>) outs(%eh33 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b34 = tensor.dim %h33, %c0 : tensor<?x?x6144xf16>
  %n34 = tensor.dim %h33, %c1 : tensor<?x?x6144xf16>
  %eh34 = tensor.empty(%b34, %n34) : tensor<?x?x6144xf16>
  %h34 = hip.sigmoid(%ctx) ins(%h33 : tensor<?x?x6144xf16>) outs(%eh34 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b35 = tensor.dim %h34, %c0 : tensor<?x?x6144xf16>
  %n35 = tensor.dim %h34, %c1 : tensor<?x?x6144xf16>
  %eh35 = tensor.empty(%b35, %n35) : tensor<?x?x6144xf16>
  %h35 = hip.sigmoid(%ctx) ins(%h34 : tensor<?x?x6144xf16>) outs(%eh35 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b36 = tensor.dim %h35, %c0 : tensor<?x?x6144xf16>
  %n36 = tensor.dim %h35, %c1 : tensor<?x?x6144xf16>
  %eh36 = tensor.empty(%b36, %n36) : tensor<?x?x6144xf16>
  %h36 = hip.sigmoid(%ctx) ins(%h35 : tensor<?x?x6144xf16>) outs(%eh36 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b37 = tensor.dim %h36, %c0 : tensor<?x?x6144xf16>
  %n37 = tensor.dim %h36, %c1 : tensor<?x?x6144xf16>
  %eh37 = tensor.empty(%b37, %n37) : tensor<?x?x6144xf16>
  %h37 = hip.sigmoid(%ctx) ins(%h36 : tensor<?x?x6144xf16>) outs(%eh37 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b38 = tensor.dim %h37, %c0 : tensor<?x?x6144xf16>
  %n38 = tensor.dim %h37, %c1 : tensor<?x?x6144xf16>
  %eh38 = tensor.empty(%b38, %n38) : tensor<?x?x6144xf16>
  %h38 = hip.sigmoid(%ctx) ins(%h37 : tensor<?x?x6144xf16>) outs(%eh38 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %b39 = tensor.dim %h38, %c0 : tensor<?x?x6144xf16>
  %n39 = tensor.dim %h38, %c1 : tensor<?x?x6144xf16>
  %eh39 = tensor.empty(%b39, %n39) : tensor<?x?x6144xf16>
  %h39 = hip.sigmoid(%ctx) ins(%h38 : tensor<?x?x6144xf16>) outs(%eh39 : tensor<?x?x6144xf16>) : tensor<?x?x6144xf16>
  %g = tensor.cast %h39 : tensor<?x?x6144xf16> to tensor<?x?x6144xf16>
  %b = tensor.dim %h39, %c0 : tensor<?x?x6144xf16>
  %n = tensor.dim %h39, %c1 : tensor<?x?x6144xf16>
  %ni = arith.index_cast %n : index to i64
  %nr = arith.index_cast %ni : i64 to index
  %slice = tensor.extract_slice %h39[0, 0, 0] [%b, %nr, 6144] [1, 1, 1] : tensor<?x?x6144xf16> to tensor<?x?x6144xf16>
  %expanded = tensor.expand_shape %slice [[0], [1], [2, 3]] output_shape [%b, %nr, 48, 128] : tensor<?x?x6144xf16> into tensor<?x?x48x128xf16>
  %er = tensor.empty(%b, %nr) : tensor<?x?x48x128xf16>
  %r = hip.sigmoid(%ctx) ins(%expanded : tensor<?x?x48x128xf16>) outs(%er : tensor<?x?x48x128xf16>) : tensor<?x?x48x128xf16>
  %u = tensor.collapse_shape %r [[0], [1], [2, 3]] : tensor<?x?x48x128xf16> into tensor<?x?x6144xf16>
  %eg = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %gf = hip.cast(%ctx) ins(%g : tensor<?x?x6144xf16>) outs(%eg : tensor<?x?x6144xf32>) {to = 1 : i64} : tensor<?x?x6144xf32>
  %eu = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %uf = hip.cast(%ctx) ins(%u : tensor<?x?x6144xf16>) outs(%eu : tensor<?x?x6144xf32>) {to = 1 : i64} : tensor<?x?x6144xf32>
  %es = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %s = hip.sigmoid(%ctx) ins(%gf : tensor<?x?x6144xf32>) outs(%es : tensor<?x?x6144xf32>) : tensor<?x?x6144xf32>
  %ea = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %a = hip.mul(%ctx) ins(%gf, %s : tensor<?x?x6144xf32>, tensor<?x?x6144xf32>) outs(%ea : tensor<?x?x6144xf32>) -> tensor<?x?x6144xf32>
  %ey = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %y = hip.mul(%ctx) ins(%a, %uf : tensor<?x?x6144xf32>, tensor<?x?x6144xf32>) outs(%ey : tensor<?x?x6144xf32>) -> tensor<?x?x6144xf32>
  %eo = tensor.empty(%b, %n) : tensor<?x?x6144xf16>
  %o = hip.cast(%ctx) ins(%y : tensor<?x?x6144xf32>) outs(%eo : tensor<?x?x6144xf16>) {to = 10 : i64} : tensor<?x?x6144xf16>
  return %o : tensor<?x?x6144xf16>
}

// -----

// CHECK-LABEL: func.func @reject_unknown_slice_extent
// CHECK: hip.sigmoid
// CHECK: hip.mul
// CHECK-NOT: hip.swiglu
func.func @reject_unknown_slice_extent(%ctx: !hip.context, %g: tensor<?x?x6144xf16>) -> tensor<?x?x6144xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b = tensor.dim %g, %c0 : tensor<?x?x6144xf16>
  %n = tensor.dim %g, %c1 : tensor<?x?x6144xf16>
  %larger = arith.addi %n, %c1 : index
  %u = tensor.extract_slice %g[0, 0, 0] [%b, %larger, 6144] [1, 1, 1] : tensor<?x?x6144xf16> to tensor<?x?x6144xf16>
  %eg = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %gf = hip.cast(%ctx) ins(%g : tensor<?x?x6144xf16>) outs(%eg : tensor<?x?x6144xf32>) {to = 1 : i64} : tensor<?x?x6144xf32>
  %eu = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %uf = hip.cast(%ctx) ins(%u : tensor<?x?x6144xf16>) outs(%eu : tensor<?x?x6144xf32>) {to = 1 : i64} : tensor<?x?x6144xf32>
  %es = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %s = hip.sigmoid(%ctx) ins(%gf : tensor<?x?x6144xf32>) outs(%es : tensor<?x?x6144xf32>) : tensor<?x?x6144xf32>
  %ea = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %a = hip.mul(%ctx) ins(%gf, %s : tensor<?x?x6144xf32>, tensor<?x?x6144xf32>) outs(%ea : tensor<?x?x6144xf32>) -> tensor<?x?x6144xf32>
  %ey = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %y = hip.mul(%ctx) ins(%a, %uf : tensor<?x?x6144xf32>, tensor<?x?x6144xf32>) outs(%ey : tensor<?x?x6144xf32>) -> tensor<?x?x6144xf32>
  %eo = tensor.empty(%b, %n) : tensor<?x?x6144xf16>
  %o = hip.cast(%ctx) ins(%y : tensor<?x?x6144xf32>) outs(%eo : tensor<?x?x6144xf16>) {to = 10 : i64} : tensor<?x?x6144xf16>
  return %o : tensor<?x?x6144xf16>
}

// -----

// CHECK-LABEL: func.func @reject_merged_dynamic_extent
// CHECK: hip.mul
// CHECK-NOT: hip.swiglu
func.func @reject_merged_dynamic_extent(%ctx: !hip.context, %g: tensor<?x?x6144xf16>, %view: tensor<?x?x?x6144xf16>) -> tensor<?x?x6144xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %b = tensor.dim %g, %c0 : tensor<?x?x6144xf16>
  %n = tensor.dim %g, %c1 : tensor<?x?x6144xf16>
  %u = tensor.collapse_shape %view [[0], [1, 2], [3]] : tensor<?x?x?x6144xf16> into tensor<?x?x6144xf16>
  %eg = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %gf = hip.cast(%ctx) ins(%g : tensor<?x?x6144xf16>) outs(%eg : tensor<?x?x6144xf32>) {to = 1 : i64} : tensor<?x?x6144xf32>
  %eu = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %uf = hip.cast(%ctx) ins(%u : tensor<?x?x6144xf16>) outs(%eu : tensor<?x?x6144xf32>) {to = 1 : i64} : tensor<?x?x6144xf32>
  %es = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %s = hip.sigmoid(%ctx) ins(%gf : tensor<?x?x6144xf32>) outs(%es : tensor<?x?x6144xf32>) : tensor<?x?x6144xf32>
  %ea = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %a = hip.mul(%ctx) ins(%gf, %s : tensor<?x?x6144xf32>, tensor<?x?x6144xf32>) outs(%ea : tensor<?x?x6144xf32>) -> tensor<?x?x6144xf32>
  %ey = tensor.empty(%b, %n) : tensor<?x?x6144xf32>
  %y = hip.mul(%ctx) ins(%a, %uf : tensor<?x?x6144xf32>, tensor<?x?x6144xf32>) outs(%ey : tensor<?x?x6144xf32>) -> tensor<?x?x6144xf32>
  %eo = tensor.empty(%b, %n) : tensor<?x?x6144xf16>
  %o = hip.cast(%ctx) ins(%y : tensor<?x?x6144xf32>) outs(%eo : tensor<?x?x6144xf16>) {to = 10 : i64} : tensor<?x?x6144xf16>
  return %o : tensor<?x?x6144xf16>
}
