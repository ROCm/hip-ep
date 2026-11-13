// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-fusion-transform --split-input-file %s | FileCheck %s

// A GELU written out as primitive HIP ops becomes one hip.gelu, whichever
// frontend produced the chain: the scalar constants may be hip.constant,
// arith.constant or tensor.splat, x^3 may be hip.pow or x*x*x, and every
// multiply and add commutes. Chains that are not exactly GELU, or whose
// temporaries have another user, stay as they are.

// CHECK-LABEL: func.func @tanh_hip_const
// CHECK-NOT: hip.tanh
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) {approximate = "tanh"}
func.func @tanh_hip_const(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %k3 = hip.constant {value = dense<0.044715> : tensor<f32>} : tensor<f32>
  %ks = hip.constant {value = dense<0.7978845608> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x2 = tensor.empty() : tensor<8x16xf32>
  %x2 = hip.mul(%ctx) ins(%x, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x2 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x3 = tensor.empty() : tensor<8x16xf32>
  %x3 = hip.mul(%ctx) ins(%x2, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_c = tensor.empty() : tensor<8x16xf32>
  %c = hip.mul(%ctx) ins(%x3, %k3 : tensor<8x16xf32>, tensor<f32>) outs(%e_c : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_s = tensor.empty() : tensor<8x16xf32>
  %s = hip.add(%ctx) ins(%x, %c : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_s : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_t = tensor.empty() : tensor<8x16xf32>
  %t = hip.mul(%ctx) ins(%s, %ks : tensor<8x16xf32>, tensor<f32>) outs(%e_t : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_th = tensor.empty() : tensor<8x16xf32>
  %th = hip.tanh(%ctx) ins(%t : tensor<8x16xf32>) outs(%e_th : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%th, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @tanh_arith_const
// CHECK-NOT: hip.tanh
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) {approximate = "tanh"}
func.func @tanh_arith_const(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = arith.constant dense<0.5> : tensor<f32>
  %one = arith.constant dense<1.0> : tensor<f32>
  %k3 = arith.constant dense<0.044715> : tensor<f32>
  %ks = arith.constant dense<0.7978845608> : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x2 = tensor.empty() : tensor<8x16xf32>
  %x2 = hip.mul(%ctx) ins(%x, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x2 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x3 = tensor.empty() : tensor<8x16xf32>
  %x3 = hip.mul(%ctx) ins(%x2, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_c = tensor.empty() : tensor<8x16xf32>
  %c = hip.mul(%ctx) ins(%x3, %k3 : tensor<8x16xf32>, tensor<f32>) outs(%e_c : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_s = tensor.empty() : tensor<8x16xf32>
  %s = hip.add(%ctx) ins(%x, %c : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_s : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_t = tensor.empty() : tensor<8x16xf32>
  %t = hip.mul(%ctx) ins(%s, %ks : tensor<8x16xf32>, tensor<f32>) outs(%e_t : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_th = tensor.empty() : tensor<8x16xf32>
  %th = hip.tanh(%ctx) ins(%t : tensor<8x16xf32>) outs(%e_th : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%th, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @tanh_splat_const
// CHECK-NOT: hip.tanh
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) {approximate = "tanh"}
func.func @tanh_splat_const(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half_s = arith.constant 0.5 : f32
  %half = tensor.splat %half_s : tensor<f32>
  %one_s = arith.constant 1.0 : f32
  %one = tensor.splat %one_s : tensor<f32>
  %k3_s = arith.constant 0.044715 : f32
  %k3 = tensor.splat %k3_s : tensor<f32>
  %ks_s = arith.constant 0.7978845608 : f32
  %ks = tensor.splat %ks_s : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x2 = tensor.empty() : tensor<8x16xf32>
  %x2 = hip.mul(%ctx) ins(%x, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x2 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x3 = tensor.empty() : tensor<8x16xf32>
  %x3 = hip.mul(%ctx) ins(%x2, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_c = tensor.empty() : tensor<8x16xf32>
  %c = hip.mul(%ctx) ins(%x3, %k3 : tensor<8x16xf32>, tensor<f32>) outs(%e_c : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_s = tensor.empty() : tensor<8x16xf32>
  %s = hip.add(%ctx) ins(%x, %c : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_s : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_t = tensor.empty() : tensor<8x16xf32>
  %t = hip.mul(%ctx) ins(%s, %ks : tensor<8x16xf32>, tensor<f32>) outs(%e_t : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_th = tensor.empty() : tensor<8x16xf32>
  %th = hip.tanh(%ctx) ins(%t : tensor<8x16xf32>) outs(%e_th : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%th, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @tanh_truncf_splat_const
// CHECK-NOT: hip.tanh
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) {approximate = "tanh"}
func.func @tanh_truncf_splat_const(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half_d = arith.constant 0.5 : f64
  %half_s = arith.truncf %half_d : f64 to f32
  %half = tensor.splat %half_s : tensor<f32>
  %one_d = arith.constant 1.0 : f64
  %one_s = arith.truncf %one_d : f64 to f32
  %one = tensor.splat %one_s : tensor<f32>
  %k3_d = arith.constant 0.044715 : f64
  %k3_s = arith.truncf %k3_d : f64 to f32
  %k3 = tensor.splat %k3_s : tensor<f32>
  %ks_d = arith.constant 0.7978845608 : f64
  %ks_s = arith.truncf %ks_d : f64 to f32
  %ks = tensor.splat %ks_s : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x2 = tensor.empty() : tensor<8x16xf32>
  %x2 = hip.mul(%ctx) ins(%x, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x2 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x3 = tensor.empty() : tensor<8x16xf32>
  %x3 = hip.mul(%ctx) ins(%x2, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_c = tensor.empty() : tensor<8x16xf32>
  %c = hip.mul(%ctx) ins(%x3, %k3 : tensor<8x16xf32>, tensor<f32>) outs(%e_c : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_s = tensor.empty() : tensor<8x16xf32>
  %s = hip.add(%ctx) ins(%x, %c : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_s : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_t = tensor.empty() : tensor<8x16xf32>
  %t = hip.mul(%ctx) ins(%s, %ks : tensor<8x16xf32>, tensor<f32>) outs(%e_t : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_th = tensor.empty() : tensor<8x16xf32>
  %th = hip.tanh(%ctx) ins(%t : tensor<8x16xf32>) outs(%e_th : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%th, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @tanh_swapped_operands
// CHECK-NOT: hip.tanh
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) {approximate = "tanh"}
func.func @tanh_swapped_operands(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %k3 = hip.constant {value = dense<0.044715> : tensor<f32>} : tensor<f32>
  %ks = hip.constant {value = dense<0.7978845608> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%half, %x : tensor<f32>, tensor<8x16xf32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x2 = tensor.empty() : tensor<8x16xf32>
  %x2 = hip.mul(%ctx) ins(%x, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x2 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x3 = tensor.empty() : tensor<8x16xf32>
  %x3 = hip.mul(%ctx) ins(%x, %x2 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_c = tensor.empty() : tensor<8x16xf32>
  %c = hip.mul(%ctx) ins(%x3, %k3 : tensor<8x16xf32>, tensor<f32>) outs(%e_c : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_s = tensor.empty() : tensor<8x16xf32>
  %s = hip.add(%ctx) ins(%c, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_s : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_t = tensor.empty() : tensor<8x16xf32>
  %t = hip.mul(%ctx) ins(%s, %ks : tensor<8x16xf32>, tensor<f32>) outs(%e_t : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_th = tensor.empty() : tensor<8x16xf32>
  %th = hip.tanh(%ctx) ins(%t : tensor<8x16xf32>) outs(%e_th : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%one, %th : tensor<f32>, tensor<8x16xf32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%p, %h : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @tanh_pow_cube
// CHECK-NOT: hip.tanh
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) {approximate = "tanh"}
func.func @tanh_pow_cube(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %k3 = hip.constant {value = dense<0.044715> : tensor<f32>} : tensor<f32>
  %ks = hip.constant {value = dense<0.7978845608> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x3 = tensor.empty() : tensor<8x16xf32>
  %x3 = hip.pow(%ctx) ins(%x : tensor<8x16xf32>) outs(%e_x3 : tensor<8x16xf32>) {exponent = 3.0 : f64} : tensor<8x16xf32>
  %e_c = tensor.empty() : tensor<8x16xf32>
  %c = hip.mul(%ctx) ins(%x3, %k3 : tensor<8x16xf32>, tensor<f32>) outs(%e_c : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_s = tensor.empty() : tensor<8x16xf32>
  %s = hip.add(%ctx) ins(%x, %c : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_s : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_t = tensor.empty() : tensor<8x16xf32>
  %t = hip.mul(%ctx) ins(%s, %ks : tensor<8x16xf32>, tensor<f32>) outs(%e_t : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_th = tensor.empty() : tensor<8x16xf32>
  %th = hip.tanh(%ctx) ins(%t : tensor<8x16xf32>) outs(%e_th : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%th, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @tanh_f16_constants
// CHECK-NOT: hip.tanh
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) {approximate = "tanh"}
func.func @tanh_f16_constants(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %k3 = hip.constant {value = dense<0.04471> : tensor<f32>} : tensor<f32>
  %ks = hip.constant {value = dense<0.7979> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x2 = tensor.empty() : tensor<8x16xf32>
  %x2 = hip.mul(%ctx) ins(%x, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x2 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x3 = tensor.empty() : tensor<8x16xf32>
  %x3 = hip.mul(%ctx) ins(%x2, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_c = tensor.empty() : tensor<8x16xf32>
  %c = hip.mul(%ctx) ins(%x3, %k3 : tensor<8x16xf32>, tensor<f32>) outs(%e_c : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_s = tensor.empty() : tensor<8x16xf32>
  %s = hip.add(%ctx) ins(%x, %c : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_s : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_t = tensor.empty() : tensor<8x16xf32>
  %t = hip.mul(%ctx) ins(%s, %ks : tensor<8x16xf32>, tensor<f32>) outs(%e_t : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_th = tensor.empty() : tensor<8x16xf32>
  %th = hip.tanh(%ctx) ins(%t : tensor<8x16xf32>) outs(%e_th : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%th, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @not_gelu_tanh_coefficient
// CHECK-NOT: hip.gelu
// CHECK: hip.tanh
func.func @not_gelu_tanh_coefficient(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %k3 = hip.constant {value = dense<0.05> : tensor<f32>} : tensor<f32>
  %ks = hip.constant {value = dense<0.7978845608> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x2 = tensor.empty() : tensor<8x16xf32>
  %x2 = hip.mul(%ctx) ins(%x, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x2 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x3 = tensor.empty() : tensor<8x16xf32>
  %x3 = hip.mul(%ctx) ins(%x2, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_c = tensor.empty() : tensor<8x16xf32>
  %c = hip.mul(%ctx) ins(%x3, %k3 : tensor<8x16xf32>, tensor<f32>) outs(%e_c : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_s = tensor.empty() : tensor<8x16xf32>
  %s = hip.add(%ctx) ins(%x, %c : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_s : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_t = tensor.empty() : tensor<8x16xf32>
  %t = hip.mul(%ctx) ins(%s, %ks : tensor<8x16xf32>, tensor<f32>) outs(%e_t : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_th = tensor.empty() : tensor<8x16xf32>
  %th = hip.tanh(%ctx) ins(%t : tensor<8x16xf32>) outs(%e_th : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%th, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @not_gelu_tanh_shared
// CHECK-NOT: hip.gelu
// CHECK: hip.tanh
func.func @not_gelu_tanh_shared(%ctx: !hip.context, %x: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %k3 = hip.constant {value = dense<0.044715> : tensor<f32>} : tensor<f32>
  %ks = hip.constant {value = dense<0.7978845608> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x2 = tensor.empty() : tensor<8x16xf32>
  %x2 = hip.mul(%ctx) ins(%x, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x2 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_x3 = tensor.empty() : tensor<8x16xf32>
  %x3 = hip.mul(%ctx) ins(%x2, %x : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_x3 : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_c = tensor.empty() : tensor<8x16xf32>
  %c = hip.mul(%ctx) ins(%x3, %k3 : tensor<8x16xf32>, tensor<f32>) outs(%e_c : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_s = tensor.empty() : tensor<8x16xf32>
  %s = hip.add(%ctx) ins(%x, %c : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_s : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_t = tensor.empty() : tensor<8x16xf32>
  %t = hip.mul(%ctx) ins(%s, %ks : tensor<8x16xf32>, tensor<f32>) outs(%e_t : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_th = tensor.empty() : tensor<8x16xf32>
  %th = hip.tanh(%ctx) ins(%t : tensor<8x16xf32>) outs(%e_th : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%th, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y, %th : tensor<8x16xf32>, tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @erf_div
// CHECK-NOT: hip.erf
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) : tensor<8x16xf32>
// CHECK-NOT: approximate
func.func @erf_div(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %kd = hip.constant {value = dense<1.4142135624> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_d = tensor.empty() : tensor<8x16xf32>
  %d = hip.div(%ctx) ins(%x, %kd : tensor<8x16xf32>, tensor<f32>) outs(%e_d : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_er = tensor.empty() : tensor<8x16xf32>
  %er = hip.erf(%ctx) ins(%d : tensor<8x16xf32>) outs(%e_er : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%er, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @erf_mul_reciprocal
// CHECK-NOT: hip.erf
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) : tensor<8x16xf32>
// CHECK-NOT: approximate
func.func @erf_mul_reciprocal(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %kd = hip.constant {value = dense<0.7071067812> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_d = tensor.empty() : tensor<8x16xf32>
  %d = hip.mul(%ctx) ins(%x, %kd : tensor<8x16xf32>, tensor<f32>) outs(%e_d : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_er = tensor.empty() : tensor<8x16xf32>
  %er = hip.erf(%ctx) ins(%d : tensor<8x16xf32>) outs(%e_er : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%er, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @erf_arith_const
// CHECK-NOT: hip.erf
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) : tensor<8x16xf32>
// CHECK-NOT: approximate
func.func @erf_arith_const(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = arith.constant dense<0.5> : tensor<f32>
  %one = arith.constant dense<1.0> : tensor<f32>
  %kd = arith.constant dense<1.4142135624> : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_d = tensor.empty() : tensor<8x16xf32>
  %d = hip.div(%ctx) ins(%x, %kd : tensor<8x16xf32>, tensor<f32>) outs(%e_d : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_er = tensor.empty() : tensor<8x16xf32>
  %er = hip.erf(%ctx) ins(%d : tensor<8x16xf32>) outs(%e_er : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%er, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @erf_splat_const
// CHECK-NOT: hip.erf
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) : tensor<8x16xf32>
// CHECK-NOT: approximate
func.func @erf_splat_const(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half_s = arith.constant 0.5 : f32
  %half = tensor.splat %half_s : tensor<f32>
  %one_s = arith.constant 1.0 : f32
  %one = tensor.splat %one_s : tensor<f32>
  %kd_s = arith.constant 1.4142135624 : f32
  %kd = tensor.splat %kd_s : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_d = tensor.empty() : tensor<8x16xf32>
  %d = hip.div(%ctx) ins(%x, %kd : tensor<8x16xf32>, tensor<f32>) outs(%e_d : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_er = tensor.empty() : tensor<8x16xf32>
  %er = hip.erf(%ctx) ins(%d : tensor<8x16xf32>) outs(%e_er : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%er, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @erf_truncf_splat_const
// CHECK-NOT: hip.erf
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) : tensor<8x16xf32>
// CHECK-NOT: approximate
func.func @erf_truncf_splat_const(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half_d = arith.constant 0.5 : f64
  %half_s = arith.truncf %half_d : f64 to f32
  %half = tensor.splat %half_s : tensor<f32>
  %one_d = arith.constant 1.0 : f64
  %one_s = arith.truncf %one_d : f64 to f32
  %one = tensor.splat %one_s : tensor<f32>
  %kd_d = arith.constant 1.4142135624 : f64
  %kd_s = arith.truncf %kd_d : f64 to f32
  %kd = tensor.splat %kd_s : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_d = tensor.empty() : tensor<8x16xf32>
  %d = hip.div(%ctx) ins(%x, %kd : tensor<8x16xf32>, tensor<f32>) outs(%e_d : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_er = tensor.empty() : tensor<8x16xf32>
  %er = hip.erf(%ctx) ins(%d : tensor<8x16xf32>) outs(%e_er : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%er, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @erf_swapped_operands
// CHECK-NOT: hip.erf
// CHECK-NOT: hip.mul
// CHECK: hip.gelu(%{{.*}}) ins(%{{.*}} : tensor<8x16xf32>) outs(%{{.*}} : tensor<8x16xf32>) : tensor<8x16xf32>
// CHECK-NOT: approximate
func.func @erf_swapped_operands(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %kd = hip.constant {value = dense<1.4142135624> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%half, %x : tensor<f32>, tensor<8x16xf32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_d = tensor.empty() : tensor<8x16xf32>
  %d = hip.div(%ctx) ins(%x, %kd : tensor<8x16xf32>, tensor<f32>) outs(%e_d : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_er = tensor.empty() : tensor<8x16xf32>
  %er = hip.erf(%ctx) ins(%d : tensor<8x16xf32>) outs(%e_er : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%one, %er : tensor<f32>, tensor<8x16xf32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%p, %h : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @not_gelu_erf_divisor
// CHECK-NOT: hip.gelu
// CHECK: hip.erf
func.func @not_gelu_erf_divisor(%ctx: !hip.context, %x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %kd = hip.constant {value = dense<2.0> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_d = tensor.empty() : tensor<8x16xf32>
  %d = hip.div(%ctx) ins(%x, %kd : tensor<8x16xf32>, tensor<f32>) outs(%e_d : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_er = tensor.empty() : tensor<8x16xf32>
  %er = hip.erf(%ctx) ins(%d : tensor<8x16xf32>) outs(%e_er : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%er, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: func.func @not_gelu_erf_shared
// CHECK-NOT: hip.gelu
// CHECK: hip.erf
func.func @not_gelu_erf_shared(%ctx: !hip.context, %x: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  %half = hip.constant {value = dense<0.5> : tensor<f32>} : tensor<f32>
  %one = hip.constant {value = dense<1.0> : tensor<f32>} : tensor<f32>
  %kd = hip.constant {value = dense<1.4142135624> : tensor<f32>} : tensor<f32>
  %e_h = tensor.empty() : tensor<8x16xf32>
  %h = hip.mul(%ctx) ins(%x, %half : tensor<8x16xf32>, tensor<f32>) outs(%e_h : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_d = tensor.empty() : tensor<8x16xf32>
  %d = hip.div(%ctx) ins(%x, %kd : tensor<8x16xf32>, tensor<f32>) outs(%e_d : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_er = tensor.empty() : tensor<8x16xf32>
  %er = hip.erf(%ctx) ins(%d : tensor<8x16xf32>) outs(%e_er : tensor<8x16xf32>) : tensor<8x16xf32>
  %e_p = tensor.empty() : tensor<8x16xf32>
  %p = hip.add(%ctx) ins(%er, %one : tensor<8x16xf32>, tensor<f32>) outs(%e_p : tensor<8x16xf32>) -> tensor<8x16xf32>
  %e_y = tensor.empty() : tensor<8x16xf32>
  %y = hip.mul(%ctx) ins(%h, %p : tensor<8x16xf32>, tensor<8x16xf32>) outs(%e_y : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %y, %er : tensor<8x16xf32>, tensor<8x16xf32>
}
