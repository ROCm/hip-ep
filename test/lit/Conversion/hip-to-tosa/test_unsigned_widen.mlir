// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// An 8- or 16-bit unsigned max, min, equal or less is computed in i32:
// unsigned_cast widens both operands, and the i32 op orders them as unsigned
// values. A max or min narrows its result back with unsigned_cast; a
// comparison already gives i1. Every op then keeps its meaning when the kernel
// is relabeled to signless, so the kernel reaches rocMLIR as i8.

// RUN: hip-mlir-opt --convert-hip-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @ui8_max_scalar
// CHECK-SAME: (%[[A:.*]]: tensor<2x8xi8>) -> tensor<2x8xi8>
// CHECK-DAG: %[[AW:.*]] = tosa.custom %[[A]] {{.*}}operator_name = "unsigned_cast"{{.*}} -> tensor<2x8xi32>
// CHECK-DAG: %[[CW:.*]] = tosa.custom %{{.*}} {{.*}}operator_name = "unsigned_cast"{{.*}} -> tensor<1x1xi32>
// CHECK: %[[M:.*]] = tosa.maximum %[[AW]], %[[CW]] : (tensor<2x8xi32>, tensor<1x1xi32>) -> tensor<2x8xi32>
// CHECK: %[[N:.*]] = tosa.custom %[[M]] {{.*}}operator_name = "unsigned_cast"{{.*}} : (tensor<2x8xi32>) -> tensor<2x8xi8>
// CHECK: return %[[N]]
// CHECK-NOT: ui8
func.func @ui8_max_scalar(%a: tensor<2x8xui8>) -> tensor<2x8xui8>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %c = hip.constant {value = dense<128> : tensor<ui8>} : tensor<ui8>
  %init = tensor.empty() : tensor<2x8xui8>
  %r = hip.max(%ctx) ins(%a, %c : tensor<2x8xui8>, tensor<ui8>)
         outs(%init : tensor<2x8xui8>) : tensor<2x8xui8>
  return %r : tensor<2x8xui8>
}

// The scalar on the left, and a min.
// CHECK-LABEL: func.func @ui8_min_scalar_lhs
// CHECK-SAME: -> tensor<2x8xi8>
// CHECK: tosa.minimum %{{.*}}, %{{.*}} : (tensor<1x1xi32>, tensor<2x8xi32>) -> tensor<2x8xi32>
// CHECK: tosa.custom %{{.*}} {{.*}}operator_name = "unsigned_cast"{{.*}} : (tensor<2x8xi32>) -> tensor<2x8xi8>
// CHECK-NOT: ui8
func.func @ui8_min_scalar_lhs(%a: tensor<2x8xui8>) -> tensor<2x8xui8>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %c = hip.constant {value = dense<200> : tensor<ui8>} : tensor<ui8>
  %init = tensor.empty() : tensor<2x8xui8>
  %r = hip.min(%ctx) ins(%c, %a : tensor<ui8>, tensor<2x8xui8>)
         outs(%init : tensor<2x8xui8>) : tensor<2x8xui8>
  return %r : tensor<2x8xui8>
}

// Clip on ui8 arrives as a max then a min; both widen, and the kernel ends
// signless.
// CHECK-LABEL: func.func @ui8_clip
// CHECK-SAME: (%{{.*}}: tensor<4xi8>) -> tensor<4xi8>
// CHECK: tosa.maximum {{.*}} -> tensor<4xi32>
// CHECK: tosa.minimum {{.*}} -> tensor<4xi32>
// CHECK-NOT: ui8
func.func @ui8_clip(%a: tensor<4xui8>) -> tensor<4xui8>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %lo = hip.constant {value = dense<16> : tensor<ui8>} : tensor<ui8>
  %hi = hip.constant {value = dense<240> : tensor<ui8>} : tensor<ui8>
  %e0 = tensor.empty() : tensor<4xui8>
  %m = hip.max(%ctx) ins(%a, %lo : tensor<4xui8>, tensor<ui8>)
         outs(%e0 : tensor<4xui8>) : tensor<4xui8>
  %e1 = tensor.empty() : tensor<4xui8>
  %r = hip.min(%ctx) ins(%m, %hi : tensor<4xui8>, tensor<ui8>)
         outs(%e1 : tensor<4xui8>) : tensor<4xui8>
  return %r : tensor<4xui8>
}

// A 16-bit unsigned max widens the same way.
// CHECK-LABEL: func.func @ui16_max
// CHECK-SAME: -> tensor<4xi16>
// CHECK: tosa.maximum {{.*}} -> tensor<4xi32>
// CHECK-NOT: ui16
func.func @ui16_max(%a: tensor<4xui16>, %b: tensor<4xui16>) -> tensor<4xui16>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %init = tensor.empty() : tensor<4xui16>
  %r = hip.max(%ctx) ins(%a, %b : tensor<4xui16>, tensor<4xui16>)
         outs(%init : tensor<4xui16>) : tensor<4xui16>
  return %r : tensor<4xui16>
}

// A 32-bit unsigned max would need i64, which rocMLIR kernels do not take:
// it stays a plain tosa.maximum, and the kernel keeps ui32.
// CHECK-LABEL: func.func @ui32_max_keeps_type
// CHECK-SAME: tensor<4xui32>
// CHECK: tosa.maximum {{.*}} -> tensor<4xui32>
func.func @ui32_max_keeps_type(%a: tensor<4xui32>, %b: tensor<4xui32>)
    -> tensor<4xui32> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %init = tensor.empty() : tensor<4xui32>
  %r = hip.max(%ctx) ins(%a, %b : tensor<4xui32>, tensor<4xui32>)
         outs(%init : tensor<4xui32>) : tensor<4xui32>
  return %r : tensor<4xui32>
}

// A signless max is unchanged.
// CHECK-LABEL: func.func @i8_max_unchanged
// CHECK: tosa.maximum {{.*}} : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
// CHECK-NOT: unsigned_cast
func.func @i8_max_unchanged(%a: tensor<4xi8>, %b: tensor<4xi8>) -> tensor<4xi8>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %init = tensor.empty() : tensor<4xi8>
  %r = hip.max(%ctx) ins(%a, %b : tensor<4xi8>, tensor<4xi8>)
         outs(%init : tensor<4xi8>) : tensor<4xi8>
  return %r : tensor<4xi8>
}

// A ui8 less compares in i32 with its operands swapped, so 200 < 100 stays
// false.
// CHECK-LABEL: func.func @ui8_less
// CHECK-SAME: (%[[A:.*]]: tensor<2x8xi8>) -> tensor<2x8xi1>
// CHECK-DAG: %[[AW:.*]] = tosa.custom %[[A]] {{.*}}operator_name = "unsigned_cast"{{.*}} -> tensor<2x8xi32>
// CHECK-DAG: %[[CW:.*]] = tosa.custom %{{.*}} {{.*}}operator_name = "unsigned_cast"{{.*}} -> tensor<1x1xi32>
// CHECK: %[[L:.*]] = tosa.greater %[[CW]], %[[AW]] : (tensor<1x1xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
// CHECK: return %[[L]]
// CHECK-NOT: ui8
func.func @ui8_less(%a: tensor<2x8xui8>) -> tensor<2x8xi1>
    attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %c = hip.constant {value = dense<100> : tensor<ui8>} : tensor<ui8>
  %init = tensor.empty() : tensor<2x8xi1>
  %r = hip.less(%ctx) ins(%a, %c : tensor<2x8xui8>, tensor<ui8>)
         outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}

// CHECK-LABEL: func.func @ui16_equal
// CHECK-SAME: (%[[A:.*]]: tensor<2x8xi16>, %[[B:.*]]: tensor<2x8xi16>) -> tensor<2x8xi1>
// CHECK-DAG: %[[AW:.*]] = tosa.custom %[[A]] {{.*}}operator_name = "unsigned_cast"{{.*}} -> tensor<2x8xi32>
// CHECK-DAG: %[[BW:.*]] = tosa.custom %[[B]] {{.*}}operator_name = "unsigned_cast"{{.*}} -> tensor<2x8xi32>
// CHECK: %[[E:.*]] = tosa.equal %[[AW]], %[[BW]] : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
// CHECK: return %[[E]]
func.func @ui16_equal(%a: tensor<2x8xui16>, %b: tensor<2x8xui16>)
    -> tensor<2x8xi1> attributes {rock.kernel} {
  %ctx = ub.poison : !hip.context
  %init = tensor.empty() : tensor<2x8xi1>
  %r = hip.equal(%ctx) ins(%a, %b : tensor<2x8xui16>, tensor<2x8xui16>)
         outs(%init : tensor<2x8xi1>) : tensor<2x8xi1>
  return %r : tensor<2x8xi1>
}
