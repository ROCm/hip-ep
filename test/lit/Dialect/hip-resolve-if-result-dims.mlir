// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// `--hip-resolve-if-result-dims` on the post-conversion spelling of decoder
// head reshapes: a size read back from a shape tensor built as
// where(equal(target, 0), shape, target) (Reshape's "0 copies the input dim"),
// and a -1 entry inferred by division. Both must become caller-side host
// arithmetic on the captured dims, with no readback, and the division must be
// guarded because the untaken branch's size is computed too.

// RUN: hip-mlir-opt --hip-resolve-if-result-dims --cse --split-input-file %s | FileCheck %s

module {
  func.func @main_graph(%ctx: !hip.context, %c: i1, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @then_n0 else @else_n1 {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    %r = hip.if(%ctx, %c) outs(%init : tensor<?x8x64xf32>)
        captures(%x : tensor<?x?xf32>) -> (tensor<?x8x64xf32>)
        then @then_n0 else @else_n1 {num_outputs = 1 : i32}
    return %r : tensor<?x8x64xf32>
  }

  func.func private @then_n0(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?x?xf32>
    %n64 = arith.index_cast %n : index to i64
    %c8 = arith.constant 8 : i64
    %c64 = arith.constant 64 : i64
    %shape = tensor.from_elements %n64, %c8, %c64 : tensor<3xi64>
    %target = arith.constant dense<[0, 8, 64]> : tensor<3xi64>
    %zeros = arith.constant dense<0> : tensor<3xi64>
    %eq_init = tensor.empty() : tensor<3xui8>
    %eq = hip.equal(%ctx) ins(%target, %zeros : tensor<3xi64>, tensor<3xi64>)
        outs(%eq_init : tensor<3xui8>) : tensor<3xui8>
    %w_init = tensor.empty() : tensor<3xi64>
    %w = hip.where(%ctx) ins(%eq, %shape, %target
        : tensor<3xui8>, tensor<3xi64>, tensor<3xi64>)
        outs(%w_init : tensor<3xi64>) : tensor<3xi64>
    %first = tensor.extract_slice %w[0] [1] [1] : tensor<3xi64> to tensor<1xi64>
    %scalar = tensor.collapse_shape %first [] : tensor<1xi64> into tensor<i64>
    %rows = hip.readback_scalar(%ctx, %scalar : tensor<i64>) -> i64
    %rows_idx = arith.index_cast %rows : i64 to index
    %y = tensor.reshape %x(%w)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    %out_init = tensor.empty(%rows_idx) : tensor<?x8x64xf32>
    %out = hip.neg(%ctx) ins(%y : tensor<?x8x64xf32>)
        outs(%out_init : tensor<?x8x64xf32>) : tensor<?x8x64xf32>
    return %out : tensor<?x8x64xf32>
  }

  // -1 inference: rows = (n * m) / m with a run-time divisor.
  func.func private @else_n1(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %x, %c0 : tensor<?x?xf32>
    %m = tensor.dim %x, %c1 : tensor<?x?xf32>
    %total = arith.muli %n, %m : index
    %rows = arith.divsi %total, %m : index
    %rows_i64 = arith.index_cast %rows : index to i64
    %c8 = arith.constant 8 : i64
    %c64 = arith.constant 64 : i64
    %shape = tensor.from_elements %rows_i64, %c8, %c64 : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}

// CHECK-LABEL: func.func @main_graph(
// CHECK-SAME: %{{.*}}: !hip.context, %[[COND:.*]]: i1, %[[X:.*]]: tensor<?x?xf32>)
// CHECK-NOT: hip.readback_scalar
// CHECK: %[[N:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
// CHECK: %[[THEN:.*]] = arith.select %{{.*}}, %[[N]], %{{.*}} : index
// CHECK: %[[M:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
// CHECK: %[[TOTAL:.*]] = arith.muli %[[N]], %[[M]] : index
// CHECK: %[[IS_ZERO:.*]] = arith.cmpi eq, %[[M]], %{{.*}} : index
// CHECK: %[[IS_NEG1:.*]] = arith.cmpi eq, %[[M]], %{{.*}} : index
// CHECK: %[[UNSAFE:.*]] = arith.ori %[[IS_ZERO]], %[[IS_NEG1]] : i1
// CHECK: %[[NEG:.*]] = arith.subi %{{.*}}, %[[TOTAL]] : index
// CHECK: %[[DIVISOR:.*]] = arith.select %[[UNSAFE]], %{{.*}}, %[[M]] : index
// CHECK: %[[QUOT:.*]] = arith.divsi %[[TOTAL]], %[[DIVISOR]] : index
// CHECK: %[[ELSE:.*]] = arith.select %[[IS_NEG1]], %[[NEG]], %[[QUOT]] : index
// CHECK: %[[SIZE:.*]] = arith.select %[[COND]], %{{.*}}, %[[ELSE]] : index
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?x8x64xf32>
// CHECK-NOT: hip.if_result_dim
// CHECK: hip.if
// CHECK-LABEL: func.func private @then_n0
// CHECK: hip.readback_scalar

// -----

// A shape entry narrowed to i32 before it reaches the shape tensor keeps the
// truncation: the caller re-extends the narrowed value instead of reusing the
// index.
module {
  func.func @narrow_from_elements(%ctx: !hip.context, %c: i1,
                                  %x: tensor<?x?xf32>) -> tensor<?x8x64xf32> {
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @narrow_then else @narrow_else {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    %r = hip.if(%ctx, %c) outs(%init : tensor<?x8x64xf32>)
        captures(%x : tensor<?x?xf32>) -> (tensor<?x8x64xf32>)
        then @narrow_then else @narrow_else {num_outputs = 1 : i32}
    return %r : tensor<?x8x64xf32>
  }

  func.func private @narrow_then(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?x?xf32>
    %n32 = arith.index_cast %n : index to i32
    %c8 = arith.constant 8 : i32
    %c64 = arith.constant 64 : i32
    %shape = tensor.from_elements %n32, %c8, %c64 : tensor<3xi32>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }

  func.func private @narrow_else(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape = arith.constant dense<[4, 8, 64]> : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}

// CHECK-LABEL: func.func @narrow_from_elements(
// CHECK-SAME: %{{.*}}: !hip.context, %[[COND:.*]]: i1, %[[X:.*]]: tensor<?x?xf32>)
// CHECK: %[[N:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
// CHECK: %[[N32:.*]] = arith.index_cast %[[N]] : index to i32
// CHECK: %[[THEN:.*]] = arith.index_cast %[[N32]] : i32 to index
// CHECK: %[[SIZE:.*]] = arith.select %[[COND]], %[[THEN]], %{{.*}} : index
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?x8x64xf32>

// -----

// hip.cast to i32 truncates and sign-extends, rather than passing the i64
// value through.
module {
  func.func @narrow_hip_cast(%ctx: !hip.context, %c: i1,
                             %x: tensor<?x?xf32>) -> tensor<?x8x64xf32> {
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @cast_then else @cast_else {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    %r = hip.if(%ctx, %c) outs(%init : tensor<?x8x64xf32>)
        captures(%x : tensor<?x?xf32>) -> (tensor<?x8x64xf32>)
        then @cast_then else @cast_else {num_outputs = 1 : i32}
    return %r : tensor<?x8x64xf32>
  }

  func.func private @cast_then(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?x?xf32>
    %n64 = arith.index_cast %n : index to i64
    %c8 = arith.constant 8 : i64
    %c64 = arith.constant 64 : i64
    %shape = tensor.from_elements %n64, %c8, %c64 : tensor<3xi64>
    %cast_init = tensor.empty() : tensor<3xi32>
    %shape32 = hip.cast(%ctx) ins(%shape : tensor<3xi64>)
        outs(%cast_init : tensor<3xi32>) {to = 6 : i64} : tensor<3xi32>
    %y = tensor.reshape %x(%shape32)
        : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }

  func.func private @cast_else(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape = arith.constant dense<[4, 8, 64]> : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}

// CHECK-LABEL: func.func @narrow_hip_cast(
// CHECK-SAME: %{{.*}}: !hip.context, %[[COND:.*]]: i1, %[[X:.*]]: tensor<?x?xf32>)
// CHECK: %[[N:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
// CHECK: %[[N32:.*]] = arith.index_cast %[[N]] : index to i32
// CHECK: %[[THEN:.*]] = arith.index_cast %[[N32]] : i32 to index
// CHECK: %[[SIZE:.*]] = arith.select %[[COND]], %[[THEN]], %{{.*}} : index
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?x8x64xf32>
// CHECK-NOT: hip.if_result_dim
// CHECK: hip.if

// -----

// hip.div on a shape tensor: the untaken branch's divisor may be zero (an
// empty input), and its size is computed anyway, so the divisor is guarded.
module {
  func.func @guarded_hip_div(%ctx: !hip.context, %c: i1,
                             %x: tensor<?x?xf32>) -> tensor<?x8x64xf32> {
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @div_then else @div_else {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    %r = hip.if(%ctx, %c) outs(%init : tensor<?x8x64xf32>)
        captures(%x : tensor<?x?xf32>) -> (tensor<?x8x64xf32>)
        then @div_then else @div_else {num_outputs = 1 : i32}
    return %r : tensor<?x8x64xf32>
  }

  func.func private @div_then(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %n = tensor.dim %x, %c0 : tensor<?x?xf32>
    %m = tensor.dim %x, %c1 : tensor<?x?xf32>
    %n64 = arith.index_cast %n : index to i64
    %m64 = arith.index_cast %m : index to i64
    %one = arith.constant 1 : i64
    %c8 = arith.constant 8 : i64
    %c64 = arith.constant 64 : i64
    %num = tensor.from_elements %n64, %c8, %c64 : tensor<3xi64>
    %den = tensor.from_elements %m64, %one, %one : tensor<3xi64>
    %q_init = tensor.empty() : tensor<3xi64>
    %q = hip.div(%ctx) ins(%num, %den : tensor<3xi64>, tensor<3xi64>)
        outs(%q_init : tensor<3xi64>) : tensor<3xi64>
    %y = tensor.reshape %x(%q)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }

  func.func private @div_else(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape = arith.constant dense<[4, 8, 64]> : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}

// CHECK-LABEL: func.func @guarded_hip_div(
// CHECK-SAME: %{{.*}}: !hip.context, %[[COND:.*]]: i1, %[[X:.*]]: tensor<?x?xf32>)
// CHECK-DAG: %[[N:.*]] = tensor.dim %[[X]], %c0
// CHECK-DAG: %[[M:.*]] = tensor.dim %[[X]], %c1
// CHECK: %[[IS_ZERO:.*]] = arith.cmpi eq, %[[M]], %{{.*}} : index
// CHECK: %[[IS_NEG1:.*]] = arith.cmpi eq, %[[M]], %{{.*}} : index
// CHECK: %[[UNSAFE:.*]] = arith.ori %[[IS_ZERO]], %[[IS_NEG1]] : i1
// CHECK: %[[DIVISOR:.*]] = arith.select %[[UNSAFE]], %{{.*}}, %[[M]] : index
// CHECK: %[[QUOT:.*]] = arith.divsi %[[N]], %[[DIVISOR]] : index
// CHECK: %[[NEG:.*]] = arith.subi %{{.*}}, %[[N]] : index
// CHECK: %[[THEN:.*]] = arith.select %[[IS_NEG1]], %[[NEG]], %[[QUOT]] : index
// CHECK: %[[SIZE:.*]] = arith.select %[[COND]], %[[THEN]], %{{.*}} : index
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?x8x64xf32>

// -----

// i32 shape arithmetic that overflows before the cast to i64: the branch
// computes Mul(n, n) + 1 in 32 bits, so each step wraps to i32 in the caller
// too rather than evaluating at index width.
module {
  func.func @narrow_overflow_before_widening(%ctx: !hip.context, %c: i1,
      %x: tensor<?x?xf32>) -> tensor<?x8x64xf32> {
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @ovf_then else @ovf_else {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    %r = hip.if(%ctx, %c) outs(%init : tensor<?x8x64xf32>)
        captures(%x : tensor<?x?xf32>) -> (tensor<?x8x64xf32>)
        then @ovf_then else @ovf_else {num_outputs = 1 : i32}
    return %r : tensor<?x8x64xf32>
  }

  func.func private @ovf_then(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?x?xf32>
    %n64 = arith.index_cast %n : index to i64
    %c8 = arith.constant 8 : i64
    %c64 = arith.constant 64 : i64
    %shape = tensor.from_elements %n64, %c8, %c64 : tensor<3xi64>
    %i32_init = tensor.empty() : tensor<3xi32>
    %s32 = hip.cast(%ctx) ins(%shape : tensor<3xi64>)
        outs(%i32_init : tensor<3xi32>) {to = 6 : i64} : tensor<3xi32>
    %sq = hip.mul(%ctx) ins(%s32, %s32 : tensor<3xi32>, tensor<3xi32>)
        outs(%i32_init : tensor<3xi32>) -> tensor<3xi32>
    %one = arith.constant dense<1> : tensor<3xi32>
    %sum = hip.add(%ctx) ins(%sq, %one : tensor<3xi32>, tensor<3xi32>)
        outs(%i32_init : tensor<3xi32>) -> tensor<3xi32>
    %i64_init = tensor.empty() : tensor<3xi64>
    %wide = hip.cast(%ctx) ins(%sum : tensor<3xi32>)
        outs(%i64_init : tensor<3xi64>) {to = 7 : i64} : tensor<3xi64>
    %y = tensor.reshape %x(%wide)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }

  func.func private @ovf_else(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape = arith.constant dense<[4, 8, 64]> : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}

// CHECK-LABEL: func.func @narrow_overflow_before_widening(
// CHECK-SAME: %{{.*}}: !hip.context, %[[COND:.*]]: i1, %[[X:.*]]: tensor<?x?xf32>)
// CHECK: %[[N:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
// CHECK: %[[N32:.*]] = arith.index_cast %[[N]] : index to i32
// CHECK: %[[NW:.*]] = arith.index_cast %[[N32]] : i32 to index
// CHECK: %[[SQ:.*]] = arith.muli %[[NW]], %[[NW]] : index
// CHECK: %[[SQ32:.*]] = arith.index_cast %[[SQ]] : index to i32
// CHECK: %[[SQW:.*]] = arith.index_cast %[[SQ32]] : i32 to index
// CHECK: %[[SUM:.*]] = arith.addi %[[SQW]], %{{.*}} : index
// CHECK: %[[SUM32:.*]] = arith.index_cast %[[SUM]] : index to i32
// CHECK: %[[THEN:.*]] = arith.index_cast %[[SUM32]] : i32 to index
// CHECK: %[[SIZE:.*]] = arith.select %[[COND]], %[[THEN]], %{{.*}} : index
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?x8x64xf32>

// -----

// An unsigned shape constant is zero-extended: ui8 255 cast to i64 is 255,
// not the -1 its bit pattern reads as when sign-extended.
module {
  func.func @unsigned_constant(%ctx: !hip.context, %c: i1,
                               %x: tensor<?x?xf32>) -> tensor<?x8x64xf32> {
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @u8_then else @u8_else {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    %r = hip.if(%ctx, %c) outs(%init : tensor<?x8x64xf32>)
        captures(%x : tensor<?x?xf32>) -> (tensor<?x8x64xf32>)
        then @u8_then else @u8_else {num_outputs = 1 : i32}
    return %r : tensor<?x8x64xf32>
  }

  func.func private @u8_then(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape8 = hip.constant {value = dense<[255, 8, 64]> : tensor<3xui8>}
        : tensor<3xui8>
    %i64_init = tensor.empty() : tensor<3xi64>
    %shape = hip.cast(%ctx) ins(%shape8 : tensor<3xui8>)
        outs(%i64_init : tensor<3xi64>) {to = 7 : i64} : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }

  func.func private @u8_else(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape = arith.constant dense<[4, 8, 64]> : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}

// CHECK-LABEL: func.func @unsigned_constant(
// CHECK-SAME: %{{.*}}: !hip.context, %[[COND:.*]]: i1, %{{.*}}: tensor<?x?xf32>)
// CHECK-DAG: %[[C255:.*]] = arith.constant 255 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-NOT: arith.constant -1 : index
// CHECK: %[[SIZE:.*]] = arith.select %[[COND]], %[[C255]], %[[C4]] : index
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?x8x64xf32>

// -----

// Gather(Shape(x), Cast(0)): the constant index reaches the Gather through a
// hip.cast to i32. The rebuilt cast folds, so the index is still a constant
// and the size resolves to dim 0 of x.
module {
  func.func @casted_gather_index(%ctx: !hip.context, %c: i1,
                                 %x: tensor<?x?xf32>) -> tensor<?x8x64xf32> {
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @gather_then else @gather_else {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    %r = hip.if(%ctx, %c) outs(%init : tensor<?x8x64xf32>)
        captures(%x : tensor<?x?xf32>) -> (tensor<?x8x64xf32>)
        then @gather_then else @gather_else {num_outputs = 1 : i32}
    return %r : tensor<?x8x64xf32>
  }

  func.func private @gather_then(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?x?xf32>
    %n64 = arith.index_cast %n : index to i64
    %c8 = arith.constant 8 : i64
    %c64 = arith.constant 64 : i64
    %shape = tensor.from_elements %n64, %c8, %c64 : tensor<3xi64>
    %idx64 = arith.constant dense<0> : tensor<1xi64>
    %idx_init = tensor.empty() : tensor<1xi32>
    %idx = hip.cast(%ctx) ins(%idx64 : tensor<1xi64>)
        outs(%idx_init : tensor<1xi32>) {to = 6 : i64} : tensor<1xi32>
    %rows_init = tensor.empty() : tensor<1xi64>
    %rows = hip.gather(%ctx) ins(%shape, %idx : tensor<3xi64>, tensor<1xi32>)
        outs(%rows_init : tensor<1xi64>) {axis = 0 : i64} : tensor<1xi64>
    %base = arith.constant dense<[1, 8, 64]> : tensor<3xi64>
    %target = tensor.insert_slice %rows into %base[0] [1] [1]
        : tensor<1xi64> into tensor<3xi64>
    %y = tensor.reshape %x(%target)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }

  func.func private @gather_else(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape = arith.constant dense<[4, 8, 64]> : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}

// CHECK-LABEL: func.func @casted_gather_index(
// CHECK-SAME: %{{.*}}: !hip.context, %[[COND:.*]]: i1, %[[X:.*]]: tensor<?x?xf32>)
// CHECK: %[[N:.*]] = tensor.dim %[[X]], %{{.*}} : tensor<?x?xf32>
// CHECK: %[[SIZE:.*]] = arith.select %[[COND]], %[[N]], %{{.*}} : index
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?x8x64xf32>
// CHECK-NOT: hip.if_result_dim
