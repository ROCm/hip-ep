/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MaterializeInitTensorsPass.cpp - Materialize placeholder inits -----===//
//
// Rewrites each pool domain into the virtual 3-region form: every
// hipsr.placeholder shape region becomes an scf.execute_region yielding
// !shape.shape, every placeholder result becomes a tensor.empty built from
// that shape, and the data ops keep their order.
//
// Before:
//   %init = hipsr.placeholder(%ctx) ins(%a, %b) : tensor<?x512xf16>
//       shape_region { ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
//         hipsr.shape_yield2 %result_shape : !shape.shape }
//   %0 = hipsr.matmul(%ctx) ins(%a, %b) outs(%init) : tensor<?x512xf16>
// After:
//   %shape = scf.execute_region -> !shape.shape { ... }
//   %init = tensor.empty(%d0) : tensor<?x512xf16>
//   %0 = hipsr.matmul(%ctx) ins(%a, %b) outs(%init) : tensor<?x512xf16>
//
// The phases run per domain: collect and group the placeholders, build the
// shape regions, build the tensor allocations, then replace and erase.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_MATERIALIZEINITTENSORSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct MaterializeInitTensorsPass
    : impl::MaterializeInitTensorsPassBase<MaterializeInitTensorsPass> {
  void runOnOperation() override {
    // Each domain is materialized on its own; the per-domain phases hook in
    // here.
    getOperation().walk([](PoolDomainOp) {});
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
