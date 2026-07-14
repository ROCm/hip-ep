/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipsr.cpp - Convert ONNX dialect to the hipsr dialect --------===//
//
// Converts ONNX dialect IR into hipsr dialect IR (tensor DPS). ONNX ops are
// matched by name via the generic MLIR Operation API, so no onnx-mlir headers
// are required. Each shaped op fills its shape region through
// ShapeRegionInterface::populateShapeRegion(), so this pass exercises that
// single-source-of-truth path.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERTONNXTOHIPSRPASS
#include "hip/Conversion/Passes.h.inc"

namespace {

struct ConvertOnnxToHipsrPass
    : impl::ConvertOnnxToHipsrPassBase<ConvertOnnxToHipsrPass> {
  void runOnOperation() override {
    ::mlir::RewritePatternSet patterns(&getContext());
    // Per-op ONNX -> hipsr conversion patterns are registered by follow-up
    // layers (e.g. Cast) that add both the hipsr op and its pattern here.

    // Same driver/config as convert-onnx-to-hip (greedy, ExistingOps): ONNX ops
    // are matched by name and only the ops present on entry are rewritten, so
    // generated hipsr / shape-region IR is left untouched.
    ::mlir::GreedyRewriteConfig config;
    config.setStrictness(::mlir::GreedyRewriteStrictness::ExistingOps);
    // The one hipsr-specific delta vs convert-onnx-to-hip: keep constant CSE
    // off. A shaped op's populateShapeRegion emits its output-shape computation
    // (including index constants) inside the op's shape region, which is
    // deliberately NOT IsolatedFromAbove so it can reference the op's operands.
    // The greedy driver's constant CSE would hoist those constants to the
    // nearest isolated ancestor (the enclosing func) and dedup them, moving
    // them out of the region. ShapeRegionInterface's scoping verifier requires
    // the region to be self-contained (uses only op operands or in-region
    // values) so it stays relocatable as the single source of truth for shape;
    // hoisting would violate that. Body-level constant CSE is not this
    // conversion pass's job anyway -- a later -cse/-canonicalize handles it.
    config.enableConstantCSE(false);
    if (::mlir::failed(::mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
