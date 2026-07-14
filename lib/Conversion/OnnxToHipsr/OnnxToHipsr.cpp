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

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERTONNXTOHIPSRPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct ConvertOnnxToHipsrPass
    : impl::ConvertOnnxToHipsrPassBase<ConvertOnnxToHipsrPass> {
  void runOnOperation() override {
    ::mlir::RewritePatternSet patterns(&getContext());
    // Per-op ONNX -> hipsr conversion patterns are registered by follow-up
    // layers (e.g. Cast) that add both the hipsr op and its pattern here.

    ::mlir::GreedyRewriteConfig config;
    config.setStrictness(::mlir::GreedyRewriteStrictness::ExistingOps);
    // Do not CSE constants across region boundaries: the index constants a
    // shaped op emits inside its shape region must stay there (the region's
    // scoping verifier only allows op operands / in-region values), so they
    // must not be hoisted to merge with identical constants in the parent.
    config.enableConstantCSE(false);
    if (::mlir::failed(::mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
