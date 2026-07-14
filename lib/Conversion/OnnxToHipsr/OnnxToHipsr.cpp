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

    // The greedy driver's constant CSE would hoist index constants out of the
    // shape region to the nearest isolated ancestor (typically the enclosing
    // func) and deduplicate them.
    //
    // However, ShapeRegionInterface requires a shape region to be
    // self-contained: it may only reference the current Op's operands or values
    // defined within the region itself. This ensures the shape region remains
    // the single source of truth for shape computation and can be relocated as
    // a whole.
    //
    // Hoisting constants out of the region would violate this requirement. For
    // example, with CSE on the %c0 below is hoisted to the func entry, so the
    // region uses a value defined outside it and the verifier rejects the op:
    //
    //   func.func @f(%in: tensor<?x8xf32>) -> tensor<?x8xf16> {
    //     %c0 = arith.constant 0 : index          // hoisted out of the region
    //     %init = tensor.empty(%d) : tensor<?x8xf16>
    //     %0 = hipsr.cast ins(%in ...) outs(%init ...) -> ... shape_region {
    //       %s  = shape.shape_of %in : tensor<?x8xf32> -> tensor<2xindex>
    //       %e0 = shape.get_extent %s, %c0 : ...   // uses %c0 from outside
    //       hipsr.shape_yield %e0
    //     }
    //     return %0 : tensor<?x8xf16>
    //   }
    //
    // so enableConstantCSE is set to false.
    config.enableConstantCSE(false);
    if (::mlir::failed(::mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
