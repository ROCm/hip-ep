/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Transforms/Passes.h"

#include "RocMlir/RocMlirFusion.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::hip {
#define GEN_PASS_DEF_FUSEROCMLIRPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

#define DEBUG_TYPE "fuse-rocmlir"

namespace {

// The anchor+pointwise outlining is expressed as three composable PDLL
// patterns in RocMlir/RocMlirFusion.pdll; their native bodies (the func.func
// creation/editing that cannot be declarative) live in RocMlirFusion.cpp.
// mlir-pdll -x cpp compiles them into populateRocMlirFusionPatterns.
class FuseROCMlirPass : public impl::FuseROCMlirPassBase<FuseROCMlirPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph")
      return;

    RewritePatternSet patterns(&getContext());
    rocmlir::populateRocMlirFusionPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

}; // namespace mlir::hip
