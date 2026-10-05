/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- RocMlirFusion.h ----------------------------------------------------===//
//
// Declarations for the native constraints/rewrites referenced by
// RocMlirFusion.pdll and the entry point that adds the generated patterns to a
// RewritePatternSet. Implementations live in RocMlirFusion.cpp; the .pdll file
// contains only one-line trampolines into these functions.
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::hip::rocmlir {

// --- native PDL constraints (result-free; match surface only) ---------------
bool isPointwiseOp(Operation *op);
bool isPointwiseChainTerminus(Operation *op);
bool hasSinglePointwiseConsumer(Operation *op);
bool isFusableRocMlirAnchor(Operation *op);

// --- native PDL rewrites ----------------------------------------------------
void outlinePointwise(PatternRewriter &rewriter, Operation *op);
void fusePointwiseIntoConsumer(PatternRewriter &rewriter, Operation *op);
void fuseAnchorIntoConsumer(PatternRewriter &rewriter, Operation *op);

// Adds the three generated PDLL patterns (RocMlirFusion.pdll) to `patterns`.
void populateRocMlirFusionPatterns(RewritePatternSet &patterns);

} // namespace mlir::hip::rocmlir
