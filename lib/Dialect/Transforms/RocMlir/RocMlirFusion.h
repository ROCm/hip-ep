/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- RocMlirFusion.h ----------------------------------------------------===//
//
// Declarations for the native constraint/rewrite referenced by
// RocMlirFusion.pdll and the entry point that adds the generated pattern to a
// RewritePatternSet. Implementations live in RocMlirFusion.cpp; the .pdll file
// contains only one-line trampolines into these functions.
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::hip::rocmlir {

// Fusable anchor: matmul/conv/gemm, first operand is hip.context, all tensor
// operands statically shaped.
bool isFusableRocMlirAnchor(Operation *op);

// Walks the trailing pointwise chain of `anchor`, outlines the subgraph into a
// rock-kernel func.func, and replaces it with a hip.rocmlir dispatch.
void outlineRocMlirSubgraph(PatternRewriter &rewriter, Operation *anchor);

// Adds the generated PDLL pattern (RocMlirFusion.pdll) to `patterns`.
void populateRocMlirFusionPatterns(RewritePatternSet &patterns);

} // namespace mlir::hip::rocmlir
