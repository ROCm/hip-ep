/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- qdq_fusion_pass.hpp ------------------------------------------------===//
//
// PDLL fusion pass loader for hip-ep (AMDMIGraphX approach).
//
// Loads PDLL patterns compiled to MLIR bytecode and applies them to the IR.
// Based on AMDMIGraphX-Private/src/amdxcgc/pdl/amdcgc_fusion_pass.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/SourceMgr.h"

namespace hip {
namespace pdl {

// Apply PDL patterns from pdlBytecodeFile to the module.
// Returns true on success, false on failure (file not found / parse error).
inline bool applyPDLPatterns(mlir::ModuleOp module, 
                             llvm::StringRef pdlBytecodeFile) {
    if (pdlBytecodeFile.empty()) {
        return true; // no-op
    }

    mlir::MLIRContext* ctx = module.getContext();

    // Register PDL dialects if not already loaded
    ctx->loadDialect<mlir::pdl::PDLDialect>();
    ctx->loadDialect<mlir::pdl_interp::PDLInterpDialect>();

    // Load the compiled PDL bytecode (produced by mlir-pdll at build time)
    mlir::ParserConfig parseConfig(ctx);
    mlir::OwningOpRef<mlir::ModuleOp> pdlModule =
        mlir::parseSourceFile<mlir::ModuleOp>(pdlBytecodeFile, parseConfig);
    
    if (!pdlModule) {
        llvm::errs() << "Failed to load PDL bytecode from: " 
                     << pdlBytecodeFile << "\n";
        return false;
    }

    // Build pattern set from PDL module
    mlir::PDLPatternModule pdlPatterns(std::move(pdlModule));
    
    mlir::RewritePatternSet patterns(ctx);
    patterns.add(std::move(pdlPatterns));
    
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    // Apply patterns greedily to all functions
    mlir::GreedyRewriteConfig config;
    config.setMaxIterations(10); // Allow multiple fusion rounds
    
    if (mlir::failed(mlir::applyPatternsGreedily(module, frozen, config))) {
        llvm::errs() << "PDL pattern application failed\n";
        return false;
    }

    return true;
}

} // namespace pdl
} // namespace hip
