/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Runtime/SchemeBindings.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_SCHEMECONVERSIONPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Generic pattern that wraps Scheme-based matching and rewriting
struct SchemePattern : public RewritePattern {
  SchemePattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Set the rewriter context so Scheme FFI can access it
    setCurrentRewriter(&rewriter, op);

    // Call Scheme pattern function which will:
    // 1. Check if the operation matches
    // 2. If yes, call FFI functions to create/replace operations
    // 3. Return true/false for success
    bool matched = callSchemePattern(op);

    // Clear the rewriter context
    clearCurrentRewriter();

    return matched ? success() : failure();
  }

private:
  // Forward declaration - will be defined after we load the Scheme script
  bool callSchemePattern(Operation *op) const;
};

struct SchemeConversionPass : public impl::SchemeConversionPassBase<SchemeConversionPass> {
  using impl::SchemeConversionPassBase<SchemeConversionPass>::SchemeConversionPassBase;

  void runOnOperation() override {
    SchemeLogLevel level = parseLogLevel(logLevel);
    if (!initializeSchemeRuntime(level)) {
      signalPassFailure();
      return;
    }

    // Load the Scheme script
    std::string modulePath = llvm::sys::fs::getMainExecutable(nullptr, (void*)&initializeSchemeRuntime);
    llvm::SmallString<256> scriptPath(modulePath);
    llvm::sys::path::remove_filename(scriptPath);

    if (llvm::sys::path::filename(scriptPath) == "bin")
      llvm::sys::path::remove_filename(scriptPath);

    llvm::sys::path::append(scriptPath, "lib", "scheme", scriptName);

    if (!loadSchemeScript(scriptPath.c_str())) {
      signalPassFailure();
      return;
    }

    // Apply greedy pattern rewriting with Scheme patterns
    RewritePatternSet patterns(&getContext());
    patterns.add<SchemePattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// Call Scheme pattern - this will be implemented properly
bool SchemePattern::callSchemePattern(Operation *op) const {
  // For now, just call the Scheme apply-pattern function
  // This requires adding a C++ → Scheme bridge function
  // TODO: Implement this properly
  return false;
}

} // namespace
} // namespace hipsr
} // namespace mlir
