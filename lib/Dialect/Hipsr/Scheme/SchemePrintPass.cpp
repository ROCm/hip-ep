/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "SchemeRuntime.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_SCHEMEPRINTPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct SchemePrintPass : public impl::SchemePrintPassBase<SchemePrintPass> {
  using impl::SchemePrintPassBase<SchemePrintPass>::SchemePrintPassBase;

  void runOnOperation() override {
    if (!initializeSchemeRuntime()) {
      signalPassFailure();
      return;
    }

    ModuleOp module = getOperation();

    // Call Scheme to initialize the pass
    std::vector<SchemeValue> initArgs = {
      makeSchemeString(module.getName().value_or("<unnamed>").str().c_str())
    };
    callSchemeFunction("pass-initialize", initArgs);

    // Walk operations and call Scheme for each one
    // Collect operations first, then process to avoid calling Scheme from lambda
    std::vector<Operation*> ops;
    module.walk([&ops](Operation *op) {
      ops.push_back(op);
    });

    llvm::errs() << "Processing " << ops.size() << " operations...\n";
    for (size_t i = 0; i < ops.size(); i++) {
      Operation* op = ops[i];
      llvm::errs() << "  Op " << i << ": " << op->getName() << "\n";
      llvm::errs() << "  Calling Scheme directly...\n";
      callSchemeCallback(nullptr, op);  // Use the callback helper
      llvm::errs() << "  Done\n";
    }

    // Call Scheme to finalize the pass
    std::vector<SchemeValue> finalizeArgs = {};
    callSchemeFunction("pass-finalize", finalizeArgs);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
