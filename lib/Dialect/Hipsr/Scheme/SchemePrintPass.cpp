/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "SchemeRuntime.h"
#include <string>

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

    // Call Scheme function to initialize the pass
    std::vector<SchemeValue> initArgs = {
      makeSchemeString(module.getName().value_or("<unnamed>").str().c_str())
    };
    callSchemeFunction("pass-initialize", initArgs);

    // Walk all operations and call Scheme function for each
    module.walk([](Operation *op) {
      std::string genericFormStr;
      llvm::raw_string_ostream os(genericFormStr);
      op->print(os, OpPrintingFlags().printGenericOpForm());
      os.flush();

      std::vector<SchemeValue> args = {
        makeSchemeString(op->getName().getStringRef().str().c_str()),
        makeSchemeInteger(op->getNumOperands()),
        makeSchemeInteger(op->getNumResults()),
        makeSchemeString(genericFormStr.c_str())
      };

      // Call Scheme to process this operation
      // The Scheme function decides what to do (print, transform, etc.)
      callSchemeFunction("process-operation", args);
    });

    // Call Scheme function to finalize the pass
    std::vector<SchemeValue> finalizeArgs = {};
    callSchemeFunction("pass-finalize", finalizeArgs);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
