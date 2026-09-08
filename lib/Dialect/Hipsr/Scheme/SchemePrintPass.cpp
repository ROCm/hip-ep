/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
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

    llvm::errs() << "\n=== Scheme-based MLIR Printer ===\n";
    llvm::errs() << "Module: " << module.getName().value_or("<unnamed>") << "\n\n";

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

      std::string result = callSchemeFunction("format-operation", args);
      if (!result.empty()) {
        llvm::errs() << result;
      }
    });

    llvm::errs() << "=== End Scheme Printer ===\n\n";
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
