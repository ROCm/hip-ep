/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace mlir {
namespace hipsr {

bool initializeSchemeRuntime();
void printOperation(const char* opName, int numOperands, int numResults, const char* genericForm);

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

      printOperation(
        op->getName().getStringRef().str().c_str(),
        op->getNumOperands(),
        op->getNumResults(),
        genericFormStr.c_str()
      );
    });

    llvm::errs() << "=== End Scheme Printer ===\n\n";
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
