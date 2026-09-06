/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- SchemePrintPass.cpp - Print MLIR in generic form from Scheme -------===//
//
// MLIR pass that uses Scheme to pretty-print operations in generic form
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Scheme/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace hipsr {

bool initializeSchemeRuntime(); // from SchemeRuntime.cpp

namespace {

struct SchemePrintPass
    : public PassWrapper<SchemePrintPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SchemePrintPass)

  StringRef getArgument() const final { return "scheme-print"; }
  StringRef getDescription() const final {
    return "Print MLIR operations in generic form using Scheme";
  }

  void runOnOperation() override {
    // Initialize Scheme runtime (with rime embedded)
    if (!initializeSchemeRuntime()) {
      signalPassFailure();
      return;
    }

    ModuleOp module = getOperation();

    llvm::errs() << "\n=== Scheme-based MLIR Printer ===\n";
    llvm::errs() << "Module: " << module.getName().value_or("<unnamed>")
                 << "\n\n";

    // Walk all operations and print in generic form
    module.walk([](Operation *op) {
      llvm::errs() << "Operation: \"" << op->getName().getStringRef() << "\"\n";
      llvm::errs() << "  Operands: " << op->getNumOperands() << "\n";
      llvm::errs() << "  Results: " << op->getNumResults() << "\n";

      // Print in generic form
      llvm::errs() << "  Generic form: ";
      op->print(llvm::errs(), OpPrintingFlags().printGenericOpForm());
      llvm::errs() << "\n\n";
    });

    llvm::errs() << "=== End Scheme Printer ===\n\n";
  }
};

} // namespace

std::unique_ptr<Pass> createSchemePrintPass() {
  return std::make_unique<SchemePrintPass>();
}

void registerHipsrSchemePasses() {
  registerPass(
      []() -> std::unique_ptr<Pass> { return createSchemePrintPass(); });
}

} // namespace hipsr
} // namespace mlir
