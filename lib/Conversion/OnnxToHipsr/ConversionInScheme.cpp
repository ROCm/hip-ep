/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ConversionInScheme.cpp - Generic conversion via Scheme --------------===//
//
// General-purpose conversion pass that loads Scheme-defined patterns.
// The Scheme module is configurable via the 'module' option.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/Scheme/Runtime/ChezSchemeInterpreter.h"
#include "hip/Dialect/Hipsr/Scheme/Runtime/SchemeMlirBindings.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERSIONINSCHEMEPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct ConversionInSchemePass
    : impl::ConversionInSchemePassBase<ConversionInSchemePass> {
  using impl::ConversionInSchemePassBase<ConversionInSchemePass>::ConversionInSchemePassBase;

  void runOnOperation() override {
    // Ensure required dialects are loaded
    getContext().loadDialect<mlir::hipsr::HipsrDialect>();
    getContext().loadDialect<mlir::onnx::OnnxDialect>();
    getContext().loadDialect<mlir::func::FuncDialect>();

    // Check if module name is specified
    if (moduleName.empty()) {
      emitError(getOperation().getLoc(),
                "Scheme module name not specified. Use --conversion-in-scheme=\"module=<name>\"");
      signalPassFailure();
      return;
    }

    // Initialize Scheme runtime if needed
    if (!ChezSchemeInterpreter::isInitialized()) {
      SchemeLogLevel level = ChezSchemeInterpreter::parseLogLevel(logLevel);
      ChezSchemeInterpreter::initialize(level);
    }

    // Set log level
    SchemeLogLevel level = ChezSchemeInterpreter::parseLogLevel(logLevel);
    ChezSchemeInterpreter::setLogLevel(level);

    // Import the specified Scheme module
    // Convert slash notation to space notation for R6RS library names
    // e.g., "passes/onnx-to-hipsr" -> "passes onnx-to-hipsr"
    std::string libraryName = moduleName;
    std::replace(libraryName.begin(), libraryName.end(), '/', ' ');
    std::string importCode = "(import (" + libraryName + "))";

    llvm::errs() << "[ConversionInScheme] About to import: " << importCode << "\n";

    if (!ChezSchemeInterpreter::eval(importCode.c_str())) {
      emitError(getOperation().getLoc(), "Failed to import (")
        << moduleName << ") module";
      signalPassFailure();
      return;
    }

    llvm::errs() << "[ConversionInScheme] Import successful, calling run-pass\n";

    // Call the Scheme run-pass function
    ModuleOp module = getOperation();
    ChezSchemeInterpreter::callPassFunction("run-pass", module);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
