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
#include "hip/Dialect/Hipsr/Scheme/Runtime/SchemeBindings.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERSIONINSCHEMEPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct ConversionInSchemePass
    : impl::ConversionInSchemePassBase<ConversionInSchemePass> {
  using impl::ConversionInSchemePassBase<ConversionInSchemePass>::ConversionInSchemePassBase;

  void runOnOperation() override {
    // Check if module name is specified
    if (moduleName.empty()) {
      emitError(getOperation().getLoc(),
                "Scheme module name not specified. Use --conversion-in-scheme=\"module=<name>\"");
      signalPassFailure();
      return;
    }

    auto* hipsrDialect = getContext().getLoadedDialect<HipsrDialect>();
    if (!hipsrDialect) {
      emitError(getOperation().getLoc(), "HipsrDialect not loaded");
      signalPassFailure();
      return;
    }

    ChezSchemeInterpreter* interpreter = hipsrDialect->getSchemeInterpreter();
    if (!interpreter || !interpreter->isInitialized()) {
      emitError(getOperation().getLoc(), "Scheme interpreter not initialized");
      signalPassFailure();
      return;
    }

    // Set log level
    SchemeLogLevel level = parseLogLevel(logLevel);
    setSchemeLogLevel(level);

    // Import the specified Scheme module
    std::string importCode = "(import (" + moduleName + "))";
    // Set library-directories before importing
    std::string libdirCode = "(library-directories (cons \"/home/build/hip-ep-chez/lib/scheme\" (library-directories)))";
    if (!interpreter->evaluateCode(libdirCode.c_str())) {
      emitWarning(getOperation().getLoc(), "Failed to set library-directories");
    }

    if (!interpreter->evaluateCode(importCode.c_str())) {
      emitError(getOperation().getLoc(), "Failed to import (")
        << moduleName << ") module";
      signalPassFailure();
      return;
    }

    // Call the Scheme run-pass function
    ModuleOp module = getOperation();
    callSchemePassFunction("run-pass", module);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
