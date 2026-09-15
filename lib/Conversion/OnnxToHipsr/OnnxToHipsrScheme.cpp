/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipsrScheme.cpp - ONNX to HipSR via Scheme patterns ----------===//
//
// Same as OnnxToHipsr.cpp but uses Scheme-defined patterns instead of C++.
// This demonstrates writing MLIR conversion patterns in Scheme without
// recompiling C++.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/Scheme/Runtime/ChezSchemeInterpreter.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERTONNXTOHIPSRSCHEMEPASS
#include "hip/Conversion/Passes.h.inc"

namespace {

struct ConvertOnnxToHipsrSchemePass
    : impl::ConvertOnnxToHipsrSchemePassBase<ConvertOnnxToHipsrSchemePass> {
  void runOnOperation() override {
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

    // Import the onnx-to-hipsr Scheme module
    if (!interpreter->evaluateCode("(import (onnx-to-hipsr))")) {
      emitError(getOperation().getLoc(), "Failed to import (onnx-to-hipsr) module");
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
