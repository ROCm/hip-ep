/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "Runtime/SchemeBindings.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_SCHEMESCRIPTPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct SchemeScriptPass : public impl::SchemeScriptPassBase<SchemeScriptPass> {
  using impl::SchemeScriptPassBase<SchemeScriptPass>::SchemeScriptPassBase;

  void runOnOperation() override {
    SchemeLogLevel level = parseLogLevel(logLevel);
    if (!initializeSchemeRuntime(level)) {
      signalPassFailure();
      return;
    }

    // Import the R6RS module
    // initializeSchemeRuntime already set (library-directories) relative to executable
    // Now we just need to (import (module-name))
    std::string importCode = "(import (" + moduleName + "))";

    if (!evaluateSchemeCode(importCode.c_str())) {
      getOperation().emitError("Failed to import R6RS module: ") << moduleName;
      signalPassFailure();
      return;
    }

    ModuleOp module = getOperation();

    // Call run-pass function exported by the module
    // This is a read-only analysis pass - Scheme code can query IR
    // but should not modify it (no rewriter access)
    callSchemePassFunction("run-pass", module);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
