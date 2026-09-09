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

    // Load the pure Scheme pass implementation
    // TODO: Make script path configurable via pass option
    const char* scriptPath = SOURCE_DIR "/lib/Dialect/Hipsr/Scheme/print-pass.scm";
    if (!loadSchemeScript(scriptPath)) {
      signalPassFailure();
      return;
    }

    // Call the Scheme pass entry point with the module
    ModuleOp module = getOperation();
    callSchemePassFunction("run-pass", module);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
