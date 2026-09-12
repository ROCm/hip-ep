/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
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

    // Find R6RS entry point script relative to this library's location
    std::string modulePath = llvm::sys::fs::getMainExecutable(nullptr, (void*)&initializeSchemeRuntime);
    llvm::SmallString<256> scriptPath(modulePath);
    llvm::sys::path::remove_filename(scriptPath);

    if (llvm::sys::path::filename(scriptPath) == "bin")
      llvm::sys::path::remove_filename(scriptPath);

    // R6RS entry point: scriptName should be like "PrintPass.scm"
    // which imports R6RS libraries via (import (mlir ffi)) etc.
    llvm::sys::path::append(scriptPath, "lib", "scheme", scriptName);

    if (!loadSchemeScript(scriptPath.c_str())) {
      signalPassFailure();
      return;
    }

    ModuleOp module = getOperation();

    // Call Scheme entry point function
    // This is a read-only analysis pass - Scheme code can query IR
    // but should not modify it (no rewriter access)
    callSchemePassFunction("run-pass", module);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
