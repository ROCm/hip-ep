/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "SchemeBindings.h"
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
    if (!initializeSchemeRuntime(verbose)) {
      signalPassFailure();
      return;
    }

    // Find script relative to this library's location
    // Pass address of a function in this compilation unit so getMainExecutable
    // uses dladdr() (Unix) or GetModuleFileName() (Windows) to find the
    // module (DLL/SO) containing this code, not just the main executable.
    // This works whether we're statically linked or a dynamic plugin.
    // Expected layout: <install>/bin/hip-mlir-opt (or <install>/lib/libHipsrSchemePass.so)
    //                  <install>/lib/scheme/PrintPass.scm
    std::string modulePath = llvm::sys::fs::getMainExecutable(nullptr, (void*)&initializeSchemeRuntime);
    llvm::SmallString<256> scriptPath(modulePath);
    llvm::sys::path::remove_filename(scriptPath);  // Remove binary name

    // If we're in bin/, go up one level; if already in lib/, stay
    if (llvm::sys::path::filename(scriptPath) == "bin")
      llvm::sys::path::remove_filename(scriptPath);  // Remove bin/

    llvm::sys::path::append(scriptPath, "lib", "scheme", "PrintPass.scm");

    if (!loadSchemeScript(scriptPath.c_str())) {
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
