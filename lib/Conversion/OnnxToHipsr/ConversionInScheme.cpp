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

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERSIONINSCHEMEPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Get the absolute path to lib/scheme directory dynamically
static std::string getSchemeLibraryPath() {
  // Get the path to the current executable/library
  // Use the address of this function as a hint for getMainExecutable
  std::string execPath = llvm::sys::fs::getMainExecutable(nullptr, (void*)&getSchemeLibraryPath);
  llvm::SmallString<256> basePath(execPath);
  llvm::sys::path::remove_filename(basePath);  // Remove binary/library name

  // Common layouts:
  //   build/bin/hip-mlir-opt -> build/lib/scheme
  //   install/bin/hip-mlir-opt -> install/lib/scheme
  if (llvm::sys::path::filename(basePath) == "bin") {
    llvm::sys::path::remove_filename(basePath);  // Go up to build/install root
  }
  // basePath now points to build root or install root

  llvm::SmallString<256> schemePath = basePath;
  llvm::sys::path::append(schemePath, "lib", "scheme");

  // Make the path absolute
  if (std::error_code ec = llvm::sys::fs::make_absolute(schemePath)) {
    // If make_absolute fails, just return the path as-is
    return std::string(schemePath.c_str());
  }

  // Normalize the path (resolve .., remove redundant separators)
  llvm::sys::path::remove_dots(schemePath, /*remove_dot_dot=*/true);

  return std::string(schemePath.c_str());
}


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

    // Set library-directories before importing (use dynamic path resolution)
    std::string schemePath = getSchemeLibraryPath();
    std::string libdirCode = "(library-directories (cons \"" + schemePath + "\" (library-directories)))";
    if (!ChezSchemeInterpreter::eval(libdirCode.c_str())) {
      emitWarning(getOperation().getLoc(), "Failed to set library-directories for path: " + schemePath);
    }

    if (!ChezSchemeInterpreter::eval(importCode.c_str())) {
      emitError(getOperation().getLoc(), "Failed to import (")
        << moduleName << ") module";
      signalPassFailure();
      return;
    }

    // Call the Scheme run-pass function
    ModuleOp module = getOperation();
    ChezSchemeInterpreter::callPassFunction("run-pass", module);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
