/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Runtime/SchemeBindings.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_SCHEMESCRIPTPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Helper to add device memory space to tensor types
static Type tensorTypeInSpace(RankedTensorType type, MemorySpace space) {
  return type.cloneWithEncoding(
      MemorySpaceAttr::get(type.getContext(), space));
}

struct SchemeScriptPass : public impl::SchemeScriptPassBase<SchemeScriptPass> {
  using impl::SchemeScriptPassBase<SchemeScriptPass>::SchemeScriptPassBase;

  void runOnOperation() override {
    SchemeLogLevel level = parseLogLevel(logLevel);
    if (!initializeSchemeRuntime(level)) {
      signalPassFailure();
      return;
    }

    // Find script relative to this library's location
    std::string modulePath = llvm::sys::fs::getMainExecutable(nullptr, (void*)&initializeSchemeRuntime);
    llvm::SmallString<256> scriptPath(modulePath);
    llvm::sys::path::remove_filename(scriptPath);

    if (llvm::sys::path::filename(scriptPath) == "bin")
      llvm::sys::path::remove_filename(scriptPath);

    llvm::sys::path::append(scriptPath, "lib", "scheme", scriptName);

    if (!loadSchemeScript(scriptPath.c_str())) {
      signalPassFailure();
      return;
    }

    ModuleOp module = getOperation();

    // Set up TypeConverter to add device memory space to tensors
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](RankedTensorType type) -> Type {
      // Leave rank-0 scalars and tensors that already have encoding untouched
      if (type.getRank() == 0 || type.getEncoding()) {
        return type;
      }
      return tensorTypeInSpace(type, MemorySpace::Device);
    });

    // Set up ConversionTarget for function signature conversion
    ConversionTarget target(getContext());
    target.addLegalDialect<onnx::OnnxDialect>();  // ONNX still legal for now
    target.addLegalDialect<HipsrDialect>();
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return converter.isLegal(op); });

    // First pass: Convert function signatures to use device memory space
    RewritePatternSet signaturePatterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        signaturePatterns, converter);

    if (failed(applyPartialConversion(module, target, std::move(signaturePatterns)))) {
      module.emitError("Failed to convert function signatures");
      signalPassFailure();
      return;
    }

    // Second pass: Call Scheme patterns to transform ONNX -> HipSR
    // Function signatures now have device memory space, so Scheme patterns
    // can use function arguments directly without needing unrealized_conversion_cast
    callSchemePassFunction("run-pass", module);
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
