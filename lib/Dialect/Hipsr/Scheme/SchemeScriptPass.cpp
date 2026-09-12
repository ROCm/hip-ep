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

    // Set up ConversionTarget - make ONNX illegal, HipSR legal
    ConversionTarget target(getContext());
    target.addIllegalDialect<onnx::OnnxDialect>();
    target.addLegalOp<onnx::NoValueOp>();  // Cleaned up later
    target.addLegalDialect<HipsrDialect>();
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return converter.isLegal(op); });

    // Collect patterns by calling Scheme
    RewritePatternSet patterns(&getContext());

    // Call the Scheme pass to populate patterns
    // This is a simplified approach - just run the rewrite for now
    callSchemePassFunction("run-pass", module);

    // For now, don't run conversion - we need to refactor how Scheme patterns work
    // TODO: Make Scheme patterns return ConversionPattern instances
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir
