/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Public sample plugin used by hip-compiler's plugin loader and
// pipeline-slot tests.
//
// The point of this DLL is to exercise the public ABI surface declared
// in `include/hip/Compiler/PluginAPI.h` and `PluginRegistry.h`
// end-to-end in CI:
//
//   1. The DLL exports `hipEpGetPluginInfo` under its unmangled C name.
//   2. The struct it returns satisfies the version + name + version
//      contract that `HipEpPluginLoader::Load` validates.
//   3. The `RegisterCallbacks` function pointer fires correctly across
//      the DLL boundary with a `HipEpPluginRegistry &`.
//   4. The plugin's pass (`SamplePrintFunctionsPass`) is registered in
//      MLIR's global pass registry and runs when requested by name
//      via the `AfterConvertOnnxToHip` slot.
//
// PR 3 will additionally contribute LLVM bitcode through the same
// callback; PR 4 will contribute a sample static library.

#include "hip/Compiler/PluginAPI.h"
#include "hip/Compiler/PluginRegistry.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

namespace {

// Minimal no-op pass implemented inside the plugin to demonstrate
// the upstream-style registerPass<>() flow. We don't transform the
// IR; we emit one remark per func.func so LIT can FileCheck the
// effect.
//
// PassWrapper + final overrides of getArgument/getDescription are
// the common upstream shape for hand-written passes (TableGen
// records do the same thing for declared passes).
struct SamplePrintFunctionsPass
    : public mlir::PassWrapper<SamplePrintFunctionsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SamplePrintFunctionsPass)

  llvm::StringRef getArgument() const final {
    return "hip-ep-sample-print-functions";
  }

  llvm::StringRef getDescription() const final {
    return "Sample plugin pass: emit a remark per func.func "
           "(no transformation)";
  }

  void runOnOperation() override {
    auto fn = getOperation();
    fn.emitRemark() << "[hip-ep-sample] visited " << fn.getSymName();
  }
};

void registerCallbacks(::hip::compiler::HipEpPluginRegistry &R) {
  // Hands the pass to MLIR's global registry. Identical pattern to
  // mlir::PassRegistration<SamplePrintFunctionsPass>() in upstream
  // tools.
  R.registerPass<SamplePrintFunctionsPass>();

  // Asks the public pipeline to run the pass at the most common
  // vendor slot: right after onnx->hip conversion. The pass name
  // must match getArgument() above; Pipelines.cpp resolves it via
  // mlir::parsePassPipeline.
  R.requestPipelineSlot(::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
                        "hip-ep-sample-print-functions");
}

} // namespace

// LLVM_ATTRIBUTE_WEAK is a no-op on Windows; the CMake target sets
// `WINDOWS_EXPORT_ALL_SYMBOLS ON` so this symbol is exported under
// its unmangled C name. On non-Windows the weak attribute lets the
// same source link statically into a tool if we ever want to.
extern "C" ::hip::compiler::HipEpPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
hipEpGetPluginInfo() {
  return {
      HIP_EP_PLUGIN_API_VERSION,
      "HipEpSamplePlugin",
      "0.2.0",
      &registerCallbacks,
  };
}
