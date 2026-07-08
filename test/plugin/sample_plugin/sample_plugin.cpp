/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Sample static plugin for the plugin registrar + pipeline-slot tests.
// Exercises the public plugin surface (PluginAPI.h / PluginRegistry.h) end to
// end: registers a pass, requests a pipeline slot, contributes a minimal vendor
// dialect (via addDialectRegistration), and contributes runtime bitcode + a
// link library. Statically linked into the host, so its registrations land in
// the host's single MLIR/pass/dialect registry.

#include "hip/Compiler/PluginAPI.h"
#include "hip/Compiler/PluginRegistry.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

#include "sample_dialect.h"

#include <cstddef>

// Embedded bitcode from sample_plugin_runtime.cpp (generated
// sample_plugin_bitcode.cpp; see CMakeLists.txt). Empty (size 0) in a build
// without clang/python -- the unit test skips the round-trip in that case.
extern "C" const unsigned char kSamplePluginBitcode[];
extern "C" const std::size_t kSamplePluginBitcodeSize;

namespace {

// No-op pass: emits one remark per func.func so LIT can FileCheck the effect.
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

} // namespace

// HIP_EP_DEFINE_PLUGIN(sample) expands to
// `extern "C" void hipEpRegisterPlugin_sample(HipEpPluginRegistry &R)`; the
// generated registrar calls it once when `sample` is selected.
HIP_EP_DEFINE_PLUGIN(sample) {
  R.registerPass<SamplePrintFunctionsPass>();

  // Run the pass at the most common vendor slot (after onnx->hip). A func.func
  // pass must be nested as func.func(<arg>) per parsePassPipeline syntax;
  // <arg> matches getArgument() above.
  R.requestPipelineSlot(::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
                        "func.func(hip-ep-sample-print-functions)");

  // Contribute a minimal vendor dialect (hip_ep_sample.marker + its
  // ConvertToLLVMPatternInterface). This is the only in-tree exercise of the
  // addDialectRegistration -> loadAllDialects path and the convert-hip-to-llvm
  // hasPromisedInterface guard, so a refactor of either fails a host test.
  R.addDialectRegistration(&registerHipEpSampleDialect);

  // Contribute the embedded bitcode (process-lifetime static, as
  // addRuntimeBitcode requires); skip the empty no-clang build.
  if (kSamplePluginBitcodeSize != 0) {
    R.addRuntimeBitcode(kSamplePluginBitcode, kSamplePluginBitcodeSize);
  }

  // Contribute the sibling static lib; path + name come from CMake defines.
  // The model module never references it, so the link succeeds regardless.
  R.addLibraryPath(HIP_EP_SAMPLE_LIB_DIR);
  R.addLibrary(HIP_EP_SAMPLE_LIB_NAME);
}
