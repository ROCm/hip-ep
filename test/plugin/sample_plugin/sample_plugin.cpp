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
//   5. the plugin contributes a tiny LLVM bitcode buffer
//      (compiled at build time from `sample_plugin_runtime.cpp`)
//      through `addRuntimeBitcode`.
//   6. the plugin contributes a tiny static library (the
//      sibling `hip_ep_sample_lib` target, compiled from
//      `sample_lib.cpp`) through `addLibraryPath` + `addLibrary`.
//      Vendor-side: this is the same shape downstream plugins will
//      use to ship vendor kernel packs.

#include "hip/Compiler/PluginAPI.h"
#include "hip/Compiler/PluginRegistry.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

#include <cstddef>

// Embedded bitcode buffer compiled from sample_plugin_runtime.cpp.
// Defined in the auto-generated `sample_plugin_bitcode.cpp` (see
// sample_plugin/CMakeLists.txt). A clean degraded build that lacks
// clang/python emits an empty buffer (size == 0); the unit test
// detects that case and skips the round-trip assertion.
extern "C" const unsigned char kSamplePluginBitcode[];
extern "C" const std::size_t kSamplePluginBitcodeSize;

namespace {

// Minimal no-op pass implemented inside the plugin to demonstrate the
// registerPass<>() flow. We don't transform the IR; we emit one remark per
// func.func so LIT can FileCheck the effect.
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
  // Hand the pass to MLIR's global pass registry.
  R.registerPass<SamplePrintFunctionsPass>();

  // Asks the public pipeline to run the pass at the most common
  // vendor slot: right after onnx->hip conversion. Pipelines.cpp
  // resolves the string via mlir::parsePassPipeline into the slot's
  // MODULE-level pass manager, so the string follows --pass-pipeline
  // syntax: a func.func pass must be nested as `func.func(<arg>)`
  // (a bare op-agnostic / ModuleOp pass would use just its <arg>).
  // SamplePrintFunctionsPass is an OperationPass<func::FuncOp>, hence
  // the nesting; <arg> matches getArgument() above.
  R.requestPipelineSlot(::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
                        "func.func(hip-ep-sample-print-functions)");

  // Contribute the embedded plugin bitcode. The buffer lives in
  // this DLL's read-only data segment for the lifetime of hip-
  // compiler, which is the contract addRuntimeBitcode requires.
  // When the build did not have clang available, the buffer is
  // empty and we skip the call so the host's
  // pluginBitcodeBuffers() stays empty too.
  if (kSamplePluginBitcodeSize != 0) {
    R.addRuntimeBitcode(kSamplePluginBitcode, kSamplePluginBitcodeSize);
  }

  // Contribute the sibling sample static library. Both the
  // path and the library name come from compile-time defines
  // populated by sample_plugin/CMakeLists.txt; HIP_EP_SAMPLE_LIB_DIR
  // resolves to $<TARGET_FILE_DIR:hip_ep_sample_lib>, i.e. the
  // build-tree directory containing hip_ep_sample_lib.lib (Windows)
  // or libhip_ep_sample_lib.a (Linux). The library exposes a
  // single uniquely-named symbol; the model module never references
  // it, so the link succeeds whether or not the model module
  // declares any plugin functions.
  R.addLibraryPath(HIP_EP_SAMPLE_LIB_DIR);
  R.addLibrary(HIP_EP_SAMPLE_LIB_NAME);
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
      "0.4.0",
      &registerCallbacks,
  };
}
