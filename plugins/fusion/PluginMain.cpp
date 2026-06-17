/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PluginMain.cpp - hip-fusion plugin entry points --------------------===//
//
// The two stock MLIR plugin entry points dlsym()'d by hip-mlir-opt:
//
//   mlirGetDialectPluginInfo()  -> registers the plugin op hip.fused_mul_add
//                                  into the existing `hip` dialect AND attaches
//                                  its bufferization external model.
//   mlirGetPassPluginInfo()     -> registers the --hip-fuse-mul-add pass.
//
// §4 / R-1 demonstration
// ----------------------
// The bufferization external-model attach below is guarded by
// HIPPOC_SKIP_BUFFERIZE_ATTACH. The default build (hip_fusion_plugin) attaches
// it; the hip_fusion_plugin_nobuf build omits it. Loading the _nobuf variant
// and running one-shot-bufferize reproduces the documented opaque failure
// (`op was not bufferized`) — turning the §4 "easy-to-miss footgun" into an
// executable test.
//
// ABI note
// --------
// This plugin uses the stock MLIR plugin ABI (mlirGetDialectPluginInfo /
// mlirGetPassPluginInfo, MLIR_PLUGIN_API_VERSION 1). The bespoke
// gnpu_plugin_entry C-ABI from the requirements doc remains a future milestone.
//
//===----------------------------------------------------------------------===//

#include "plugins/fusion/FusedMulAddLowering.h"
#include "plugins/fusion/Passes.h"
#include "plugins/fusion/PipelineSlot.h"
#include "plugins/fusion/PluginOps.h"

#include "hip/Dialect/IR/HipBufferize.h" // HipDstBufferizableModel<>
#include "hip/Dialect/IR/HipDialect.h"

#include "hip/Conversion/HipToLLVM/Passes.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Config/llvm-config.h" // LLVM_VERSION_STRING
#include "llvm/Support/Compiler.h"   // LLVM_ATTRIBUTE_WEAK

using namespace mlir;

namespace {

// A minimal conversion pass that lowers hip.fused_mul_add to LLVM using the
// plugin's FusedMulAddOpLowering pattern. Intended to run AFTER the standard
// convert-hip-to-llvm pass has already lowered all in-tree hip.* ops, leaving
// only the plugin-contributed op as the sole illegal operation.
struct FusedMulAddToLLVMPass
    : public PassWrapper<FusedMulAddToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusedMulAddToLLVMPass)
  StringRef getName() const override { return "fused-mul-add-to-llvm"; }
  StringRef getArgument() const override { return "fused-mul-add-to-llvm"; }
  StringRef getDescription() const override {
    return "Lower hip.fused_mul_add to LLVM (plugin pass)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);
    typeConverter.addConversion([ctx](hip::ContextType) -> Type {
      return LLVM::LLVMPointerType::get(ctx, 0);
    });

    RewritePatternSet patterns(ctx);
    hip::populateFusedMulAddLoweringPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<hip::FusedMulAddOp>();
    target.addLegalOp<ModuleOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

static std::unique_ptr<Pass> createFusedMulAddToLLVMPass() {
  return std::make_unique<FusedMulAddToLLVMPass>();
}

// On Linux LLVM_ATTRIBUTE_WEAK makes these symbols weak so the plugin loader
// can dlsym them. On Windows we use a .def file (hip_fusion_plugin.def) to
// export them explicitly — __declspec(dllexport) conflicts with the weak
// declaration in DialectPlugin.h / PassPlugin.h.
extern "C" ::mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "hip-fusion", LLVM_VERSION_STRING,
          [](DialectRegistry *registry) {
            registry->addExtension(
                +[](MLIRContext *ctx, hip::HipDialect *dialect) {
                  // Touchpoint A: register the real hip.* op via the host hook.
                  dialect->registerPluginOps<hip::FusedMulAddOp>();
#ifndef HIPPOC_SKIP_BUFFERIZE_ATTACH
                  // Touchpoint D / R-1: attach the bufferization external
                  // model so OneShotBufferize can legalize the op.
                  hip::FusedMulAddOp::attachInterface<
                      hip::HipDstBufferizableModel<hip::FusedMulAddOp>>(*ctx);
#endif
                });
          }};
}

extern "C" ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "hip-fuse-mul-add", LLVM_VERSION_STRING,
          []() {
            // Touchpoint B: register the --hip-fuse-mul-add pass.
            hip::registerHipFuseMulAddPass();

            // Touchpoint C: declare where in the pipeline this pass runs.
            // Anchor: "convert-onnx-to-hip" (stable public anchor).
            // Position: After — run the fusion pass after ONNX→HIP conversion
            // so we have hip.mul / hip.add ops to pattern-match, and before
            // one-shot-bufferize so the fused op goes through bufferization.
            hip::PipelineSlotRegistry::get().addSlot(
                "convert-onnx-to-hip", hip::SlotPosition::After,
                []() -> std::unique_ptr<Pass> {
                  return hip::createHipFuseMulAddPass();
                },
                "hip-fuse-mul-add — fuse mul+add after ONNX→HIP conversion");

            // Touchpoint E: register a plugin-extended HipToLLVM pipeline.
            //
            // The in-tree ConvertHipToLLVM pass has no hook for plugin patterns.
            // We register a named pipeline "hip-to-llvm-with-fusion-plugin" that
            // runs the standard in-tree pass first (lowering all in-tree hip.*
            // ops to LLVM), then runs a second partial conversion that lowers
            // the remaining hip.fused_mul_add via FusedMulAddOpLowering.
            //
            // This two-pass approach avoids modifying the in-tree pass and is
            // safe because after the first pass only the plugin op remains in
            // the HipDialect — it is the only illegal op for the second pass.
            //
            // LIT tests that exercise E2E lowering use this pipeline instead of
            // the bare --convert-hip-to-llvm shorthand.
            PassPipelineRegistration<>(
                "hip-to-llvm-with-fusion-plugin",
                "HipToLLVM pipeline extended with hip.fused_mul_add lowering",
                [](OpPassManager &pm) {
                  pm.addPass(memref::createExpandStridedMetadataPass());
                  pm.addPass(hip::createConvertHipToLLVMPass());
                  pm.addPass(createFusedMulAddToLLVMPass());
                });
          }};
}
