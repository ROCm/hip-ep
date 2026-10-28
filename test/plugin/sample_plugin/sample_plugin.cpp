/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Sample static plugin for the plugin registrar + pipeline-slot tests.
// Exercises the public plugin surface (PluginAPI.h / PluginRegistry.h) end to
// end: registers a pass, requests a pipeline slot, registers a frontend
// pipeline, contributes a minimal vendor dialect (via addDialectRegistration),
// and contributes runtime bitcode + a link library. Statically linked into the
// host, so its registrations land in the host's single MLIR/pass/dialect
// registry.

#include "hip/Compiler/PluginAPI.h"
#include "hip/Compiler/PluginRegistry.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "sample_dialect.h"

#include <cstddef>
#include <memory>

// Embedded bitcode from sample_plugin_runtime.cpp (generated
// sample_plugin_bitcode.cpp; see CMakeLists.txt). Empty (size 0) in a build
// without clang/python -- the unit test skips the round-trip in that case.
extern "C" const unsigned char kSamplePluginBitcode[];
extern "C" const std::size_t kSamplePluginBitcodeSize;

namespace {

void emitSampleConstant(mlir::func::FuncOp function,
                        llvm::StringRef sourceName) {
  mlir::Block &entry = function.front();
  mlir::OpBuilder builder(function.getContext());
  builder.setInsertionPointToStart(&entry);
  auto tensorType = mlir::RankedTensorType::get({2}, builder.getF32Type());
  llvm::SmallVector<mlir::Attribute> values = {
      builder.getF32FloatAttr(101.0f),
      builder.getF32FloatAttr(102.0f),
  };
  auto value = mlir::DenseElementsAttr::get(tensorType, values);
  auto constant = mlir::hip::ConstantOp::create(builder, function.getLoc(),
                                                tensorType, value);
  auto name = builder.getStringAttr(sourceName);
  constant.setSymbolNameHintAttr(name);
  constant.setSourceNameAttr(name);

  mlir::Value context;
  for (mlir::BlockArgument argument : function.getArguments()) {
    if (mlir::isa<mlir::hip::ContextType>(argument.getType())) {
      context = argument;
      break;
    }
  }
  auto terminator = mlir::dyn_cast<mlir::func::ReturnOp>(entry.getTerminator());
  if (!context || !terminator)
    return;

  for (mlir::OpOperand &returned : terminator->getOpOperands()) {
    if (returned.get().getType() != tensorType)
      continue;
    builder.setInsertionPoint(terminator);
    mlir::Value init = mlir::tensor::EmptyOp::create(
        builder, function.getLoc(), tensorType.getShape(),
        tensorType.getElementType());
    auto biased =
        mlir::hip::AddOp::create(builder, function.getLoc(), context,
                                 returned.get(), constant.getResult(), init);
    returned.set(biased->getResult(0));
    break;
  }
}

// Emits one remark per func.func. A marked function also gets a plugin-owned
// hip.constant at the start of its entry block; because this pass runs at
// AfterConvertOnnxToHip, the production externalizer consumes it.
struct SamplePrintFunctionsPass
    : public mlir::PassWrapper<SamplePrintFunctionsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SamplePrintFunctionsPass)

  llvm::StringRef getArgument() const final {
    return "hip-ep-sample-print-functions";
  }

  llvm::StringRef getDescription() const final {
    return "Sample plugin pass: emit a remark and optional hip.constant";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::hip::HipDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto fn = getOperation();
    fn.emitRemark() << "[hip-ep-sample] visited " << fn.getSymName();
    if (fn->hasAttr("hip_ep_sample.emit_constant"))
      emitSampleConstant(fn, "sample_plugin_weight");
  }
};

// Negative fixture for the supported producer boundary. A marked function gets
// a carrier at BeforeBufferization, where the production late-carrier guard
// must reject it.
struct SampleEmitLateConstantPass
    : public mlir::PassWrapper<SampleEmitLateConstantPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SampleEmitLateConstantPass)

  llvm::StringRef getArgument() const final {
    return "hip-ep-sample-emit-late-constant";
  }

  llvm::StringRef getDescription() const final {
    return "Sample plugin negative fixture: emit a late hip.constant";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::hip::HipDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto function = getOperation();
    if (function->hasAttr("hip_ep_sample.emit_late_constant"))
      emitSampleConstant(function, "sample_plugin_late_weight");
  }
};

// Stand-in for a frontend's lowering to HIP, run only by the sample's frontend
// pipeline. A function marked hip_ep_sample.frontend_input is otherwise already
// tensor-form HIP IR; "lowering" it adds a plugin-owned carrier, as a frontend
// emits its weights, and drops the marker.
struct SampleFrontendPass
    : public mlir::PassWrapper<SampleFrontendPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SampleFrontendPass)

  llvm::StringRef getArgument() const final { return "hip-ep-sample-frontend"; }

  llvm::StringRef getDescription() const final {
    return "Sample plugin frontend pipeline: lower a marked function";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::hip::HipDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto function = getOperation();
    if (!function->removeAttr("hip_ep_sample.frontend_input"))
      return;
    emitSampleConstant(function, "sample_frontend_weight");
  }
};

// Frontend pipeline selected by `hip.compiler.frontend_pipeline`. Marker
// attributes on the module exercise both ways a builder can reject its input.
bool buildSampleFrontendPipeline(
    hip::compiler::FrontendPipelineContext &context) {
  if (context.module->hasAttr("hip_ep_sample.reject")) {
    context.errorMessage = "hip-ep-sample cannot lower this module";
    return false;
  }
  if (context.module->hasAttr("hip_ep_sample.reject_silently"))
    return false;
  context.passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<SampleFrontendPass>());
  return true;
}

} // namespace

// HIP_EP_DEFINE_PLUGIN(sample) expands to
// `extern "C" void hipEpRegisterPlugin_sample(HipEpPluginRegistry &R)`; the
// generated registrar calls it once when `sample` is selected.
HIP_EP_DEFINE_PLUGIN(sample) {
  R.registerPass<SamplePrintFunctionsPass>();
  R.registerPass<SampleEmitLateConstantPass>();

  // Run the pass at the most common vendor slot (after onnx->hip). A func.func
  // pass must be nested as func.func(<arg>) per parsePassPipeline syntax;
  // <arg> matches getArgument() above.
  R.requestPipelineSlot(::hip::compiler::PipelineSlot::AfterConvertOnnxToHip,
                        "func.func(hip-ep-sample-print-functions)");
  R.requestPipelineSlot(::hip::compiler::PipelineSlot::BeforeBufferization,
                        "func.func(hip-ep-sample-emit-late-constant)");

  // A module whose `hip.compiler.frontend_pipeline` names this id is lowered by
  // buildSampleFrontendPipeline and then the host's tensor-to-bufferized and
  // HIP-to-LLVM pipelines. The second id is registered twice on purpose:
  // selecting it must fail as ambiguous.
  R.addFrontendPipeline("hip-ep-sample", &buildSampleFrontendPipeline);
  R.addFrontendPipeline("hip-ep-sample-duplicate",
                        &buildSampleFrontendPipeline);
  R.addFrontendPipeline("hip-ep-sample-duplicate",
                        &buildSampleFrontendPipeline);

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
