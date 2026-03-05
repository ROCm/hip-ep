/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/Compiler/Pipeline.h"
#include "hip/InitAllPasses.h"
#include "hip/Support/DiskFileSystem.h"
#include "compilation_options_generated.h"

#ifdef ENABLE_ONNX_FRONTEND
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#endif

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/FileSystem.h"

namespace {

/// External model that teaches one-shot-bufferize how to handle any HIP
/// destination-passing-style op: tensor operands become memref buffers, and
/// each tensor result aliases its DPS init operand's buffer.
template <typename OpTy>
struct HipDstBufferizableModel
    : public mlir::bufferization::DstBufferizableOpInterfaceExternalModel<
          HipDstBufferizableModel<OpTy>, OpTy> {
  mlir::LogicalResult
  bufferize(mlir::Operation *op, mlir::RewriterBase &rewriter,
            const mlir::bufferization::BufferizationOptions &options,
            mlir::bufferization::BufferizationState &state) const {
    auto dstOp = mlir::cast<mlir::DestinationStyleOpInterface>(op);

    llvm::SmallVector<mlir::Value> newOperands;
    for (mlir::OpOperand &operand : op->getOpOperands()) {
      if (mlir::isa<mlir::TensorType>(operand.get().getType())) {
        mlir::FailureOr<mlir::Value> buffer =
            getBuffer(rewriter, operand.get(), options, state);
        if (mlir::failed(buffer))
          return mlir::failure();
        newOperands.push_back(*buffer);
      } else {
        newOperands.push_back(operand.get());
      }
    }

    // Recreate op with memref operands; no tensor results (writes in-place).
    OpTy::create(rewriter, op->getLoc(), mlir::TypeRange{}, newOperands,
                 op->getAttrs());

    // DPS convention: replace each tensor result with its tied init buffer.
    llvm::SmallVector<mlir::Value> replacements;
    for (mlir::OpResult result : op->getResults()) {
      if (!mlir::isa<mlir::TensorType>(result.getType())) {
        replacements.push_back(result);
        continue;
      }
      mlir::OpOperand *initOperand = dstOp.getTiedOpOperand(result);
      mlir::FailureOr<mlir::Value> initBuffer =
          getBuffer(rewriter, initOperand->get(), options, state);
      if (mlir::failed(initBuffer))
        return mlir::failure();
      replacements.push_back(*initBuffer);
    }

    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op,
                                                       replacements);
    return mlir::success();
  }
};

void registerHipBufferizableOpInterfaceModels(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, mlir::hip::HipDialect *) {
    mlir::hip::AvgPoolOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::AvgPoolOp>>(*ctx);
    mlir::hip::CastOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::CastOp>>(*ctx);
    mlir::hip::ConvOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ConvOp>>(*ctx);
    mlir::hip::GatherOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GatherOp>>(*ctx);
    mlir::hip::GemmOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GemmOp>>(*ctx);
    mlir::hip::GroupQueryAttentionOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GroupQueryAttentionOp>>(*ctx);
    mlir::hip::MatMulOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MatMulOp>>(*ctx);
    mlir::hip::MaxPoolOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MaxPoolOp>>(*ctx);
    mlir::hip::MulOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MulOp>>(*ctx);
    mlir::hip::ReduceSumOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ReduceSumOp>>(*ctx);
    mlir::hip::ReluOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ReluOp>>(*ctx);
    mlir::hip::RotaryEmbeddingOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::RotaryEmbeddingOp>>(*ctx);
    mlir::hip::SigmoidOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SigmoidOp>>(*ctx);
    mlir::hip::SimplifiedLayerNormOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SimplifiedLayerNormOp>>(*ctx);
    mlir::hip::SkipSimplifiedLayerNormOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SkipSimplifiedLayerNormOp>>(*ctx);
    mlir::hip::SubOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SubOp>>(*ctx);
  });
}

/// Pipeline options shared by both morphizen-pipeline and hip-pipeline.
/// constants-file: embedded in the DLL metadata blob by generate-interface.
/// constants-dir:  directory where OnnxToHip writes the constants binary.
struct PipelineOptions : public mlir::PassPipelineOptions<PipelineOptions> {
  Option<std::string> constantsFile{
      *this, "constants-file",
      llvm::cl::desc("Filename for constants data embedded in module metadata "
                     "(default: constants.bin)"),
      llvm::cl::init("")};
  Option<std::string> constantsDir{
      *this, "constants-dir",
      llvm::cl::desc("Directory to write constants file into (default: .)"),
      llvm::cl::init("")};
  Option<int> optLevel{*this, "opt-level",
                       llvm::cl::desc("Optimization level 0-3 (default: 2)"),
                       llvm::cl::init(2)};
  Option<bool> verbose{*this, "verbose",
                       llvm::cl::desc("Enable verbose output"),
                       llvm::cl::init(false)};
};

} // namespace

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::BuiltinDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::hip::HipDialect>();
#ifdef ENABLE_ONNX_FRONTEND
  registry.insert<mlir::ONNXDialect>();
#endif

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  registerHipBufferizableOpInterfaceModels(registry);

  mlir::hip::registerOptimizeMemRefsPass();
  mlir::hip::registerPoolAllocsPass();
  udna::compiler::registerConversionPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::bufferization::registerBufferizationPipelines();
  mlir::registerConvertBufferizationToMemRefPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return mlir::createCSEPass(); });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createCanonicalizerPass();
  });

  // compilationOpts and fs live in main() so they outlive MlirOptMain() and
  // therefore all pass executions triggered by it. Passes store
  // const CompilationOptionsT& and morphizen::FileSystem* — both must remain
  // valid for the full duration of pm.run().
  udna::compiler::CompilationOptionsT compilationOpts;
  std::unique_ptr<udna::DiskFileSystem> fs;

  // HIP→LLVM sub-pipeline: bufferization, canonicalize, pool-allocs,
  // HIP→LLVM, generate-interface. Shared with CompilerDriver.
  // Use when input is already in HIP tensor IR (after OnnxToHip).
  mlir::PassPipelineRegistration<PipelineOptions> hipPipeline(
      "hip-pipeline",
      "HIP→LLVM sub-pipeline (bufferize, canonicalize, pool, HIP→LLVM, "
      "interface)",
      [&compilationOpts, &fs](mlir::OpPassManager &pm,
                              const PipelineOptions &opts) {
        compilationOpts.constants_file = opts.constantsFile;
        compilationOpts.opt_level = opts.optLevel;
        compilationOpts.verbose = opts.verbose;

        const std::string &dir = opts.constantsDir;
        if (!dir.empty())
          llvm::sys::fs::create_directories(dir);
        fs = std::make_unique<udna::DiskFileSystem>(
            dir.empty() ? "." : dir.c_str());

        udna::compiler::compiler::populateHipPipeline(pm, compilationOpts);
      });

  // Full ONNX→HIP→LLVM→Interface pipeline.
  // Use when input is ONNX dialect MLIR (from onnx-mlir frontend).
  mlir::PassPipelineRegistration<PipelineOptions> morphizenPipeline(
      "morphizen-pipeline",
      "Complete Morphizen ONNX→HIP→LLVM→Interface pipeline",
      [&compilationOpts, &fs](mlir::OpPassManager &pm,
                              const PipelineOptions &opts) {
        compilationOpts.constants_file = opts.constantsFile;
        compilationOpts.opt_level = opts.optLevel;
        compilationOpts.verbose = opts.verbose;

        const std::string &dir = opts.constantsDir;
        if (!dir.empty())
          llvm::sys::fs::create_directories(dir);
        fs = std::make_unique<udna::DiskFileSystem>(
            dir.empty() ? "." : dir.c_str());

        udna::compiler::compiler::populateMorphizenPipeline(pm, compilationOpts,
                                                            fs.get());
      });

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "hip-mlir-opt: HIP dialect compiler driver\n", registry));
}
