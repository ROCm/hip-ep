/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "CrashHandler.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/Dialect/Transforms/Pipelines.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
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
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "hip/InitAllPasses.h"

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
    mlir::hip::ConvOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ConvOp>>(*ctx);
    mlir::hip::ConvTransposeOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ConvTransposeOp>>(*ctx);
    mlir::hip::MatmulOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MatmulOp>>(*ctx);
    mlir::hip::RmsNormOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::RmsNormOp>>(*ctx);
    mlir::hip::SkipRmsNormOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SkipRmsNormOp>>(*ctx);
    mlir::hip::RopeOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::RopeOp>>(*ctx);
    mlir::hip::MiopenAddOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MiopenAddOp>>(*ctx);
    mlir::hip::AddOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::AddOp>>(*ctx);
    mlir::hip::MulOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MulOp>>(*ctx);
    mlir::hip::MiopenSoftmaxOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MiopenSoftmaxOp>>(*ctx);
    mlir::hip::TransposeOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::TransposeOp>>(*ctx);
    mlir::hip::GatherOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GatherOp>>(*ctx);
    mlir::hip::RangeOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::RangeOp>>(*ctx);
    mlir::hip::SiluOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SiluOp>>(*ctx);
    mlir::hip::GqaOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GqaOp>>(*ctx);
    mlir::hip::CastOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::CastOp>>(*ctx);
    mlir::hip::SigmoidOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SigmoidOp>>(*ctx);
    mlir::hip::SoftplusOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SoftplusOp>>(*ctx);
    mlir::hip::GeluOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GeluOp>>(*ctx);
    mlir::hip::LeakyReluOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::LeakyReluOp>>(*ctx);
    mlir::hip::PoolOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::PoolOp>>(*ctx);
    mlir::hip::ResizeOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ResizeOp>>(*ctx);
    mlir::hip::GlobalPoolOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GlobalPoolOp>>(*ctx);
    mlir::hip::ReciprocalOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ReciprocalOp>>(*ctx);
    mlir::hip::SqrtOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SqrtOp>>(*ctx);
    mlir::hip::SubOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SubOp>>(*ctx);
    mlir::hip::ReduceSumOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ReduceSumOp>>(*ctx);
    mlir::hip::ReduceMaxOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ReduceMaxOp>>(*ctx);
    mlir::hip::MatMulNBitsOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MatMulNBitsOp>>(*ctx);
    mlir::hip::QMoEOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::QMoEOp>>(*ctx);
    mlir::hip::LinearAttentionOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::LinearAttentionOp>>(*ctx);
    mlir::hip::CausalConvWithStateOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::CausalConvWithStateOp>>(*ctx);
    mlir::hip::HipDNNGraphOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::HipDNNGraphOp>>(*ctx);
    mlir::hip::WhereOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::WhereOp>>(*ctx);
    mlir::hip::LayerNormOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::LayerNormOp>>(*ctx);
    mlir::hip::MinOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MinOp>>(*ctx);
    mlir::hip::NegOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::NegOp>>(*ctx);
    mlir::hip::EqualOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::EqualOp>>(*ctx);
    mlir::hip::DivOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::DivOp>>(*ctx);
    mlir::hip::NotOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::NotOp>>(*ctx);
    mlir::hip::AndOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::AndOp>>(*ctx);
    mlir::hip::CosOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::CosOp>>(*ctx);
    mlir::hip::SinOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SinOp>>(*ctx);
    mlir::hip::ExpOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ExpOp>>(*ctx);
    mlir::hip::CumSumOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::CumSumOp>>(*ctx);
    mlir::hip::PadOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::PadOp>>(*ctx);
    mlir::hip::TileOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::TileOp>>(*ctx);
    mlir::hip::ExpandOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ExpandOp>>(*ctx);
    mlir::hip::ReduceProdOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ReduceProdOp>>(*ctx);
    mlir::hip::LessOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::LessOp>>(*ctx);
    mlir::hip::GatherNDOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GatherNDOp>>(*ctx);
    mlir::hip::SignOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SignOp>>(*ctx);
    mlir::hip::ModOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ModOp>>(*ctx);
    mlir::hip::SliceOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SliceOp>>(*ctx);
    mlir::hip::ScatterNDOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::ScatterNDOp>>(*ctx);
    mlir::hip::MultiHeadAttentionOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MultiHeadAttentionOp>>(*ctx);
    mlir::hip::NonZeroOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::NonZeroOp>>(*ctx);
    mlir::hip::SizeOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SizeOp>>(*ctx);
    mlir::hip::LoopOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::LoopOp>>(*ctx);
  });
}

} // namespace

int main(int argc, char **argv) {
  hip::install_crash_handlers("hip-mlir-opt");
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
  registry.insert<hip::compiler::detail::OnnxStubDialect>();

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  registerHipBufferizableOpInterfaceModels(registry);

  mlir::hip::registerHipPasses();
  mlir::hip::registerHipPipelines();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createOutlineOnnxToHipDNNPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::hip::createOnnxLoopOutlinePass();
  });
  mlir::bufferization::registerBufferizationPasses();
  mlir::bufferization::registerBufferizationPipelines();
  mlir::registerConvertBufferizationToMemRefPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  // Registered so that LIT tests and end-to-end pipelines can fold
  // `tensor.dim` / `memref.dim` of HIP op results through the reify
  // implementation. Used in `hip-matmul-reify-shapes.mlir`.
  mlir::memref::registerResolveShapedTypeResultDimsPass();
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return mlir::createCSEPass(); });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::createCanonicalizerPass();
  });

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "hip-mlir-opt: HIP dialect compiler driver\n", registry));
}
