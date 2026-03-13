/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/Dialect/Transforms/Pipelines.h"

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
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

/// Minimal ONNX dialect stub that claims the "onnx" namespace and permits
/// unknown operations.  This allows hip-mlir-opt to parse generic-syntax
/// ONNX MLIR (e.g. "onnx.MatMul"(...)) without requiring the full onnx-mlir
/// dialect library.
class OnnxStubDialect : public mlir::Dialect {
public:
  explicit OnnxStubDialect(mlir::MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx,
                mlir::TypeID::get<OnnxStubDialect>()) {
    allowUnknownOperations();
  }
  static constexpr llvm::StringLiteral getDialectNamespace() { return "onnx"; }
};

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
    mlir::hip::HipblasltMatmulOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::HipblasltMatmulOp>>(*ctx);
    mlir::hip::RmsNormOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::RmsNormOp>>(*ctx);
    mlir::hip::SkipRmsNormOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SkipRmsNormOp>>(*ctx);
    mlir::hip::RopeOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::RopeOp>>(*ctx);
    mlir::hip::MiopenAddOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MiopenAddOp>>(*ctx);
    mlir::hip::MulOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MulOp>>(*ctx);
    mlir::hip::MiopenSoftmaxOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::MiopenSoftmaxOp>>(*ctx);
    mlir::hip::TransposeOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::TransposeOp>>(*ctx);
    mlir::hip::GatherOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GatherOp>>(*ctx);
    mlir::hip::SiluOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::SiluOp>>(*ctx);
    mlir::hip::GqaOp::attachInterface<
        HipDstBufferizableModel<mlir::hip::GqaOp>>(*ctx);
  });
}

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
  registry.insert<OnnxStubDialect>();

  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  registerHipBufferizableOpInterfaceModels(registry);

  mlir::hip::registerHipPasses();
  mlir::hip::registerHipPipelines();
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

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "hip-mlir-opt: HIP dialect compiler driver\n", registry));
}
