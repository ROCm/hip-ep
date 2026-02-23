/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_BUFFERIZE_H
#define HIP_BUFFERIZE_H

#include "HipDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace hip {

template <typename OpTy>
struct HipDstBufferizableModel
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          HipDstBufferizableModel<OpTy>, OpTy> {
  LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                          const bufferization::BufferizationOptions& options,
                          bufferization::BufferizationState& state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);

    SmallVector<Value> newOperands;
    for (OpOperand& operand : op->getOpOperands()) {
      if (isa<TensorType>(operand.get().getType())) {
        FailureOr<Value> buffer =
            getBuffer(rewriter, operand.get(), options, state);
        if (failed(buffer))
          return failure();
        newOperands.push_back(*buffer);
      } else {
        newOperands.push_back(operand.get());
      }
    }

    OpTy::create(rewriter, op->getLoc(), TypeRange{}, newOperands,
                 op->getAttrs());

    SmallVector<Value> replacements;
    for (OpResult result : op->getResults()) {
      if (!isa<TensorType>(result.getType())) {
        replacements.push_back(result);
        continue;
      }
      OpOperand* initOperand = dstOp.getTiedOpOperand(result);
      FailureOr<Value> initBuffer =
          getBuffer(rewriter, initOperand->get(), options, state);
      if (failed(initBuffer))
        return failure();
      replacements.push_back(*initBuffer);
    }

    bufferization::replaceOpWithBufferizedValues(rewriter, op, replacements);
    return success();
  }
};

inline void
registerHipBufferizableOpInterfaceModels(DialectRegistry& registry) {
  registry.addExtension(+[](MLIRContext* ctx, HipDialect*) {
    HipblasltMatmulOp::attachInterface<
        HipDstBufferizableModel<HipblasltMatmulOp>>(*ctx);
    MiopenRmsNormOp::attachInterface<
        HipDstBufferizableModel<MiopenRmsNormOp>>(*ctx);
    MiopenSkipRmsNormOp::attachInterface<
        HipDstBufferizableModel<MiopenSkipRmsNormOp>>(*ctx);
    MiopenRopeOp::attachInterface<HipDstBufferizableModel<MiopenRopeOp>>(
        *ctx);
    MiopenAddOp::attachInterface<HipDstBufferizableModel<MiopenAddOp>>(*ctx);
    MiopenMulOp::attachInterface<HipDstBufferizableModel<MiopenMulOp>>(*ctx);
    MiopenSoftmaxOp::attachInterface<
        HipDstBufferizableModel<MiopenSoftmaxOp>>(*ctx);
    TransposeOp::attachInterface<HipDstBufferizableModel<TransposeOp>>(*ctx);
    GatherOp::attachInterface<HipDstBufferizableModel<GatherOp>>(*ctx);
    SiluOp::attachInterface<HipDstBufferizableModel<SiluOp>>(*ctx);
    GqaOp::attachInterface<HipDstBufferizableModel<GqaOp>>(*ctx);
  });
}

}  // namespace hip
}  // namespace mlir

#endif  // HIP_BUFFERIZE_H
