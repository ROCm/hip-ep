/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- BufferizableOpInterfaceImpl.cpp - hipsr bufferization models -------===//
//
// BufferizableOpInterface models for the hipsr dialect. A new model belongs in
// the anonymous namespace below and is attached in the single registration
// function at the bottom.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/BufferizableOpInterfaceImpl.h"

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::hipsr;

namespace mlir {
namespace hipsr {
namespace {

// hipsr.preserve_shape only records that `$shape` describes `$data`. It has no
// results and no init operand, so it is not DPS and cannot reuse
// DstBufferizableOpInterfaceExternalModel. It touches no memory, so nothing
// needs copying and a later DPS op can write the buffer in place.
struct PreserveShapeBufferizableModel
    : public BufferizableOpInterface::ExternalModel<
          PreserveShapeBufferizableModel, PreserveShapeOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto preserveOp = cast<PreserveShapeOp>(op);
    FailureOr<Value> dataBuf =
        getBuffer(rewriter, preserveOp.getData(), options, state);
    if (failed(dataBuf)) {
      return failure();
    }
    replaceOpWithNewBufferizedOp<PreserveShapeOp>(
        rewriter, op, preserveOp.getShape(), *dataBuf);
    return success();
  }
};

} // namespace
} // namespace hipsr
} // namespace mlir

void mlir::hipsr::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HipsrDialect *) {
    PreserveShapeOp::attachInterface<PreserveShapeBufferizableModel>(*ctx);
  });
}
