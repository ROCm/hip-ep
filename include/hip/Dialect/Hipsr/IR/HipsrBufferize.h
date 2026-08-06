/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPSR_BUFFERIZE_H
#define HIPSR_BUFFERIZE_H

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace hipsr {

// Bufferization model for hipsr.preserve_shape.
// This op is not DPS: it has no results and no init operand.
// DstBufferizableOpInterfaceExternalModel does not apply,
// so the three analysis queries are implemented manually.
//
// The op only records that `$shape` describes `$data`.
// It does not read, write, or create aliases, so all queries return
// "nothing happens here".
// This avoids unnecessary buffer copies and allows later DPS ops
// to write the buffer in place.

struct PreserveShapeBufferizableModel
    : public bufferization::BufferizableOpInterface::ExternalModel<
          PreserveShapeBufferizableModel, PreserveShapeOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const bufferization::AnalysisState &) const {
    return false;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const bufferization::AnalysisState &) const {
    return false;
  }
  bufferization::AliasingValueList
  getAliasingValues(Operation *, OpOperand &,
                    const bufferization::AnalysisState &) const {
    return {};
  }
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto preserveOp = cast<PreserveShapeOp>(op);
    // The op can accept both memref and tensor types.
    // This means the op may already be in the final form after bufferization,
    // In this case, we don't need to bufferize it again.
    if (!isa<TensorType>(preserveOp.getData().getType()))
      return success();

    FailureOr<Value> dataBuf =
        getBuffer(rewriter, preserveOp.getData(), options, state);
    if (failed(dataBuf))
      return failure();
    PreserveShapeOp::create(rewriter, op->getLoc(), preserveOp.getShape(),
                            *dataBuf);
    rewriter.eraseOp(op);
    return success();
  }
};

inline void
registerHipsrBufferizableOpInterfaceModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HipsrDialect *) {
    PreserveShapeOp::attachInterface<PreserveShapeBufferizableModel>(*ctx);
  });
}

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_BUFFERIZE_H
