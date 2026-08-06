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

// Bufferization model for hipsr.preserve_shape. The op is not DPS -- it has no
// results and no init operand -- so DstBufferizableOpInterfaceExternalModel
// does not apply and the three analysis queries have to be answered by hand.
//
// It records that `$shape` describes `$data` and touches neither, so all three
// report "nothing happens here". That is what keeps the op from ever forcing a
// copy on the buffer it names: with no read, no write and no aliasing value,
// the One-Shot analysis has no conflict to resolve, and a DPS op downstream is
// still free to write that buffer in place. The op then names whichever buffer
// its `$data` tensor was folded into, which is the reason it exists.
//
// The Write effect the op reports through MemoryEffectOpInterface is a separate
// mechanism, there only to keep DCE off a result-less op. Bufferization never
// consults it; it reads the queries below.
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
