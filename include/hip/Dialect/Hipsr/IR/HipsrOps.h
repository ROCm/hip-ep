/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_OPS_H
#define HIPSR_OPS_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrOps.h.inc"

namespace mlir {
namespace hipsr {

// return Outputs if op is a hipsr.compute op, or the DPS inits if op is a
// hipsr DPS op. return empty range if op is not a hipsr op.
::mlir::OperandRange getHipsrDestinationOperands(::mlir::Operation *op);

// return the data operands of a hipsr op, which sit between the context and
// the destinations. return empty range if op is not a hipsr op.
::mlir::OperandRange getHipsrInputOperands(::mlir::Operation *op);

// return the shape-graph value holding value's shape: value itself when it is
// already a legal placeholder input, otherwise the destination its producer
// writes into. return value unchanged when no destination matches.
::mlir::Value getShapeGraphCounterpart(::mlir::Value value);

// compute: use is in Outputs
// dps: use is in Init
bool isHipsrDestinationOperand(::mlir::OpOperand &use);

// Returns the result that aliases this destination operand.
// DPS pattern: outs()[i] and result[i] occupy the same buffer.
// hipsr.compute is not DPS but stores its results in the same positions.
// Returns a null OpResult if the operand is not a destination.
//
// Example: %r0, %r1 = hipsr.foo outs(%init0, %init1)
//          getResultForDestination(%init0) -> %r0
//          getResultForDestination(%init1) -> %r1
//
// Example: %r = hipsr.compute outs(%d : tensor<?x256xf16>) : tensor<?xf16>
//          getResultForDestination(%d) -> %r
::mlir::OpResult getResultForDestination(::mlir::OpOperand &use);

// Example: getExtentTensorTypeForRank(ctx, 2) -> tensor<2xindex>
::mlir::RankedTensorType getExtentTensorTypeForRank(::mlir::MLIRContext *ctx,
                                                    int64_t rank);

// Takes a ranked data tensor, not a shape or a buffer.
//
// Example: %d : tensor<?x4xf16> -> tensor<2xindex>
::mlir::RankedTensorType getExtentTensorTypeOf(::mlir::Value data);

// Broadcasting left-pads, so the result is as long as the longest input. Every
// input must state its extent count.
//
// Example: {tensor<3xindex>, tensor<1xindex>} -> tensor<3xindex>
::mlir::RankedTensorType
getBroadcastExtentTensorType(::mlir::ValueRange shapes);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_OPS_H
