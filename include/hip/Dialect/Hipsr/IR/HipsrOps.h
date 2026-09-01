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

// return the result held in the destination use names, or null if none.
::mlir::OpResult getHipsrResultHeldIn(::mlir::OpOperand &use);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_OPS_H
