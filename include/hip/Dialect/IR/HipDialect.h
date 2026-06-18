/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_H
#define HIP_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "hip/Dialect/IR/HipDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hip/Dialect/IR/HipTypes.h.inc"

// Generated header for the `HipDpsOp` C++ interface class (TableGen def
// `HipDpsOpInterface`). Must precede HipOps.h.inc — every Hip_DpsOp's
// generated declaration references `HipDpsOp::Trait`.
#include "hip/Dialect/IR/HipDpsOpInterface.h.inc"

// Generated header for the `OpStateOpInterface` C++ interface class. Must
// precede HipOps.h.inc — stateful ops reference `OpStateOpInterface::Trait`.
#include "hip/Dialect/IR/HipOpStateInterface.h.inc"

namespace mlir {
namespace hip {
/// Shared helper for `OpStateOpInterface::generateOpStateInit` bodies: declare
/// (or look up) the extern construct symbol `ctorSymbol` with signature
/// `(RuntimeState*, i64 x N) -> OpState*`, emit the call passing `statePtr`
/// plus `i64Args` as constants, and return the constructed pointer. The caller
/// (--generate-op-state-init) stores the result into op_states[slot]. Defined
/// in lib/Dialect/IR/HipOpStateInterface.cpp.
::mlir::Value emitOpStateConstruct(::mlir::OpBuilder &builder,
                                   ::mlir::Location loc, ::mlir::Value statePtr,
                                   ::llvm::StringRef ctorSymbol,
                                   ::llvm::ArrayRef<int64_t> i64Args);
} // namespace hip
} // namespace mlir

#define GET_OP_CLASSES
#include "hip/Dialect/IR/HipOps.h.inc"

#endif // HIP_DIALECT_H
