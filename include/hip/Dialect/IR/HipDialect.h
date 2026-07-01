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

// HIP memory-space enum + attribute (#hip.mem<device|host|pinned|managed>).
// The enum header must precede the attribute header: the generated
// MemorySpaceAttr class references the MemorySpaceKind enum.
#include "hip/Dialect/IR/HipEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hip/Dialect/IR/HipAttributes.h.inc"

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
/// `(RuntimeState*, i32 slot, i64 x N) -> i8`, emit the call passing
/// `statePtr`, the `slot` constant, and `i64Args` as constants, and return the
/// i8 result (vestigial, always 0). The construct symbol stores the built state
/// into op_states[slot] itself (via hipdnn_ep_op_state_set), so the caller
/// (--generate-op-state-init) does not emit a separate store. Defined in
/// lib/Dialect/IR/HipOpStateInterface.cpp.
::mlir::Value emitOpStateConstruct(::mlir::OpBuilder &builder,
                                   ::mlir::Location loc, ::mlir::Value statePtr,
                                   int32_t slot, ::llvm::StringRef ctorSymbol,
                                   ::llvm::ArrayRef<int64_t> i64Args);

/// Memory-space operand predicates backing the Hip_TensorOr{Device,Host,
/// Pinned,Managed}MemRef type constraints (see HipOps.td). One predicate per
/// `#hip.mem<...>` kind:
///
/// - `isDeviceCompatibleMemRef`  : memref carrying `#hip.mem<device>`
/// - `isHostCompatibleMemRef`    : memref carrying `#hip.mem<host>`
/// - `isPinnedCompatibleMemRef`  : memref carrying `#hip.mem<pinned>`
/// - `isManagedCompatibleMemRef` : memref carrying `#hip.mem<managed>`
///
/// TRANSITIONAL: the current pipeline does not yet stamp a `#hip.mem<...>`
/// space onto memrefs, so a memref with NO hip memory space currently
/// satisfies ALL four predicates. This relaxed check is controlled by a single
/// toggle (`kAcceptUnspecifiedMemorySpace` in HipDialect.cpp); flip it to
/// enforce that every constrained operand carries an explicit space.
bool isDeviceCompatibleMemRef(::mlir::Type type);
bool isHostCompatibleMemRef(::mlir::Type type);
bool isPinnedCompatibleMemRef(::mlir::Type type);
bool isManagedCompatibleMemRef(::mlir::Type type);
} // namespace hip
} // namespace mlir

#define GET_OP_CLASSES
#include "hip/Dialect/IR/HipOps.h.inc"

#endif // HIP_DIALECT_H
