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

#include "llvm/ADT/StringRef.h"

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
/// Discardable attribute names stamped on `hip.alloc_output` by
/// `hip-use-output-allocator` and consumed by `AllocOutputOpLowering`. They
/// describe the ONNX / func.return ("ABI") output shape when it is a
/// rank-reduced view of the internal compute buffer -- e.g. a rank-3 vision
/// buffer whose ONNX output is returned rank-2 through `memref.collapse_shape`.
///
/// The mapping is captured while the collapse reassociation is still explicit,
/// i.e. BEFORE `expand-strided-metadata` decomposes collapse_shape into
/// reinterpret_cast + extract_strided_metadata (which erases the reassociation
/// and re-defines the external dims *after* the alloc, so the HIP->LLVM
/// lowering can no longer re-derive them without violating SSA dominance at the
/// alloc site). The lowering then re-computes each external dim from the
/// internal alloc sizes (which dominate) -- static dims from the attribute,
/// dynamic dims as the runtime product of the internal dims they fold.
///   kAbiShapeAttrName  : DenseI64ArrayAttr -- external shape, one entry per
///                        external dim (ShapedType::kDynamic marks dynamic).
///   kAbiGroupsAttrName : DenseI64ArrayAttr -- number of consecutive internal
///                        dims folded into each external dim (contiguous
///                        collapse reassociation; entries sum to internal rank).
inline constexpr ::llvm::StringLiteral kAbiShapeAttrName = "hipdnn.abi_shape";
inline constexpr ::llvm::StringLiteral kAbiGroupsAttrName = "hipdnn.abi_groups";

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
} // namespace hip
} // namespace mlir

#define GET_OP_CLASSES
#include "hip/Dialect/IR/HipOps.h.inc"

#endif // HIP_DIALECT_H
