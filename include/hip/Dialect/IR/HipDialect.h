/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_H
#define HIP_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
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

namespace mlir {
namespace OpTrait {
namespace hip {
/// Op trait marking ops whose result EXTENT depends on input *data values*
/// (not just input shapes) — e.g. ONNX NonZero, whose trailing output dim is
/// the count of non-zero elements (knowable only after scanning the data).
/// Generic passes and the EP-metadata builder query this with
/// `op->hasTrait<mlir::OpTrait::hip::DataDependentResult>()` instead of
/// name-matching, so adding a future data-dependent op (Unique, Compress,
/// dynamic TopK, …) needs no edits to the generic resolution logic. Must be
/// declared before HipOps.h.inc, whose generated op class lists the trait.
template <typename ConcreteType>
class DataDependentResult
    : public TraitBase<ConcreteType, DataDependentResult> {};
} // namespace hip
} // namespace OpTrait
} // namespace mlir

#define GET_OP_CLASSES
#include "hip/Dialect/IR/HipOps.h.inc"

#endif // HIP_DIALECT_H
