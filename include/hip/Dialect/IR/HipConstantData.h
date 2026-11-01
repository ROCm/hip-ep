/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_CONSTANT_DATA_H
#define HIP_DIALECT_IR_HIP_CONSTANT_DATA_H

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace hip {

/// Read the payload of a `hip.constant` carrier as a `DenseElementsAttr` of
/// `type`, whatever source it holds.
///
/// "Compile-time constant" is not the same as "inline `value`": the importer
/// externalizes initializers past a small byte threshold, so an ordinary weight
/// reaches the compiler as a file-backed byte range or a memory-address range
/// rather than an inline attribute.
///
/// Memory-address carriers are never resolved. That address is process-local
/// and valid only while the producer that recorded it is live, which a pass
/// cannot establish; textual pass invocation would dereference it blind.
///
/// Fails (without diagnostics) when the payload does not match `type` or the
/// source cannot be read.
FailureOr<DenseElementsAttr> readConstantData(ConstantOp constant,
                                              RankedTensorType type);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_CONSTANT_DATA_H
