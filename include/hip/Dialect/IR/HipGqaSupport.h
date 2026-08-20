/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_GQA_SUPPORT_H
#define HIP_DIALECT_IR_HIP_GQA_SUPPORT_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace hip {

enum class GqaKvCacheMode {
  Unquantized,
  Int8PerChannel,
};

/// Types and optional-presence facts used to validate the generated GQA
/// runtime ABI. A null Type represents an omitted optional operand.
struct GqaFeatureTypes {
  Type query;
  Type key;
  Type value;
  Type pastKey;
  Type pastValue;
  Type output;
  Type presentKey;
  Type presentValue;
  Type kScale;
  Type vScale;
};

/// Validate the GQA features whose runtime interpretation depends on attributes
/// and element types. This helper emits no IR and is shared by conversion,
/// verification, reification, and lowering so failure is mutation-free.
FailureOr<GqaKvCacheMode>
verifyGqaFeatureSupport(StringRef kQuantType, StringRef vQuantType,
                        int64_t kvCacheBitWidth, int64_t rotaryInterleaved,
                        int64_t kvNumHeads, const GqaFeatureTypes &types,
                        function_ref<InFlightDiagnostic()> emitError);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_GQA_SUPPORT_H
