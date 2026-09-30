/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Mirrors (mlir core value): Value and ValueArrayRef primitives.

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Value.h"

extern "C" {

uint64_t mlir_value_get_defining_op(uint64_t value) {
  if (!value) return 0;
  MlirValue cVal{reinterpret_cast<const void*>(value)};
  return reinterpret_cast<uint64_t>(unwrap(cVal).getDefiningOp());
}

int mlir_value_is_block_argument(uint64_t value) {
  if (!value) return 0;
  mlir::Value val = unwrap(MlirValue{reinterpret_cast<const void*>(value)});
  return mlir::isa<mlir::BlockArgument>(val) ? 1 : 0;
}

int mlir_value_get_result_number(uint64_t value) {
  if (!value) return -1;
  mlir::Value val = unwrap(MlirValue{reinterpret_cast<const void*>(value)});
  auto result = mlir::dyn_cast<mlir::OpResult>(val);
  if (!result) return -1;
  return static_cast<int>(result.getResultNumber());
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerValueBindings() {
  Sregister_symbol("mlir_value_get_defining_op",   (void*)::mlir_value_get_defining_op);
  Sregister_symbol("mlir_value_is_block_argument", (void*)::mlir_value_is_block_argument);
  Sregister_symbol("mlir_value_get_result_number", (void*)::mlir_value_get_result_number);
}

} // namespace hipsr
} // namespace mlir
