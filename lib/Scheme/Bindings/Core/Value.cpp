/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Mirrors (mlir core value): Value and ValueArrayRef primitives.

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"

extern "C" {

// Return the Operation* that defines a Value (i.e. the op whose result this is).
// value: Value* as uptr (via MLIR C-API opaque pointer)
// Returns: Operation* as uptr, or 0 if the value is a block argument (no defining op).
uint64_t mlir_value_get_defining_op(uint64_t value) {
  if (!value) return 0;
  MlirValue cVal{reinterpret_cast<const void*>(value)};
  return reinterpret_cast<uint64_t>(unwrap(cVal).getDefiningOp());
}

// Test whether a Value is a block argument (rather than an op result).
// value: Value* as uptr
// Returns: 1 if the value is a BlockArgument, 0 if it is an OpResult or null.
int mlir_value_is_block_argument(uint64_t value) {
  if (!value) return 0;
  mlir::Value val = unwrap(MlirValue{reinterpret_cast<const void*>(value)});
  return mlir::isa<mlir::BlockArgument>(val) ? 1 : 0;
}

// Return the zero-based result index of an OpResult Value within its defining op.
// value: Value* as uptr; must be an OpResult (not a BlockArgument)
// Returns: result index (>= 0), or -1 if the value is a BlockArgument or null.
int mlir_value_get_result_number(uint64_t value) {
  if (!value) return -1;
  mlir::Value val = unwrap(MlirValue{reinterpret_cast<const void*>(value)});
  auto result = mlir::dyn_cast<mlir::OpResult>(val);
  if (!result) return -1;
  return static_cast<int>(result.getResultNumber());
}

// Return the number of uses of a Value.
// Useful for single-use guards (e.g. hip-op-single-use? in Scheme).
uint64_t mlir_value_num_uses(uint64_t val_ptr) {
  if (!val_ptr) return 0;
  auto val = mlir::Value::getFromOpaquePointer(reinterpret_cast<const void*>(val_ptr));
  return static_cast<uint64_t>(std::distance(val.use_begin(), val.use_end()));
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerValueBindings() {
  Sregister_symbol("mlir_value_get_defining_op",   (void*)::mlir_value_get_defining_op);
  Sregister_symbol("mlir_value_is_block_argument", (void*)::mlir_value_is_block_argument);
  Sregister_symbol("mlir_value_get_result_number", (void*)::mlir_value_get_result_number);
  Sregister_symbol("mlir_value_num_uses",          (void*)::mlir_value_num_uses);
}

} // namespace hipsr
} // namespace mlir
