/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip dialect-specific FFI bindings.
//
// Only put here what genuinely requires hip dialect ODS-generated accessors
// or hip-specific type APIs not reachable through generic MLIR primitives.
// All other hip-specific logic belongs in scheme/mlir/hip/fusion.sls,
// implemented in Scheme using the general primitives in Core/.

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include <limits>
#include <optional>

// Reuse inline helpers from the PDLL fusion layer — single source of truth.
// The header lives in lib/Dialect/Transforms/fusion_pattern/ which is added
// as a private include directory for HipsrSchemeRuntime below.
#include "fusion_pattern/hip_fusion_transform.hpp"

using namespace hip::fusion_transform;

extern "C" {

// Extract the float value of a splat hip.constant scale operand.
// val_ptr: Value* (opaque) — the scale Value.
// Returns the float as double; NaN when not a splat float constant.
double hip_extract_splat_scale(uint64_t val_ptr) {
  if (!val_ptr) return std::numeric_limits<double>::quiet_NaN();
  auto val = mlir::Value::getFromOpaquePointer(reinterpret_cast<const void*>(val_ptr));
  std::optional<float> scale = tryHipSplatScale(val);
  if (!scale) return std::numeric_limits<double>::quiet_NaN();
  return static_cast<double>(*scale);
}

// Build a tensor.empty whose element type comes from result_type and whose
// dynamic dimensions are read off shape_source.
// rw_ptr:           RewriterBase* as uptr
// result_type_ptr:  RankedTensorType* opaque ptr (the desired type)
// shape_source_ptr: Value* opaque ptr (shape donor)
// Returns: the tensor.empty Value* as uptr, or 0 on failure.
uint64_t hip_build_init(uint64_t rw_ptr, uint64_t result_type_ptr,
                         uint64_t shape_source_ptr) {
  if (!rw_ptr || !result_type_ptr || !shape_source_ptr) return 0;
  auto* rw = reinterpret_cast<mlir::RewriterBase*>(rw_ptr);
  auto  resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(result_type_ptr)));
  auto  shapeSource = mlir::Value::getFromOpaquePointer(
      reinterpret_cast<const void*>(shape_source_ptr));
  if (!resultType || !shapeSource) return 0;

  // Cast rw to PatternRewriter for buildInitValue (it only uses OpBuilder API).
  auto* pr = static_cast<mlir::PatternRewriter*>(rw);
  mlir::Value init = buildInitValue(*pr, resultType, shapeSource);
  if (!init) return 0;
  return reinterpret_cast<uint64_t>(init.getAsOpaquePointer());
}

// Clone a layout op (hip.transpose, tensor.collapse_shape, etc.) replacing
// its dequantized float operand with the original quantized input and
// rebuilding any DPS init for the quantized result type.
// rw_ptr:      RewriterBase* as uptr
// dq_ptr:      hip.dequantize_linear Operation* as uptr
// layout_ptr:  the layout Operation* to clone (matched by :any)
// q_ptr:       hip.quantize_linear Operation* as uptr (determines result type)
// Returns: new Operation*'s result Value* as uptr, or 0 on failure.
uint64_t hip_create_requantized_layout_op(uint64_t rw_ptr, uint64_t dq_ptr,
                                           uint64_t layout_ptr, uint64_t q_ptr) {
  if (!rw_ptr || !dq_ptr || !layout_ptr || !q_ptr) return 0;
  auto* rw     = reinterpret_cast<mlir::RewriterBase*>(rw_ptr);
  auto* layout = reinterpret_cast<mlir::Operation*>(layout_ptr);
  auto* qOp    = reinterpret_cast<mlir::Operation*>(q_ptr);

  auto dq = mlir::dyn_cast<mlir::hip::DequantizeLinearOp>(
      reinterpret_cast<mlir::Operation*>(dq_ptr));
  if (!dq) return 0;

  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
      qOp->getResult(0).getType());
  if (!resultType) return 0;

  // Replace the dequantized float operand with the original quantized input.
  llvm::SmallVector<mlir::Value> operands(layout->getOperands());
  for (mlir::Value& operand : operands)
    if (operand == dq.getResult(0))
      operand = dq.getInput();

  // Rebuild the DPS init for the quantized result type if present.
  if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(layout)) {
    mlir::OpOperand& init = dps.getDpsInitsMutable()[0];
    auto* pr = static_cast<mlir::PatternRewriter*>(rw);
    operands[init.getOperandNumber()] = buildInitValue(*pr, resultType, init.get());
  }

  mlir::OperationState state(layout->getLoc(), layout->getName());
  state.addOperands(operands);
  state.addTypes(resultType);
  state.addAttributes(layout->getAttrs());
  mlir::Operation* newOp = rw->create(state);
  return reinterpret_cast<uint64_t>(newOp->getResult(0).getAsOpaquePointer());
}

// Returns 1 if the zero-point of op is extractable (absent or a splat constant),
// 0 otherwise. Absent → treated as 0, which is always extractable.
int hip_extractable_qdq_zeropoint(uint64_t op_ptr) {
  if (!op_ptr) return 0;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  return tryHipQdqZeropoint(op, 0).has_value() ? 1 : 0;
}

// Extracts the zero-point of op as i64. Returns absent_value if absent.
// Returns INT64_MIN on extraction failure (non-constant zero point).
int64_t hip_extract_qdq_zeropoint_i64(uint64_t op_ptr, int64_t absent_value) {
  if (!op_ptr) return absent_value;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  std::optional<int64_t> zp = tryHipQdqZeropoint(op, absent_value);
  return zp.value_or(INT64_MIN);
}

// Extracts the logical quantized bit-width: 4 if packed_int4, else the storage
// integer width. Returns 0 on failure.
int64_t hip_qdq_value_bits(uint64_t op_ptr) {
  if (!op_ptr) return 0;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto dq = mlir::dyn_cast<mlir::hip::DequantizeLinearOp>(op);
  if (!dq) {
    auto q = mlir::dyn_cast<mlir::hip::QuantizeLinearOp>(op);
    if (!q) return 0;
  }
  mlir::IntegerType intType = getQdqQuantizedElementType(op);
  if (!intType) return 0;
  if (auto dq2 = mlir::dyn_cast<mlir::hip::DequantizeLinearOp>(op))
    return dq2.getPackedInt4() ? 4 : static_cast<int64_t>(intType.getWidth());
  return static_cast<int64_t>(intType.getWidth());
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerHipFusionBindings() {
  Sregister_symbol("hip_extract_splat_scale",            (void*)::hip_extract_splat_scale);
  Sregister_symbol("hip_build_init",                     (void*)::hip_build_init);
  Sregister_symbol("hip_create_requantized_layout_op",   (void*)::hip_create_requantized_layout_op);
  Sregister_symbol("hip_extractable_qdq_zeropoint",      (void*)::hip_extractable_qdq_zeropoint);
  Sregister_symbol("hip_extract_qdq_zeropoint_i64",      (void*)::hip_extract_qdq_zeropoint_i64);
  Sregister_symbol("hip_qdq_value_bits",                 (void*)::hip_qdq_value_bits);
}

} // namespace hipsr
} // namespace mlir
