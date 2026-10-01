/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Mirrors (mlir core operation): operation inspection, mutation, attr access,
// walk, and diagnostic emission.

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "hip/Scheme/Bindings/LockedSchemeObject.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

extern "C" {

// Return the registered name of an operation (e.g. "onnx.Cast", "hipsr.min").
// op: Operation* as uptr
// Returns: pointer into op's name storage; valid for the lifetime of the op.
//          Returns "" for null op.
const char* mlir_operation_get_name(uint64_t op) {
  if (!op) return "";
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return cppOp->getName().getStringRef().data();
}

// Return the MLIRContext that owns this operation.
// op: Operation* as uptr
// Returns: MLIRContext* as uptr, or 0 for null op.
uint64_t mlir_operation_get_context(uint64_t op) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return reinterpret_cast<uint64_t>(cppOp->getContext());
}

// Return the number of operands (Value inputs) of an operation.
// op: Operation* as uptr
// Returns: operand count, or 0 for null op.
int64_t mlir_operation_num_operands(uint64_t op) {
  if (!op) return 0;
  return reinterpret_cast<mlir::Operation*>(op)->getNumOperands();
}

// Return the number of results (Value outputs) of an operation.
// op: Operation* as uptr
// Returns: result count, or 0 for null op.
int64_t mlir_operation_num_results(uint64_t op) {
  if (!op) return 0;
  return reinterpret_cast<mlir::Operation*>(op)->getNumResults();
}

// Return the operand Value at position index (MLIR C-API wrapped pointer).
// op: Operation* as uptr
// index: zero-based operand index
// Returns: Value* as uptr (via MLIR C-API wrap), or 0 if out of range.
uint64_t mlir_operation_get_operand(uint64_t op, int64_t index) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  if (index < 0 || index >= (int64_t)cppOp->getNumOperands()) return 0;
  mlir::Value val = cppOp->getOperand(index);
  MlirValue cVal = wrap(val);
  return reinterpret_cast<uint64_t>(const_cast<void*>(cVal.ptr));
}

// Return the result Value at position index (MLIR C-API wrapped pointer).
// op: Operation* as uptr
// index: zero-based result index
// Returns: Value* as uptr (via MLIR C-API wrap), or 0 if out of range.
uint64_t mlir_operation_get_result(uint64_t op, int64_t index) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  if (index < 0 || index >= (int64_t)cppOp->getNumResults()) return 0;
  mlir::Value val = cppOp->getResult(index);
  MlirValue cVal = wrap(val);
  return reinterpret_cast<uint64_t>(const_cast<void*>(cVal.ptr));
}

// Return the nearest parent Operation* of op (the op that contains this op's
// block), or null if op is at the top level.
// op_ptr: Operation* as ptr (raw Chez Scheme pointer)
// Returns: parent Operation* as ptr, or nullptr.
ptr mlir_operation_get_parent(ptr op_ptr) {
  if (!op_ptr) return nullptr;
  return static_cast<mlir::Operation*>(op_ptr)->getParentOp();
}

// Return the operand Value at position index as a raw opaque Value pointer
// (via Value::getAsOpaquePointer, not the MLIR C-API wrap).
// op_ptr: Operation* as ptr
// index: zero-based operand index
// Returns: Value* opaque pointer, or nullptr if out of range.
ptr mlir_operation_get_operand_value(ptr op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  if (index < 0 || index >= (int)op->getNumOperands()) return nullptr;
  return const_cast<void*>(op->getOperand(index).getAsOpaquePointer());
}

// Return the result Value at position index as a raw opaque Value pointer.
// op_ptr: Operation* as ptr
// index: zero-based result index
// Returns: Value* opaque pointer, or nullptr if out of range.
ptr mlir_operation_get_result_value(ptr op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  if (index < 0 || index >= (int)op->getNumResults()) return nullptr;
  return const_cast<void*>(op->getResult(index).getAsOpaquePointer());
}

// Return the source location of an operation as a raw opaque Location pointer
// (via Location::getAsOpaquePointer).
// op_ptr: Operation* as ptr
// Returns: Location opaque pointer, or nullptr for null op.
ptr mlir_operation_get_loc(ptr op_ptr) {
  if (!op_ptr) return nullptr;
  return const_cast<void*>(static_cast<mlir::Operation*>(op_ptr)->getLoc().getAsOpaquePointer());
}

// Return the block argument of the enclosing func.func at position index.
// Walks up the parent chain to find the nearest func::FuncOp, then returns
// its block argument. Used to locate the HipSR context argument by convention.
// op_ptr: Operation* as ptr (any op nested inside a FuncOp)
// index: zero-based argument index of the FuncOp
// Returns: Value* opaque pointer of the block argument, or nullptr if not found.
ptr mlir_operation_get_block_argument(ptr op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  while (op && !llvm::isa<mlir::func::FuncOp>(op))
    op = op->getParentOp();
  if (!op) return nullptr;
  auto funcOp = llvm::cast<mlir::func::FuncOp>(op);
  if (index < 0 || index >= (int)funcOp.getNumArguments()) return nullptr;
  return const_cast<void*>(funcOp.getArgument(index).getAsOpaquePointer());
}

// Walk all operations nested inside op (post-order) and call callback on each.
// The callback is a Scheme procedure (lambda (op-uptr) ...).
// The callback object is GC-locked for the duration of the walk.
// op: root Operation* as uptr
// callback: Scheme procedure ptr
void mlir_operation_walk(uint64_t op, ptr callback) {
  if (!op) return;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  mlir::hipsr::LockedSchemeObject locked(callback);
  cppOp->walk([&locked](mlir::Operation* walkOp) {
    ptr schemeOp = Sunsigned64(reinterpret_cast<uint64_t>(walkOp));
    Scall1(locked.get(), schemeOp);
  });
}

// Return the number of DPS (Destination-Passing Style) init operands.
// DPS ops carry pre-allocated output buffers as "init" operands.
// op_ptr: Operation* as uptr; returns 0 if op does not implement DPS interface.
int mlir_operation_num_dps_inits(uint64_t op_ptr) {
  if (!op_ptr) return 0;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
  if (!dpsOp) return 0;
  return static_cast<int>(dpsOp.getNumDpsInits());
}

// Return the DPS init Value at position index as a raw opaque Value pointer.
// op_ptr: Operation* as uptr
// index: zero-based init index
// Returns: Value* opaque pointer, or 0 if out of range or not a DPS op.
uint64_t mlir_operation_get_dps_init_value(uint64_t op_ptr, int index) {
  if (!op_ptr) return 0;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
  if (!dpsOp) return 0;
  if (index < 0 || index >= static_cast<int>(dpsOp.getNumDpsInits())) return 0;
  return reinterpret_cast<uint64_t>(dpsOp.getDpsInits()[index].getAsOpaquePointer());
}

// Replace the operand at position index with a new Value.
// op_ptr: Operation* as uptr
// index: zero-based operand index
// value: replacement Value* as uptr (MLIR C-API wrapped)
void mlir_operation_set_operand(uint64_t op_ptr, int index, uint64_t value) {
  if (!op_ptr || !value) return;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  mlir::Value val = unwrap(MlirValue{reinterpret_cast<const void*>(value)});
  op->setOperand(static_cast<unsigned>(index), val);
}

// Test whether all results of an operation have no uses (the op is dead).
// op_ptr: Operation* as uptr
// Returns: 1 if use_empty() is true (op is dead), 0 otherwise.
//          Returns 1 for null op (conservative: treat null as dead).
int mlir_operation_use_empty(uint64_t op_ptr) {
  if (!op_ptr) return 1;
  return reinterpret_cast<mlir::Operation*>(op_ptr)->use_empty() ? 1 : 0;
}

// Attribute access

// Return the string value of a named StringAttr, or "" if absent or wrong type.
// The returned pointer is owned by the attribute storage (valid while the op lives).
// op_ptr: Operation* as uptr
// attr_name: attribute dictionary key
const char* mlir_operation_get_string_attr(uint64_t op_ptr, const char* attr_name) {
  if (!op_ptr) return "";
  auto attr = reinterpret_cast<mlir::Operation*>(op_ptr)->getAttrOfType<mlir::StringAttr>(attr_name);
  if (!attr) return "";
  return attr.getValue().data();
}

// Return the sign-extended integer value of a named IntegerAttr.
// op_ptr: Operation* as uptr
// attr_name: attribute dictionary key
// default_val: returned when the attribute is absent or the wrong type
int64_t mlir_operation_get_integer_attr(uint64_t op_ptr, const char* attr_name, int64_t default_val) {
  if (!op_ptr) return default_val;
  if (auto attr = reinterpret_cast<mlir::Operation*>(op_ptr)->getAttrOfType<mlir::IntegerAttr>(attr_name))
    return attr.getValue().getSExtValue();
  return default_val;
}

// Return a named integer-array attribute as a Scheme list of integers.
// Accepts DenseI64ArrayAttr or ArrayAttr of IntegerAttr elements.
// op_ptr: Operation* as uptr
// attr_name: attribute dictionary key
// Returns: Scheme list (ptr), or Snil if absent or elements are not integers.
ptr mlir_operation_get_integer_array_attr(uint64_t op_ptr, const char* attr_name) {
  if (!op_ptr) return Snil;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  if (auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(attr_name)) {
    ptr list = Snil;
    for (int i = (int)attr.size() - 1; i >= 0; --i)
      list = Scons(Sinteger(attr[i]), list);
    return list;
  }
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>(attr_name)) {
    ptr list = Snil;
    for (int i = (int)attr.size() - 1; i >= 0; --i) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr[i]);
      if (!intAttr) return Snil;
      list = Scons(Sinteger(intAttr.getInt()), list);
    }
    return list;
  }
  return Snil;
}

// Set a named attribute to an IntegerAttr of IndexType on an operation.
// Deprecated: prefer mlir_operation_set_attribute with a pre-built attr.
// op_ptr: Operation* as uptr
// attr_name: attribute dictionary key
// value: integer value to encode as IndexType
void mlir_operation_set_index_attr(uint64_t op_ptr, const char* attr_name, int64_t value) {
  if (!op_ptr) return;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op_ptr);
  cppOp->setAttr(attr_name,
      mlir::IntegerAttr::get(mlir::IndexType::get(cppOp->getContext()), value));
}

// Set a named attribute to a DenseI64ArrayAttr built from a Scheme list.
// Deprecated: prefer mlir_operation_set_attribute with a pre-built attr.
// op_ptr: Operation* as uptr
// attr_name: attribute dictionary key
// values_list: Scheme list of integer Scheme objects
void mlir_operation_set_dense_i64_array(uint64_t op_ptr, const char* attr_name, ptr values_list) {
  if (!op_ptr) return;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  llvm::SmallVector<int64_t> values;
  for (ptr cur = static_cast<ptr>(values_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) break;
    values.push_back(Sinteger_value(Scar(cur)));
  }
  op->setAttr(attr_name, mlir::DenseI64ArrayAttr::get(op->getContext(), values));
}

// Set a named attribute to a DenseI32ArrayAttr built from a Scheme list.
// Deprecated: prefer mlir_operation_set_attribute with a pre-built attr.
// op_ptr: Operation* as uptr
// attr_name: attribute dictionary key
// values_list: Scheme list of integer Scheme objects (truncated to i32)
void mlir_operation_set_dense_i32_array(uint64_t op_ptr, const char* attr_name, ptr values_list) {
  if (!op_ptr) return;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  llvm::SmallVector<int32_t> values;
  for (ptr cur = static_cast<ptr>(values_list); cur != Snil; cur = Scdr(cur)) {
    if (!Spairp(cur)) break;
    values.push_back(static_cast<int32_t>(Sinteger_value(Scar(cur))));
  }
  op->setAttr(attr_name, mlir::DenseI32ArrayAttr::get(op->getContext(), values));
}

// Set a named f32 FloatAttr on an operation.
void mlir_operation_set_f32_attr(uint64_t op_ptr, const char* name, double value) {
  if (!op_ptr) return;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  op->setAttr(name, mlir::FloatAttr::get(mlir::Float32Type::get(op->getContext()),
                                          static_cast<float>(value)));
}

// Set a named i64 IntegerAttr (signless) on an operation.
void mlir_operation_set_i64_attr(uint64_t op_ptr, const char* name, int64_t value) {
  if (!op_ptr) return;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  op->setAttr(name, mlir::IntegerAttr::get(
                       mlir::IntegerType::get(op->getContext(), 64), value));
}

// Set a named UnitAttr on an operation (marks a boolean-style flag as present).
void mlir_operation_set_unit_attr(uint64_t op_ptr, const char* name) {
  if (!op_ptr) return;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  op->setAttr(name, mlir::UnitAttr::get(op->getContext()));
}

// Copy a named attribute from src_op to dst_op under a (possibly different) name.
// No-op if the attribute is absent on src_op.
// dst_op_ptr: destination Operation* as uptr
// dst_name: key to set on dst_op
// src_op_ptr: source Operation* as uptr
// src_name: key to read from src_op
void mlir_operation_copy_attr(uint64_t dst_op_ptr, const char* dst_name,
                               uint64_t src_op_ptr, const char* src_name) {
  if (!dst_op_ptr || !src_op_ptr) return;
  auto* dst = reinterpret_cast<mlir::Operation*>(dst_op_ptr);
  auto* src = reinterpret_cast<mlir::Operation*>(src_op_ptr);
  auto attr = src->getAttr(src_name);
  if (attr) dst->setAttr(dst_name, attr);
}

// Test whether a named attribute is present in the operation's attribute dict.
// op_ptr: Operation* as uptr
// attr_name: attribute dictionary key
// Returns: 1 if present, 0 if absent or op is null.
int mlir_operation_has_attr(uint64_t op_ptr, const char* attr_name) {
  if (!op_ptr) return 0;
  return reinterpret_cast<mlir::Operation*>(op_ptr)->hasAttr(attr_name) ? 1 : 0;
}

// Diagnostics — route messages through MLIR's diagnostic engine so they
// appear in ORT's error output with the op's source location attached.
// Falls back to mlir_log_* when op_ptr is 0 (no op context available).

// Emit an error diagnostic attached to the operation's source location.
// op_ptr: Operation* as uptr, or 0 to fall back to mlir_log_error
// msg: diagnostic message string
void mlir_emit_error(uint64_t op_ptr, const char *msg) {
  if (!op_ptr) { mlir_log_error(msg); return; }
  reinterpret_cast<mlir::Operation*>(op_ptr)->emitError(msg);
}

// Emit a warning diagnostic attached to the operation's source location.
// op_ptr: Operation* as uptr, or 0 to fall back to mlir_log_warning
// msg: diagnostic message string
void mlir_emit_warning(uint64_t op_ptr, const char *msg) {
  if (!op_ptr) { mlir_log_warning(msg); return; }
  reinterpret_cast<mlir::Operation*>(op_ptr)->emitWarning(msg);
}

// Emit a remark (informational) diagnostic attached to the operation's source location.
// op_ptr: Operation* as uptr, or 0 to fall back to mlir_log_info
// msg: diagnostic message string
void mlir_emit_remark(uint64_t op_ptr, const char *msg) {
  if (!op_ptr) { mlir_log_info(msg); return; }
  reinterpret_cast<mlir::Operation*>(op_ptr)->emitRemark(msg);
}

// Erase an operation directly from the IR, without going through a rewriter.
// Use only outside of active conversion patterns (e.g. post-pass cleanup).
// op_ptr: Operation* as uptr; no-op for null.
void mlir_op_erase(uint64_t op_ptr) {
  if (!op_ptr) return;
  reinterpret_cast<mlir::Operation*>(op_ptr)->erase();
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerOperationBindings() {
  Sregister_symbol("mlir_operation_get_name",              (void*)::mlir_operation_get_name);
  Sregister_symbol("mlir_operation_get_context",           (void*)::mlir_operation_get_context);
  Sregister_symbol("mlir_operation_num_operands",          (void*)::mlir_operation_num_operands);
  Sregister_symbol("mlir_operation_num_results",           (void*)::mlir_operation_num_results);
  Sregister_symbol("mlir_operation_get_operand",           (void*)::mlir_operation_get_operand);
  Sregister_symbol("mlir_operation_get_result",            (void*)::mlir_operation_get_result);
  Sregister_symbol("mlir_operation_get_parent",            (void*)::mlir_operation_get_parent);
  Sregister_symbol("mlir_operation_get_operand_value",     (void*)::mlir_operation_get_operand_value);
  Sregister_symbol("mlir_operation_get_result_value",      (void*)::mlir_operation_get_result_value);
  Sregister_symbol("mlir_operation_get_loc",               (void*)::mlir_operation_get_loc);
  Sregister_symbol("mlir_operation_get_block_argument",    (void*)::mlir_operation_get_block_argument);
  Sregister_symbol("mlir_operation_walk",                  (void*)::mlir_operation_walk);
  Sregister_symbol("mlir_operation_num_dps_inits",         (void*)::mlir_operation_num_dps_inits);
  Sregister_symbol("mlir_operation_get_dps_init_value",    (void*)::mlir_operation_get_dps_init_value);
  Sregister_symbol("mlir_operation_set_operand",           (void*)::mlir_operation_set_operand);
  Sregister_symbol("mlir_operation_use_empty",             (void*)::mlir_operation_use_empty);
  Sregister_symbol("mlir_operation_get_string_attr",       (void*)::mlir_operation_get_string_attr);
  Sregister_symbol("mlir_operation_get_integer_attr",      (void*)::mlir_operation_get_integer_attr);
  Sregister_symbol("mlir_operation_get_integer_array_attr",(void*)::mlir_operation_get_integer_array_attr);
  Sregister_symbol("mlir_operation_set_f32_attr",           (void*)::mlir_operation_set_f32_attr);
  Sregister_symbol("mlir_operation_set_i64_attr",           (void*)::mlir_operation_set_i64_attr);
  Sregister_symbol("mlir_operation_set_unit_attr",          (void*)::mlir_operation_set_unit_attr);
  Sregister_symbol("mlir_operation_set_index_attr",         (void*)::mlir_operation_set_index_attr);
  Sregister_symbol("mlir_operation_set_dense_i64_array",   (void*)::mlir_operation_set_dense_i64_array);
  Sregister_symbol("mlir_operation_set_dense_i32_array",   (void*)::mlir_operation_set_dense_i32_array);
  Sregister_symbol("mlir_operation_copy_attr",             (void*)::mlir_operation_copy_attr);
  Sregister_symbol("mlir_operation_has_attr",              (void*)::mlir_operation_has_attr);
  Sregister_symbol("mlir_emit_error",                      (void*)::mlir_emit_error);
  Sregister_symbol("mlir_emit_warning",                    (void*)::mlir_emit_warning);
  Sregister_symbol("mlir_emit_remark",                     (void*)::mlir_emit_remark);
  Sregister_symbol("mlir_op_erase",                        (void*)::mlir_op_erase);
}

} // namespace hipsr
} // namespace mlir
