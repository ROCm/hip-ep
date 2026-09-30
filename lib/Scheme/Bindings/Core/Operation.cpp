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

const char* mlir_operation_get_name(uint64_t op) {
  if (!op) return "";
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return cppOp->getName().getStringRef().data();
}

uint64_t mlir_operation_get_context(uint64_t op) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  return reinterpret_cast<uint64_t>(cppOp->getContext());
}

int64_t mlir_operation_num_operands(uint64_t op) {
  if (!op) return 0;
  return reinterpret_cast<mlir::Operation*>(op)->getNumOperands();
}

int64_t mlir_operation_num_results(uint64_t op) {
  if (!op) return 0;
  return reinterpret_cast<mlir::Operation*>(op)->getNumResults();
}

uint64_t mlir_operation_get_operand(uint64_t op, int64_t index) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  if (index < 0 || index >= (int64_t)cppOp->getNumOperands()) return 0;
  mlir::Value val = cppOp->getOperand(index);
  MlirValue cVal = wrap(val);
  return reinterpret_cast<uint64_t>(const_cast<void*>(cVal.ptr));
}

uint64_t mlir_operation_get_result(uint64_t op, int64_t index) {
  if (!op) return 0;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  if (index < 0 || index >= (int64_t)cppOp->getNumResults()) return 0;
  mlir::Value val = cppOp->getResult(index);
  MlirValue cVal = wrap(val);
  return reinterpret_cast<uint64_t>(const_cast<void*>(cVal.ptr));
}

ptr mlir_operation_get_parent(ptr op_ptr) {
  if (!op_ptr) return nullptr;
  return static_cast<mlir::Operation*>(op_ptr)->getParentOp();
}

ptr mlir_operation_get_operand_value(ptr op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  if (index < 0 || index >= (int)op->getNumOperands()) return nullptr;
  return const_cast<void*>(op->getOperand(index).getAsOpaquePointer());
}

ptr mlir_operation_get_result_value(ptr op_ptr, int index) {
  if (!op_ptr) return nullptr;
  mlir::Operation* op = static_cast<mlir::Operation*>(op_ptr);
  if (index < 0 || index >= (int)op->getNumResults()) return nullptr;
  return const_cast<void*>(op->getResult(index).getAsOpaquePointer());
}

ptr mlir_operation_get_loc(ptr op_ptr) {
  if (!op_ptr) return nullptr;
  return const_cast<void*>(static_cast<mlir::Operation*>(op_ptr)->getLoc().getAsOpaquePointer());
}

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

void mlir_operation_walk(uint64_t op, ptr callback) {
  if (!op) return;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op);
  mlir::hipsr::LockedSchemeObject locked(callback);
  cppOp->walk([&locked](mlir::Operation* walkOp) {
    ptr schemeOp = Sunsigned64(reinterpret_cast<uint64_t>(walkOp));
    Scall1(locked.get(), schemeOp);
  });
}

int mlir_operation_num_dps_inits(uint64_t op_ptr) {
  if (!op_ptr) return 0;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
  if (!dpsOp) return 0;
  return static_cast<int>(dpsOp.getNumDpsInits());
}

uint64_t mlir_operation_get_dps_init_value(uint64_t op_ptr, int index) {
  if (!op_ptr) return 0;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op);
  if (!dpsOp) return 0;
  if (index < 0 || index >= static_cast<int>(dpsOp.getNumDpsInits())) return 0;
  return reinterpret_cast<uint64_t>(dpsOp.getDpsInits()[index].getAsOpaquePointer());
}

void mlir_operation_set_operand(uint64_t op_ptr, int index, uint64_t value) {
  if (!op_ptr || !value) return;
  mlir::Operation* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  mlir::Value val = unwrap(MlirValue{reinterpret_cast<const void*>(value)});
  op->setOperand(static_cast<unsigned>(index), val);
}

int mlir_operation_use_empty(uint64_t op_ptr) {
  if (!op_ptr) return 1;
  return reinterpret_cast<mlir::Operation*>(op_ptr)->use_empty() ? 1 : 0;
}

// Attribute access
const char* mlir_operation_get_string_attr(uint64_t op_ptr, const char* attr_name) {
  if (!op_ptr) return "";
  auto attr = reinterpret_cast<mlir::Operation*>(op_ptr)->getAttrOfType<mlir::StringAttr>(attr_name);
  if (!attr) return "";
  return attr.getValue().data();
}

int64_t mlir_operation_get_integer_attr(uint64_t op_ptr, const char* attr_name, int64_t default_val) {
  if (!op_ptr) return default_val;
  if (auto attr = reinterpret_cast<mlir::Operation*>(op_ptr)->getAttrOfType<mlir::IntegerAttr>(attr_name))
    return attr.getValue().getSExtValue();
  return default_val;
}

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

void mlir_operation_set_index_attr(uint64_t op_ptr, const char* attr_name, int64_t value) {
  if (!op_ptr) return;
  mlir::Operation* cppOp = reinterpret_cast<mlir::Operation*>(op_ptr);
  cppOp->setAttr(attr_name,
      mlir::IntegerAttr::get(mlir::IndexType::get(cppOp->getContext()), value));
}

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

void mlir_operation_copy_attr(uint64_t dst_op_ptr, const char* dst_name,
                               uint64_t src_op_ptr, const char* src_name) {
  if (!dst_op_ptr || !src_op_ptr) return;
  auto* dst = reinterpret_cast<mlir::Operation*>(dst_op_ptr);
  auto* src = reinterpret_cast<mlir::Operation*>(src_op_ptr);
  auto attr = src->getAttr(src_name);
  if (attr) dst->setAttr(dst_name, attr);
}

int mlir_operation_has_attr(uint64_t op_ptr, const char* attr_name) {
  if (!op_ptr) return 0;
  return reinterpret_cast<mlir::Operation*>(op_ptr)->hasAttr(attr_name) ? 1 : 0;
}

// Diagnostics
void mlir_emit_error(uint64_t op_ptr, const char *msg) {
  if (!op_ptr) { mlir_log_error(msg); return; }
  reinterpret_cast<mlir::Operation*>(op_ptr)->emitError(msg);
}

void mlir_emit_warning(uint64_t op_ptr, const char *msg) {
  if (!op_ptr) { mlir_log_warning(msg); return; }
  reinterpret_cast<mlir::Operation*>(op_ptr)->emitWarning(msg);
}

void mlir_emit_remark(uint64_t op_ptr, const char *msg) {
  if (!op_ptr) { mlir_log_info(msg); return; }
  reinterpret_cast<mlir::Operation*>(op_ptr)->emitRemark(msg);
}

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
  Sregister_symbol("mlir_operation_set_index_attr",        (void*)::mlir_operation_set_index_attr);
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
