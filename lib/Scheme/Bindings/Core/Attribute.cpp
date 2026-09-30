/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Mirrors (mlir core attribute): attribute construction and op-level get/set.
// All mlir_make_attr_* functions share the uniform signature
//   (uint64_t ctx_ptr, ptr value) → uint64_t
// so Scheme can discover them dynamically via foreign-entry?.

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

extern "C" {

uint64_t mlir_make_attr_i64(uint64_t ctx_ptr, ptr value) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                             Sinteger64_value(value))
          .getAsOpaquePointer());
}

uint64_t mlir_make_attr_index(uint64_t ctx_ptr, ptr value) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                             Sinteger64_value(value))
          .getAsOpaquePointer());
}

uint64_t mlir_make_attr_i32_array(uint64_t ctx_ptr, ptr value) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  llvm::SmallVector<int32_t> vec;
  for (ptr cur = value; cur != Snil; cur = Scdr(cur))
    vec.push_back(static_cast<int32_t>(Sfixnum_value(Scar(cur))));
  return reinterpret_cast<uint64_t>(
      mlir::DenseI32ArrayAttr::get(ctx, vec).getAsOpaquePointer());
}

uint64_t mlir_make_attr_i64_array(uint64_t ctx_ptr, ptr value) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  llvm::SmallVector<int64_t> vec;
  for (ptr cur = value; cur != Snil; cur = Scdr(cur))
    vec.push_back(Sinteger64_value(Scar(cur)));
  return reinterpret_cast<uint64_t>(
      mlir::DenseI64ArrayAttr::get(ctx, vec).getAsOpaquePointer());
}

// value: Scheme list (result-type-uptr key-string data-addr data-size)
// data-addr is a raw memory address as integer; caller keeps backing memory alive.
uint64_t mlir_make_attr_dense_resource(uint64_t /*ctx_ptr*/, ptr value) {
  auto result_type_ptr = Sunsigned64_value(Scar(value));
  const char* key      = Sstring_data(Scar(Scdr(value)));
  int64_t data_addr    = Sinteger64_value(Scar(Scdr(Scdr(value))));
  int64_t data_size    = Sinteger64_value(Scar(Scdr(Scdr(Scdr(value)))));

  auto baseType   = mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(result_type_ptr));
  auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(baseType);
  if (!resultType) return 0;

  llvm::ArrayRef<char> data = {
      reinterpret_cast<const char*>(static_cast<uintptr_t>(data_addr)),
      static_cast<size_t>(data_size)};
  return reinterpret_cast<uint64_t>(
      mlir::DenseResourceElementsAttr::get(resultType, key,
          mlir::UnmanagedAsmResourceBlob::allocateInferAlign(data))
          .getAsOpaquePointer());
}

uint64_t mlir_operation_get_attribute(uint64_t op_ptr, const char* name) {
  auto *op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto attr = op->getAttr(name);
  if (!attr) return 0;
  return reinterpret_cast<uint64_t>(attr.getAsOpaquePointer());
}

void mlir_operation_set_attribute(uint64_t op_ptr, const char* name, uint64_t attr_ptr) {
  auto *op   = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto  attr = mlir::Attribute::getFromOpaquePointer(
      reinterpret_cast<const void*>(attr_ptr));
  op->setAttr(name, attr);
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerAttributeBindings() {
  Sregister_symbol("mlir_make_attr_i64",            (void*)::mlir_make_attr_i64);
  Sregister_symbol("mlir_make_attr_index",          (void*)::mlir_make_attr_index);
  Sregister_symbol("mlir_make_attr_i32_array",      (void*)::mlir_make_attr_i32_array);
  Sregister_symbol("mlir_make_attr_i64_array",      (void*)::mlir_make_attr_i64_array);
  Sregister_symbol("mlir_make_attr_dense_resource", (void*)::mlir_make_attr_dense_resource);
  Sregister_symbol("mlir_operation_get_attribute",  (void*)::mlir_operation_get_attribute);
  Sregister_symbol("mlir_operation_set_attribute",  (void*)::mlir_operation_set_attribute);
}

} // namespace hipsr
} // namespace mlir
