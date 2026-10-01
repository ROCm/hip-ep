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
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include <string>

extern "C" {

// Create an IntegerAttr<i64> with the given integer value.
// ctx_ptr:  MLIRContext* as uptr
// value:    Scheme integer (fixnum or bignum) — the 64-bit integer value
// Returns:  Attribute opaque ptr (Attribute::getAsOpaquePointer())
uint64_t mlir_make_attr_i64(uint64_t ctx_ptr, ptr value) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                             Sinteger64_value(value))
          .getAsOpaquePointer());
}

// Create an IntegerAttr<IndexType> with the given integer value.
// Used for attributes that must carry MLIR's platform-sized index type
// (e.g. shape.const_size "value").
// ctx_ptr:  MLIRContext* as uptr
// value:    Scheme integer — the index value
// Returns:  Attribute opaque ptr
uint64_t mlir_make_attr_index(uint64_t ctx_ptr, ptr value) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(
      mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                             Sinteger64_value(value))
          .getAsOpaquePointer());
}

// Create a DenseI32ArrayAttr from a Scheme list of fixnum integers.
// ctx_ptr:  MLIRContext* as uptr
// value:    Scheme list of fixnums — the i32 elements
// Returns:  Attribute opaque ptr
uint64_t mlir_make_attr_i32_array(uint64_t ctx_ptr, ptr value) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  llvm::SmallVector<int32_t> vec;
  for (ptr cur = value; cur != Snil; cur = Scdr(cur))
    vec.push_back(static_cast<int32_t>(Sfixnum_value(Scar(cur))));
  return reinterpret_cast<uint64_t>(
      mlir::DenseI32ArrayAttr::get(ctx, vec).getAsOpaquePointer());
}

// Create a DenseI64ArrayAttr from a Scheme list of integers.
// ctx_ptr:  MLIRContext* as uptr
// value:    Scheme list of integers (fixnum or bignum) — the i64 elements
// Returns:  Attribute opaque ptr
uint64_t mlir_make_attr_i64_array(uint64_t ctx_ptr, ptr value) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  llvm::SmallVector<int64_t> vec;
  for (ptr cur = value; cur != Snil; cur = Scdr(cur))
    vec.push_back(Sinteger64_value(Scar(cur)));
  return reinterpret_cast<uint64_t>(
      mlir::DenseI64ArrayAttr::get(ctx, vec).getAsOpaquePointer());
}

// Extract a C++ std::string from a Chez Scheme string object.
// Sstring_data is not available in this Chez Scheme version; iterate chars.
static std::string schemeStringToStd(ptr s) {
  iptr len = Sstring_length(s);
  std::string result(static_cast<size_t>(len), '\0');
  for (iptr i = 0; i < len; ++i)
    result[i] = static_cast<char>(Sstring_ref(s, i));
  return result;
}

// Create a DenseResourceElementsAttr wrapping a pre-existing raw memory region.
// This is the generic blob attr — caller is responsible for keeping backing
// memory alive (UnmanagedAsmResourceBlob takes a non-owning reference).
//
// ctx_ptr is unused (ctx is derived from result_type).
// value:   Scheme list of four elements:
//            (result-type-uptr  key-string  data-addr-integer  data-size-integer)
//   result-type-uptr: RankedTensorType opaque ptr — the shaped type of the resource
//   key-string:       Scheme string — unique blob key (e.g. "mem|0x..." or "file|...|4")
//   data-addr:        Scheme integer — raw memory address of the data bytes
//   data-size:        Scheme integer — byte count
// Returns:  Attribute opaque ptr, or 0 if result-type is not a RankedTensorType.
uint64_t mlir_make_attr_dense_resource(uint64_t /*ctx_ptr*/, ptr value) {
  auto result_type_ptr = Sunsigned64_value(Scar(value));
  std::string key_str  = schemeStringToStd(Scar(Scdr(value)));
  const char* key      = key_str.c_str();
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

// Get a named attribute from an operation as an opaque Attribute pointer.
// op_ptr:  Operation* as uptr
// name:    attribute name string
// Returns: Attribute opaque ptr, or 0 if the attribute is absent.
uint64_t mlir_operation_get_attribute(uint64_t op_ptr, const char* name) {
  auto *op = reinterpret_cast<mlir::Operation*>(op_ptr);
  auto attr = op->getAttr(name);
  if (!attr) return 0;
  return reinterpret_cast<uint64_t>(attr.getAsOpaquePointer());
}

// Set a named attribute on an operation from an opaque Attribute pointer.
// op_ptr:   Operation* as uptr
// name:     attribute name string
// attr_ptr: Attribute opaque ptr (from any mlir_make_attr_* or get_attribute)
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
