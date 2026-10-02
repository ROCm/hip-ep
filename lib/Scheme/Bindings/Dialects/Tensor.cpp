/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"


#define DEBUG_TYPE "scheme-tensor-bindings"

// Note: scheme.h included via SchemeMlirBindings.h -> ChezSchemeInterpreter.h

extern "C" {

// Clone a RankedTensorType with a HipSR MemorySpaceAttr encoding.
// type_ptr:  RankedTensorType* as opaque ptr (ptr, not uint64_t)
// space_int: mlir::hipsr::MemorySpace enum value (0=Host, 1=Device)
// Returns:   new type with the memory space set; returns type_ptr unchanged
//            if the type is not a RankedTensorType (and logs an error)
ptr mlir_type_set_memory_space(ptr type_ptr, int space_int) {
  mlir::Type type = mlir::Type::getFromOpaquePointer(type_ptr);
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType) {
    mlir_log_error("mlir_type_set_memory_space: Type is not a RankedTensorType");
    return type_ptr;
  }

  mlir::hipsr::MemorySpace space = static_cast<mlir::hipsr::MemorySpace>(space_int);
  auto newType = tensorType.cloneWithEncoding(
      mlir::hipsr::MemorySpaceAttr::get(tensorType.getContext(), space));

  return const_cast<void*>(newType.getAsOpaquePointer());
}

// Return the shape of a RankedTensorType as a Scheme list of integers.
// Uses kDynamic (very negative int64) for dynamic dimensions.
// type_ptr: RankedTensorType* as opaque ptr
// Returns:  Scheme list of exact integers, or Snil if not a ranked tensor
ptr mlir_type_get_shape(ptr type_ptr) {
  if (!type_ptr) return Snil;
  mlir::Type type = mlir::Type::getFromOpaquePointer(type_ptr);
  if (auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    llvm::ArrayRef<int64_t> shape = tensorType.getShape();
    // Convert to Scheme list
    ptr list = Snil;
    for (int i = shape.size() - 1; i >= 0; --i) {
      list = Scons(Sinteger(shape[i]), list);
    }
    return list;
  }
  return Snil;
}

// Return the MLIR type of a Value.
// value_ptr: Value* as opaque ptr
// Returns:   Type* as opaque ptr, or null if value_ptr is null
ptr mlir_value_get_type(ptr value_ptr) {
  if (!value_ptr) return nullptr;
  mlir::Value value = mlir::Value::getFromOpaquePointer(value_ptr);
  return const_cast<void*>(value.getType().getAsOpaquePointer());
}

//===----------------------------------------------------------------------===//
// Phase 2: Operation/Value Navigation FFI
//===----------------------------------------------------------------------===//

// Return 1 if type_ptr is a RankedTensorType, 0 otherwise.
// type_ptr: Type* as uptr
int mlir_type_is_ranked_tensor(uint64_t type_ptr) {
  if (!type_ptr) return 0;
  mlir::Type type = mlir::Type::getFromOpaquePointer(reinterpret_cast<void*>(type_ptr));
  return mlir::isa<mlir::RankedTensorType>(type) ? 1 : 0;
}

// Get rank of RankedTensorType
// Returns rank, or -1 if not a ranked tensor
int64_t mlir_type_get_rank(uint64_t type_ptr) {
  if (!type_ptr) return -1;
  mlir::Type type = mlir::Type::getFromOpaquePointer(reinterpret_cast<void*>(type_ptr));
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType) return -1;
  return tensorType.getRank();
}

// Get element type of tensor type
// Returns Type* as unsigned-64, or 0 if not a tensor
uint64_t mlir_type_get_element_type(uint64_t type_ptr) {
  if (!type_ptr) return 0;
  mlir::Type type = mlir::Type::getFromOpaquePointer(reinterpret_cast<void*>(type_ptr));
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType) return 0;
  return reinterpret_cast<uint64_t>(const_cast<void*>(tensorType.getElementType().getAsOpaquePointer()));
}



// Attach any MLIR attribute as the encoding of a RankedTensorType.
// Dialect-agnostic: works with any attribute type (HipSR MemorySpaceAttr,
// DLTI attrs, custom attrs, etc.).
// type_ptr: RankedTensorType* as uptr; returns 0 if not a ranked tensor
// attr_ptr: Attribute* (opaque) as uptr — the encoding to set
// Returns:  new RankedTensorType with the encoding attached, as opaque type uptr
uint64_t mlir_tensor_type_with_encoding(uint64_t type_ptr, uint64_t attr_ptr) {
  if (!type_ptr || !attr_ptr) return 0;
  auto baseType  = mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(type_ptr));
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(baseType);
  if (!tensorType) return 0;
  auto attr = mlir::Attribute::getFromOpaquePointer(reinterpret_cast<const void*>(attr_ptr));
  return reinterpret_cast<uint64_t>(
      tensorType.cloneWithEncoding(attr).getAsOpaquePointer());
}

// Clone a RankedTensorType with the HipSR Host memory space encoding.
// Produces a host tensor type (tensor<..., #hipsr.mem<host>>).
// type_ptr: RankedTensorType* as uptr; returns type_ptr unchanged if not ranked tensor
// Returns:  new type with host encoding as opaque type uptr
uint64_t mlir_tensor_type_in_host_space(uint64_t type_ptr) {
  if (!type_ptr) return 0;
  mlir::Type type = mlir::Type::getFromOpaquePointer(reinterpret_cast<void*>(type_ptr));
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType) return type_ptr;
  auto newType = tensorType.cloneWithEncoding(
      mlir::hipsr::MemorySpaceAttr::get(tensorType.getContext(), mlir::hipsr::MemorySpace::Host));
  return reinterpret_cast<uint64_t>(const_cast<void*>(newType.getAsOpaquePointer()));
}

//===----------------------------------------------------------------------===//
// MLIR Dialect Conversion Primitives
//===----------------------------------------------------------------------===//

// Return the encoding attribute of a RankedTensorType (e.g. #hipsr.mem<device>).
// type_ptr: RankedTensorType* as uptr
// Returns:  Attribute* as opaque uptr, or 0 if the type has no encoding
uint64_t mlir_type_get_encoding(uint64_t type_ptr) {
  if (!type_ptr) return 0;
  mlir::Type type = mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(type_ptr));
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType) return 0;
  mlir::Attribute enc = tensorType.getEncoding();
  if (!enc) return 0;
  return reinterpret_cast<uint64_t>(enc.getAsOpaquePointer());
}

// Return 1 if type_ptr is a RankedTensorType with HipSR Device memory space, 0 otherwise.
// type_ptr: Type* as uptr
int mlir_type_is_device_tensor(uint64_t type_ptr) {
  if (!type_ptr) return 0;
  auto type = mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(type_ptr));
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType) return 0;
  auto enc = mlir::dyn_cast_or_null<mlir::hipsr::MemorySpaceAttr>(tensorType.getEncoding());
  return (enc && enc.getValue() == mlir::hipsr::MemorySpace::Device) ? 1 : 0;
}

// Returns 1 if the named attribute exists on the operation.

} // extern "C"

namespace mlir {
namespace hipsr {

void registerTensorBindings() {
  Sregister_symbol("mlir_type_set_memory_space", (void*)::mlir_type_set_memory_space);
  Sregister_symbol("mlir_type_get_shape", (void*)::mlir_type_get_shape);
  Sregister_symbol("mlir_value_get_type", (void*)::mlir_value_get_type);
  Sregister_symbol("mlir_type_is_ranked_tensor", (void*)::mlir_type_is_ranked_tensor);
  Sregister_symbol("mlir_type_get_rank", (void*)::mlir_type_get_rank);
  Sregister_symbol("mlir_type_get_element_type", (void*)::mlir_type_get_element_type);
  Sregister_symbol("mlir_tensor_type_with_encoding",   (void*)::mlir_tensor_type_with_encoding);
  Sregister_symbol("mlir_tensor_type_in_host_space",   (void*)::mlir_tensor_type_in_host_space);
  Sregister_symbol("mlir_type_get_encoding", (void*)::mlir_type_get_encoding);
  Sregister_symbol("mlir_type_is_device_tensor", (void*)::mlir_type_is_device_tensor);
}

} // namespace hipsr
} // namespace mlir
