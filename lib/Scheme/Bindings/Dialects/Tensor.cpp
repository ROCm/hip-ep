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

// Type shape query - returns Scheme list
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

// Get type from value
ptr mlir_value_get_type(ptr value_ptr) {
  if (!value_ptr) return nullptr;
  mlir::Value value = mlir::Value::getFromOpaquePointer(value_ptr);
  return const_cast<void*>(value.getType().getAsOpaquePointer());
}

//===----------------------------------------------------------------------===//
// Phase 2: Operation/Value Navigation FFI
//===----------------------------------------------------------------------===//

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

// Clone tensor type with device memory space
// Returns new Type* as unsigned-64, or original if not a ranked tensor
uint64_t mlir_tensor_type_in_device_space(uint64_t type_ptr) {
  if (!type_ptr) return 0;
  mlir::Type type = mlir::Type::getFromOpaquePointer(reinterpret_cast<void*>(type_ptr));
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType) return type_ptr; // Return original if not a tensor

  // Use tensorTypeInSpace from OnnxToHipsrUtils
  auto newType = tensorType.cloneWithEncoding(
      mlir::hipsr::MemorySpaceAttr::get(tensorType.getContext(), mlir::hipsr::MemorySpace::Device));

  return reinterpret_cast<uint64_t>(const_cast<void*>(newType.getAsOpaquePointer()));
}

//===----------------------------------------------------------------------===//
// MLIR Dialect Conversion Primitives
//===----------------------------------------------------------------------===//

// Helper: Populate Cast conversion patterns
// This is kept as a helper since it's a reusable component
uint64_t mlir_type_get_encoding(uint64_t type_ptr) {
  if (!type_ptr) return 0;
  mlir::Type type = mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(type_ptr));
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!tensorType) return 0;
  mlir::Attribute enc = tensorType.getEncoding();
  if (!enc) return 0;
  return reinterpret_cast<uint64_t>(enc.getAsOpaquePointer());
}

// Mark ONNX dialect illegal (except NoValueOp)
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
  Sregister_symbol("mlir_tensor_type_in_device_space", (void*)::mlir_tensor_type_in_device_space);
  Sregister_symbol("mlir_type_get_encoding", (void*)::mlir_type_get_encoding);
  Sregister_symbol("mlir_type_is_device_tensor", (void*)::mlir_type_is_device_tensor);
}

} // namespace hipsr
} // namespace mlir
