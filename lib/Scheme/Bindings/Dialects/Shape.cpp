/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#define DEBUG_TYPE "scheme-shape-bindings"

// Note: scheme.h included via SchemeMlirBindings.h -> ChezSchemeInterpreter.h

extern "C" {

// Return mlir::shape::ShapeType (the !shape.shape type) for the given context.
// ctx_ptr: MLIRContext* as uptr
// Returns: ShapeType as opaque type uptr, or 0 if ctx_ptr is null
uint64_t mlir_get_shape_shape_type(uint64_t ctx_ptr) {
  if (!ctx_ptr) return 0;
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(
      mlir::shape::ShapeType::get(ctx).getAsOpaquePointer());
}

// Return mlir::shape::SizeType (the !shape.size type) for the given context.
// ctx_ptr: MLIRContext* as uptr
// Returns: SizeType as opaque type uptr, or 0 if ctx_ptr is null
uint64_t mlir_get_shape_size_type(uint64_t ctx_ptr) {
  if (!ctx_ptr) return 0;
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(
      mlir::shape::SizeType::get(ctx).getAsOpaquePointer());
}

// Return mlir::shape::WitnessType (the !shape.witness type) for the given context.
// ctx_ptr: MLIRContext* as uptr
// Returns: WitnessType as opaque type uptr, or 0 if ctx_ptr is null
uint64_t mlir_get_shape_witness_type(uint64_t ctx_ptr) {
  if (!ctx_ptr) return 0;
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(
      mlir::shape::WitnessType::get(ctx).getAsOpaquePointer());
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerShapeBindings() {
  Sregister_symbol("mlir_get_shape_shape_type",   (void*)::mlir_get_shape_shape_type);
  Sregister_symbol("mlir_get_shape_size_type",    (void*)::mlir_get_shape_size_type);
  Sregister_symbol("mlir_get_shape_witness_type", (void*)::mlir_get_shape_witness_type);
}

} // namespace hipsr
} // namespace mlir
