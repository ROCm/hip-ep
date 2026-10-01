/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"


#define DEBUG_TYPE "scheme-hipsr-bindings"

// Note: scheme.h included via SchemeMlirBindings.h -> ChezSchemeInterpreter.h

extern "C" {

// mlir_get_hipsr_context_arg was removed: the Scheme implementation in
// (mlir hipsr helpers) using mlir-operation-get-block-argument is sufficient.
void mlir_populate_cast_conversion_patterns(
    uint64_t converter_ptr, uint64_t patterns_ptr, uint64_t ctx_ptr) {
  if (!converter_ptr || !patterns_ptr || !ctx_ptr) return;

  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);
  auto* patterns = reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr);
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);

  mlir::hipsr::populateCastConversionPatterns(*converter, *patterns, ctx);
}

// Create a RewritePatternSet
// Returns RewritePatternSet* as uint64_t (opaque handle for Scheme)
void mlir_placeholder_set_barrier_type(uint64_t op_ptr) {
  if (!op_ptr) return;
  auto* op = reinterpret_cast<mlir::Operation*>(op_ptr);
  op->setAttr("placeholder_type",
      mlir::hipsr::PlaceholderTypeAttr::get(op->getContext(),
                                             mlir::hipsr::PlaceholderType::Barrier));
}

// Copy a named attribute from src_op to dst_op. No-op if attr is absent on src.

uint64_t mlir_get_hipsr_context_type(uint64_t ctx_ptr) {
  if (!ctx_ptr) return 0;
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(
      mlir::hipsr::ContextType::get(ctx).getAsOpaquePointer());
}

// Create a HipSR device memory space attribute.
uint64_t mlir_hipsr_make_device_space_attr(uint64_t ctx_ptr) {
  auto *ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  return reinterpret_cast<uint64_t>(
      mlir::hipsr::MemorySpaceAttr::get(ctx, mlir::hipsr::MemorySpace::Device)
          .getAsOpaquePointer());
}

// mlir_hipsr_load_file_map — HipSR-specific: memory-map a file via HipsrDialect.
// Returns the buffer start address as uptr, or 0 if the file cannot be mapped.
uint64_t mlir_hipsr_load_file_map(uint64_t ctx_ptr, const char* path) {
  auto *ctx     = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);
  auto *dialect = ctx->getLoadedDialect<mlir::hipsr::HipsrDialect>();
  llvm::MemoryBuffer *buf = dialect->getOrLoadFileMap(path);
  if (!buf) return 0;
  return reinterpret_cast<uint64_t>(buf->getBufferStart());
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerHipsrBindings() {
  Sregister_symbol("mlir_populate_cast_conversion_patterns", (void*)::mlir_populate_cast_conversion_patterns);
  Sregister_symbol("mlir_placeholder_set_barrier_type", (void*)::mlir_placeholder_set_barrier_type);
  Sregister_symbol("mlir_get_hipsr_context_type", (void*)::mlir_get_hipsr_context_type);
  Sregister_symbol("mlir_hipsr_make_device_space_attr", (void*)::mlir_hipsr_make_device_space_attr);
  Sregister_symbol("mlir_hipsr_load_file_map",          (void*)::mlir_hipsr_load_file_map);
}

} // namespace hipsr
} // namespace mlir
