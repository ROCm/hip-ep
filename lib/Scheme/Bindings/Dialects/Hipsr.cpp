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

// Mirrors kOrtMemAddrTag in OnnxToHip.cpp.
static constexpr llvm::StringLiteral kOrtMemAddrTag = "*/_ORT_MEM_ADDR_/*";

// Create hipsr.constant from ORT in-memory data (zero-copy).
// addr_as_i64: raw memory address encoded as int64. size: byte count.
// Returns result Value* as uptr, or 0 on failure.
uint64_t mlir_build_hipsr_constant_from_ort_mem(
    uint64_t rewriter_ptr, uint64_t loc_op_ptr, uint64_t result_type_ptr,
    int64_t addr_as_i64, int64_t size) {
  auto *rewriter  = reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr);
  auto *loc_op    = reinterpret_cast<mlir::Operation*>(loc_op_ptr);
  auto  baseType  = mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(result_type_ptr));
  auto  resultType = llvm::dyn_cast<mlir::RankedTensorType>(baseType);
  if (!resultType) return 0;

  rewriter->setInsertionPoint(loc_op);
  std::string key = "mem|0x" + llvm::utohexstr(static_cast<uint64_t>(addr_as_i64), /*LowerCase=*/true);
  llvm::ArrayRef<char> data = {
      reinterpret_cast<const char*>(static_cast<uintptr_t>(addr_as_i64)),
      static_cast<size_t>(size)};
  auto value = mlir::DenseResourceElementsAttr::get(
      resultType, key,
      mlir::UnmanagedAsmResourceBlob::allocateInferAlign(data));
  auto *newOp = rewriter->create<mlir::hipsr::ConstantOp>(
      loc_op->getLoc(), resultType, value,
      mlir::IntegerAttr(), mlir::IntegerAttr(), mlir::IntegerAttr());
  return reinterpret_cast<uint64_t>(newOp->getResult(0).getAsOpaquePointer());
}

// Create hipsr.constant from a file-backed memory-mapped resource.
// Returns result Value* as uptr, or 0 if the file cannot be memory-mapped.
uint64_t mlir_build_hipsr_constant_from_file(
    uint64_t rewriter_ptr, uint64_t loc_op_ptr, uint64_t result_type_ptr,
    const char* location, int64_t offset, int64_t size) {
  auto *rewriter  = reinterpret_cast<mlir::RewriterBase*>(rewriter_ptr);
  auto *loc_op    = reinterpret_cast<mlir::Operation*>(loc_op_ptr);
  auto  baseType  = mlir::Type::getFromOpaquePointer(reinterpret_cast<const void*>(result_type_ptr));
  auto  resultType = llvm::dyn_cast<mlir::RankedTensorType>(baseType);
  if (!resultType) return 0;

  auto *dialect = loc_op->getContext()->getLoadedDialect<mlir::hipsr::HipsrDialect>();
  llvm::MemoryBuffer *buf = dialect->getOrLoadFileMap(location);
  if (!buf) return 0;

  rewriter->setInsertionPoint(loc_op);
  std::string key = (llvm::Twine("file|") + location + "|" + llvm::Twine(offset)).str();
  llvm::ArrayRef<char> data = {buf->getBufferStart() + offset, static_cast<size_t>(size)};
  auto value = mlir::DenseResourceElementsAttr::get(
      resultType, key,
      mlir::UnmanagedAsmResourceBlob::allocateInferAlign(data));
  auto *newOp = rewriter->create<mlir::hipsr::ConstantOp>(
      loc_op->getLoc(), resultType, value,
      mlir::IntegerAttr(), mlir::IntegerAttr(), mlir::IntegerAttr());
  return reinterpret_cast<uint64_t>(newOp->getResult(0).getAsOpaquePointer());
}

} // extern "C"

namespace mlir {
namespace hipsr {

void registerHipsrBindings() {
  Sregister_symbol("mlir_populate_cast_conversion_patterns", (void*)::mlir_populate_cast_conversion_patterns);
  Sregister_symbol("mlir_placeholder_set_barrier_type", (void*)::mlir_placeholder_set_barrier_type);
  Sregister_symbol("mlir_get_hipsr_context_type", (void*)::mlir_get_hipsr_context_type);
  Sregister_symbol("mlir_build_hipsr_constant_from_ort_mem",  (void*)::mlir_build_hipsr_constant_from_ort_mem);
  Sregister_symbol("mlir_build_hipsr_constant_from_file",     (void*)::mlir_build_hipsr_constant_from_file);
}

} // namespace hipsr
} // namespace mlir
