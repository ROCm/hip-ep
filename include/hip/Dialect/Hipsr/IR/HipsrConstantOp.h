/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_CONSTANT_OP_H
#define HIPSR_CONSTANT_OP_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cassert>

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.h.inc"

namespace mlir {
// Forward-declared to keep the LLVM-conversion headers out of this widely
// included op header; only the pattern definition in HipsrConstantOp.cpp needs
// their full definitions.
class LLVMTypeConverter;
class RewritePatternSet;

namespace hipsr {

// Out-of-line template: needs the complete ConstantOp / attribute / dialect
// types declared above.
//
// CONTRACT: MorphiZen-generated constants only. Works both before and after
// externalization (value/source are never removed -- offset/size only mark the
// constants-file location). Fails fast on forms MorphiZen never emits
// (splat-optimized DenseElementsAttr, DenseResourceElementsAttr); add real APIs
// if that changes.
template <typename T>::llvm::ArrayRef<T> ConstantOp::getDataValues() {
  // Inline dense value. MorphiZen writes FULL tensor data
  // (numElements * elemByteWidth), never splat-optimized.
  if (auto dense =
          ::llvm::dyn_cast_or_null<::mlir::DenseElementsAttr>(getValueAttr())) {
    if (dense.isSplat()) {
      llvm_unreachable("splat DenseElementsAttr not supported "
                       "(MorphiZen never emits splat-optimized constants)");
    }
    ::llvm::ArrayRef<char> raw = dense.getRawData();
    return ::llvm::ArrayRef<T>(reinterpret_cast<const T *>(raw.data()),
                               raw.size() / sizeof(T));
  }

  // Resource-backed inline value: not emitted by MorphiZen.
  if (::llvm::isa_and_nonnull<::mlir::DenseResourceElementsAttr>(
          getValueAttr())) {
    llvm_unreachable("DenseResourceElementsAttr not supported "
                     "(MorphiZen never emits this)");
  }

  // External source: raw pointer (mem) or memory-mapped file (cached).
  if (::mlir::Attribute src = getSourceAttr()) {
    if (auto mem = ::llvm::dyn_cast<MemSourceAttr>(src)) {
      const T *p =
          reinterpret_cast<const T *>(static_cast<uintptr_t>(mem.getAddress()));
      return ::llvm::ArrayRef<T>(p, mem.getSize() / sizeof(T));
    }
    if (auto file = ::llvm::dyn_cast<FileSourceAttr>(src)) {
      auto *dialect = getContext()->getLoadedDialect<HipsrDialect>();
      assert(dialect && "HipsrDialect not loaded");
      ::llvm::MemoryBuffer *buf =
          dialect->getOrLoadFileMap(file.getPath().getValue());
      assert(buf && "failed to load file for FileSourceAttr");
      const T *p =
          reinterpret_cast<const T *>(buf->getBufferStart() + file.getOffset());
      return ::llvm::ArrayRef<T>(p, file.getSize() / sizeof(T));
    }
  }

  // The verifier guarantees value XOR source, so this is unreachable.
  llvm_unreachable("ConstantOp has neither value nor source");
}

/// Registers the hipsr.constant -> LLVM lowering pattern. Defined out-of-line
/// in HipsrConstantOp.cpp so the op owns its lowering; aggregated by
/// convert-hip-to-llvm through HipsrDialect's ConvertToLLVMPatternInterface.
void populateHipsrConstantLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_CONSTANT_OP_H
