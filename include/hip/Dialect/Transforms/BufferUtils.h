/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- BufferUtils.h - Shared buffer analysis utilities -------------------===//
//
// Utilities shared by PoolAllocs, OptimizeMemRefs, and LowerAllocs.
// Centralizes byte-size computation, alignment, and alias-aware liveness
// queries using MLIR's BufferViewFlowAnalysis.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_TRANSFORMS_BUFFERUTILS_H
#define HIP_DIALECT_TRANSFORMS_BUFFERUTILS_H

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace hip {

/// Whether static strides describe one flat run. Unit dimensions span no
/// distance, so their strides do not affect contiguity.
inline bool isContiguousRowMajor(ArrayRef<int64_t> shape,
                                 ArrayRef<int64_t> strides) {
  if (shape.size() != strides.size())
    return false;
  int64_t expected = 1;
  for (auto [extent, stride] : llvm::reverse(llvm::zip(shape, strides))) {
    if (extent < 0)
      return false;
    if (extent == 1)
      continue;
    if (stride != expected || llvm::MulOverflow(expected, extent, expected))
      return false;
  }
  return true;
}

/// Bytes one element of \p type occupies in memory. The element bit width is
/// rounded up to whole bytes, so a sub-byte element such as i1 occupies one
/// byte. Runtime kernels store one byte per i1 element (Less/And/Not/Where
/// masks); sizing an allocation from its total bit count instead would pack
/// i1 eight-per-byte and let such a write spill into the next live buffer.
int64_t getElementByteSize(MemRefType type);

/// Byte size for a fully-static memref type; returns 0 when any dim is dynamic.
/// Computed from getElementByteSize, so i1 memrefs get one byte per element.
int64_t getStaticByteSize(MemRefType type);

/// Emit arith ops for: llvm::alignTo(value, alignment).
/// Produces: ((value + alignment - 1) / alignment) * alignment.
/// Returns \p value unchanged if alignment <= 1.
Value emitAlignUp(OpBuilder &builder, Location loc, Value value,
                  int64_t alignment);

/// Find the highest block-local operation index among all transitive users
/// of \p allocResult, following view-like aliases via \p aliasAnalysis.
/// memref.dealloc ops are excluded so that lifetimes reflect data usage,
/// not administrative cleanup.
///
/// Users in nested regions are resolved to their ancestor in \p block
/// via Block::findAncestorOpInBlock.
unsigned findLastAliasedUseIndex(Value allocResult,
                                 const BufferViewFlowAnalysis &aliasAnalysis,
                                 Block &block,
                                 const DenseMap<Operation *, unsigned> &opIndex,
                                 unsigned blockSize);

/// Find the last Operation* among all transitive users of \p allocResult,
/// following view-like aliases via \p aliasAnalysis.  Resolves nested-region
/// users to their ancestor in \p entryBlock.
Operation *findLastAliasedUser(Value allocResult,
                               const BufferViewFlowAnalysis &aliasAnalysis,
                               Block &entryBlock);

/// Return true if any value in the transitive alias set of \p root
/// (as determined by \p aliasAnalysis) is contained in \p valueSet.
bool isAliasInSet(Value root, const BufferViewFlowAnalysis &aliasAnalysis,
                  const DenseSet<Value> &valueSet);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_TRANSFORMS_BUFFERUTILS_H
