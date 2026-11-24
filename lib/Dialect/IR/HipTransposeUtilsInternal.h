/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_TRANSPOSE_UTILS_INTERNAL_H
#define HIP_DIALECT_IR_HIP_TRANSPOSE_UTILS_INTERNAL_H

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::hip::detail {

/// The `perm` attribute as integers. Fails on anything the verifier would have
/// rejected, so a malformed op is left alone rather than rewritten on a guess.
bool getPermutation(TransposeOp transpose, SmallVectorImpl<int64_t> &perm);

/// Whether permuting `source` by `perm` gives `result`'s shape and element
/// type. A valid permutation value can still disagree with the declared tensor
/// types, so a rewrite built from them must reject that case. A dimension
/// dynamic on either side cannot be compared and is accepted.
bool permutesIntoResultType(Value source, Value result,
                            ArrayRef<int64_t> perm);

/// Add the pattern that turns a transpose moving only size-1 dimensions into a
/// reshape (HipTransposeUnitDimCanonicalize.cpp). Called from
/// TransposeOp::getCanonicalizationPatterns, the op's single registration point.
void populateUnitDimTransposePatterns(RewritePatternSet &patterns);

} // namespace mlir::hip::detail

#endif // HIP_DIALECT_IR_HIP_TRANSPOSE_UTILS_INTERNAL_H
