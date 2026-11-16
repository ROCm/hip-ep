/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPTOLLVM_RUNTIMEACCESS_H
#define HIP_CONVERSION_HIPTOLLVM_RUNTIMEACCESS_H

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::hip {

// Native runtime guarantees do not apply to an undeclared alternative.
class RuntimeAccessOverrides {
public:
  explicit RuntimeAccessOverrides(RewritePatternSet &alternatives) {
    for (const auto &pattern : alternatives.getNativePatterns()) {
      if (auto root = pattern->getRootKind())
        roots.insert(*root);
      else
        unknownRoot = true;
    }
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
    unknownRoot |= bool(alternatives.getPDLPatterns().getModule());
#endif
  }

  bool contains(OperationName name) const {
    return unknownRoot || roots.contains(name);
  }

private:
  DenseSet<OperationName> roots;
  bool unknownRoot = false;
};

} // namespace mlir::hip

#endif
