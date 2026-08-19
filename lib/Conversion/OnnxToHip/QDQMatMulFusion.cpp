/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- QDQMatMulFusion.cpp ------------------------------------------------===//
//
// PDLL-based fusion pattern for QDQ MatMul.
//
// The pattern is defined declaratively in PDLL/QDQMatMulFusion.pdll and
// compiled to C++ during the build process:
//
//   QDQMatMulFusion.pdll --[mlir-pdll]--> QDQMatMulFusion.cpp.inc
//
// This demonstrates how PDLL can replace hand-written C++ pattern matching
// code with a declarative pattern description language.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"
#include "mlir/Parser/Parser.h"

// Include PDLL-generated pattern code
// Generated during build by CMake custom command
#include "QDQMatMulFusion.cpp.inc"

namespace mlir {
namespace hip {

void populateQDQMatMulFusionPatterns(RewritePatternSet &patterns) {
  // Use PDLL-generated pattern registration function
  // This replaces the hand-written C++ pattern that was here before
  populateGeneratedPDLLPatterns(patterns);
}

} // namespace hip
} // namespace mlir
