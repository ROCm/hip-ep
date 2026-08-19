/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- QDQMatMulFusion.cpp ------------------------------------------------===//
//
// PDLL-based fusion patterns for QDQ MatMul (AMDMIGraphX approach).
//
// The patterns are defined in PDLL/QDQMatMulFusion.pdll and compiled to
// MLIR bytecode at build time. This file loads and applies them at runtime.
//
// Build workflow:
//   QDQMatMulFusion.pdll --[mlir-pdll -x mlir]--> QDQMatMulFusion.pdl.mlir
//
// Runtime workflow:
//   Load .pdl.mlir bytecode -> Parse to PDLPatternModule -> Apply patterns
//
// This follows the AMDMIGraphX-Private approach for better flexibility:
// - Patterns can be updated without rebuilding (just replace .pdl.mlir file)
// - Cleaner separation between pattern definition and C++ code
// - Easier to add complex native rewrite callbacks if needed
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#ifdef ENABLE_PDLL_FUSION
#include "pdl/qdq_fusion_pass.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#endif

namespace mlir {
namespace hip {

void populateQDQMatMulFusionPatterns(RewritePatternSet &patterns) {
#ifdef ENABLE_PDLL_FUSION
    // Note: This function is called during pattern population, but
    // PDLL patterns work at the module level, not pattern-by-pattern.
    // 
    // The actual PDL pattern application happens in a separate pass
    // that loads the bytecode and applies it to the entire module.
    // 
    // For now, this is a placeholder. A proper integration would:
    // 1. Create a pass that calls hip::pdl::applyPDLPatterns()
    // 2. Register that pass in the pass pipeline
    // 3. The pass loads QDQMATMUL_FUSION_PDL_FILE at runtime
    //
    // TODO: Integrate PDL pattern loading into the conversion pipeline
    
    // Placeholder: C++ fallback pattern (for when PDLL is disabled)
    // In a full implementation, this would be replaced by PDL-only
#else
    // PDLL disabled - use C++ fallback pattern
    // (Original C++ pattern implementation would go here)
#endif
}

#ifdef ENABLE_PDLL_FUSION
// Example of how to apply PDL patterns at the module level:
// This would be called from a pass, not from populateQDQMatMulFusionPatterns
bool applyQDQFusionPDL(mlir::ModuleOp module) {
    constexpr const char* pdlFile = QDQMATMUL_FUSION_PDL_FILE;
    return ::hip::pdl::applyPDLPatterns(module, pdlFile);
}
#endif

} // namespace hip
} // namespace mlir
