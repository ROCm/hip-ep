/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"

#define DEBUG_TYPE "scheme-func-bindings"

// Note: scheme.h included via SchemeMlirBindings.h -> ChezSchemeInterpreter.h

extern "C" {

void mlir_populate_return_conversion_patterns(
    uint64_t converter_ptr, uint64_t patterns_ptr, uint64_t ctx_ptr) {
  if (!converter_ptr) {
    llvm::errs() << "[SCHEME FFI ERROR] mlir_populate_return_conversion_patterns: converter_ptr is null!\n";
    return;
  }
  if (!patterns_ptr) {
    llvm::errs() << "[SCHEME FFI ERROR] mlir_populate_return_conversion_patterns: patterns_ptr is null!\n";
    return;
  }
  if (!ctx_ptr) {
    llvm::errs() << "[SCHEME FFI ERROR] mlir_populate_return_conversion_patterns: ctx_ptr is null!\n";
    return;
  }

  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);
  auto* patterns = reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr);
  auto* ctx = reinterpret_cast<mlir::MLIRContext*>(ctx_ptr);

  mlir::hipsr::populateReturnConversionPatterns(*converter, *patterns, ctx);
}

// Helper: Populate FuncOp type conversion pattern
void mlir_populate_func_type_conversion_pattern(
    uint64_t patterns_ptr, uint64_t converter_ptr) {
  if (!patterns_ptr || !converter_ptr) return;

  auto* patterns = reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr);
  auto* converter = reinterpret_cast<mlir::TypeConverter*>(converter_ptr);

  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(*patterns, *converter);
}

// Helpers: Populate conversion patterns for remaining ONNX ops
#define DEFINE_POPULATE_PATTERNS(name, fn) \
  void name(uint64_t converter_ptr, uint64_t patterns_ptr, uint64_t ctx_ptr) { \
    if (!converter_ptr || !patterns_ptr || !ctx_ptr) return; \
    mlir::hipsr::fn( \
      *reinterpret_cast<mlir::TypeConverter*>(converter_ptr), \
      *reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr), \
      reinterpret_cast<mlir::MLIRContext*>(ctx_ptr)); \
  }

DEFINE_POPULATE_PATTERNS(mlir_populate_matmul_conversion_patterns,    populateMatMulConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_expand_conversion_patterns,     populateExpandConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_min_conversion_patterns,        populateMinConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_shape_conversion_patterns,      populateShapeConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_reshape_conversion_patterns,    populateReshapeConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_unsqueeze_conversion_patterns,  populateUnsqueezeConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_equal_conversion_patterns,      populateEqualConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_transpose_conversion_patterns,  populateTransposeConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_gather_conversion_patterns,     populateGatherConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_slice_conversion_patterns,      populateSliceConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_scatter_nd_conversion_patterns, populateScatterNDConversionPatterns)
DEFINE_POPULATE_PATTERNS(mlir_populate_nonzero_conversion_patterns,    populateNonZeroConversionPatterns)

#undef DEFINE_POPULATE_PATTERNS

// Constant patterns take no ctx (type converter only)

} // extern "C"

namespace mlir {
namespace hipsr {

void registerFuncBindings() {
  Sregister_symbol("mlir_populate_return_conversion_patterns", (void*)::mlir_populate_return_conversion_patterns);
  Sregister_symbol("mlir_populate_func_type_conversion_pattern", (void*)::mlir_populate_func_type_conversion_pattern);
}

} // namespace hipsr
} // namespace mlir
