/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Scheme/Bindings/SchemeMlirBindings.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"
#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "scheme-onnx-bindings"

// Note: scheme.h included via SchemeMlirBindings.h -> ChezSchemeInterpreter.h

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

extern "C" {

// onnx.Return → func.return conversion (belongs with ONNX patterns, not func dialect)
void mlir_populate_return_conversion_patterns(
    uint64_t converter_ptr, uint64_t patterns_ptr, uint64_t ctx_ptr) {
  if (!converter_ptr || !patterns_ptr || !ctx_ptr) return;
  mlir::hipsr::populateReturnConversionPatterns(
    *reinterpret_cast<mlir::TypeConverter*>(converter_ptr),
    *reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr),
    reinterpret_cast<mlir::MLIRContext*>(ctx_ptr));
}

void mlir_populate_constant_conversion_patterns(
    uint64_t converter_ptr, uint64_t patterns_ptr, uint64_t /*ctx_ptr*/) {
  if (!converter_ptr || !patterns_ptr) return;
  mlir::hipsr::populateOnnxToHipsrConstantPatterns(
    *reinterpret_cast<mlir::TypeConverter*>(converter_ptr),
    *reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr));
}

// Helper: Erase dead NoValue operations
void mlir_conversion_target_add_illegal_onnx(uint64_t target_ptr) {
  if (!target_ptr) return;
  auto* target = reinterpret_cast<mlir::ConversionTarget*>(target_ptr);
  target->addIllegalDialect<mlir::onnx::OnnxDialect>();
  target->addLegalOp<mlir::onnx::NoValueOp>();
}

// Mark HipSR dialect legal

} // extern "C"

namespace mlir {
namespace hipsr {

void registerOnnxBindings() {
  Sregister_symbol("mlir_populate_return_conversion_patterns",             (void*)::mlir_populate_return_conversion_patterns);
  Sregister_symbol("mlir_populate_matmul_conversion_patterns",            (void*)::mlir_populate_matmul_conversion_patterns);
  Sregister_symbol("mlir_populate_expand_conversion_patterns",            (void*)::mlir_populate_expand_conversion_patterns);
  Sregister_symbol("mlir_populate_min_conversion_patterns",               (void*)::mlir_populate_min_conversion_patterns);
  Sregister_symbol("mlir_populate_shape_conversion_patterns",             (void*)::mlir_populate_shape_conversion_patterns);
  Sregister_symbol("mlir_populate_reshape_conversion_patterns",           (void*)::mlir_populate_reshape_conversion_patterns);
  Sregister_symbol("mlir_populate_unsqueeze_conversion_patterns",         (void*)::mlir_populate_unsqueeze_conversion_patterns);
  Sregister_symbol("mlir_populate_equal_conversion_patterns",             (void*)::mlir_populate_equal_conversion_patterns);
  Sregister_symbol("mlir_populate_transpose_conversion_patterns",         (void*)::mlir_populate_transpose_conversion_patterns);
  Sregister_symbol("mlir_populate_gather_conversion_patterns",            (void*)::mlir_populate_gather_conversion_patterns);
  Sregister_symbol("mlir_populate_slice_conversion_patterns",             (void*)::mlir_populate_slice_conversion_patterns);
  Sregister_symbol("mlir_populate_scatter_nd_conversion_patterns",        (void*)::mlir_populate_scatter_nd_conversion_patterns);
  Sregister_symbol("mlir_populate_nonzero_conversion_patterns",           (void*)::mlir_populate_nonzero_conversion_patterns);
  Sregister_symbol("mlir_populate_constant_conversion_patterns",          (void*)::mlir_populate_constant_conversion_patterns);
}

} // namespace hipsr
} // namespace mlir
