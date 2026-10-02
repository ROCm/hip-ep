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

// DEFINE_POPULATE_PATTERNS generates populate_* FFI functions.
// Each generated function adds C++ conversion patterns for one ONNX op to a
// RewritePatternSet, using the provided TypeConverter and MLIRContext.
// All generated functions share the same signature:
//   converter_ptr: TypeConverter* as uptr
//   patterns_ptr:  RewritePatternSet* as uptr — patterns are appended in-place
//   ctx_ptr:       MLIRContext* as uptr
// None of them return a value; they are no-ops if any pointer is null.
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

// Populate the onnx.Return → func.return conversion pattern.
// Placed here (not in Func.cpp) because onnx.Return is an ONNX-dialect concept.
// converter_ptr: TypeConverter* as uptr
// patterns_ptr:  RewritePatternSet* as uptr
// ctx_ptr:       MLIRContext* as uptr
void mlir_populate_return_conversion_patterns(
    uint64_t converter_ptr, uint64_t patterns_ptr, uint64_t ctx_ptr) {
  if (!converter_ptr || !patterns_ptr || !ctx_ptr) return;
  mlir::hipsr::populateReturnConversionPatterns(
    *reinterpret_cast<mlir::TypeConverter*>(converter_ptr),
    *reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr),
    reinterpret_cast<mlir::MLIRContext*>(ctx_ptr));
}

// Populate C++ patterns for onnx.Constant lowering (external data path).
// The Scheme patterns handle the inline value path; these C++ patterns are the
// fallback for DenseResourceElementsAttr from file or ORT memory-mapped data.
// converter_ptr: TypeConverter* as uptr
// patterns_ptr:  RewritePatternSet* as uptr
// ctx_ptr:       unused (context is derived from the pattern set)
void mlir_populate_constant_conversion_patterns(
    uint64_t converter_ptr, uint64_t patterns_ptr, uint64_t /*ctx_ptr*/) {
  if (!converter_ptr || !patterns_ptr) return;
  mlir::hipsr::populateOnnxToHipsrConstantPatterns(
    *reinterpret_cast<mlir::TypeConverter*>(converter_ptr),
    *reinterpret_cast<mlir::RewritePatternSet*>(patterns_ptr));
}


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
