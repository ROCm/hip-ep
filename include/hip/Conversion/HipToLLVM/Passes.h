//===- Passes.h - HIP to LLVM conversion pass declarations ------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_HIPTOLLVM_PASSES_H
#define HIP_CONVERSION_HIPTOLLVM_PASSES_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace mlir {
namespace hip {

#define GEN_PASS_DECL_CONVERTHIPTOLLVMPASS
#include "hip/Conversion/Passes.h.inc"

/// Populate \p patterns with all HIP-dialect lowerings to LLVM dialect.
/// External clients can compose this with their own type converter / patterns
/// to drive a partial conversion themselves; the in-tree
/// `ConvertHipToLLVMPass` calls this and then bundles standard
/// func/memref/arith/cf-to-LLVM patterns to minimize unrealized casts at the
/// memref/LLVM boundary.
void populateConvertHipToLLVMPatterns(const LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_HIPTOLLVM_PASSES_H
