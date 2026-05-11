//===- Passes.h - ONNX to HIP conversion pass declarations ------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIP_PASSES_H
#define HIP_CONVERSION_ONNXTOHIP_PASSES_H

#include "hip/Dialect/Transforms/Pipelines.h"

#include <memory>

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;
} // namespace mlir

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir {
namespace hip {

#define GEN_PASS_DECL_CONVERTONNXTOHIPPASS
#include "hip/Conversion/Passes.h.inc"

/// Creates a pass with an external FileSystem for constant storage.
/// When \p fs is non-null, externalized constants are written via \p fs
/// instead of a DiskFileSystem rooted at the output directory.
///
/// This overload exists in addition to the TableGen-generated
/// `createConvertOnnxToHipPass()` because the FileSystem pointer cannot
/// be expressed as a TableGen `Option`.
std::unique_ptr<Pass> createConvertOnnxToHipPass(
    morphizen::FileSystem* fs,
    int64_t minNumElements = kDefaultExternalizeMinNumElements,
    bool skipConstantData = false);

/// Populate \p patterns with all ONNX-to-HIP compute-op rewrite patterns
/// (MatMul, Conv, Norm, Gqa, ...).  External clients can call this to drive
/// the conversion through their own greedy/dialect-conversion driver
/// without instantiating the full `ConvertOnnxToHipPass`.  The pass itself
/// uses these patterns plus its own bookkeeping for constants and module
/// metadata.
void populateConvertOnnxToHipPatterns(RewritePatternSet& patterns,
                                      MLIRContext* ctx);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_PASSES_H
