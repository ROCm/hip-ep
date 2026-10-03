/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
void populateReduceMinConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<OnnxReductionToHip<mlir::hip::ReduceMinOp>>(ctx,
                                                           "onnx.ReduceMin");
}

} // namespace hip
} // namespace mlir
