/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {

void populateReduceSumConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<OnnxReductionToHip<mlir::hip::ReduceSumOp>>(ctx,
                                                           "onnx.ReduceSum");
}

} // namespace hip
} // namespace mlir
