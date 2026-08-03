/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {

void populateReduceL2ConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<OnnxReductionToHip<mlir::hip::ReduceL2Op>>(ctx, "onnx.ReduceL2");
}

} // namespace hip
} // namespace mlir
