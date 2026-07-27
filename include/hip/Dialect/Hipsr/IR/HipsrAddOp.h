/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_ADD_OP_H
#define HIPSR_ADD_OP_H

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;

namespace hipsr {

void populateHipsrAddLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_ADD_OP_H
