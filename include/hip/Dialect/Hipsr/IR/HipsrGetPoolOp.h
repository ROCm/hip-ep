/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_GET_POOL_OP_H
#define HIPSR_GET_POOL_OP_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;

namespace hipsr {

void populateHipsrGetPoolLoweringPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_GET_POOL_OP_H
