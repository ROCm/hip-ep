/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_SCHEMEMLIR_BINDINGS_H
#define LIB_DIALECT_HIPSR_SCHEME_SCHEMEMLIR_BINDINGS_H

#include "hip/Dialect/Hipsr/Scheme/Runtime/ChezSchemeInterpreter.h"

namespace mlir {
class Operation;
class Value;
class Type;
class Attribute;

namespace hipsr {

// MLIR C++ to Scheme conversions - wrap MLIR objects as foreign pointers
ptr makeSchemeOperation(mlir::Operation* op);
ptr makeSchemeValue(mlir::Value val);
ptr makeSchemeType(mlir::Type type);
ptr makeSchemeAttribute(mlir::Attribute attr);

// Register all MLIR foreign functions accessible from Scheme
void registerMlirForeignFunctions();

} // namespace hipsr
} // namespace mlir

#endif
