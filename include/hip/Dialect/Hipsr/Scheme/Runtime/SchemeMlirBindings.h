/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef LIB_DIALECT_HIPSR_SCHEME_SCHEMEMLIR_BINDINGS_H
#define LIB_DIALECT_HIPSR_SCHEME_SCHEMEMLIR_BINDINGS_H

#include "hip/Dialect/Hipsr/Scheme/Runtime/ChezSchemeInterpreter.h"

namespace mlir {
namespace hipsr {

// Register all MLIR foreign functions accessible from Scheme
void registerMlirForeignFunctions();

} // namespace hipsr
} // namespace mlir

#endif
