/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_GGML_TRANSFORMS_PASSES_H
#define HIP_DIALECT_GGML_TRANSFORMS_PASSES_H

#include "hip/Dialect/Ggml/IR/GgmlDialect.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ggml {

#define GEN_PASS_DECL_GGMLSYMBOLIZESHAPESPASS
#include "hip/Dialect/Ggml/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hip/Dialect/Ggml/Transforms/Passes.h.inc"

} // namespace ggml
} // namespace mlir

#endif // HIP_DIALECT_GGML_TRANSFORMS_PASSES_H
