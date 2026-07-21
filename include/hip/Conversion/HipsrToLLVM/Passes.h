/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPSRTOLLVM_PASSES_H
#define HIP_CONVERSION_HIPSRTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace hipsr {

// Per-conversion declaration. createConvertHipsrToLLVMPass() is declared by
// GEN_PASS_DECL (defined by GEN_PASS_DEF in HipsrToLLVM.cpp). Registration
// lives in the aggregate hip/Conversion/Passes.h.
#define GEN_PASS_DECL_CONVERTHIPSRTOLLVMPASS
#include "hip/Conversion/Passes.h.inc"

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_HIPSRTOLLVM_PASSES_H
