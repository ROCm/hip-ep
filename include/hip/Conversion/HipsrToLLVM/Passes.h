/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPSRTOLLVM_PASSES_H
#define HIP_CONVERSION_HIPSRTOLLVM_PASSES_H

namespace mlir {
class DialectRegistry;
namespace hipsr {

// Attach the ConvertToLLVMPatternInterface to HipsrDialect so the standard LLVM
// conversion driver discovers and applies hipsr lowering without a dedicated
// pass. Call when populating the DialectRegistry, before conversion runs.
void registerConvertHipsrToLLVMInterface(DialectRegistry &registry);

} // namespace hipsr
} // namespace mlir

#endif // HIP_CONVERSION_HIPSRTOLLVM_PASSES_H
